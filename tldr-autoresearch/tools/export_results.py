#!/usr/bin/env python3
"""Derive the small result tables in results/ from a run's data directory. Stdlib only; read-only on data/.

    python3 tools/export_results.py [--data DIR] [--out DIR] [--until 2026-10-04T12:00:00Z]

--data defaults to $RLTLDR_ROOT/data ($RLTLDR_ROOT defaults to the checkout that contains tools/), --out to
results/ next to tools/. --until drops everything that finished after that UTC time, so a snapshot of a run that
is still going (the frozen-v5 ablation) can be re-derived exactly.

Inputs (all under --data):
  RL run (training harness; policy updated every 8 experiments)
    attempts/*/rollout.json   one per experiment (driver): runs, keep decision, policy version, insight
    ledger.jsonl, calib_ledger.jsonl   trusted runs (runner); the baseline calibration runs
    metrics_trainer.jsonl     one line per policy update (trainer)
    insight_blocklist.json    insights retracted after the fact (never trained on)
  head-to-head (h2h): two continuous upstream-style pi sessions, base model vs RL policy v5
    h2h/{base,v5}/ledger.jsonl     trusted runs: the only source of val_bpb and validity
    h2h/{base,v5}/repo/results.tsv the agent's own log (untrusted): used only for its keep/discard decisions,
                                   mapped onto runs by git commit (the ledger records the agent's HEAD per run)
    h2h/{base,v5}/status.json      supervisor start time
  frozen-v5 ablation (frz): the training harness with v5 frozen, x1 = insights on, x2 = insights off
    h2h/{x1,x2}/rl/{attempts/*/rollout.json,ledger.jsonl,calib_ledger.jsonl}, h2h/{x1,x2}/rl_config.json

Outputs (--out): rl_run_experiments.tsv, rl_run_insights.tsv, h2h_runs.tsv, frz_runs.tsv, summary.tsv.
val_bpb is bits per byte on autoresearch's pinned validation shard (prepare.evaluate_bpb, computed by the trusted
eval bootstrap); lower is better.
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import stat
import statistics
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("RLTLDR_ROOT") or CODE

sys.dont_write_bytecode = True   # importing ar_run must not leave __pycache__ in the checkout
try:                        # the harness's own definition of an invalid measurement
    sys.path.insert(0, os.path.join(CODE, "tools"))
    from ar_run import TRUSTED_FATAL  # noqa: E402
    INVALID_FLAGS = set(TRUSTED_FATAL) | {"prepare_py_modified"}
except Exception:           # pragma: no cover - same list as tools/h2h_dashboard/build.py
    INVALID_FLAGS = {"no_trusted_eval", "multiple_evals", "val_leak", "non_causal", "eval_unverifiable",
                     "over_time_budget", "prepare_py_modified"}

H2H_ARMS = ("base", "v5")
FRZ_ARMS = ("x1", "x2")
# The kept g0002-a6 put torch.compile(mode="max-autotune-no-cudagraphs") into train.py, which overrides the
# harness's no-autotune setting; run-to-run noise is estimated separately before and after it.
AUTOTUNE_FROM = "g0002-a6"
TSV_MAX = 4 << 20           # an agent-written results.tsv larger than this is ignored
HEX = re.compile(r"[0-9a-f]{4,40}")


# ---------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------
def jl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def rj(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def epoch(x):
    """ISO-8601 string or epoch number -> epoch seconds (None if unparsable)."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return dt.datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else ""


def f6(x):
    return f"{x:.6f}" if isinstance(x, (int, float)) else ""


def f2(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else ""


def clean(s, n=300):
    return re.sub(r"\s+", " ", "" if s is None else str(s)).strip()[:n]


def first_line_json(path):
    try:
        with open(path) as f:
            return json.loads(f.readline())
    except (OSError, ValueError):
        return None


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact test p for the 2x2 table [[a, b], [c, d]]."""
    r1, c1, n = a + b, a + c, a + b + c + d

    def p(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


def write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(clean(r.get(h, ""), 400) for h in header) + "\n")


def valid_run(e):
    return (e.get("status") == "ok" and isinstance(e.get("val_bpb"), (int, float))
            and not INVALID_FLAGS.intersection(e.get("flags") or []))


# ---------------------------------------------------------------------------------------------------------
# training-harness runs (RL run and frozen-v5 arms share the driver's rollout.json format)
# ---------------------------------------------------------------------------------------------------------
def load_attempts(data, until):
    """One row per finished experiment, in order; hours are measured from the first experiment's pi session."""
    rows, t0 = [], None
    for p in sorted(glob.glob(os.path.join(data, "attempts", "*", "rollout.json"))):
        r = rj(p)
        if not r or not r.get("t_end") or (until and r["t_end"] > until):
            continue
        sess = first_line_json(os.path.join(os.path.dirname(p), "events.jsonl")) or {}
        t_start = epoch(sess.get("timestamp"))
        if t0 is None and t_start:
            t0 = t_start
        parent = r.get("parent") or {}
        keep = r.get("status") == "keep"
        out = r.get("outcome") or {}
        rows.append({
            "attempt_id": r["attempt_id"], "group": r.get("group"), "k": r.get("k"),
            "policy": f"v{(r.get('policy') or {}).get('version', '?')}",
            "conditioned": int(bool(r.get("conditioned"))),
            "insights_in_context": len(r.get("insights_in_context") or []),
            "screen_val_bpb": r.get("val_bpb"), "confirm_val_bpb": r.get("confirm_mean"),
            "best_before": parent.get("val_bpb"),
            "decision": r.get("status"), "reward": r.get("reward"), "success": bool(r.get("success")),
            "best_after": r.get("confirm_mean") if keep else parent.get("val_bpb"),
            "flags": ",".join(out.get("flags") or []),
            "num_steps": (out.get("metrics") or {}).get("num_steps"),
            "t_end": r["t_end"], "agent_wall_s": (r.get("pi") or {}).get("wall_s"),
            "insight_wall_s": (r.get("insight") or {}).get("wall_s"),
            "hint": (r.get("insight") or {}).get("hint"),
            "context": r.get("insights_in_context") or [],
            "desc": r.get("desc"),
        })
    for x in rows:
        x["hours"] = (x["t_end"] - t0) / 3600 if t0 else None
    return rows, t0


def attempt_table(rows, arm=None):
    out = []
    for i, x in enumerate(rows, 1):
        d = {"n": i, "attempt_id": x["attempt_id"], "group": x["group"], "k": x["k"], "policy": x["policy"],
             "insights_in_context": x["insights_in_context"], "screen_val_bpb": f6(x["screen_val_bpb"]),
             "confirm_val_bpb": f6(x["confirm_val_bpb"]), "best_before": f6(x["best_before"]),
             "decision": x["decision"], "reward": x["reward"], "best_after": f6(x["best_after"]),
             "num_steps": x["num_steps"], "flags": x["flags"], "t_end_utc": iso(x["t_end"]),
             "hours": f2(x["hours"]), "agent_min": f2((x["agent_wall_s"] or 0) / 60),
             "desc": x["desc"]}
        if arm:
            d["arm"] = arm
        out.append(d)
    return out


EXP_COLS = ["n", "attempt_id", "group", "k", "policy", "insights_in_context", "screen_val_bpb", "confirm_val_bpb",
            "best_before", "decision", "reward", "best_after", "num_steps", "flags", "t_end_utc", "hours",
            "agent_min", "desc"]


def baseline_runs(data, until):
    """The unmodified train.py's calibration runs (calib ledger + the main ledger's baseline entry)."""
    runs = {}
    for e in jl(os.path.join(data, "calib_ledger.jsonl")) + jl(os.path.join(data, "ledger.jsonl")):
        if (e.get("attempt_id") or "").startswith(("baseline", "noise")) and valid_run(e):
            if not until or (epoch(e.get("ts_end")) or 0) <= until:
                runs[e["run_id"]] = e["val_bpb"]
    return list(runs.values())


def baseline_steps(data, until):
    out = []
    for e in jl(os.path.join(data, "calib_ledger.jsonl")) + jl(os.path.join(data, "ledger.jsonl")):
        if (e.get("attempt_id") or "").startswith(("baseline", "noise")) and valid_run(e):
            if not until or (epoch(e.get("ts_end")) or 0) <= until:
                out.append((e.get("metrics") or {}).get("num_steps"))
    return out


def best_after_n(rows, n, baseline):
    best = baseline
    for x in rows[:n]:
        best = x["best_after"]
    return best


# ---------------------------------------------------------------------------------------------------------
# RL run
# ---------------------------------------------------------------------------------------------------------
def rl_run(data, until, out, S):
    rows, t0 = load_attempts(data, until)
    base = baseline_runs(data, until)
    write_tsv(os.path.join(out, "rl_run_experiments.tsv"), EXP_COLS, attempt_table(rows))

    # insights: one per failed experiment (the last one of a group is written but never injected)
    blocked = set((rj(os.path.join(data, "insight_blocklist.json"), {}) or {}).get("hints") or [])
    copies = {}
    for x in rows:
        for c in x["context"]:
            key = (x["group"], c.get("hint"))
            copies[key] = copies.get(key, 0) + 1
    ins = []
    for x in rows:
        if not x["hint"]:
            continue
        ins.append({"attempt_id": x["attempt_id"], "group": x["group"], "k": x["k"], "policy": x["policy"],
                    "screen_val_bpb": f6(x["screen_val_bpb"]), "best_before": f6(x["best_before"]),
                    "n_words": len(x["hint"].split()), "has_digits": int(bool(re.search(r"\d", x["hint"]))),
                    "injected_copies": copies.get((x["group"], x["hint"]), 0),
                    "retracted": int(x["hint"] in blocked), "hint": x["hint"]})
    write_tsv(os.path.join(out, "rl_run_insights.tsv"),
              ["attempt_id", "group", "k", "policy", "screen_val_bpb", "best_before", "n_words", "has_digits",
               "injected_copies", "retracted", "hint"], ins)

    sec = "rl_run"
    add = lambda m, v, n="", note="": S.append((sec, m, v, n, note))  # noqa: E731
    bmean = statistics.mean(base) if base else None
    add("baseline_val_bpb", f6(bmean), len(base), "unmodified train.py, mean of calibration runs (no autotune)")
    if len(base) > 1:
        add("baseline_sd", f"{statistics.stdev(base):.6f}", len(base), "run-to-run sd, unmodified train.py")
    add("baseline_num_steps", " ".join(str(s) for s in baseline_steps(data, until)), "",
        "optimizer steps the unmodified train.py gets in the 5-minute budget on this GPU")
    keeps = [x for x in rows if x["decision"] == "keep"]
    add("n_experiments", len(rows), "", "one fresh pi session per experiment")
    add("n_keeps", len(keeps), len(rows), "confirmed: fresh-compile re-run beats best by keep_margin 0.0005")
    if not rows:
        return rows, bmean, None
    final = keeps[-1] if keeps else None
    best = final["best_after"] if final else bmean
    i_best = rows.index(final) + 1 if final else 0
    add("final_best_val_bpb", f6(best), "", f"unselected confirmation re-run of {final['attempt_id'] if final else '-'}")
    add("experiments_to_final_best", i_best)
    add("hours_to_final_best", f2(final["hours"]) if final else "", "", "from the first experiment's session start")
    add("hours_total", f2(rows[-1]["hours"]), len(rows))
    if bmean:
        add("gain_total", f6(bmean - best), "", f"{100 * (bmean - best) / bmean:.1f}% relative")
        batch = [x for x in keeps if re.search(r"batch", x["desc"] or "", re.I)]
        g_batch = sum(x["best_before"] - x["best_after"] for x in batch)
        add("gain_from_batch_halving", f6(g_batch), len(batch),
            f"{100 * g_batch / (bmean - best):.0f}% of the total gain")
    led = jl(os.path.join(data, "ledger.jsonl"))
    led = [e for e in led if not until or (epoch(e.get("ts_end")) or 0) <= until]
    ids = {x["attempt_id"] for x in rows}
    att_runs = [e for e in led if (e.get("attempt_id") or "").split("-confirm")[0] in ids]
    add("gpu_training_runs", sum(e.get("status") != "refused" for e in att_runs), "",
        "agent runs + confirmation re-runs (refused re-run requests excluded)")
    add("refused_rerun_requests", sum(e.get("status") == "refused" for e in att_runs))
    runs_to_best = None
    if final:
        done = [e for e in att_runs if e.get("status") != "refused"]
        last = max(i for i, e in enumerate(done) if e["attempt_id"].split("-confirm")[0] == final["attempt_id"])
        runs_to_best = last + 1
        add("gpu_training_runs_to_final_best", runs_to_best, "", "including confirmation re-runs")
    add("unfinished_attempt_runs", sum(1 for e in led if (e.get("attempt_id") or "").startswith("g")
                                       and e["attempt_id"].split("-confirm")[0] not in ids), "",
        "runs of an attempt still in progress when the run was stopped (excluded)")
    agent_runs = [e for e in att_runs if e.get("status") != "refused" and "-confirm" not in e["attempt_id"]]
    add("agent_runs_crashed_or_invalid", sum(not valid_run(e) for e in agent_runs), len(agent_runs))

    # noise from screen/confirm pairs (the screen run is selected, so this is an upper-ish estimate)
    pairs = [x for x in rows if x["confirm_val_bpb"] is not None]
    for name, sel in (("before_autotune", [x for x in pairs if x["attempt_id"] < AUTOTUNE_FROM]),
                      ("with_autotune", [x for x in pairs if x["attempt_id"] >= AUTOTUNE_FROM])):
        if sel:
            d = [x["screen_val_bpb"] - x["confirm_val_bpb"] for x in sel]
            sig = math.sqrt(sum(v * v for v in d) / len(d) / 2)
            add(f"noise_sigma_{name}", f"{sig:.6f}", len(sel), "per-run sd = rms(screen - confirm) / sqrt(2)")
    rev = [x for x in pairs if x["decision"] != "keep"]
    add("apparent_wins_reversed_by_confirmation", len(rev), len(pairs), ",".join(x["attempt_id"] for x in rev))

    # success by policy version and with/without insights in context
    by_v = {}
    for x in rows:
        s = by_v.setdefault(x["policy"], [0, 0])
        s[0] += x["success"]
        s[1] += 1
    add("keeps_by_policy_version", " ".join(f"{v}:{s[0]}/{s[1]}" for v, s in sorted(by_v.items())))
    v0 = by_v.get("v0", [0, 0])
    later = [sum(s[0] for v, s in by_v.items() if v != "v0"), sum(s[1] for v, s in by_v.items() if v != "v0")]
    add("fisher_p_v0_vs_later", f"{fisher_two_sided(v0[0], v0[1] - v0[0], later[0], later[1] - later[0]):.2f}",
        len(rows), f"v0 {v0[0]}/{v0[1]} vs v1+ {later[0]}/{later[1]}")
    w = [x for x in rows if x["conditioned"]]
    wo = [x for x in rows if not x["conditioned"]]
    ws, wos = sum(x["success"] for x in w), sum(x["success"] for x in wo)
    add("keeps_with_insights", f"{ws}/{len(w)}")
    add("keeps_without_insights", f"{wos}/{len(wo)}", "",
        "insights are injected only while the group's running success rate is <= 50% (not randomised)")
    add("fisher_p_with_vs_without_insights",
        f"{fisher_two_sided(ws, len(w) - ws, wos, len(wo) - wos):.2f}", len(rows))
    add("insights_written", len(ins), "", f"{sum(int(i['retracted']) for i in ins)} retracted")
    if ins:
        add("insight_mean_words", f"{statistics.mean(i['n_words'] for i in ins):.1f}", len(ins))
        add("insights_with_digits", sum(i["has_digits"] for i in ins), len(ins))

    tr = [m for m in jl(os.path.join(data, "metrics_trainer.jsonl")) if not until or (m.get("t") or 0) <= until]
    if tr:
        add("policy_updates", len(tr), "", "LoRA r=32, Dr.GRPO + 0.5 * insight-SFT, one per group of 8")
        add("grpo_tokens_total", sum((m.get("data") or {}).get("n_grpo_tokens", 0) for m in tr), len(tr))
        add("sft_tokens_total", sum((m.get("data") or {}).get("n_sft_tokens", 0) for m in tr), len(tr))
        add("optimizer_steps_total", sum((m.get("update") or {}).get("steps", 0) for m in tr), len(tr))
        add("update_seconds_mean", f"{statistics.mean(m.get('t_update', 0) for m in tr):.0f}", len(tr))
    return rows, bmean, runs_to_best


# ---------------------------------------------------------------------------------------------------------
# head-to-head
# ---------------------------------------------------------------------------------------------------------
def read_agent_tsv(path):
    """The agent's results.tsv (untrusted): bounded read of a regular file, rows of commit/val/status/desc."""
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_size > TSV_MAX:
            return []
        with open(path, "rb") as f:
            raw = f.read(TSV_MAX)
    except OSError:
        return []
    rows = []
    for line in raw.decode("utf-8", "replace").splitlines():
        cells = [c.strip() for c in line.split("\t")]
        if len(cells) < 4 or cells[0].lower() == "commit":
            continue
        try:
            val = float(cells[1])
        except ValueError:
            val = None
        rows.append({"commit": cells[0].lower()[:40], "val": val if val and val > 0 else None,
                     "status": cells[3].lower()[:20]})
    return rows


def h2h(data, until, out, S, rl_hours_to_best, rl_runs_to_best=None):
    table, S_pairs = [], []
    for arm in H2H_ARMS:
        d = os.path.join(data, "h2h", arm)
        led = [e for e in jl(os.path.join(d, "ledger.jsonl")) if e.get("status") != "refused"
               and (not until or (epoch(e.get("ts_end")) or 0) <= until)]
        if not led:
            continue
        status = rj(os.path.join(d, "status.json"), {}) or {}
        t0 = epoch(status.get("started_at")) or epoch(led[0].get("ts"))
        runs = []
        for i, e in enumerate(led, 1):
            kind = "valid" if valid_run(e) else ("crash" if e.get("status") != "ok" else "invalid")
            head = str(e.get("agent_head") or "").lower()
            runs.append({"arm": arm, "run": i, "kind": kind, "val": e.get("val_bpb") if kind == "valid" else None,
                         "t_end": epoch(e.get("ts_end")), "head": head if HEX.fullmatch(head) else "",
                         "steps": (e.get("metrics") or {}).get("num_steps"), "flags": e.get("flags") or [],
                         "empty_diff": "empty_diff" in (e.get("flags") or []), "desc": e.get("desc"),
                         "train_sha": e.get("train_sha256"),
                         "decision": ""})
        # map the agent's results.tsv rows onto runs: by commit (ledger agent_head), else by the 6-decimal value
        used, keeps = set(), []
        for row in read_agent_tsv(os.path.join(d, "repo", "results.tsv")):
            c = row["commit"]
            cands = [r for r in runs if HEX.fullmatch(c) and r["head"].startswith(c)]
            if not cands and row["val"] is not None:
                cands = [r for r in runs if r["val"] is not None and abs(r["val"] - row["val"]) < 1.5e-6]
            free = [r for r in cands if r["run"] not in used]
            if not cands:
                continue
            r = free[0] if free else cands[-1]
            used.add(r["run"])
            r["decision"] = row["status"]
            if row["status"] == "keep" and r["val"] is not None:
                keeps.append(r)
        for r in runs:
            table.append({"arm": arm, "run": r["run"], "t_end_utc": iso(r["t_end"]),
                          "hours": f2((r["t_end"] - t0) / 3600 if r["t_end"] and t0 else None),
                          "status": r["kind"], "val_bpb": f6(r["val"]), "num_steps": r["steps"] or "",
                          "flags": ",".join(r["flags"]), "agent_decision": r["decision"], "desc": r["desc"]})

        sec = f"h2h_{arm}"
        add = lambda m, v, n="", note="": S.append((sec, m, v, n, note))  # noqa: E731
        hours = (max(r["t_end"] for r in runs if r["t_end"]) - t0) / 3600
        base = next((r for r in runs if r["kind"] == "valid" and r["empty_diff"]), None)
        exp = [r for r in runs if r is not base]          # experiment runs: everything but the baseline run
        valid = [r for r in exp if r["kind"] == "valid"]
        keeps = [r for r in keeps if r is not base]
        stopped = sum("interrupted" in r["flags"] for r in exp)
        add("start_utc", iso(t0))
        add("end_utc", iso(t0 + hours * 3600), "", "last run finished (stopped by the operator)")
        add("hours", f2(hours))
        add("runs", len(exp), "", f"excluding the baseline run: valid {len(valid)}, "
                                  f"crash {sum(r['kind'] == 'crash' for r in exp) - stopped}, "
                                  f"interrupted at stop {stopped}, invalid {sum(r['kind'] == 'invalid' for r in exp)}")
        add("runs_per_hour", f2(len(exp) / hours) if hours else "")
        add("baseline_val_bpb", f6(base["val"]) if base else "", 1, "first run, unmodified train.py on this GPU")
        cal = [e for sub in ("calibration", os.path.join("calibration", "c2_fix_validation"))
               for e in jl(os.path.join(data, "h2h", sub, arm, "ledger.jsonl")) if valid_run(e)]
        if cal:
            add("gpu_calibration_baselines", " ".join(f6(e["val_bpb"]) for e in cal), len(cal),
                "unmodified train.py on this arm's GPU before the start")
        add("agent_logged_rows", sum(1 for r in exp if r["decision"]), len(exp), "results.tsv rows matched to runs")
        add("agent_keeps", len(keeps), "", "excluding the baseline row")
        reps = {}
        for r in valid:
            reps.setdefault(r["train_sha"], []).append(r["val"])
        pairs = [v[:2] for v in reps.values() if len(v) >= 2]
        S_pairs.extend(pairs)
        add("repeat_pairs", len(pairs), "", "same train.py measured twice (the agent's own re-checks)")
        if keeps:
            add("final_kept_val_bpb", f6(keeps[-1]["val"]), 1,
                f"trusted value of the agent's latest keep (run {keeps[-1]['run']}); single run, selected")
            add("best_kept_val_bpb", f6(min(r["val"] for r in keeps)), len(keeps), "min over the agent's keeps")
        if valid:
            b = min(valid, key=lambda r: r["val"])
            add("best_single_run_val_bpb", f6(b["val"]), len(valid), f"min over all valid runs (run {b['run']})")
        if rl_hours_to_best and valid:
            early = [r for r in valid if r["t_end"] and (r["t_end"] - t0) / 3600 <= rl_hours_to_best]
            if early:
                add("best_single_run_within_rl_time_to_best", f6(min(r["val"] for r in early)), len(early),
                    f"first {rl_hours_to_best:.2f} h, the RL run's time to its final best")
        if rl_runs_to_best and valid:
            early = [r for r in exp[:rl_runs_to_best] if r["kind"] == "valid"]
            if early:
                add("best_single_run_within_rl_runs_to_best", f6(min(r["val"] for r in early)), len(early),
                    f"first {rl_runs_to_best} runs, the RL run's GPU runs to its final best")
    if S_pairs:
        sig = math.sqrt(sum((a - b) ** 2 for a, b in S_pairs) / len(S_pairs) / 2)
        S.append(("h2h", "noise_sigma_repeat_pairs", f"{sig:.6f}", len(S_pairs),
                  "per-run sd = rms(run1 - run2) / sqrt(2), both arms, autotune on"))
    write_tsv(os.path.join(out, "h2h_runs.tsv"),
              ["arm", "run", "t_end_utc", "hours", "status", "val_bpb", "num_steps", "flags", "agent_decision",
               "desc"], table)


# ---------------------------------------------------------------------------------------------------------
# frozen-v5 ablation
# ---------------------------------------------------------------------------------------------------------
def frz(data, until, out, S, rl_rows, rl_base):
    table, arms = [], {}
    for arm in FRZ_ARMS:
        d = os.path.join(data, "h2h", arm, "rl")
        rows, t0 = load_attempts(d, until)
        if not rows:
            continue
        cfg = rj(os.path.join(data, "h2h", arm, "rl_config.json"), {}) or {}
        arms[arm] = rows
        table += attempt_table(rows, arm)
        base = baseline_runs(d, until)
        sec = f"frz_{arm}"
        add = lambda m, v, n="", note="": S.append((sec, m, v, n, note))  # noqa: E731
        add("insights_enabled", int(bool(cfg.get("insights_enabled", True))))
        add("gpu_minor", cfg.get("agent_gpu_minor", ""))
        add("snapshot_utc", iso(max(x["t_end"] for x in rows)), "", "last finished experiment included")
        add("baseline_val_bpb", f6(statistics.mean(base)) if base else "", len(base),
            "mean of calibration runs, unmodified train.py")
        add("n_experiments", len(rows))
        add("n_keeps", sum(x["decision"] == "keep" for x in rows), len(rows))
        add("best_val_bpb", f6(rows[-1]["best_after"]), "", "unselected confirmation re-run of the latest keep")
        add("hours", f2(rows[-1]["hours"]), len(rows))
        add("insights_written", sum(1 for x in rows if x["hint"]))
        led = [e for e in jl(os.path.join(d, "ledger.jsonl")) if not until or (epoch(e.get("ts_end")) or 0) <= until]
        bad = [e for e in led if e.get("status") == "ok" and not valid_run(e)]
        add("integrity_flagged_runs", len(bad), len(led), ",".join(f"{e.get('attempt_id')}:{'+'.join(e.get('flags') or [])}"
                                                               for e in bad))
        add("refused_rerun_requests", sum(e.get("status") == "refused" for e in led))
    write_tsv(os.path.join(out, "frz_runs.tsv"), ["arm"] + EXP_COLS, table)
    if arms:   # matched comparison: best after the same number of experiments (RL run: base policy v0 at first)
        n = min(len(r) for r in arms.values())
        sec = "frz_matched"
        S.append((sec, "experiments_compared", n, "", "per arm, from the same baseline train.py"))
        if rl_rows:
            S.append((sec, "rl_run_best_after_n", f6(best_after_n(rl_rows, n, rl_base)), n,
                      "RL run, insights on, policy v0 for the first group"))
        for arm, rows in arms.items():
            S.append((sec, f"{arm}_best_after_n", f6(best_after_n(rows, n, None)), n,
                      "frozen v5, insights " + ("on" if arm == "x1" else "off")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(CODE, "results"))
    ap.add_argument("--until", default=None, help="ignore everything that finished after this UTC time (ISO)")
    a = ap.parse_args()
    until = epoch(a.until) if a.until else None
    if a.until and until is None:
        ap.error(f"cannot parse --until {a.until!r}")
    if not os.path.isdir(a.data):
        ap.error(f"no data directory at {a.data} (set RLTLDR_ROOT or pass --data)")
    data, out = os.path.realpath(a.data), os.path.realpath(a.out)
    if out == data or out.startswith(data + os.sep):
        ap.error("--out must not be inside the data directory (the run's data is read-only)")
    os.makedirs(a.out, exist_ok=True)
    S = []
    rl_rows, rl_base, rl_runs_to_best = rl_run(a.data, until, a.out, S)
    keeps = [x for x in rl_rows if x["decision"] == "keep"]
    h2h(a.data, until, a.out, S, keeps[-1]["hours"] if keeps else None, rl_runs_to_best)
    frz(a.data, until, a.out, S, rl_rows, rl_base)
    with open(os.path.join(a.out, "summary.tsv"), "w") as f:
        f.write("section\tmetric\tvalue\tn\tnote\n")
        for row in S:
            f.write("\t".join(clean(c, 400) for c in row) + "\n")
    for row in S:
        print("  ".join(str(c) for c in row if c != ""))
    print(f"wrote {len(S)} summary rows and the tables to {a.out}")


if __name__ == "__main__":
    main()
