#!/usr/bin/env python3
"""Tests for the h2h dashboard (tools/h2h_dashboard/build.py) on synthetic data in a temp dir.

Generates ~40 runs per arm shaped like real data/ledger.jsonl lines (ok / crash / oom / timeout / invalid /
refused, agent_head, flags, metrics, last_step), the agent's results.tsv (keep/discard/crash rows, a typo'd
value, a row for an unknown commit, a CSV-ish header), status.json (epoch and ISO timestamps), events.jsonl,
calls.jsonl (with token id arrays) and runner_state.json. The v5 arm's repo is a real git repo with one commit
per experiment so that the train.py-content fallback (loose git objects, no git binary) is exercised.

    /usr/bin/python3 tests/test_h2h_dashboard.py            # run the checks
    /usr/bin/python3 tests/test_h2h_dashboard.py --keep DIR # also leave the data + dashboard in DIR
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(CODE, "tools", "h2h_dashboard", "build.py")
PY = "/usr/bin/python3"
GIT_ENV = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(t))


def ledger_entry(t, wall, head, train_sha, desc, status, val, flags, steps, seq, error_tail=None):
    """One line in the shape tools/ar_run.py writes."""
    ok = status == "ok"
    return {
        "run_id": time.strftime("%Y%m%d-%H%M%S", time.gmtime(t)) + "-%06x" % seq, "attempt_id": None, "ts": iso(t),
        "wall_s": wall, "head": "8721165" + "0" * 33, "branch": "HEAD", "train_sha256": train_sha,
        "train_ast_sha256": hashlib.sha256(train_sha.encode()).hexdigest(), "parent_train_sha256": "0" * 64,
        "parent_rev": "8721165" + "0" * 33, "prepare_sha256": "4f2b" + "0" * 60, "desc": desc, "status": status,
        "returncode": 0 if ok else 1, "val_bpb": val if ok else None,
        "metrics": {"val_bpb": val, "training_seconds": 300.1, "total_seconds": wall - 3, "peak_vram_mb": 25277.5,
                    "mfu_percent": 22.5, "total_tokens_M": 254.9, "num_steps": steps, "num_params_M": 57.7,
                    "depth": 9, "t_train_trusted": 302.4, "val_bpb_stdout": round(val, 6) if val else None,
                    "t_to_eval": 333.1} if ok else {},
        "error_tail": error_tail, "flags": flags, "diff_path": "/x/diff.patch", "log_path": "/x/run.log",
        "gpu": None, "max_runs": 0, "timeout_s": 660, "mode": "committed", "diff_lines": 0 if "empty_diff" in flags else 7,
        "trusted": {"valid": ok, "evals": []}, "ts_end": iso(t + wall),
        "last_step": {"step": steps - 1, "pct": 100.0, "loss": 2.82, "dt_ms": 155, "tok_per_sec": 844380 + seq,
                      "epoch": 1, "remaining_s": 0},
        "seq": seq, "agent_head": head,
    }


def make_arm(root, arm, t0, seed, n=40, improve=0.0016, real_git=False):
    """Write one arm's files; returns the expectations the checks need."""
    rnd = random.Random(seed)
    d = os.path.join(root, "data", "h2h", arm)
    repo = os.path.join(d, "repo")
    os.makedirs(repo)
    if real_git:
        subprocess.run(["git", "init", "-q", "-b", "autoresearch/h2h", repo], check=True, env=GIT_ENV)

    def commit(content, msg):
        if not real_git:
            return hashlib.sha1(f"{arm}{msg}{rnd.random()}".encode()).hexdigest()
        with open(os.path.join(repo, "train.py"), "w") as f:
            f.write(content)
        subprocess.run(["git", "-C", repo, "add", "train.py"], check=True, env=GIT_ENV)
        subprocess.run(["git", "-C", repo, "commit", "-q", "-m", msg], check=True, env=GIT_ENV)
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], check=True, env=GIT_ENV,
                              capture_output=True, text=True).stdout.strip()

    ledger, tsv = [], ["commit\tval_bpb\tmemory_gb\tstatus\tdescription"]
    exp = {"keeps": [], "crash": 0, "invalid": 0, "valid": 0, "refused": 0, "match": {}}
    base_val = 1.0800 + (0.0002 if arm == "v5" else 0.0)
    kept_val, t, seq, prev_head = None, t0 + 240, 0, None
    ideas = ["increase depth 8 -> 9", "lower matrix LR 0.04 -> 0.03", "warmdown ratio 0.5 -> 0.7",
             "RoPE base 10000 -> 50000", "SwiGLU MLP instead of ReLU^2", "batch 2^19 -> 2^18 tokens",
             "untie embedding LR", "QK-norm before RoPE", "value embeddings on every layer",
             "drop final logit softcap", "weight decay 0.0 -> 0.1 on matrices", "window pattern SSSL -> SSL"]
    for k in range(n):
        seq += 1
        desc = "baseline" if k == 0 else rnd.choice(ideas) + f" (try {k})"
        content = f"# train.py for {arm} experiment {k}\nprint('hello')\n"
        head = commit(content, desc)
        train_sha = hashlib.sha256(content.encode()).hexdigest()
        wall = 340 + rnd.random() * 20
        r = rnd.random()
        forced = (arm == "base" and k in (9, 12)) or (arm == "v5" and k in (7, 21))   # special cases below
        if forced:
            r = 0.5
        status, val, flags, err = "ok", None, [], None
        if k == 0:
            val, flags = base_val, ["empty_diff"]
        elif r < 0.10:
            status, err = rnd.choice([("crash", "Traceback (most recent call last):\n  File \"train.py\", line 88\n"
                                      "RuntimeError: shape mismatch [8, 512] vs [8, 640]\nlast step: none"),
                                     ("oom", "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12 GiB"),
                                     ("timeout", "killed: exceeded the 660s wall-clock limit\nstep 1234 (98%)")])
            wall = 660 if status == "timeout" else 90
        elif r < 0.15:
            val, flags = (kept_val or base_val) - 0.004, ["over_time_budget", "wall_clock_excessive"]
        else:
            drift = (kept_val or base_val) - improve * rnd.random() * (1.0 if rnd.random() < 0.45 else -1.5)
            val = drift + rnd.gauss(0, 0.0004)
            if rnd.random() < 0.06 and not forced:
                val = 1.31 + rnd.random() * 0.2          # a blow-up: above the chart's scale
        steps = 1945 + rnd.randint(-15, 15) if status == "ok" else 0
        head_logged = head
        # special cases the matcher must handle
        if arm == "v5" and k in (7, 21):
            head_logged = prev_head                      # dirty-tree run: the commit came after the run
        if arm == "base" and k == 12:
            head_logged = None                           # old ar_run without agent_head
        e = ledger_entry(t, round(wall, 1), head_logged, train_sha, desc, status, val, flags, steps, seq, err)
        ledger.append(e)
        if k == 25:                                      # a refused submission (no run index)
            ledger.append(ledger_entry(t + wall + 5, 0.0, head, train_sha, desc + " (again)", "refused", None,
                                       ["gpu_missing"], 0, seq + 1000, "GPU is not available"))
            exp["refused"] += 1
        t += wall + 60 + rnd.random() * 240
        valid = status == "ok" and not flags or flags == ["empty_diff"]
        exp["valid"] += valid
        exp["crash"] += status != "ok"
        exp["invalid"] += status == "ok" and "over_time_budget" in flags
        if k == n - 1:
            break                                        # the agent has not logged the newest run yet
        short = head[:7]
        if status != "ok":
            tsv.append(f"{short}\t0.000000\t0.0\tcrash\t{desc}")
            exp["match"][k] = "head"
        elif not valid:
            tsv.append(f"{short}\t0.000000\t0.0\tcrash\t{desc} (invalid: over time)")
        else:
            keep = kept_val is None or val < kept_val
            logged = val + (0.001 if (arm == "base" and k == 9) else 0.0)     # one typo'd value
            tsv.append(f"{short}\t{logged:.6f}\t24.7\t{'keep' if keep else 'discard'}\t{desc}")
            exp["match"][k] = ("head" if head_logged == head else ("train_sha" if real_git else "value"))
            if keep:
                kept_val = val
                exp["keeps"].append(val)
        prev_head = head
    tsv.insert(5, "deadbee\t0.990000\t24.0\tkeep\tunknown commit the ledger never saw")
    with open(os.path.join(d, "ledger.jsonl"), "w") as f:
        for e in ledger:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(repo, "results.tsv"), "w") as f:
        f.write("\n".join(tsv) + "\n")
    t_end = t
    # supervisor files
    st = {"started_at": t0 if arm == "base" else iso(t0), "pid": 4242, "n_prompts": 9, "n_nudges": 8,
          "n_restarts": 1 if arm == "v5" else 0, "n_compactions": 3, "n_runsh_calls": n + 1,
          "last_event_ts": t_end - 30, "context_tokens": 61234, "state": "agent working"}
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump(st, f)
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        tt = t0
        while tt < t_end:
            f.write(json.dumps({"ts": tt, "type": rnd.choice(["message_end", "tool_execution_start",
                                                               "tool_execution_end"])}) + "\n")
            tt += 30 + rnd.random() * 90
        f.write(json.dumps({"ts": t_end - 30, "type": "tool_execution_start",
                            "tool": "bash", "args": "./run.sh \"next idea\" > run.log 2>&1"}) + "\n")
    calls = []
    tt, ctx = t0 + 5, 4000
    while tt < t_end:
        nc = rnd.randint(20, 3000)
        rate = (46.0 if arm == "base" else 45.5) + rnd.gauss(0, 1)
        calls.append({"ts_start": tt, "ts_end": tt + nc / rate + 0.4, "arm": arm, "served_model": f"h2h-{arm}",
                      "status": "ok" if rnd.random() > 0.02 else "upstream_stalled", "finish_reason": "stop",
                      "n_prompt": ctx, "n_completion": nc, "ttft_s": 0.4, "decode_tok_s": rate,
                      "prompt_ids": list(range(min(ctx, 3000))), "completion_ids": list(range(nc)),
                      "logprobs": [-0.1] * nc})
        ctx = ctx + nc + 800 if ctx < 100000 else 20000
        tt += nc / rate + 20 + rnd.random() * 60
    with open(os.path.join(d, "calls.jsonl"), "w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")
    if arm == "v5":
        with open(os.path.join(d, "runner_state.json"), "w") as f:
            json.dump({"job": {"desc": "next idea: SwiGLU + depth 10", "t0": t_end - 120, "pid": 1}}, f)
    exp.update(n=n, calls=calls, t_end=t_end, dir=d, repo=repo)
    return exp


def run_build(root, *args, timeout=60):
    env = {**os.environ, "RLTLDR_ROOT": root}
    env.pop("H2H_CONFIG", None)
    r = subprocess.run([PY, BUILD, *args], env=env, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr
    return r.stdout


def load_doc(path):
    html = open(path).read()
    a = html.index('<script id="run-data" type="application/json">') + len('<script id="run-data" type="application/json">')
    b = html.index("</script>", a)
    return html, json.loads(html[a:b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=None)
    a = ap.parse_args()
    root = a.keep or tempfile.mkdtemp(prefix="h2h_dash_")
    if a.keep and os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    t0 = time.time() - 5.8 * 3600
    exp = {"base": make_arm(root, "base", t0, 1, improve=0.0012),
           "v5": make_arm(root, "v5", t0 + 4, 2, improve=0.0019, real_git=True)}
    out = os.path.join(root, "dashboard_h2h", "index.html")
    passed = []

    def check(name, cond, detail=""):
        assert cond, f"FAIL {name} {detail}"
        passed.append(name)
        print(f"ok   {name}" + (f"  ({detail})" if detail else ""))

    t = time.time()
    print(run_build(root).strip())
    check("build runs", os.path.exists(out), f"{time.time() - t:.2f}s")
    html, doc = load_doc(out)
    check("placeholder replaced", "/*__DATA__*/" not in html)
    check("no raw '<' in payload", "<" not in html[html.index('type="application/json">') + 24:html.index("</script>")])
    arms = {x["name"]: x for x in doc["arms"]}
    check("both arms present", set(arms) == {"base", "v5"})
    for name, e in exp.items():
        s, runs = arms[name]["summary"], arms[name]["runs"]
        check(f"{name}: run count", s["n_runs"] == e["n"], f"{s['n_runs']} runs")
        check(f"{name}: valid/crash/invalid", (s["n_valid"], s["n_crash"], s["n_invalid"]) ==
              (e["valid"], e["crash"], e["invalid"]), f"{s['n_valid']}/{s['n_crash']}/{s['n_invalid']}")
        check(f"{name}: refused not indexed", s["n_refused"] == e["refused"] and
              all(r["i"] is None for r in runs if r["kind"] == "refused"))
        check(f"{name}: frontier = agent keeps", [round(f["val"], 9) for f in arms[name]["frontier"]] ==
              [round(v, 9) for v in e["keeps"]], f"{len(e['keeps'])} keeps")
        check(f"{name}: current best = latest keep", abs(s["best_kept"]["val"] - e["keeps"][-1]) < 1e-12,
              f"{s['best_kept']['val']:.6f}")
        bv = min(r["val"] for r in runs if r["kind"] == "valid")
        check(f"{name}: best valid single run", abs(s["best_valid"]["val"] - bv) < 1e-12)
        check(f"{name}: unmatched keep row counted", s["n_tsv_unmatched"] == 1)
        check(f"{name}: newest run not logged yet", runs[-1]["dec"] is None)
        idx = [r for r in runs if r["kind"] != "refused"]
        for k, how in e["match"].items():
            check(f"{name}: run {k + 1} matched by {how}", idx[k]["match"] == how, f"got {idx[k]['match']}")
        check(f"{name}: calibration = first baseline run", arms[name]["calib"]["i"] == 1)
        check(f"{name}: calls aggregated", s["n_calls"] == len(e["calls"]) and
              s["tokens_out"] == sum(c["n_completion"] for c in e["calls"]), f"{s['n_calls']} calls")
        check(f"{name}: decode tok/s plausible", 40 < s["decode_tok_s"] < 52, f"{s['decode_tok_s']:.1f}")
        check(f"{name}: supervisor counters", s["nudges"] == 8 and s["compactions"] == 3)
        check(f"{name}: t0 parsed (epoch or ISO)", abs(arms[name]["t0"] - (t0 if name == "base" else t0 + 4)) < 2)
    base_runs = [r for r in arms["base"]["runs"] if r["kind"] != "refused"]
    check("typo'd logged value kept as claim", abs(base_runs[9]["claimed"] - base_runs[9]["val"] - 0.001) < 1e-6)
    check("in-flight run from runner_state", arms["v5"]["inflight"]["desc"].startswith("next idea"))
    check("live", doc["live"])
    check("invalid runs have no plotted value", all(r["val"] is None for x in doc["arms"] for r in x["runs"]
                                                     if r["kind"] != "valid"))

    # incremental calls cache: append, rebuild, totals follow; the offset ends on a line boundary
    extra = {"ts_start": time.time(), "ts_end": time.time() + 1, "status": "ok", "n_prompt": 10,
             "n_completion": 100, "decode_tok_s": 50.0, "ttft_s": 0.3}
    with open(os.path.join(exp["base"]["dir"], "calls.jsonl"), "a") as f:
        f.write(json.dumps(extra) + "\n" + '{"ts_start": 1, "n_compl')            # + a torn line
    cache = json.load(open(os.path.join(root, "dashboard_h2h", ".calls_cache.json")))
    run_build(root)
    _, doc2 = load_doc(out)
    s2 = {x["name"]: x for x in doc2["arms"]}["base"]["summary"]
    check("incremental calls: appended call counted", s2["n_calls"] == len(exp["base"]["calls"]) + 1 and
          s2["tokens_out"] == sum(c["n_completion"] for c in exp["base"]["calls"]) + 100)
    cache2 = json.load(open(os.path.join(root, "dashboard_h2h", ".calls_cache.json")))
    p = os.path.join(exp["base"]["dir"], "calls.jsonl")
    check("incremental calls: torn line left for later", cache2[p]["off"] == os.path.getsize(p) - len('{"ts_start": 1, "n_compl')
          and cache2[p]["off"] > cache[p]["off"])
    t = time.time()
    run_build(root)
    check("incremental rebuild is fast", time.time() - t < 3, f"{time.time() - t:.2f}s")

    # hostile results.tsv: a FIFO must not hang the build, a symlink is not followed, garbage is ignored
    v5repo = exp["v5"]["repo"]
    os.rename(os.path.join(v5repo, "results.tsv"), os.path.join(v5repo, "results.tsv.bak"))
    os.mkfifo(os.path.join(v5repo, "results.tsv"))
    run_build(root, timeout=30)
    _, doc3 = load_doc(out)
    a3 = {x["name"]: x for x in doc3["arms"]}["v5"]
    check("FIFO results.tsv: no hang, treated as empty", a3["summary"]["n_tsv"] == 0 and a3["summary"]["n_runs"] == 40)
    os.remove(os.path.join(v5repo, "results.tsv"))
    os.symlink(os.path.join(v5repo, "results.tsv.bak"), os.path.join(v5repo, "results.tsv"))
    run_build(root)
    _, doc4 = load_doc(out)
    check("symlinked results.tsv not followed", {x["name"]: x for x in doc4["arms"]}["v5"]["summary"]["n_tsv"] == 0)
    os.remove(os.path.join(v5repo, "results.tsv"))
    with open(os.path.join(v5repo, "results.tsv"), "wb") as f:
        f.write(b"\x00\xff garbage\nnot\ta\nabc</script><!--<script>\t1.0\t1\tkeep\tx</script>\n")
    run_build(root)
    html5, doc5 = load_doc(out)
    check("garbage results.tsv tolerated", {x["name"]: x for x in doc5["arms"]}["v5"]["summary"]["n_tsv"] == 1)
    check("script-breaking text stays inside the JSON", html5.count("</script>") == html.count("</script>"))
    os.remove(os.path.join(v5repo, "results.tsv"))
    os.rename(os.path.join(v5repo, "results.tsv.bak"), os.path.join(v5repo, "results.tsv"))

    # missing files: an empty arm and a missing data dir both build
    shutil.rmtree(exp["v5"]["dir"])
    run_build(root)
    _, doc6 = load_doc(out)
    check("missing arm dir builds", {x["name"]: x for x in doc6["arms"]}["v5"]["summary"]["n_runs"] == 0)
    empty = tempfile.mkdtemp(prefix="h2h_dash_empty_")
    run_build(empty)
    check("empty root builds", os.path.exists(os.path.join(empty, "dashboard_h2h", "index.html")))
    shutil.rmtree(empty)

    # --status text (used by ctl_h2h.sh status) and --data override
    txt = run_build(root, "--status")
    check("--status prints both arms", "[base]" in txt and "[v5]" in txt and "best kept" in txt, txt.splitlines()[0])
    alt = os.path.join(root, "alt_out", "x.html")
    run_build(root, "--data", os.path.join(root, "data", "h2h"), "--out", alt)
    check("--data/--out override", os.path.exists(alt))

    # regenerate the full v5 arm for anyone inspecting --keep output
    if a.keep:
        make_arm(root, "v5", t0 + 4, 2, improve=0.0019, real_git=True)
        print(run_build(root).strip())
    else:
        shutil.rmtree(root)
    print(f"\nALL {len(passed)} CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
