"""
committee.py: the uncertainty committee of the hypothesis framework. Task-agnostic.

K independent world models ("members") forecast every queued idea before it runs. Each member
is a separate LLM call with its own lens and its own persistent world model; members never see
each other's forecasts (shown the majority, members anchor on it). Every member is scored on
every experiment it forecast, so the committee's uncertainty is a measured quantity:

    disagreement   per idea: members' ranges on the objective that do not overlap (conflict),
                   entropy of their verdict votes and of their improve/no-improve votes
    counterexample a result outside EVERY member's range: the shared world model lacks a mechanism
    calibration    hit rate of each member and of the consensus; whether disagreement actually
                   predicts consensus misses on this task (AUROC), measured as results come in

Actions (all automatic, all journaled):
    ideas          veto an idea that all members forecast as clearly harmful (one veto in
                   VETO_AUDIT_EVERY is kept as an audit, which measures the veto's false rejects);
                   raise the priority of the most disputed idea (it decides between world models),
                   only while disagreement is shown to predict misses; ideate sees the forecasts
                   and is asked to sharpen or drop disputed and doomed ideas
    hypotheses     counterexamples trigger a revise, with the counterexamples and the members'
                   cruxes as input; members' votes on each open hypothesis go to revise
    literature     a counterexample triggers a literature request on it (rate-limited)
    members        a member that keeps missing is replaced by a fresh one built from the
                   counterexamples; when members stop disagreeing, one is reseeded with a new lens

State (workspace root, untracked by git):
    committee/members.json   committee/W1.md ...    each member's world model
    committee/forecasts.jsonl  committee/ledger.jsonl  committee/rounds.jsonl  committee/report.md
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import research as R  # noqa: E402

DIR = os.path.join(R.ROOT, "committee")
MEMBERS_FILE = os.path.join(DIR, "members.json")
FORECASTS_FILE = os.path.join(DIR, "forecasts.jsonl")
LEDGER_FILE = os.path.join(DIR, "ledger.jsonl")
ROUNDS_FILE = os.path.join(DIR, "rounds.jsonl")
REPORT_FILE = os.path.join(DIR, "report.md")
STATE_FILE = os.path.join(DIR, "state.json")

MODELS = os.environ.get("RESEARCH_COMMITTEE_MODELS", "claude-opus-5-5,claude-sonnet-5-5").split(",")
EFFORT = os.environ.get("RESEARCH_COMMITTEE_EFFORT", "medium")
VETO_AUDIT_EVERY = 4  # every 4th veto is kept in the queue as an audit
VETO_MIN_SCORED, VETO_MIN_HIT = 5, 0.5  # the veto acts only once the committee has shown it can forecast
RECIPE_CHARS = 60000  # the editable files at the branch tip, as members see them
PRUNE_WINDOW, PRUNE_HITRATE = 8, 0.25  # a member that hits <= 25% of its last 8 forecasts is replaced
COLLAPSE_ROUNDS = 3  # rounds in a row with no disagreement on any idea -> reseed a member
AUROC_MIN, AUROC_N = 0.55, 12  # disagreement steers priorities only while it predicts misses
LIT_EVERY = 8  # at most one committee literature request per 8 experiments
COUNTEREXAMPLES_FOR_REVISE = 2  # counterexamples since the last revise that make one due

LENSES = {
    "mechanism": "Explain the results by the causal mechanism inside the system: what each change does to "
    "the process that produces the metrics, and why.",
    "empirical": "Explain the results only by regularities measured in the results table. Trust measured "
    "differences over theory; extrapolate trends; distrust anything never measured here.",
    "cost": "Account for where each metric's value comes from: the budget each part of the process consumes. "
    "A change helps only if what it gains outweighs what it costs elsewhere.",
    "skeptic": "Assume most changes are null within run-to-run noise and that past gains partly regress. "
    "Predict a change only for a strong reason, and say how large the noise is.",
}
RESEED_LENSES = {
    "contrarian": "Look for the explanation the current hypotheses and the obvious reading of the results "
    "miss. Prefer a mechanism nobody has proposed yet that fits all the results, including the surprising ones.",
    "counterexample": "Start from the results the previous world models failed to predict. Build the "
    "simplest world model that explains them and everything else measured.",
}

FORECAST_SCHEMA = {
    "type": "object",
    "properties": {
        "world_model": {"type": "string"},
        "forecasts": {"type": "array", "items": {"type": "object", "properties": {
            "idea": {"type": "string"},
            "metrics": {"type": "array", "items": {"type": "object", "properties": {
                "metric": {"type": "string"}, "lo": {"type": "number"}, "point": {"type": "number"},
                "hi": {"type": "number"}}, "required": ["metric", "lo", "point", "hi"]}},
            "verdict": {"type": "string", "enum": ["supports", "refutes", "inconclusive"]},
            "reason": {"type": "string"}},
            "required": ["idea", "metrics", "verdict", "reason"]}},
        "hypotheses": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "outcome": {"type": "string", "enum": ["supported", "refuted", "unknown"]}},
            "required": ["id", "outcome"]}},
        "cruxes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["forecasts", "hypotheses", "cruxes"],
}  # fmt: skip


# ---------------------------------------------------------------------------
# config and state
# ---------------------------------------------------------------------------


def config():
    return (R.read_json(R.FRAMEWORK_FILE, {}) or {}).get("committee") or {}


def enabled():
    return bool(config())


def k():
    return int(config().get("k", 4))


def load_ar():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ar_for_committee", os.path.join(R.ROOT, "ar.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    out = []
    for line in R.read_text(path).splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def append_jsonl(path, rows):
    os.makedirs(DIR, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def members(active_only=True):
    ms = R.read_json(MEMBERS_FILE, []) or []
    return [m for m in ms if m["status"] == "active"] if active_only else ms


def state():
    return R.read_json(STATE_FILE, {}) or {}


def set_state(**kw):
    R.write_json(STATE_FILE, {**state(), **kw})


def numeric_metrics(t):
    """The metrics members forecast: the objective, the infeasible ordering and numeric constraints."""
    out = [t["objective"]["metric"]]
    if t.get("infeasible_order"):
        out.append(t["infeasible_order"]["metric"])
    for c in t.get("constraints", []):
        if not isinstance(c["value"], bool):
            out.append(c["metric"])
    return list(dict.fromkeys(out))


def tip_metrics(ar, t):
    tip = R.tip()
    return ar.row_metrics(t, tip) if tip else None


def model_for(n):
    return MODELS[(n - 1) % len(MODELS)]


def new_member(ms, lens, why):
    n = max([int(m["id"][1:]) for m in ms], default=0) + 1
    m = {"id": f"W{n}", "lens": lens, "model": model_for(n), "status": "active", "born": R.last_exp(),
         "why": why, "created": time.time()}  # fmt: skip
    ms.append(m)
    return m


def ensure_members():
    ms = members(active_only=False)
    if not ms:
        for lens in list(LENSES)[: k()] + list(RESEED_LENSES)[: max(0, k() - len(LENSES))]:
            new_member(ms, lens, "initial committee")
        R.write_json(MEMBERS_FILE, ms)
    return members()


def wm_path(mid):
    return os.path.join(DIR, f"{mid}.md")


# ---------------------------------------------------------------------------
# scoring: every forecast made before an experiment started is checked against its result
# ---------------------------------------------------------------------------


def last_forecasts_before(fcs, idea, ts):
    """Each member's newest forecast for this idea made before ts."""
    out = {}
    for f in fcs:
        if f["idea"] == idea and f["ts"] < ts:
            out[f["member"]] = f
    return out


def entropy(counts):
    n = sum(counts)
    if n == 0 or len(counts) < 2:
        return 0.0
    h = -sum(c / n * math.log(c / n) for c in counts if c)
    return abs(h / math.log(len(counts)))


def conflict(ranges):
    """Fraction of member pairs whose ranges do not overlap (0 = all compatible)."""
    pairs = [(a, b) for i, a in enumerate(ranges) for b in ranges[i + 1:]]
    if not pairs:
        return 0.0
    return sum(a[2] < b[0] or b[2] < a[0] for a, b in pairs) / len(pairs)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2) if n else None


def idea_stats(ar, t, tipm, fs):
    """Disagreement and votes over one idea's forecasts {member: forecast}."""
    obj = t["objective"]["metric"]
    ranges = [f["metrics"][obj] for f in fs.values() if obj in f["metrics"]]
    verdicts = [f["verdict"] for f in fs.values()]
    improve, harm = 0, 0
    for f in fs.values():
        point = {m: v[1] for m, v in f["metrics"].items()}
        cand = {**(tipm or {}), **point}  # metrics a member does not forecast are taken from the tip
        if ar.valid(t, cand) and ar.improves(t, cand, tipm)[0]:
            improve += 1
        if confident_harm(ar, t, tipm, f):
            harm += 1
    n = len(fs)
    return {
        "n": n,
        "conflict": round(conflict(ranges), 3),
        "verdict_entropy": round(entropy([verdicts.count(v) for v in ("supports", "refutes", "inconclusive")]), 3),
        "improve_votes": improve,
        "improve_entropy": round(entropy([improve, n - improve]), 3),
        "harm_votes": harm,
        "median": [median([r[i] for r in ranges]) for i in range(3)] if ranges else None,
    }


def confident_harm(ar, t, tipm, f):
    """The member's whole range is on the wrong side of the tip: the comparison ar.py makes, with ranges.
    Tip meets the constraints: a range that surely breaks one, or an objective range entirely worse.
    Tip does not: a range entirely worse on the infeasible ordering."""
    if not tipm:
        return False
    tip_ok = ar.valid(t, tipm) and ar.feasible(t, tipm)
    if tip_ok:
        for c in t.get("constraints", []):
            r = f["metrics"].get(c["metric"])
            if r and not isinstance(c["value"], bool):
                if c["op"] in (">=", ">") and r[2] < c["value"] or c["op"] in ("<=", "<") and r[0] > c["value"]:
                    return True
    spec = t["objective"] if tip_ok else (t.get("infeasible_order") or t["objective"])
    r, b = f["metrics"].get(spec["metric"]), tipm.get(spec["metric"])
    if r and isinstance(b, (int, float)) and not isinstance(b, bool):
        return r[0] > b if spec["direction"] == "minimize" else r[2] < b
    return False


def score_new():
    """Ledger rows for logged experiments whose idea was forecast before it started."""
    ar, t = load_ar(), R.read_json(R.TASK_FILE)
    done = {row["exp"] for row in read_jsonl(LEDGER_FILE)}
    fcs = read_jsonl(FORECASTS_FILE)
    preds = {p["exp"]: p for p in R.read_tsv(R.PRED_FILE, R.PRED_COLS)} if os.path.exists(R.PRED_FILE) else {}
    vetoes = state().get("audits", [])
    new = []
    for row in R.results():
        if row["exp"] in done or row["status"] == "crash" or row.get("tag", "-") in ("", "-"):
            continue
        run = R.read_json(os.path.join(R.RUNS_DIR, f"{row['exp']}.json"), {}) or {}
        fs = last_forecasts_before(fcs, row["tag"], run.get("started", 0))
        if not fs:
            continue
        measured = run.get("metrics", {})
        tipm = None
        before = [r for r in R.results() if R.exp_num(r["exp"]) < R.exp_num(row["exp"]) and r["status"] == "keep"]
        if before:
            tipm = ar.row_metrics(t, before[-1])
        hits, misses_all = {}, []
        for metric in numeric_metrics(t):
            v = measured.get(metric)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            inside = {mid: f["metrics"][metric][0] <= v <= f["metrics"][metric][2]
                      for mid, f in fs.items() if metric in f["metrics"]}  # fmt: skip
            hits[metric] = inside
            if inside and not any(inside.values()):
                misses_all.append(metric)
        st = idea_stats(ar, t, tipm, fs)
        obj = t["objective"]["metric"]
        v, med = measured.get(obj), st["median"]
        cons_hit = med is not None and isinstance(v, (int, float)) and med[0] <= v <= med[2]
        half = (med[2] - med[0]) / 2 if med else 0
        z = (v - med[1]) / half if med and half > 0 and isinstance(v, (int, float)) else None
        agent = preds.get(row["exp"], {}).get("pred", "")
        new.append({
            "exp": row["exp"], "idea": row["tag"], "status": row["status"], "measured": measured,
            "forecasts": {mid: {"metrics": f["metrics"], "verdict": f["verdict"], "round": f["round"]} for mid, f in fs.items()},
            "hits": hits, "counterexample": misses_all, "stats": st, "consensus_hit": cons_hit,
            "z": round(z, 3) if z is not None else None,
            "agent_hit": (f"{obj}:hit" in agent) if f"{obj}:" in agent else None,
            "audit": row["tag"] in vetoes, "description": row["description"][:300],
        })  # fmt: skip
    append_jsonl(LEDGER_FILE, new)
    return new


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def health():
    """Calibration of the committee, from the ledger."""
    led = read_jsonl(LEDGER_FILE)
    t = R.read_json(R.TASK_FILE)
    obj = t["objective"]["metric"]
    out = {"scored": len(led)}
    if not led:
        return out
    out["consensus_hit"] = sum(r["consensus_hit"] for r in led) / len(led)
    agent = [r["agent_hit"] for r in led if r["agent_hit"] is not None]
    out["agent_hit"] = sum(agent) / len(agent) if agent else None
    out["counterexamples"] = sum(bool(r["counterexample"]) for r in led)
    dis = [r["stats"]["conflict"] + r["stats"]["verdict_entropy"] for r in led]
    out["auroc_disagreement_vs_miss"] = auroc(dis, [not r["consensus_hit"] for r in led])
    out["auroc_improve_votes_vs_keep"] = auroc([r["stats"]["improve_votes"] / max(1, r["stats"]["n"]) for r in led],
                                               [r["status"] == "keep" for r in led])  # fmt: skip
    per = {}
    for r in led:
        for mid, ok in r["hits"].get(obj, {}).items():
            per.setdefault(mid, []).append(ok)
    out["members"] = {mid: {"n": len(v), "hit": sum(v) / len(v), "recent": sum(v[-PRUNE_WINDOW:]) / len(v[-PRUNE_WINDOW:])}
                      for mid, v in per.items()}  # fmt: skip
    audits = [r for r in led if r["audit"]]
    out["veto_audits"] = [(r["exp"], r["status"]) for r in audits]
    zs = [r["z"] for r in led[-5:] if r["z"] is not None]
    out["recent_bias_z"] = sum(zs) / len(zs) if zs else None
    return out


def veto_active(h=None):
    """(active, why): the veto acts only on a committee that has shown it can forecast, and stops if
    its audits show it rejects ideas that would have been kept."""
    h = h or health()
    if h.get("scored", 0) < VETO_MIN_SCORED:
        return False, f"veto inactive until {VETO_MIN_SCORED} results are scored"
    if h.get("consensus_hit", 0) < VETO_MIN_HIT:
        return False, f"veto inactive: consensus ranges hit {h['consensus_hit']:.0%} (< {VETO_MIN_HIT:.0%})"
    kept = [e for e, st in h.get("veto_audits", []) if st == "keep"]
    if kept and len(kept) * 3 >= len(h["veto_audits"]):
        return False, f"veto stopped: audits {', '.join(kept)} were kept (false rejects)"
    return True, "veto active"


def disagreement_trusted(h=None):
    h = h or health()
    a = h.get("auroc_disagreement_vs_miss")
    return h.get("scored", 0) < AUROC_N or a is None or a >= AUROC_MIN


# ---------------------------------------------------------------------------
# a round: every member updates its world model (on new evidence) and forecasts the queue
# ---------------------------------------------------------------------------


def scorecard(mid, n=12):
    lines = []
    for r in read_jsonl(LEDGER_FILE)[-40:]:
        f = r["forecasts"].get(mid)
        if not f:
            continue
        parts = []
        for metric, rng in f["metrics"].items():
            v = r["measured"].get(metric)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                parts.append(f"{metric}: you [{rng[0]:g}, {rng[1]:g}, {rng[2]:g}] measured {v:g} "
                             f"{'hit' if rng[0] <= v <= rng[2] else 'MISS'}")  # fmt: skip
        lines.append(f"{r['exp']} ({r['idea']}, {r['status']}): " + "; ".join(parts) + f"; you voted {f['verdict']}")
    return "\n".join(lines[-n:]) or "(nothing scored yet)"


def counterexamples(n=6):
    rows = [r for r in read_jsonl(LEDGER_FILE) if r["counterexample"]][-n:]
    out = []
    for r in rows:
        ms = ", ".join(f"{m}={r['measured'].get(m)}" for m in r["counterexample"])
        out.append(f"{r['exp']} ({r['idea']}): {r['description'][:200]} -> measured {ms}, outside every member's range")
    return out


def recipe(t):
    """The editable files at the branch tip: what every idea is a change to."""
    tip = R.tip()
    ref = tip["commit"] if tip else "HEAD"

    def git(*a):
        return subprocess.run(["git", *a], cwd=R.ROOT, capture_output=True, text=True).stdout

    out, used = [], 0
    for path in git("ls-tree", "-r", "--name-only", ref, "--", *t.get("editable", [])).split():
        body = git("show", f"{ref}:{path}")
        if "\0" in body:
            continue
        body = body[: max(0, RECIPE_CHARS - used)]
        used += len(body)
        out.append(f"--- {path} ---\n{body}")
        if used >= RECIPE_CHARS:
            out.append("(truncated)")
            break
    return "\n".join(out) or "(no editable files)"


def member_prompt(m, ideas, t, update):
    obj = t["objective"]
    names = numeric_metrics(t)
    tip = R.tip()
    tipm = {k2: tip[k2] for k2 in names if tip and tip.get(k2, "") != ""}
    lens = {**LENSES, **RESEED_LENSES}[m["lens"]]
    wm = R.read_text(wm_path(m["id"]))
    queue = "\n".join(f"{i['id']}\t{i['hypothesis']}\t{i['description']}" for i in ideas)
    cex = counterexamples() if m["lens"] in ("counterexample", "contrarian") else []
    hypos = [h for h in R.read_tsv(R.HYPO_FILE, R.HYPO_COLS) if h["status"] == "open"]
    tree = "\n".join(f"{h['id']}\t{h['statement']}" for h in hypos) or "(none)"
    return f"""ROLE: committee-member. You are member {m['id']}, one of several independent forecasters of an
autonomous research loop. You never see the other members; disagreeing with them is useful.
Do not use tools.

YOUR LENS: {lens}

{R.inputs_task()}
The forecast metrics: {', '.join(names)}. The current best (the branch tip): {json.dumps(tipm) or 'none yet'}.

=== RESULTS (results.tsv: every experiment, measured by the task) ===
{R.read_text(R.RESULTS_FILE) or '(no experiments yet)'}
=== THE CURRENT BEST (the editable files at the branch tip; every idea is a change to these) ===
{recipe(t)}
=== OPEN HYPOTHESES ===
{tree}
=== YOUR WORLD MODEL (from your previous round) ===
{wm or '(you are a new member: build your world model from the results)'}
=== YOUR SCORECARD (your past forecasts against what was measured) ===
{scorecard(m['id'])}
{('=== COUNTEREXAMPLES (results no member predicted) ===' + chr(10) + chr(10).join(cex) + chr(10)) if cex else ''}
=== IDEAS TO FORECAST (id, hypothesis, description) ===
{queue or '(none)'}

{'1. Update your world model with the new results and your misses: the mechanisms you believe, each with the experiments that support it, the run-to-run noise, and what you are unsure of. At most 600 words; return it whole as world_model.' + chr(10) if update else ''}2. For EVERY idea above (use its id exactly, e.g. "I002"): if it were run next on top of the
   current best, what would be measured?
   Give each forecast metric as lo, point, hi: an 80% range (honest about noise: about 8 in 10 of
   your ranges should contain the measured value). Vote whether the result will support or refute
   the idea's hypothesis (or be inconclusive). One-line reason.
3. For each open hypothesis: will it end up supported, refuted, or is it unknown to you?
4. Cruxes: 1-3 questions whose answers would most change your forecasts (the {obj['metric']} objective
   is to {obj['direction']})."""


def call_member(m, ideas, t, update):
    out, cost = R.llm(f"committee-{m['id']}", member_prompt(m, ideas, t, update), FORECAST_SCHEMA,
                      model=m["model"], effort=EFFORT)  # fmt: skip
    return m, out, cost


def run_round(kind, ideas_filter=None):
    """kind 'update' (after an experiment) or 'forecast' (new ideas only). Returns a summary."""
    t = R.read_json(R.TASK_FILE)
    ar = load_ar()
    os.makedirs(DIR, exist_ok=True)
    scored = score_new() if kind == "update" else []
    replaced = maintain() if kind == "update" else []
    ms = ensure_members()
    queued = [i for i in R.read_tsv(R.IDEAS_FILE, R.IDEA_COLS) if i["status"] == "queued"]
    if ideas_filter is not None:
        queued = [i for i in queued if i["id"] in ideas_filter]
    if not queued and kind == "forecast":
        return "no new ideas to forecast"
    names = set(numeric_metrics(t))
    rnd = len(read_jsonl(ROUNDS_FILE)) + 1
    with ThreadPoolExecutor(len(ms)) as pool:
        futs = [pool.submit(call_member, m, queued, t, kind == "update") for m in ms]
        res, failed = [], []
        for f in futs:
            try:
                res.append(f.result())
            except Exception as exc:  # a failed member sits this round out
                failed.append(f"{type(exc).__name__}: {str(exc)[:120]}")
    now, rows, cost, cruxes, hvotes = time.time(), [], 0.0, {}, {}
    ids = {i["id"] for i in queued}
    for m, out, c in res:
        cost += c
        if kind == "update" and out.get("world_model"):
            with open(wm_path(m["id"]), "w") as f:
                f.write(f"<!-- {m['id']} ({m['lens']}, {m['model']}), after {R.last_exp()} -->\n" + out["world_model"])
        for fc in out.get("forecasts", []):
            met = {x["metric"]: sorted([x["lo"], x["point"], x["hi"]]) for x in fc["metrics"] if x["metric"] in names}
            met = {k2: [v[0], v[1], v[2]] for k2, v in met.items()}
            m_id = re.match(r"\s*(I\d+)", fc["idea"])  # some models append the description to the id
            iid = m_id.group(1) if m_id else fc["idea"]
            if iid in ids and met:
                rows.append({"round": rnd, "ts": now, "after": R.last_exp(), "member": m["id"], "idea": iid,
                             "metrics": met, "verdict": fc["verdict"], "reason": fc["reason"][:300]})  # fmt: skip
        cruxes[m["id"]] = out.get("cruxes", [])[:3]
        for hv in out.get("hypotheses", []):
            hvotes.setdefault(hv["id"], {})[m["id"]] = hv["outcome"]
    append_jsonl(FORECASTS_FILE, rows)
    tipm = tip_metrics(ar, t)
    stats = {}
    for i in queued:
        fs = {r["member"]: r for r in rows if r["idea"] == i["id"]}
        if fs:
            stats[i["id"]] = idea_stats(ar, t, tipm, fs)
    actions = act(stats, len(res), kind)
    collapsed = bool(stats) and all(s["conflict"] == 0 and s["verdict_entropy"] == 0 for s in stats.values())
    if kind == "update":
        set_state(collapse_streak=(state().get("collapse_streak", 0) + 1) if collapsed else 0)
    append_jsonl(ROUNDS_FILE, [{"round": rnd, "ts": now, "after": R.last_exp(), "kind": kind,
                                "members": [m["id"] for m, _, _ in res], "failed": failed, "cost": round(cost, 4),
                                "stats": stats, "cruxes": cruxes, "hypotheses": hvotes, "actions": actions,
                                "scored": [r["exp"] for r in scored], "replaced": replaced}])  # fmt: skip
    write_report()
    cex = [f"{r['exp']} outside every range ({', '.join(r['counterexample'])})" for r in scored if r["counterexample"]]
    summary = (f"round {rnd} ({kind}): {len(res)}/{len(ms)} members, {len(stats)} ideas forecast"
               + (f"; scored {', '.join(r['exp'] for r in scored)}" if scored else "")
               + (f"; COUNTEREXAMPLE {'; '.join(cex)}" if cex else "")
               + (f"; {'; '.join(actions)}" if actions else "") + (f"; {'; '.join(replaced)}" if replaced else "")
               + (f"; {len(failed)} member call(s) failed" if failed else ""))  # fmt: skip
    R.journal("committee", cost, summary)
    return summary


def act(stats, n_members, kind):
    """Veto clearly harmful ideas (with audits) and raise the most disputed one. Returns what was done."""
    if n_members < 3 or not stats:
        return []
    done, st = [], state()
    h = health()
    veto, _ = veto_active(h)
    with R.locked():
        ideas = R.read_tsv(R.IDEAS_FILE, R.IDEA_COLS)
        for i in ideas:
            s = stats.get(i["id"])
            if not s or i["status"] != "queued" or i["id"] in st.get("audits", []):
                continue
            if veto and s["improve_votes"] == 0 and s["harm_votes"] >= math.ceil(0.75 * s["n"]):
                count = st.get("vetoes", 0) + 1
                st["vetoes"] = count
                if count % VETO_AUDIT_EVERY == 0:
                    st.setdefault("audits", []).append(i["id"])
                    i["note"] = (i["note"] + " | " if i["note"] else "") + "committee veto AUDIT: run it to measure the veto"
                    done.append(f"veto audit {i['id']}")
                else:
                    i["status"] = "dropped"
                    i["note"] = f"committee veto: {s['harm_votes']}/{s['n']} members forecast clear harm, none an improvement"
                    done.append(f"vetoed {i['id']}")
        if kind == "update" and disagreement_trusted(h):
            live = [(s["conflict"] + s["verdict_entropy"], iid) for iid, s in stats.items()
                    if s["improve_votes"] > 0 and any(i["id"] == iid and i["status"] == "queued" for i in ideas)]  # fmt: skip
            if live:
                info, iid = max(live)
                if info >= 0.8:
                    for i in ideas:
                        if i["id"] == iid and int(i["priority"] or 0) < 5 and "most disputed" not in i["note"]:
                            i["priority"] = str(int(i["priority"] or 0) + 1)
                            i["note"] = (i["note"] + " | " if i["note"] else "") + f"committee: most disputed (info {info:.2f}), +1 priority"
                            done.append(f"raised {iid} (disputed, info {info:.2f})")
        R.write_tsv(R.IDEAS_FILE, R.IDEA_COLS, ideas)
    set_state(vetoes=st.get("vetoes", 0), audits=st.get("audits", []))
    return done


def maintain():
    """Replace a member that keeps missing; reseed one when the committee has stopped disagreeing."""
    ms = members(active_only=False)
    if not ms:
        return []
    h = health()
    active = [m for m in ms if m["status"] == "active"]
    out = []
    per = h.get("members", {})
    worst = sorted([(v["recent"], mid) for mid, v in per.items()
                    if v["n"] >= PRUNE_WINDOW and v["recent"] <= PRUNE_HITRATE and any(m["id"] == mid for m in active)])  # fmt: skip
    reason = None
    if worst:
        reason = f"{worst[0][1]} hit {worst[0][0]:.0%} of its last {PRUNE_WINDOW} forecasts"
        victim = worst[0][1]
    elif state().get("collapse_streak", 0) >= COLLAPSE_ROUNDS:
        victim = min(active, key=lambda m: (per.get(m["id"], {}).get("hit", 1.0), m["id"]))["id"]
        reason = f"no disagreement for {COLLAPSE_ROUNDS} rounds: reseeding {victim}"
        set_state(collapse_streak=0)
    if reason:
        for m in ms:
            if m["id"] == victim:
                m["status"], m["retired"] = "retired", reason
        used = {m["lens"] for m in ms if m["status"] == "active"}
        lens = next((x for x in RESEED_LENSES if x not in used), "counterexample")
        nm = new_member(ms, lens, reason)
        out.append(f"replaced {victim} by {nm['id']} ({lens}): {reason}")
        R.write_json(MEMBERS_FILE, ms)
    return out


# ---------------------------------------------------------------------------
# what the rest of the framework reads
# ---------------------------------------------------------------------------


def latest_round(kind=None):
    rs = [r for r in read_jsonl(ROUNDS_FILE) if kind is None or r["kind"] == kind]
    return rs[-1] if rs else None


def current_forecasts():
    """{idea: {member: forecast}} from each member's newest forecast per idea."""
    out = {}
    for f in read_jsonl(FORECASTS_FILE):
        out.setdefault(f["idea"], {})[f["member"]] = f
    return out


def revise_reasons(last_revise_exp):
    """Reasons from the committee that make a revise due."""
    led = [r for r in read_jsonl(LEDGER_FILE) if R.exp_num(r["exp"]) > last_revise_exp and r["counterexample"]]
    if len(led) >= COUNTEREXAMPLES_FOR_REVISE:
        return [f"{len(led)} counterexamples (results outside every committee member's range: "
                f"{', '.join(r['exp'] for r in led)})"]  # fmt: skip
    return []


def lit_question():
    """A literature request on the newest counterexample, if one is due (rate-limited)."""
    led = read_jsonl(LEDGER_FILE)
    if not led or not led[-1]["counterexample"]:
        return None
    last = state().get("lit_after", "")
    if last and R.exp_num(R.last_exp()) - R.exp_num(last) < LIT_EVERY:
        return None
    r = led[-1]
    rnd = latest_round("update") or {}
    cr = [c for cs in (rnd.get("cruxes") or {}).values() for c in cs][:6]
    ranges = "; ".join(f"{mid} {f['metrics']}" for mid, f in r["forecasts"].items())
    ms = ", ".join(f"{m}={r['measured'].get(m)}" for m in r["counterexample"])
    q = (f"What mechanism explains this result, which none of our forecasters predicted? Experiment {r['exp']}: "
         f"{r['description'][:300]} Measured {ms}; forecast ranges were {ranges[:400]}. "
         f"Open cruxes: {' | '.join(cr)[:600]}")  # fmt: skip
    set_state(lit_after=R.last_exp())
    return q


def inputs_for(role):
    """The committee's evidence for ideate or revise ('' before any round)."""
    rnd = latest_round()
    if not rnd:
        return ""
    h = health()
    lines = [f"=== UNCERTAINTY COMMITTEE ({len(members())} independent world models; their forecasts are scored on every result) ==="]
    lines.append(fmt_health(h))
    cex = counterexamples()
    if cex:
        lines.append("Counterexamples (results outside every member's range: a mechanism is missing):")
        lines += [f"  {c}" for c in cex]
    if role == "ideate":
        fc, ideas = current_forecasts(), {i["id"]: i for i in R.read_tsv(R.IDEAS_FILE, R.IDEA_COLS)}
        ar, t = load_ar(), R.read_json(R.TASK_FILE)
        tipm = tip_metrics(ar, t)
        lines.append("Forecasts for the queued ideas (median 80% range on the objective; conflict = share of member "
                     "pairs whose ranges do not overlap; votes):")  # fmt: skip
        for iid, fs in fc.items():
            if ideas.get(iid, {}).get("status") != "queued":
                continue
            s = idea_stats(ar, t, tipm, fs)
            reasons = " / ".join(f"{mid}: {f['reason'][:120]}" for mid, f in fs.items())
            lines.append(f"  {iid}: median {s['median']}, conflict {s['conflict']}, improve votes {s['improve_votes']}/{s['n']}, "
                         f"harm votes {s['harm_votes']}/{s['n']}, verdict entropy {s['verdict_entropy']} -- {reasons}")  # fmt: skip
        vetoed = [i for i in ideas.values() if i["note"].startswith("committee veto:")][-5:]
        if vetoed:
            lines.append("Vetoed (members forecast clear harm): " + "; ".join(f"{i['id']} {i['description'][:80]}" for i in vetoed))
    lines.append("Members' cruxes (what would most change their forecasts):")
    for mid, cs in (rnd.get("cruxes") or {}).items():
        lines += [f"  {mid}: {c}" for c in cs]
    if role == "revise":
        lines.append("Members' votes on the open hypotheses (supported/refuted/unknown):")
        for hid, votes in (rnd.get("hypotheses") or {}).items():
            vs = list(votes.values())
            lines.append(f"  {hid}: " + ", ".join(f"{o} {vs.count(o)}" for o in ("supported", "refuted", "unknown") if vs.count(o)))
    return "\n".join(lines) + "\n"


def fmt_health(h):
    if not h.get("scored"):
        return "Calibration: nothing scored yet."
    a = h.get("auroc_disagreement_vs_miss")
    b = h.get("auroc_improve_votes_vs_keep")
    parts = [f"{h['scored']} results scored", f"consensus range hit {h['consensus_hit']:.0%} (target ~80%)"]
    if h.get("agent_hit") is not None:
        parts.append(f"experimenter's sealed range hit {h['agent_hit']:.0%}")
    parts.append(f"counterexamples {h['counterexamples']}")
    if a is not None:
        parts.append(f"disagreement predicts misses: AUROC {a:.2f}" + ("" if disagreement_trusted(h) else " (not informative: priorities ignore it)"))
    if b is not None:
        parts.append(f"improve votes pick keeps: AUROC {b:.2f}")
    if h.get("recent_bias_z") is not None:
        parts.append(f"recent bias {h['recent_bias_z']:+.2f} half-ranges")
    parts.append(veto_active(h)[1] + (f" (audits: {h['veto_audits']})" if h.get("veto_audits") else ""))
    return "Calibration: " + "; ".join(parts) + "."


def write_report():
    h = health()
    ms = members(active_only=False)
    lines = ["# Uncertainty committee", "", fmt_health(h), "", "## Members"]
    for m in ms:
        sc = h.get("members", {}).get(m["id"])
        perf = f"hit {sc['hit']:.0%} of {sc['n']}, recent {sc['recent']:.0%}" if sc else "not scored yet"
        lines.append(f"- {m['id']} {m['status']} ({m['lens']}, {m['model']}, since {m['born']}): {perf}"
                     + (f"; retired: {m['retired']}" if m.get("retired") else ""))  # fmt: skip
    lines += ["", "## Queued ideas", "idea | median range | conflict | improve votes | harm votes | verdict entropy", "---|---|---|---|---|---"]
    ar, t = load_ar(), R.read_json(R.TASK_FILE)
    tipm = tip_metrics(ar, t)
    ideas = {i["id"]: i for i in R.read_tsv(R.IDEAS_FILE, R.IDEA_COLS)}
    for iid, fs in current_forecasts().items():
        if ideas.get(iid, {}).get("status") == "queued":
            s = idea_stats(ar, t, tipm, fs)
            lines.append(f"{iid} | {s['median']} | {s['conflict']} | {s['improve_votes']}/{s['n']} | {s['harm_votes']}/{s['n']} | {s['verdict_entropy']}")
    cex = counterexamples()
    if cex:
        lines += ["", "## Counterexamples"] + [f"- {c}" for c in cex]
    rnd = latest_round()
    if rnd:
        lines += ["", "## Cruxes (latest round)"] + [f"- {mid}: {c}" for mid, cs in rnd["cruxes"].items() for c in cs]
        lines += ["", "## Votes on open hypotheses"]
        for hid, votes in rnd["hypotheses"].items():
            lines.append(f"- {hid}: " + ", ".join(f"{mid} {o}" for mid, o in votes.items()))
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


def status_lines():
    """For `research.py status`."""
    if not latest_round():
        return ["committee: no round yet"]
    h = health()
    out = [f"committee ({len(members())} members): {fmt_health(h)}"]
    cex = counterexamples(2)
    if cex:
        out.append("  newest counterexample: " + cex[-1][:220])
    return out


def after_seal(idea, started):
    """Shown once the experimenter has sealed a pre-registration: the committee's forecast for it."""
    fs = last_forecasts_before(read_jsonl(FORECASTS_FILE), idea, started)
    if not fs:
        return []
    ar, t = load_ar(), R.read_json(R.TASK_FILE)
    s = idea_stats(ar, t, tip_metrics(ar, t), fs)
    out = [f"committee forecast for {idea} (made before launch): median {t['objective']['metric']} range {s['median']}, "
           f"conflict {s['conflict']}, improve votes {s['improve_votes']}/{s['n']}, verdict entropy {s['verdict_entropy']}"]  # fmt: skip
    for mid, f in fs.items():
        out.append(f"  {mid}: {f['metrics']} {f['verdict']}: {f['reason'][:160]}")
    return out
