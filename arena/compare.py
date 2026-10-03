"""Compare arms of an arena on the task's own terms (task-agnostic).

    python3 compare.py <arena dir> [--runs]

For each arm: experiments run, the best run that meets the task's constraints (by the objective),
the best run so far by the infeasible ordering, when each was reached, whether the agent is
alive, and an isolation audit (did an arm's agent touch another arm's workspace?). Metrics are
re-read from each run's log with the task's own regexes, independent of the agents' bookkeeping.
"""

import datetime as dt
import glob
import importlib.util
import json
import os
import re
import sys


def load_ar(ws):
    spec = importlib.util.spec_from_file_location("ar_" + os.path.basename(ws), os.path.join(ws, "ar.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audit(arena, arm, others):
    """Tool calls / tool outputs in this arm's session that mention another arm's workspace."""
    path = os.path.join(arena, "logs", f"{arm}.jsonl")
    pats = [re.compile(re.escape(os.path.join(arena, o)) + r"\b|\.\./" + re.escape(o) + r"\b") for o in others]
    hits = []
    for line in open(path) if os.path.exists(path) else []:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (e.get("message") or {}).get("content")
        for c in content if isinstance(content, list) else []:
            text = json.dumps(c.get("input") if c.get("type") == "tool_use" else c.get("content"))
            if c.get("type") in ("tool_use", "tool_result") and any(p.search(text) for p in pats):
                hits.append(" ".join(text.split())[:160])
    return hits


def main():
    arena = os.path.abspath(sys.argv[1])
    start = dt.datetime.fromisoformat(open(os.path.join(arena, "START")).read().strip())
    now = dt.datetime.now(dt.UTC)
    arms = sorted(d for d in os.listdir(arena) if os.path.exists(os.path.join(arena, d, "task.json")))
    print(f"arena {arena}: started {start:%H:%M} UTC, {(now - start).total_seconds() / 3600:.2f} h ago")
    for arm in arms:
        ws = os.path.join(arena, arm)
        ar = load_ar(ws)
        arm_start = start  # an arm that joined later is timed from its own START
        if os.path.exists(os.path.join(ws, "START")):
            arm_start = dt.datetime.fromisoformat(open(os.path.join(ws, "START")).read().strip())
        t = json.load(open(os.path.join(ws, "task.json")))
        obj, order = t["objective"], t.get("infeasible_order") or t["objective"]
        runs = []
        for log in sorted(glob.glob(os.path.join(ws, "runs", "E*.log"))):
            exp = os.path.basename(log)[:-4]
            m = ar.extract_metrics(t, open(log, errors="replace").read())
            done = os.path.exists(os.path.join(ws, "runs", f"{exp}.json"))
            at = (os.path.getmtime(log) - arm_start.timestamp()) / 3600
            runs.append({"exp": exp, "m": m, "ok": ar.valid(t, m), "feasible": ar.valid(t, m) and ar.feasible(t, m),
                         "done": done, "at": at})  # fmt: skip
        sign = 1 if obj["direction"] == "minimize" else -1
        feas = [r for r in runs if r["feasible"]]
        best = min(feas, key=lambda r: sign * r["m"][obj["metric"]], default=None)
        osign = 1 if order["direction"] == "minimize" else -1
        valid = [r for r in runs if r["ok"]]
        best_order = min(valid, key=lambda r: osign * r["m"][order["metric"]], default=None)
        idle = now.timestamp() - os.path.getmtime(os.path.join(arena, "logs", f"{arm}.jsonl")) \
            if os.path.exists(os.path.join(arena, "logs", f"{arm}.jsonl")) else None
        hits = audit(arena, arm, [a for a in arms if a != arm])
        print(f"\n== {arm} (since {arm_start:%H:%M}, {(now - arm_start).total_seconds() / 3600:.2f} h): {len(runs)} runs ({sum(not r['ok'] for r in runs)} without metrics); "
              f"agent log idle {idle:.0f}s; isolation {'clean' if not hits else f'FLAGGED x{len(hits)}'}"
              if idle is not None else f"\n== {arm}: {len(runs)} runs")  # fmt: skip
        if best:
            vals = ", ".join(f"{k}={v}" for k, v in best["m"].items())
            print(f"   best meeting constraints: {best['exp']} at {best['at']:.2f} h: {vals}")
        else:
            print("   best meeting constraints: none yet")
        if best_order:
            print(f"   best {order['metric']} overall: {best_order['m'][order['metric']]} ({best_order['exp']})")
        if "--runs" in sys.argv:
            for r in runs:
                vals = " ".join(f"{k}={v}" for k, v in r["m"].items())
                print(f"     {r['at']:5.2f}h {r['exp']} {'feasible' if r['feasible'] else '        '} {vals}")
        for h in hits[:3]:
            print(f"   isolation: {h}")


if __name__ == "__main__":
    main()
