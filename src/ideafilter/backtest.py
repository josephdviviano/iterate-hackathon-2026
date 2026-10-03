"""Backtest the idea-filter committee on the evaluated arms of the speedrun programme.

For each arm the members see the recipe, the directives, and every finding
that does not mention the arm's sweep (leave-one-sweep-out). Predictions are
appended to a JSONL file so a run can resume. Scoring: AUROC and Brier of the
mean P(positive) against the measured label, disagreement against error, and
the sweeps a reject threshold would have saved without losing a positive.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from .committee import aggregate, predict

S = Path("/private/tmp/claude-501/-Users-jdv-code-iterate/086a2bad-2d8b-489a-b75f-0f68d03c3e20/scratchpad")
LOOP = S / "loop"


def findings() -> list[dict]:
    out = []
    for p in sorted((LOOP / "tickets/PROGRAMME/findings").glob("F-*.yaml")):
        d = yaml.safe_load(p.read_text())
        out.append({"id": d["finding_id"], "observation": d.get("observation", ""), "interpretation": d.get("interpretation", ""),
                    "decision": d.get("decision_implication", ""), "kind": d.get("result_kind", "")})
    return out


def sweep_code(sweep: str) -> str:
    m = re.match(r"([a-z]+)(\d+)", sweep)
    return (m.group(1).upper() + m.group(2)) if m else sweep.upper()


def context_for(sweep: str, fnd: list[dict]) -> str:
    code = sweep_code(sweep)
    keep = [f for f in fnd if not re.search(rf"\b{code}\b", f["observation"] + f["interpretation"] + f["decision"])]
    recipe = (LOOP / "ctx/config.py").read_text()
    lab = (LOOP / "ctx/lab_config.py").read_text() if (LOOP / "ctx/lab_config.py").exists() else ""
    directives = (LOOP / "ctx/DIRECTIVES.md").read_text()
    lines = [f"- {f['id']} ({f['kind']}): {f['observation']} Interpretation: {f['interpretation']}" for f in keep]
    return ("## Recipe parameters (the submitted defaults)\n\n```python\n" + recipe + "\n```\n\n## Lab recipe levers (the names used in sweep arms; "
            "a lever absent from the submitted defaults is OFF in the control)\n\n```python\n" + lab + "\n```\n\n## Team directives\n\n" + directives +
            "\n\n## Prior findings (every one that does not concern this sweep)\n\n" + "\n".join(lines))


def describe(arm: dict) -> str:
    changes = "; ".join(f"{k} = {v} (control: {arm['base'].get(k, 'the lever is off, or the submitted default')})" for k, v in arm["levers"].items())
    return f"Change to the recipe, tested in one sweep arm of {arm['n']} seeds: {changes}."


def run(args) -> None:
    arms = json.load(open(S / "ideas_dataset.json"))
    if args.min_seeds:
        arms = [a for a in arms if a["n"] >= args.min_seeds]
    if args.sweeps:
        arms = [a for a in arms if a["sweep"] in args.sweeps]
    out = S / f"backtest_{args.tag}.jsonl"
    done = {json.loads(l)["key"] for l in out.read_text().splitlines()} if out.exists() else set()
    fnd = findings()
    ctx_cache: dict = {}
    print(f"{len(arms)} arms, {len(done)} done")
    for i, arm in enumerate(arms):
        key = f"{arm['sweep']}/{arm['arm']}"
        if key in done:
            continue
        ctx = ctx_cache.setdefault(arm["sweep"], context_for(arm["sweep"], fnd))
        preds = predict(describe(arm), ctx, model=args.model, parallel=4)
        agg = aggregate(preds)
        rec = {"key": key, "sweep": arm["sweep"], "levers": arm["levers"], "n": arm["n"], "dpp": arm["dpp"], "dtime": arm["dtime"],
               "label": arm["label"], **agg}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{i+1}/{len(arms)} {key:34s} label {arm['label']:8s} dpp {arm['dpp']:+.2f}  p_mean {agg['p_mean']:.2f} spread {agg['spread']:.2f}", flush=True)


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg: return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def score(args) -> None:
    recs = [json.loads(l) for l in (S / f"backtest_{args.tag}.jsonl").read_text().splitlines()]
    y = [r["label"] == "positive" for r in recs]; p = [r["p_mean"] for r in recs]
    print(f"n={len(recs)} positives={sum(y)}")
    print(f"AUROC p_mean: {auroc(p, y)}  Brier: {sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(recs):.3f}  base-rate Brier: {sum((sum(y)/len(y) - yi) ** 2 for yi in y)/len(y):.3f}")
    for m in recs[0]["members"]:
        name = m["member"]; pm = [next(x["p_positive"] for x in r["members"] if x["member"] == name) for r in recs]
        print(f"  member {name:10s} AUROC {auroc(pm, y)}")
    err = [abs(pi - yi) for pi, yi in zip(p, y)]; spread = [r["spread"] for r in recs]
    print(f"AUROC of spread vs |error|>0.5: {auroc(spread, [e > 0.5 for e in err])}")
    for thr in (0.1, 0.2, 0.3, 0.4):
        rej = [r for r in recs if r["p_mean"] < thr and r["p_max"] < thr + 0.2]
        lost = sum(r["label"] == "positive" for r in rej)
        print(f"reject if p_mean<{thr} and p_max<{thr+0.2}: skip {len(rej)}/{len(recs)} sweeps ({len(rej)/len(recs):.0%}), positives lost {lost}")
    print("calibration buckets (p_mean -> rate):")
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        b = [yi for pi, yi in zip(p, y) if lo <= pi < lo + 0.2 + (1e-9 if lo == 0.8 else 0)]
        if b: print(f"  [{lo:.1f},{lo+0.2:.1f}) n={len(b):3d} positive rate {sum(b)/len(b):.2f}")


def describe_hyp(h: dict) -> str:
    return (f"Hypothesis {h['id']} ({h['exploration']}): {h['proposition']}\nMechanism: {h.get('mechanism')}\n"
            f"Prediction by its author: {h.get('prediction')}\nWhat would disconfirm it: {h.get('disconfirming')}\n"
            "Decide whether a sweep testing this hypothesis, with its author's best setting, would measure POSITIVE.")


def context_for_hyp(h: dict, fnd: list[dict]) -> str:
    codes = [str(e).upper() for e in (h.get("evidence_needed") or [])] + [str(h.get("resolved_by") or "")]
    resolved = h.get("resolved_by") or ""
    pat = r"\b(" + "|".join(re.escape(c) for c in codes if re.match(r"^[A-Z]+\d+$", c)) + r")\b" if any(re.match(r"^[A-Z]+\d+$", c) for c in codes) else None
    keep = [f for f in fnd if not (pat and re.search(pat, f["observation"] + f["interpretation"] + f["decision"]))]
    if resolved.startswith("F-"):
        keep = [f for f in keep if f["id"] != resolved]
    # the climb's own levers (X-002, X-003) are described across many early findings; keep only findings from later tasks
    if h["exploration"] in ("X-002", "X-003"):
        keep = [f for f in keep if int(f["id"][2:]) > 29]
    recipe = (LOOP / "ctx/config.py").read_text()
    lab = (LOOP / "ctx/lab_config.py").read_text() if (LOOP / "ctx/lab_config.py").exists() else ""
    directives = (LOOP / "ctx/DIRECTIVES.md").read_text()
    lines = [f"- {f['id']} ({f['kind']}): {f['observation']} Interpretation: {f['interpretation']}" for f in keep]
    return ("## Recipe parameters (the submitted defaults)\n\n```python\n" + recipe + "\n```\n\n## Lab recipe levers (names used in sweeps)\n\n```python\n" + lab +
            "\n```\n\n## Team directives\n\n" + directives + "\n\n## Prior findings (those that do not resolve this hypothesis)\n\n" + "\n".join(lines))


def run_hyp(args) -> None:
    hyps = [h for h in json.load(open(S / "hypotheses_labels.json")) if h["label"] in ("positive", "negative")]
    out = S / f"backtest_{args.tag}.jsonl"
    done = {json.loads(l)["key"] for l in out.read_text().splitlines()} if out.exists() else set()
    fnd = findings()
    print(f"{len(hyps)} labelled hypotheses, {len(done)} done")
    for i, h in enumerate(hyps):
        key = f"{h['exploration']}/{h['id']}"
        if key in done:
            continue
        preds = predict(describe_hyp(h), context_for_hyp(h, fnd), model=args.model, parallel=4)
        agg = aggregate(preds)
        rec = {"key": key, "sweep": h["exploration"], "levers": {"proposition": h["proposition"]}, "n": 0, "dpp": None, "dtime": None,
               "label": h["label"], "evidence": h["evidence_line"], **agg}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{i+1}/{len(hyps)} {key:12s} label {h['label']:8s} p_mean {agg['p_mean']:.2f} spread {agg['spread']:.2f} | {h['proposition'][:60]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--tag", default="arms"); r.add_argument("--model", default="sonnet")
    r.add_argument("--min-seeds", type=int, default=0); r.add_argument("--sweeps", nargs="*"); r.set_defaults(fn=run)
    s = sub.add_parser("score"); s.add_argument("--tag", default="arms"); s.set_defaults(fn=score)
    h = sub.add_parser("run-hyp"); h.add_argument("--tag", default="hyp"); h.add_argument("--model", default="sonnet"); h.set_defaults(fn=run_hyp)
    a = ap.parse_args(); a.fn(a)
