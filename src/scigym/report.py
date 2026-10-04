"""Arms side by side: reaction score, trajectory error, admission, and disagreement against error.

    uv run python -m scigym.report --model gptoss
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from bioprot.metrics import bootstrap
from .loop import ARMS, ART

PAPER = {"Gemini-2.5-Pro": (0.1817, 0.3212), "GPT-4.1": (0.1740, 0.4611), "Claude-3.7-Sonnet": (0.1688, 0.3615),
         "Claude-3.5-Haiku": (0.0530, 0.6281)}  # RMS F1, STE on the 137 small systems, 20 iterations (Duan et al., 2025)


def load(arm: str, model: str, tag: str = "") -> list[dict]:
    d = ART / f"{arm}_{model}{tag}"
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []
    return [r for r in rows if "error" not in r]


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or not y.any():
        return None
    return float((s[y][:, None] > s[~y][None, :]).mean() + 0.5 * (s[y][:, None] == s[~y][None, :]).mean())


def summarise(rows: list[dict]) -> dict:
    f1 = np.array([r["rms_medoid"]["f1"] if r["rms_medoid"] else 0.0 for r in rows])
    f1m = np.array([r["rms_majority"]["f1"] for r in rows])
    best = np.array([r["rms_best_member"] or 0.0 for r in rows])
    s = np.array([r["ste_medoid"] for r in rows])
    sp = np.array([r["ste_partial"] for r in rows])
    ids = np.array([r["system"] for r in rows])
    ci = lambda x: list(bootstrap(ids, lambda i: float(np.mean(x[i])), 1000))
    spread = [r["spread"] for r in rows]
    wrong = [f < 0.5 for f in f1]
    return {"n": len(rows), "rms_f1": float(f1.mean()), "rms_f1_ci": ci(f1), "rms_f1_majority": float(f1m.mean()),
            "rms_f1_best_member": float(best.mean()), "ste": float(s.mean()), "ste_ci": ci(s), "ste_partial": float(sp.mean()),
            "any_admitted": float(np.mean([r["any_admitted"] for r in rows])),
            "admitted_members": float(np.mean([r["n_admitted"] / max(1, r["n_members"]) for r in rows])),
            "calls": float(np.mean([r["calls"] for r in rows])),
            "auroc_spread_vs_wrong": auroc([x if x is not None else 1.0 for x in spread], wrong),
            "unanimous": {"n": int(sum(1 for x in spread if x == 0)), "f1": float(np.mean([f for f, x in zip(f1, spread) if x == 0])) if any(x == 0 for x in spread) else None},
            "split": {"n": int(sum(1 for x in spread if x)), "f1": float(np.mean([f for f, x in zip(f1, spread) if x])) if any(spread) else None}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptoss")
    ap.add_argument("--tag", default="", help="directory suffix of a tolerance arm, e.g. _eps50 (set SCIGYM_EPS to match)")
    a = ap.parse_args()
    out = {}
    print("| Arm | n | RMS F1 medoid [CI] | majority | best member | STE [CI] | STE of the partial model | any admitted | admitted share | calls | AUROC spread vs F1<0.5 | unanimous n, F1 | split n, F1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    common = None
    per_arm = {arm: load(arm, a.model, a.tag) for arm in ARMS}
    for rows in per_arm.values():
        if not rows:  # an arm that was not run does not empty the paired set
            continue
        ids = {r["system"] for r in rows}
        common = ids if common is None else common & ids
    from .model import EPS
    for arm, rows in per_arm.items():
        rows = [r for r in rows if r["system"] in (common or set())]
        if not rows:
            continue
        e = summarise(rows)
        # systems where the empty model already sits inside the tolerance cannot refute anything
        informative = [r for r in rows if r["ste_partial"] > EPS]
        e["informative"] = summarise(informative) if len(informative) >= 3 else None
        out[arm] = e
        u, sp = e["unanimous"], e["split"]
        print(f"| {arm} | {e['n']} | {e['rms_f1']:.3f} [{e['rms_f1_ci'][0]:.3f}, {e['rms_f1_ci'][1]:.3f}] | {e['rms_f1_majority']:.3f} | {e['rms_f1_best_member']:.3f} | "
              f"{e['ste']:.3f} [{e['ste_ci'][0]:.3f}, {e['ste_ci'][1]:.3f}] | {e['ste_partial']:.3f} | {e['any_admitted']:.2f} | {e['admitted_members']:.2f} | {e['calls']:.0f} | "
              f"{e['auroc_spread_vs_wrong'] if e['auroc_spread_vs_wrong'] is None else round(e['auroc_spread_vs_wrong'], 2)} | {u['n']}, {u['f1'] if u['f1'] is None else round(u['f1'], 2)} | {sp['n']}, {sp['f1'] if sp['f1'] is None else round(sp['f1'], 2)} |")
    print("\nInformative systems only (the empty model's trajectory error is above the tolerance):")
    for arm, e in out.items():
        i = e.get("informative")
        if i:
            print(f"  {arm}: n {i['n']}, RMS F1 {i['rms_f1']:.3f} [{i['rms_f1_ci'][0]:.3f}, {i['rms_f1_ci'][1]:.3f}], STE {i['ste']:.3f} (partial {i['ste_partial']:.3f}), any admitted {i['any_admitted']:.2f}, AUROC {i['auroc_spread_vs_wrong']}")
    if "committee_probe" in out and "committee_fixed" in out:
        a_rows = {r["system"]: r for r in per_arm["committee_probe"]}
        b_rows = {r["system"]: r for r in per_arm["committee_fixed"]}
        ids = sorted(common)
        d = np.array([(a_rows[i]["rms_medoid"]["f1"] if a_rows[i]["rms_medoid"] else 0) - (b_rows[i]["rms_medoid"]["f1"] if b_rows[i]["rms_medoid"] else 0) for i in ids])
        lo, hi = bootstrap(np.array(ids), lambda idx: float(np.mean(d[idx])), 1000)
        out["paired_probe_minus_fixed_f1"] = {"mean": float(d.mean()), "ci": [lo, hi], "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
        print(f"\npaired, probe minus fixed, RMS F1: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], wins {int((d > 0).sum())}, losses {int((d < 0).sum())}, ties {int((d == 0).sum())}")
    print("\npaper, 137 small systems, 20 iterations (RMS F1, STE):", ", ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in PAPER.items()))
    (ART / f"summary_{a.model}{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
