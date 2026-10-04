"""Questions 2 and 3 on SciGym: is the committee's spread a calibrated uncertainty, and is
disagreement higher where the committee is wrong, at the system level and at the reaction level.

    uv run python -m scigym.calibration --model qwen
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from bioprot.conformal import aci, split_conformal
from bioprot.metrics import brier, ece
from .data import load_systems
from .loop import ART, ARMS
from .report import auroc, load


def system_level(rows: list[dict], alpha: float = 0.1, gamma: float = 0.05) -> dict:
    """One minus the spread as the committee's confidence that its answer is right (F1 at least 0.5):
    reliability, and the R22 wrapper (split conformal over systems, and the online rule in id order)."""
    rows = sorted(rows, key=lambda r: r["system"])
    f1 = np.array([r["rms_medoid"]["f1"] if r["rms_medoid"] else 0.0 for r in rows])
    spread = np.array([r["spread"] if r["spread"] is not None else 1.0 for r in rows])
    correct = f1 >= 0.5
    conf = 1 - spread
    out = {"n": int(len(rows)), "share_correct": float(correct.mean()), "ece": ece(conf, correct, 5), "brier": brier(conf, correct),
           "auroc_spread_vs_wrong": auroc(spread, ~correct)}
    order = np.argsort(conf)
    bins = np.array_split(order, 4)
    out["reliability"] = [{"mean_conf": float(conf[b].mean()), "share_correct": float(correct[b].mean()), "n": int(len(b))} for b in bins if len(b)]
    if correct.any() and (~correct).any():
        out["split_conformal"] = split_conformal(spread, correct, [r["system"] for r in rows], alpha)
        out["aci"] = aci(spread, correct, alpha, gamma)
    return out


def reaction_level(rows: list[dict], systems: dict) -> dict:
    """Every reaction any member proposed, with the share of members behind it and whether it is in
    the truth set: the effect-row flag of the ARC committee, here per reaction."""
    shares, true = [], []
    per_system = []
    for r in rows:
        truth = Counter(tuple(map(tuple, k)) for k in (rr.key for rr in systems[r["system"]].truth_reactions))
        members = [set(tuple(map(tuple, k)) for k in m["keys"]) for m in r["members"]]
        if not members:
            continue
        proposed = Counter(k for ks in members for k in ks)
        for k, n in proposed.items():
            shares.append(n / len(members))
            true.append(k in truth)
        una = [k for k, n in proposed.items() if n == len(members)]
        per_system.append({"system": r["system"], "n_proposed": len(proposed), "n_unanimous": len(una),
                           "precision_unanimous": float(np.mean([k in truth for k in una])) if una else None,
                           "precision_any": float(np.mean([k in truth for k in proposed]))})
    s, y = np.array(shares), np.array(true, bool)
    una = s >= 1 - 1e-9
    out = {"n_reactions": int(len(s)), "n_systems": len(per_system), "auroc_share_vs_true": auroc(s, y),
           "ece_share_as_p_true": ece(s, y, 5), "precision_unanimous": float(y[una].mean()) if una.any() else None,
           "n_unanimous": int(una.sum()), "precision_split": float(y[~una].mean()) if (~una).any() else None, "n_split": int((~una).sum())}
    order = np.argsort(s)
    out["reliability"] = [{"mean_share": float(s[b].mean()), "precision": float(y[b].mean()), "n": int(len(b))} for b in np.array_split(order, 4) if len(b)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--tag", default="")
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    systems = {s.id: s for s in load_systems()}
    out = {}
    for arm in ARMS:
        rows = load(arm, a.model, a.tag)
        if not rows:
            continue
        sl, rl = system_level(rows, a.alpha), reaction_level(rows, systems)
        out[arm] = {"system": sl, "reaction": rl}
        print(f"\n{arm} ({a.model}{a.tag}, n {sl['n']}, share with F1 >= 0.5: {sl['share_correct']:.2f})")
        print(f"  system level, confidence = 1 - spread: ECE {sl['ece']:.3f}, Brier {sl['brier']:.3f}, AUROC spread vs wrong {sl['auroc_spread_vs_wrong']}")
        print("  reliability (mean confidence -> share correct, n):", [(round(b["mean_conf"], 2), round(b["share_correct"], 2), b["n"]) for b in sl["reliability"]])
        if "split_conformal" in sl:
            c, o = sl["split_conformal"], sl["aci"]
            print(f"  split conformal, target {1 - a.alpha:.2f}: coverage {c['coverage']:.3f} [{c['coverage_ci'][0]:.2f}, {c['coverage_ci'][1]:.2f}], "
                  f"committed {c['committed']:.2f}, accuracy when committed {c['accuracy_when_committed']}, commits to 'right' {c['passed']:.2f} with error {c['error_among_passed']}, abstain {c['abstain']:.2f}")
            print(f"  online (ACI) in id order: coverage {o['coverage']:.3f}, committed {o['committed']:.2f}, abstain {o['abstain']:.2f}, accuracy when committed {o['accuracy_when_committed']}")
        print(f"  reaction level: {rl['n_reactions']} proposed reactions over {rl['n_systems']} systems; AUROC share vs true {rl['auroc_share_vs_true']}, "
              f"ECE of share as P(true) {rl['ece_share_as_p_true']:.3f}; precision unanimous {rl['precision_unanimous']} (n {rl['n_unanimous']}) against split {rl['precision_split']} (n {rl['n_split']})")
        print("  reaction reliability (mean share -> precision, n):", [(round(b["mean_share"], 2), round(b["precision"], 2), b["n"]) for b in rl["reliability"]])
    (ART / f"calibration_{a.model}{a.tag}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
