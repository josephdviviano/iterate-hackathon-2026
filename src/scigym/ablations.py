"""Offline ablations on the stored SciGym runs: admission against the tolerance, selective
prediction by committee spread, and the per-step curves of the probe and fixed arms.

    uv run python -m scigym.ablations --model qwen
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .loop import ART
from .report import load


def admission_vs_eps(rows: list[dict], eps_grid=(0.15, 0.2, 0.3, 0.5, 1.0)) -> list[dict]:
    """Share of final members whose worst experiment error is at or under each tolerance, and the
    reaction F1 of those members against the rest."""
    out = []
    for eps in eps_grid:
        adm, f1_adm, f1_rest = [], [], []
        for r in rows:
            for m in r["members"]:
                ok = bool(m["errors"]) and max(m["errors"]) <= eps
                adm.append(ok)
                keys = m["keys"]
        out.append({"eps": eps, "admitted_share": float(np.mean(adm)) if adm else None,
                    "systems_with_an_admitted_member": float(np.mean([any(max(m["errors"] or [9]) <= eps for m in r["members"]) for r in rows]))})
    return out


def selective_by_spread(rows: list[dict]) -> dict:
    """Keep the systems with the lowest committee spread: F1 of the kept share against all."""
    f1 = np.array([r["rms_medoid"]["f1"] if r["rms_medoid"] else 0.0 for r in rows])
    spread = np.array([r["spread"] if r["spread"] is not None else 1.0 for r in rows])
    order = np.argsort(spread, kind="stable")
    n = len(rows)
    at = lambda c: float(f1[order[:max(1, int(round(c * n)))]].mean())
    return {"n": n, "f1_all": float(f1.mean()), "f1_at_50": at(0.5), "f1_at_80": at(0.8),
            "f1_lowest_quarter_spread": at(0.25), "f1_highest_quarter_spread": float(f1[order[-max(1, n // 4):]].mean())}


def step_curves(rows: list[dict], budget: int = 4) -> list[dict]:
    """Mean reaction F1 of the medoid, spread, survivors and resynthesized members after each experiment."""
    out = []
    for t in range(1, budget + 1):
        st = [s for r in rows for s in r["steps"] if s["t"] == t]
        if not st:
            break
        out.append({"t": t, "n": len(st), "f1_medoid": float(np.mean([s["rms_medoid_f1"] or 0.0 for s in st])),
                    "f1_majority": float(np.mean([s["rms_majority_f1"] for s in st])),
                    "spread": float(np.mean([s["spread"] if s["spread"] is not None else 1.0 for s in st])),
                    "survivors": float(np.mean([s["survivors"] for s in st])), "refuted": float(np.mean([s["refuted"] for s in st])),
                    "resynthesized": float(np.mean([s["resynthesized"] for s in st])),
                    "disagreement_at_choice": float(np.mean([s["disagreement_at_choice"] for s in st if s["disagreement_at_choice"] is not None])) if any(s["disagreement_at_choice"] is not None for s in st) else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--arms", default="committee_probe,committee_fixed,single_fixed")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = {}
    per_arm = {arm: load(arm, a.model, a.tag) for arm in a.arms.split(",")}
    common = set.intersection(*[{r["system"] for r in rows} for rows in per_arm.values() if rows])
    for arm, rows in per_arm.items():
        rows = [r for r in rows if r["system"] in common]
        if not rows:
            continue
        out[arm] = {"admission": admission_vs_eps(rows), "selective": selective_by_spread(rows), "steps": step_curves(rows)}
        print(f"\n{arm} ({a.model}, n {len(rows)})")
        print("  admission by tolerance:", ", ".join(f"eps {x['eps']}: members {x['admitted_share']:.2f}, systems {x['systems_with_an_admitted_member']:.2f}" for x in out[arm]["admission"]))
        s = out[arm]["selective"]
        print(f"  selective by spread: F1 all {s['f1_all']:.3f}, most agreed 50% {s['f1_at_50']:.3f}, 80% {s['f1_at_80']:.3f}, "
              f"lowest-spread quarter {s['f1_lowest_quarter_spread']:.3f}, highest-spread quarter {s['f1_highest_quarter_spread']:.3f}")
        for st in out[arm]["steps"]:
            print(f"  step {st['t']}: F1 medoid {st['f1_medoid']:.3f} majority {st['f1_majority']:.3f} spread {st['spread']:.2f} "
                  f"survivors {st['survivors']:.2f} refuted {st['refuted']:.2f} resynthesized {st['resynthesized']:.2f}"
                  + (f" disagreement at choice {st['disagreement_at_choice']:.2f}" if st["disagreement_at_choice"] is not None else ""))
    (ART / f"ablations_{a.model}{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
