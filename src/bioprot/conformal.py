"""Conformal sets over the acceptable label from an uncertainty score: the R22 wrapper on BioProt.

    uv run python -m bioprot.conformal

The score ranks; the wrapper turns the rank into a set with a coverage
guarantee. For a plan with uncertainty u in [0, 1] the nonconformity of the
label "acceptable" is u and of "not acceptable" is 1 - u. The set holds every
label whose nonconformity is at or below the calibrated quantile: one label is
a commitment (execute, or reject), both labels is an abstention. Split
conformal calibrates on half the protocols and is repeated over random
splits; the online form is R22's adaptive conformal inference (Gibbs and
Candes, 2021) along the protocol sequence.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from .generate import ART
from .report import load_condition, uncertainty_of

SIGNALS = ("verbal", "self_consistency", "critique")


def quantile_level(n: int, alpha: float) -> float:
    """The finite-sample level ceil((n + 1)(1 - alpha)) / n, capped at 1."""
    return min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)


def nonconformity(u: np.ndarray, acceptable: np.ndarray) -> np.ndarray:
    return np.where(acceptable, u, 1 - u)


def sets_at(u: np.ndarray, q: float) -> np.ndarray:
    """Columns: acceptable in set, not acceptable in set."""
    return np.stack([u <= q, (1 - u) <= q], axis=1)


def summarise(sets: np.ndarray, acceptable: np.ndarray) -> dict:
    inset = np.where(acceptable, sets[:, 0], sets[:, 1])
    single = sets.sum(1) == 1
    passed = single & sets[:, 0]
    return {"coverage": float(inset.mean()), "committed": float(single.mean()),
            "accuracy_when_committed": float(inset[single].mean()) if single.any() else None,
            "passed": float(passed.mean()),
            "error_among_passed": float((~acceptable[passed]).mean()) if passed.any() else None,
            "abstain": float((sets.sum(1) == 2).mean()), "empty": float((sets.sum(1) == 0).mean()),
            # what the gate does to each class: a random gate at the same pass rate passes both classes equally
            "good_passed": float(passed[acceptable].mean()) if acceptable.any() else None,
            "bad_passed": float(passed[~acceptable].mean()) if (~acceptable).any() else None,
            "bad_rejected": float((single & sets[:, 1])[~acceptable].mean()) if (~acceptable).any() else None}


def split_conformal(u, acceptable, groups, alpha: float = 0.1, n_splits: int = 500, seed: int = 0) -> dict:
    u, acceptable, groups = np.asarray(u, float), np.asarray(acceptable, bool), np.asarray(groups)
    ids = np.unique(groups)
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_splits):
        cal_ids = rng.choice(ids, size=len(ids) // 2, replace=False)
        cal = np.isin(groups, cal_ids)
        q = float(np.quantile(nonconformity(u[cal], acceptable[cal]), quantile_level(int(cal.sum()), alpha)))
        runs.append(summarise(sets_at(u[~cal], q), acceptable[~cal]))
    out = {"alpha": alpha, "n_splits": n_splits, "n": int(len(u)), "n_protocols": int(len(ids))}
    for k in runs[0]:
        vals = np.array([r[k] for r in runs if r[k] is not None], float)
        out[k] = float(vals.mean()) if len(vals) else None
        out[k + "_ci"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None
    return out


def aci(u, acceptable, alpha: float = 0.1, gamma: float = 0.05) -> dict:
    """R22's online rule: the threshold is the (1 - alpha_t) quantile of the past
    realised nonconformities and alpha_t moves by gamma after each hit or miss."""
    u, acceptable = np.asarray(u, float), np.asarray(acceptable, bool)
    scores = nonconformity(u, acceptable)
    alpha_t, past, sets = alpha, [], []
    for i in range(len(u)):
        q = float(np.quantile(past, min(1.0, max(0.0, 1 - alpha_t)))) if past else 1.0
        s = sets_at(u[i:i + 1], q)[0]
        sets.append(s)
        hit = s[0] if acceptable[i] else s[1]
        alpha_t += gamma * (alpha - (0.0 if hit else 1.0))
        past.append(scores[i])
    return {"alpha": alpha, "gamma": gamma, "n": int(len(u))} | summarise(np.array(sets), acceptable)


def evaluate(items: list[dict], signal: str, alpha: float, n_splits: int) -> dict | None:
    keep = [(it, uncertainty_of(it, signal)) for it in items]
    keep = [(it, v) for it, v in keep if v is not None]
    if len(keep) < 20:
        return None
    u = np.clip([v for _, v in keep], 0, 1)
    y = np.array([it["acceptable"] for it, _ in keep])
    g = np.array([it["protocol_id"] for it, _ in keep])
    return {"split": split_conformal(u, y, g, alpha, n_splits), "online": aci(u, y, alpha)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-splits", type=int, default=500)
    ap.add_argument("--conditions", nargs="*", default=["qwen_shuf_human", "gptoss_shuf_human", "mistral_shuf_human"])
    a = ap.parse_args()
    out = {"alpha": a.alpha, "conditions": {}}
    print("| Condition | Signal | Split: coverage [CI] | committed | accuracy when committed | passed, error among passed | abstain | Online (R22 rule): coverage, committed, accuracy, abstain |")
    print("|---|---|---|---|---|---|---|---|")
    for cond in a.conditions:
        items = load_condition(ART / cond)
        base = float(np.mean([it["acceptable"] for it in items]))
        out["conditions"][cond] = {"acceptable_rate": base, "signals": {}}
        for sig in SIGNALS:
            e = evaluate(items, sig, a.alpha, a.n_splits)
            if e is None:
                continue
            out["conditions"][cond]["signals"][sig] = e
            s, o = e["split"], e["online"]
            f2 = lambda v: "n/a" if v is None else f"{v:.2f}"
            print(f"| {cond} | {sig} | {s['coverage']:.3f} [{s['coverage_ci'][0]:.2f}, {s['coverage_ci'][1]:.2f}] | {s['committed']:.2f} | "
                  f"{f2(s['accuracy_when_committed'])} | {s['passed']:.2f}, {f2(s['error_among_passed'])} | {s['abstain']:.2f} | "
                  f"{o['coverage']:.3f}, {o['committed']:.2f}, {f2(o['accuracy_when_committed'])}, {o['abstain']:.2f} |")
        print(f"  {cond}: acceptable rate {base:.2f} (error among all plans {1 - base:.2f})")
    (ART / "conformal.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
