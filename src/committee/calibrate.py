"""Calibration of the committee's uncertainty.

Inputs per held-out step: the members' predicted next states and the
observed one. Nothing else, so every method here transfers to any
environment with a committee and observations.

1. Reliability of the vote share as a probability: ECE and Brier.
2. A calibration map fit on other levels (isotonic regression of P(error)
   on disagreement and vote share), tested leave-one-level-out.
3. Adaptive conformal inference along the trajectory (Gibbs and Candès,
   2021): nonconformity is one minus the vote share of the outcome that
   happened; the threshold adapts step by step from past outcomes so that
   coverage tracks a target under drift. The output is a prediction set of
   next states; set size is the calibrated uncertainty.
4. Good-Turing missing mass: the share of members with a behaviour no
   other member has estimates the mass of hypotheses not yet sampled.
5. Agreed-but-wrong detection: among unanimous steps, does the entropy of
   the effect rows the step touches predict the error?
"""

from __future__ import annotations

import json
import math
from collections import Counter

from .committee import Committee, Member, auroc, description_length, row_uncertainty
from .evaluate import load_runs
from .experiment import condition_dir
from .loader import build_buffer, temporal_split
from .matrix import RowKey, context_signature, object_type
from .verify import canonical

LEVELS = (("ar25", 3), ("m0r0", 3), ("sk48", 2), ("ar25", 7))


def level_steps(game: str, level: int, train_frac: float = 0.4, condition: str = "committee_devin") -> list[dict]:
    """Per held-out step: vote shares over candidate outcomes, truth, disagreement, max row entropy."""
    train, test = temporal_split(build_buffer(game), level, train_frac)
    cond = condition_dir(game, level, train_frac, condition)
    members = [Member(n, s, p, description_length(s)) for n, m, s, p in load_runs(cond, train, test) if m["consistent"] and p]
    com = Committee(members)
    rows = {r["row"]: r["eta_committee"] for r in row_uncertainty(com, train, test)}
    k = len(members)
    out = []
    for i, t in enumerate(test):
        keys = [json.dumps(canonical(m.test_preds[i])) if m.test_preds[i] is not None else "<error>" for m in members]
        counts = Counter(keys)
        truth = json.dumps(canonical(t.after_objs))
        shares = {key: c / k for key, c in counts.items()}
        plural = max(shares, key=lambda x: shares[x])
        row_etas = [rows.get(str(RowKey(object_type(bo), t.action_key, context_signature(t.before_objs, bo, t.click))), 0.0)
                    for bo in t.before_objs]
        out.append({"game": game, "level": level, "step": t.step, "shares": shares, "truth": truth,
                    "plurality": plural, "plural_share": shares[plural], "truth_share": shares.get(truth, 0.0),
                    "correct": plural == truth, "disagreement": com.uniform_disagreement(i),
                    "n_groups": len(counts), "singletons": sum(1 for c in counts.values() if c == 1),
                    "max_row_eta": max(row_etas) if row_etas else 0.0, "k": k})
    return out


def ece(probs: list[float], outcomes: list[bool], n_bins: int = 5) -> float:
    """Expected calibration error with equal-mass bins."""
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    n = len(order)
    total = 0.0
    for b in range(n_bins):
        idx = order[b * n // n_bins:(b + 1) * n // n_bins]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(outcomes[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return round(total, 4)


def brier(probs: list[float], outcomes: list[bool]) -> float:
    return round(sum((p - float(o)) ** 2 for p, o in zip(probs, outcomes)) / len(probs), 4)


def selective(scores_uncertainty: list[float], correct: list[bool]) -> dict:
    """Risk-coverage: rank by uncertainty, report selective accuracy at 50% and 80% coverage and AURC."""
    order = sorted(range(len(correct)), key=lambda i: scores_uncertainty[i])
    n = len(order)
    risks = []
    wrong = 0
    for j, i in enumerate(order, start=1):
        wrong += (not correct[i])
        risks.append(wrong / j)
    def at(cov):
        j = max(1, int(round(cov * n)))
        return round(1 - risks[j - 1], 3)
    return {"selective_acc_50": at(0.5), "selective_acc_80": at(0.8), "aurc": round(sum(risks) / n, 4),
            "overall_acc": round(sum(correct) / n, 3)}


def reliability_table(steps: list[dict], prob_key: str = "plural_share", n_bins: int = 5) -> list[dict]:
    order = sorted(steps, key=lambda s: s[prob_key])
    n = len(order)
    out = []
    for b in range(n_bins):
        chunk = order[b * n // n_bins:(b + 1) * n // n_bins]
        if chunk:
            out.append({"bin": b, "n": len(chunk), "mean_prob": round(sum(s[prob_key] for s in chunk) / len(chunk), 3),
                        "accuracy": round(sum(s["correct"] for s in chunk) / len(chunk), 3)})
    return out


def loo_calibration_map(by_level: dict[str, list[dict]]) -> dict:
    """Isotonic regression of P(correct) on the vote share, fit on the other levels."""
    from sklearn.isotonic import IsotonicRegression

    results = {}
    for held, steps in by_level.items():
        train = [s for lv, ss in by_level.items() if lv != held for s in ss]
        iso = IsotonicRegression(out_of_bounds="clip").fit([s["plural_share"] for s in train], [float(s["correct"]) for s in train])
        raw = [s["plural_share"] for s in steps]
        mapped = [float(v) for v in iso.predict(raw)]
        outcomes = [s["correct"] for s in steps]
        results[held] = {"n": len(steps), "ece_raw": ece(raw, outcomes), "ece_mapped": ece(mapped, outcomes),
                         "brier_raw": brier(raw, outcomes), "brier_mapped": brier(mapped, outcomes),
                         "mapped_prob_unanimous": round(float(iso.predict([1.0])[0]), 3)}
    return results


def aci(steps: list[dict], alpha: float = 0.1, gamma: float = 0.05) -> dict:
    """Adaptive conformal inference along one level's held-out sequence.

    Score s(y) = 1 - share(y). At step t the set is every candidate outcome with s(y) <= q_t,
    where q_t is the (1 - alpha_t) empirical quantile of past scores of the realised outcomes.
    If q_t >= 1 the set is "all candidates and anything else" and the step abstains.
    alpha_t moves by gamma toward the target after each observed miss or hit."""
    alpha_t = alpha
    past: list[float] = []
    covered = 0
    sizes: list[int] = []
    abstains = 0
    singleton_hits = singleton_n = 0
    trace = []
    for s in steps:
        if past:
            srt = sorted(past)
            level = min(1.0, max(0.0, 1 - alpha_t))
            j = min(len(srt) - 1, int(math.ceil(level * (len(srt) + 1))) - 1)
            q = srt[max(0, j)]
        else:
            q = 1.0
        abstain = q >= 1.0
        cand = {y: 1 - sh for y, sh in s["shares"].items()}
        members_in = [y for y, sc in cand.items() if sc <= q]
        size = len(members_in)
        truth_score = 1 - s["truth_share"]
        hit = abstain or (truth_score <= q)
        covered += hit
        sizes.append(size if not abstain else size + 1)
        abstains += abstain
        if size == 1 and not abstain:
            singleton_n += 1
            singleton_hits += hit
        past.append(truth_score)
        trace.append({"step": s["step"], "q": round(q, 3), "size": size, "abstain": abstain, "hit": hit,
                      "alpha_t": round(alpha_t, 3)})
        alpha_t = alpha_t + gamma * (alpha - (0 if hit else 1))
    n = len(steps)
    return {"target_coverage": 1 - alpha, "gamma": gamma, "n": n, "coverage": round(covered / n, 3),
            "mean_set_size": round(sum(sizes) / n, 2), "abstain_rate": round(abstains / n, 3),
            "singleton_rate": round(singleton_n / n, 3),
            "singleton_accuracy": round(singleton_hits / singleton_n, 3) if singleton_n else None,
            "coverage_last_half": round(sum(t["hit"] for t in trace[n // 2:]) / (n - n // 2), 3), "trace": trace}


def good_turing(steps: list[dict]) -> dict:
    """Missing mass = singletons / K, per step and as the level mean, against the unanimous error."""
    k = steps[0]["k"]
    per = [s["singletons"] / k for s in steps]
    unanimous = [s for s in steps if s["n_groups"] == 1]
    return {"mean_missing_mass": round(sum(per) / len(per), 3),
            "unanimous_error": round(sum(not s["correct"] for s in unanimous) / len(unanimous), 3) if unanimous else None,
            "n_unanimous": len(unanimous)}


def agreed_but_wrong(steps: list[dict]) -> dict:
    """Among unanimous steps, can the max committee entropy of the touched rows flag the errors?"""
    un = [s for s in steps if s["n_groups"] == 1]
    errs = [not s["correct"] for s in un]
    if not any(errs) or all(errs):
        return {"n_unanimous": len(un), "n_wrong": sum(errs), "auroc_row_eta": None}
    return {"n_unanimous": len(un), "n_wrong": sum(errs), "auroc_row_eta": round(auroc([s["max_row_eta"] for s in un], errs), 3)}


def run(alpha: float = 0.1, gamma: float = 0.05, levels=LEVELS) -> dict:
    by_level = {f"{g} L{l}": level_steps(g, l) for g, l in levels}
    pooled = [s for ss in by_level.values() for s in ss]
    out = {
        "pooled": {"n": len(pooled), "ece_vote_share": ece([s["plural_share"] for s in pooled], [s["correct"] for s in pooled]),
                   "brier_vote_share": brier([s["plural_share"] for s in pooled], [s["correct"] for s in pooled]),
                   "reliability": reliability_table(pooled),
                   "selective_by_disagreement": selective([s["disagreement"] for s in pooled], [s["correct"] for s in pooled])},
        "loo_map": loo_calibration_map(by_level),
        "per_level": {},
    }
    for lv, steps in by_level.items():
        c = aci(steps, alpha, gamma)
        out["per_level"][lv] = {
            "n": len(steps),
            "ece_vote_share": ece([s["plural_share"] for s in steps], [s["correct"] for s in steps]),
            "brier_vote_share": brier([s["plural_share"] for s in steps], [s["correct"] for s in steps]),
            "selective": selective([s["disagreement"] for s in steps], [s["correct"] for s in steps]),
            "aci": {k: v for k, v in c.items() if k != "trace"},
            "good_turing": good_turing(steps),
            "agreed_but_wrong": agreed_but_wrong(steps),
        }
    out["aci_pooled"] = {"coverage": round(sum(out["per_level"][lv]["aci"]["coverage"] * out["per_level"][lv]["n"] for lv in by_level)
                                           / len(pooled), 3)}
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Calibration battery and adaptive conformal sets for the committee.")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.05)
    args = parser.parse_args(argv)
    out = run(args.alpha, args.gamma)
    Path("artifacts/calibration.json").write_text(json.dumps(out, indent=1))
    p = out["pooled"]
    print(f"pooled n={p['n']}: vote-share ECE {p['ece_vote_share']}, Brier {p['brier_vote_share']}; "
          f"selective by disagreement: acc@50% {p['selective_by_disagreement']['selective_acc_50']}, "
          f"acc@80% {p['selective_by_disagreement']['selective_acc_80']}, AURC {p['selective_by_disagreement']['aurc']}, "
          f"overall {p['selective_by_disagreement']['overall_acc']}")
    print("reliability (vote share -> accuracy):", [(r["mean_prob"], r["accuracy"], r["n"]) for r in p["reliability"]])
    print("leave-one-level-out isotonic map (ECE raw -> mapped, Brier raw -> mapped, P(correct|unanimous) mapped):")
    for lv, r in out["loo_map"].items():
        print(f"  {lv}: ECE {r['ece_raw']} -> {r['ece_mapped']}, Brier {r['brier_raw']} -> {r['brier_mapped']}, unanimous -> {r['mapped_prob_unanimous']}")
    print(f"adaptive conformal, target coverage {1 - args.alpha}, gamma {args.gamma}:")
    for lv, r in out["per_level"].items():
        a = r["aci"]; g = r["good_turing"]; w = r["agreed_but_wrong"]
        print(f"  {lv}: coverage {a['coverage']} (last half {a['coverage_last_half']}), mean set size {a['mean_set_size']}, "
              f"abstain {a['abstain_rate']}, singleton rate {a['singleton_rate']} acc {a['singleton_accuracy']} | "
              f"missing mass {g['mean_missing_mass']} vs unanimous error {g['unanimous_error']} | "
              f"agreed-but-wrong {w['n_wrong']}/{w['n_unanimous']}, row-eta AUROC {w['auroc_row_eta']}")
    print("pooled ACI coverage:", out["aci_pooled"]["coverage"])


if __name__ == "__main__":
    main()
