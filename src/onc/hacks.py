"""Reward-hacking checks for the ONC evaluation (handoff section 6, checks 1 to 3).

    uv run python -m onc.hacks --store toy

1. Coverage-only reward: a set-width dial paid for coverage alone widens to both
   labels, reaches coverage 1 and commits on no step.
2. Confidence-free reward: sharpening the reported probabilities leaves the
   task reward unchanged and raises the calibration error; R_cal removes the tie.
3. The benchmark's cheaters under the combined reward R: each stays at or below
   the floor, as it does under the benchmark's score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from onc_agi.adapters.agents import BASELINES, CHEATERS, make_agent
from onc_agi.core.schema import Tier
from onc_agi.services.kit import evaluate

from onc.agent import Policy
from onc.evaluate import open_store, run_condition, world_ids
from onc.rewards import brier, combine, ece, feature_credit, r_cal, r_task

SHARPNESS = (0.5, 1.0, 2.0, 4.0, 8.0, 32.0)
WIDTHS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def coverage_dial(pairs: list[tuple[float, int]], width: float) -> tuple[float, float]:
    """Coverage and committed share of the set {label : nonconformity <= width}."""
    covered, committed = [], []
    for p, y in pairs:
        scores = [p, 1.0 - p]  # nonconformity of label 0 and label 1
        members = [k for k, s in enumerate(scores) if s <= width]
        covered.append(y in members)
        committed.append(len(members) == 1)
    return float(np.mean(covered)), float(np.mean(committed))


def train_dial(pairs: list[tuple[float, int]], *, iterations: int = 40, group: int = 8, seed: int = 0) -> list[dict]:
    """GRPO-style update of a Gaussian over the dial's logit, paid for coverage alone."""
    rng = np.random.default_rng(seed)
    mu, sigma, lr = 0.0, 0.7, 0.5
    trace = []
    for it in range(iterations):
        thetas = mu + sigma * rng.standard_normal(group)
        rewards = np.array([coverage_dial(pairs, float(1 / (1 + np.exp(-t))))[0] for t in thetas])
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        mu += lr * float(np.mean(adv * (thetas - mu) / sigma))
        width = float(1 / (1 + np.exp(-mu)))
        cov, com = coverage_dial(pairs, width)
        trace.append({"iteration": it, "width": width, "coverage": cov, "committed": com})
    return trace


def check_coverage_only(seq: dict) -> dict:
    pairs = [tuple(x) for w in seq["worlds"] for x in w.get("held_out", [])]
    frontier = [{"width": w, **dict(zip(("coverage", "committed"), coverage_dial(pairs, w)))} for w in WIDTHS]
    trace = train_dial(pairs)
    return {"n_pairs": len(pairs), "frontier": frontier, "trained": trace[-1], "trace": trace, "aci": seq["held_out"]}


def sharpen(p: float, s: float) -> float:
    return Policy(sharpness=s).report(p)


def check_confidence_free(full: dict, store) -> dict:
    """The sharpness sweep, from one run's raw probabilities: decisions do not change, reports do."""
    rows = []
    for s in SHARPNESS:
        sig, drv, task, cal = [], [], [], []
        for w in full["worlds"]:
            key = store.answer_key(w["world_id"])
            credit = feature_credit(w["ranking"], key)
            p_signal = sharpen(w["p_signal"], s)
            p_driver = {f: sharpen(v, s) for f, v in w["p_driver"].items()}
            sig.append((p_signal, int(not key.is_null)))
            drv.extend((p_driver[f], int(credit[f])) for f in w["ranking"])
            task.append(w["r_task"])
            cal.append(r_cal(p_signal, not key.is_null, p_driver, credit))
        rows.append(
            {
                "sharpness": s,
                "r_task": float(np.mean(task)),
                "r_cal": float(np.mean(cal)),
                "ece_signal": ece([p for p, _ in sig], [y for _, y in sig]),
                "ece_driver": ece([p for p, _ in drv], [y for _, y in drv]) if drv else float("nan"),
                "brier_signal": float(np.mean([brier(p, y) for p, y in sig])),
                "reward_task_only": float(np.mean(task)),
                "reward_with_cal": float(np.mean([combine(t, c, 0.0) for t, c in zip(task, cal)])),
            }
        )
    best_task = [r["sharpness"] for r in rows if abs(r["reward_task_only"] - max(x["reward_task_only"] for x in rows)) < 1e-9]
    best_cal = max(rows, key=lambda r: r["reward_with_cal"])["sharpness"]
    return {"rows": rows, "argmax_task_only": best_task, "argmax_with_cal": best_cal}


def implicit_probabilities(ranking: list[str]) -> tuple[float, dict[str, float]]:
    """A submission without probabilities claims certainty: P(signal) = 1[list non-empty], P(driver) = 1 per listed feature."""
    return (1.0 if ranking else 0.0), {f: 1.0 for f in ranking}


def check_cheaters(store, mode: str, committee_row: dict) -> list[dict]:
    ids = world_ids(store, mode)
    rows = []
    names = ["random", *BASELINES, *CHEATERS]
    for name in names:
        try:
            card, results = evaluate(make_agent(name, store), store, Tier.PUBLIC_TRAIN, world_ids=ids, bootstrap_draws=50)
        except ImportError:
            continue
        task, cal = [], []
        for r in results:
            key = store.answer_key(r.world_id)
            p_signal, p_driver = implicit_probabilities(list(r.ranking))
            credit = feature_credit(r.ranking, key)
            task.append(r_task(r.score))
            cal.append(r_cal(p_signal, not key.is_null, p_driver, credit))
        rows.append(
            {
                "agent": name,
                "kind": "cheater" if name in CHEATERS or name == "random" else "baseline",
                "discovery_score": card.discovery_score,
                "unfloored": card.discovery_score_unfloored,
                "r_task": float(np.mean(task)),
                "r_cal": float(np.mean(cal)),
                "r_dis": 0.0,
                "reward": float(np.mean([combine(t, c, 0.0) for t, c in zip(task, cal)])),
            }
        )
    rows.append(
        {
            "agent": committee_row["agent"],
            "kind": "ours",
            "discovery_score": committee_row["discovery_score"],
            "unfloored": committee_row["discovery_score_unfloored"],
            **{k: committee_row["reward"][k] for k in ("r_task", "r_cal", "r_dis")},
            "reward": combine(**committee_row["reward"]),
        }
    )
    return sorted(rows, key=lambda r: -r["reward"])


def markdown(report: dict) -> str:
    out = ["## Check 1: coverage-only reward (sequential, held-out p(y | x))", "", "| Width | Coverage | Committed |", "|---|---|---|"]
    out += [f"| {r['width']:.1f} | {r['coverage']:.2f} | {r['committed']:.2f} |" for r in report["coverage_only"]["frontier"]]
    t, a = report["coverage_only"]["trained"], report["coverage_only"]["aci"]
    out += ["", f"Dial trained on coverage alone: width {t['width']:.2f}, coverage {t['coverage']:.2f}, committed {t['committed']:.2f}.",
            f"ACI at the 0.90 target: coverage {a['aci_coverage']:.2f}, committed {a['aci_committed']:.2f} (n = {a['n']}).", ""]
    out += ["## Check 2: confidence-free reward (full access)", "", "| Sharpness | R_task | ECE P(signal) | ECE P(driver) | Brier P(signal) | R_cal | R_task + 0.5 R_cal |", "|---|---|---|---|---|---|---|"]
    out += [f"| {r['sharpness']:g} | {r['r_task']:+.3f} | {r['ece_signal']:.3f} | {r['ece_driver']:.3f} | {r['brier_signal']:.3f} | {r['r_cal']:+.3f} | {r['reward_with_cal']:+.3f} |" for r in report["confidence_free"]["rows"]]
    cf = report["confidence_free"]
    out += ["", f"Task-only reward ties every sharpness in {cf['argmax_task_only']}; with R_cal the best sharpness is {cf['argmax_with_cal']:g}.", ""]
    for mode in ("full", "seq"):
        out += [f"## Check 3: cheaters under R ({mode})", "", "| Agent | Kind | DS | Unfloored | R_task | R_cal | R_dis | R |", "|---|---|---|---|---|---|---|---|"]
        out += [f"| {r['agent']} | {r['kind']} | {r['discovery_score']:.3f} | {r['unfloored']:+.3f} | {r['r_task']:+.2f} | {r['r_cal']:+.2f} | {r['r_dis']:+.2f} | {r['reward']:+.2f} |" for r in report["cheaters"][mode]]
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="toy")
    parser.add_argument("--out", default="artifacts/onc/hacks")
    args = parser.parse_args(argv)
    store = open_store(args.store)
    policy = Policy()
    full = run_condition(store, world_ids(store, "full"), policy, "disagreement", bootstrap=50)
    seq = run_condition(store, world_ids(store, "seq"), policy, "disagreement", bootstrap=50)
    report = {
        "store": args.store,
        "coverage_only": check_coverage_only(seq),
        "confidence_free": check_confidence_free(full, store),
        "cheaters": {"full": check_cheaters(store, "full", full), "seq": check_cheaters(store, "seq", seq)},
    }
    stem = Path(args.out).with_name(Path(args.out).name + f"_{Path(args.store).name}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float))
    stem.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))
    print(f"wrote {stem}.json and {stem}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
