"""Evaluate the committee agent on a world store and write the ONC results.

    uv run python -m onc.evaluate --store toy --mode both
    uv run python -m onc.evaluate --store artifacts/onc/dev --mode seq --acquisition disagreement,random

Each condition gives the benchmark scorecard, the calibration of P(signal),
P(driver) and the held-out p(y | x), adaptive conformal coverage with the share
of committed steps, spend, alignment diagnostics and the three reward terms.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.schema import Tier
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import PipelineAgent, evaluate

from onc.agent import CommitteeAgent, EpisodeLog, Policy, StagedCommitteeAgent, committee_analyst
from onc.conformal import aci, binary_scores
from onc.rewards import brier, ece, feature_credit, r_cal, r_task, shaping_total

ACQUISITIONS = ("disagreement", "random", "all", "pipeline", "staged")


def open_store(name: str) -> FileWorldStore:
    return FileWorldStore(fixture_store() if name == "toy" else Path(name))


def world_ids(store: FileWorldStore, mode: str, limit: int | None = None) -> list[str]:
    suffix = "-full" if mode == "full" else "-seq"
    ids = [w for w in store.world_ids(Tier.PUBLIC_TRAIN) if w.endswith(suffix)]
    if limit and limit < len(ids):
        # An even stride over the sorted ids keeps every role, null worlds included, in the sample.
        step = len(ids) / limit
        ids = [ids[int(i * step)] for i in range(limit)]
    return ids


def make_agent(policy: Policy, acquisition: str, logs: dict[str, EpisodeLog]):
    """The committee behind each acquisition design; ``logs`` collects the per-world probabilities."""
    if acquisition in ("disagreement", "random", "all"):
        agent = CommitteeAgent(Policy(**{**policy.__dict__, "acquisition": acquisition}), name=f"committee_{acquisition}")
        agent.logs = logs
        return agent
    if acquisition == "pipeline":
        return PipelineAgent("committee_pipeline", committee_analyst(policy, logs))
    if acquisition == "staged":
        return StagedCommitteeAgent(policy, logs)
    if acquisition.startswith("llm:"):
        from onc.llm_agent import LLMAgent

        return LLMAgent.from_env(acquisition[4:], logs)
    raise ValueError(acquisition)


def calibration(pairs: Sequence[tuple[float, int]]) -> dict:
    if not pairs:
        return {"n": 0}
    p = [a for a, _ in pairs]
    y = [b for _, b in pairs]
    conformal = aci(binary_scores(p), y)
    return {
        "n": len(pairs),
        "ece": ece(p, y),
        "brier": float(np.mean([brier(a, b) for a, b in pairs])),
        "aci_coverage": conformal.coverage,
        "aci_committed": conformal.committed,
    }


def run_condition(store: FileWorldStore, ids: list[str], policy: Policy, acquisition: str, *, bootstrap: int = 200) -> dict:
    logs: dict[str, EpisodeLog] = {}
    agent = make_agent(policy, acquisition, logs)
    start = time.time()
    card, results = evaluate(agent, store, Tier.PUBLIC_TRAIN, world_ids=ids, bootstrap_draws=bootstrap)
    return summarise(store, agent.name, acquisition, policy, card, results, logs, time.time() - start)


def summarise(store: FileWorldStore, name: str, acquisition: str, policy: Policy, card, results, logs: dict[str, EpisodeLog], seconds: float) -> dict:
    """One result row from a scorecard, the episode results and the per-world logs."""
    signal_pairs, driver_pairs, held_out, worlds = [], [], [], []
    for r in results:
        key = store.answer_key(r.world_id)
        log = logs[r.world_id]
        credit = feature_credit(r.ranking, key)
        signal_pairs.append((log.p_signal, int(not key.is_null)))
        driver_pairs.extend((log.p_driver.get(f, 0.0), int(credit[f])) for f in r.ranking)
        held_out.extend(log.held_out)
        task = r_task(r.score)
        cal = r_cal(log.p_signal, not key.is_null, log.p_driver, credit, log.held_out)
        dis = shaping_total(log.disagreement, log.resynthesis_steps) if log.disagreement else 0.0
        worlds.append(
            {
                "world_id": r.world_id,
                "is_null": key.is_null,
                "find": r.score.find,
                "find_signed": r.score.find_signed,
                "abstained": r.score.abstained,
                "restrained": r.score.restrained,
                "leaked": r.score.leaked,
                "spent": r.spent,
                "efficiency": r.score.efficiency,
                "reference_cost": key.reference_cost,
                "n_rows": log.n_rows,
                "steps": len(log.disagreement),
                "p_signal": log.p_signal,
                "p_driver": log.p_driver,
                "ranking": list(r.ranking),
                "truth": [p.true_feature for g in key.recoverable for p in g.parts],
                "disagreement": log.disagreement,
                "held_out": log.held_out,
                "resynthesis_steps": log.resynthesis_steps,
                "members": log.members,
                "dropped": log.dropped,
                "r_task": task,
                "r_cal": cal,
                "r_dis": dis,
                "notes": log.notes,
            }
        )
    return {
        "agent": name,
        "acquisition": acquisition,
        "policy": policy.__dict__,
        "n_worlds": len(results),
        "seconds": seconds,
        "discovery_score": card.discovery_score,
        "discovery_score_unfloored": card.discovery_score_unfloored,
        "interval": [card.interval.low, card.interval.high],
        "find": card.find,
        "find_signed": card.find_signed,
        "restraint": card.restraint,
        "strict": card.strict_discovery_score,
        "leak_rate": card.leak_rate,
        "abstention_on_signal": card.abstention_on_signal,
        "restraint_on_null": card.restraint_on_null,
        "mean_data_cost": card.mean_data_cost,
        "mean_efficiency": float(np.mean([w["efficiency"] for w in worlds])),
        "alignment": None if card.alignment is None else card.alignment.model_dump(),
        "resynthesis_events": int(sum(len(w["resynthesis_steps"]) for w in worlds)),
        "mean_steps": float(np.mean([w["steps"] for w in worlds])),
        "p_signal": calibration(signal_pairs),
        "p_driver": calibration(driver_pairs),
        "held_out": calibration(held_out),
        "reward": {k: float(np.mean([w[k] for w in worlds])) for k in ("r_task", "r_cal", "r_dis")},
        "worlds": worlds,
    }


def table(rows: list[dict]) -> str:
    head = "| Condition | DS | 95% | Find | Restraint | Strict | Leak | Cost | Eff | Acq gap | ECE P(sig) | ECE P(drv) | Brier held-out | ACI cov / commit | R_task | R_cal | R_dis |"
    out = [head, "|" + "---|" * 17]
    for r in rows:
        al = r["alignment"] or {}
        ho = r["held_out"]
        cov = f"{ho['aci_coverage']:.2f} / {ho['aci_committed']:.2f}" if ho.get("n") else "n/a"
        out.append(
            f"| {r['label']} | {r['discovery_score']:.3f} | [{r['interval'][0]:+.2f}, {r['interval'][1]:+.2f}] | {r['find']:.2f} | {r['restraint']:+.2f} | "
            f"{r['strict']:.2f} | {r['leak_rate']:.2f} | {r['mean_data_cost']:.0f} | {r['mean_efficiency']:.2f} | {al.get('acquisition_gap', float('nan')):.2f} | "
            f"{r['p_signal'].get('ece', float('nan')):.2f} | {r['p_driver'].get('ece', float('nan')):.2f} | {ho.get('brier', float('nan')):.3f} | {cov} | "
            f"{r['reward']['r_task']:+.2f} | {r['reward']['r_cal']:+.2f} | {r['reward']['r_dis']:+.2f} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="toy", help="'toy' or a store directory")
    parser.add_argument("--mode", default="both", choices=["full", "seq", "both"])
    parser.add_argument("--weighting", default="likelihood,equal")
    parser.add_argument("--acquisition", default="disagreement,random,pipeline,staged")
    parser.add_argument("--tau", type=float, default=Policy.tau)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/onc/eval")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    store = open_store(args.store)
    rows = []
    for mode in (["full", "seq"] if args.mode == "both" else [args.mode]):
        ids = world_ids(store, mode, args.limit)
        for weighting in args.weighting.split(","):
            policy = Policy(tau=args.tau, weighting=weighting, seed=args.seed)
            for acquisition in (args.acquisition.split(",") if mode == "seq" else ["disagreement"]):
                row = run_condition(store, ids, policy, acquisition)
                row["mode"], row["weighting"] = mode, weighting
                row["label"] = f"{mode} / {weighting} / {acquisition}" if mode == "seq" else f"{mode} / {weighting}"
                rows.append(row)
                print(f"{row['label']:40s} DS {row['discovery_score']:.3f} cost {row['mean_data_cost']:.0f} {row['seconds']:.0f}s", flush=True)
    stem = Path(args.out).with_name(Path(args.out).name + (f"_{args.tag}" if args.tag else "") + f"_{Path(args.store).name}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps({"store": args.store, "conditions": rows}, indent=1, default=float))
    stem.with_suffix(".md").write_text(table(rows) + "\n")
    print(table(rows))
    print(f"wrote {stem}.json and {stem}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
