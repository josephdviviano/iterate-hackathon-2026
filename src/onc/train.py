"""Train the acquisition and reporting policy with the three reward terms (handoff section 7).

    uv run python -m onc.train --store artifacts/onc/dev --arm all --iterations 15

What is trained: the policy's decision parameters, as a Gaussian over
(logit tau, log stop threshold, logit spend cap, log sharpness). The committee
is frozen: the policy can change what it buys, when it stops, what it lists
and what it reports, not the members or their weights. The update is a
group-relative policy gradient (GRPO-style): G samples per iteration on the
same batch of worlds, advantages normalised within the group, and the mean
moved along the advantage-weighted samples. Arms A to D differ only in the
reward weights.

Every number from generated worlds is a dev-world number, not a benchmark result.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from onc_agi.core.schema import Tier
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import evaluate

from onc.agent import CommitteeAgent, Policy
from onc.evaluate import open_store, run_condition, world_ids
from onc.rewards import combine, feature_credit, r_cal, r_task, shaping_total
from onc.worlds import split_ids

ARMS = {"A": (0.0, 0.0), "B": (0.5, 0.0), "C": (0.0, 0.25), "D": (0.5, 0.25)}
NAMES = ("tau", "stop_threshold", "spend_cap", "sharpness")


def arm_weights(arm: str) -> tuple[float, float]:
    """A named arm, or ``<lam_cal>/<lam_dis>`` for a grid point."""
    if arm in ARMS:
        return ARMS[arm]
    cal, dis = arm.split("/")
    return float(cal), float(dis)


GRID = [f"{c:g}/{d:g}" for c in (0.0, 0.25, 0.5, 1.0) for d in (0.0, 0.1, 0.25, 0.5)]


def to_theta(policy: Policy) -> np.ndarray:
    logit = lambda p: float(np.log(p / (1 - p)))
    return np.array([logit(policy.tau), np.log(policy.stop_threshold), logit(policy.spend_cap), np.log(policy.sharpness)])


def to_policy(theta: np.ndarray, base: Policy) -> Policy:
    sigmoid = lambda z: float(1 / (1 + np.exp(-z)))
    return Policy(
        **{
            **base.__dict__,
            "tau": min(max(sigmoid(theta[0]), 0.05), 0.95),
            "stop_threshold": float(np.exp(theta[1])),
            "spend_cap": min(max(sigmoid(theta[2]), 0.1), 1.0),
            "sharpness": float(np.exp(np.clip(theta[3], -2, 4))),
        }
    )


def episodes(args: tuple) -> list[dict]:
    """Worker: run one policy on a list of worlds and return the reward terms per world."""
    store_path, policy_dict, ids = args
    store = open_store(store_path)
    agent = CommitteeAgent(Policy(**policy_dict))
    _, results = evaluate(agent, store, Tier.PUBLIC_TRAIN, world_ids=ids, bootstrap_draws=1, with_alignment=False)
    out = []
    for r in results:
        key = store.answer_key(r.world_id)
        log = agent.logs[r.world_id]
        credit = feature_credit(r.ranking, key)
        out.append(
            {
                "world_id": r.world_id,
                "r_task": r_task(r.score),
                "r_cal": r_cal(log.p_signal, not key.is_null, log.p_driver, credit, log.held_out),
                "r_dis": shaping_total(log.disagreement, log.resynthesis_steps) if log.disagreement else 0.0,
                "find": r.score.find,
                "spent": r.spent,
                "members": len(log.members),
                "driver_sets": len({tuple(m["drivers"]) for m in log.members}),
            }
        )
    return out


def train_arm(
    arm: str, store_path: str, train_ids: list[str], *, iterations: int, group: int, batch: int, sigma: float, lr: float, seed: int, workers: int
) -> dict:
    lam_cal, lam_dis = arm_weights(arm)
    rng = np.random.default_rng(seed)
    base = Policy(seed=seed)
    mu = to_theta(base)
    trace = []
    with Pool(workers) as pool:
        for it in range(iterations):
            ids = [train_ids[i] for i in rng.choice(len(train_ids), size=min(batch, len(train_ids)), replace=False)]
            thetas = mu + sigma * rng.standard_normal((group, len(mu)))
            jobs = [(store_path, to_policy(t, base).__dict__, ids) for t in thetas]
            start = time.time()
            results = pool.map(episodes, jobs)
            rewards = np.array([np.mean([combine(e["r_task"], e["r_cal"], e["r_dis"], lam_cal, lam_dis) for e in res]) for res in results])
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            mu = mu + lr * (adv[:, None] * (thetas - mu) / sigma).mean(axis=0)
            policy = to_policy(mu, base)
            trace.append(
                {
                    "iteration": it,
                    "reward_mean": float(rewards.mean()),
                    "reward_best": float(rewards.max()),
                    "policy": {k: policy.__dict__[k] for k in NAMES},
                    "members": float(np.mean([e["members"] for res in results for e in res])),
                    "driver_sets": float(np.mean([e["driver_sets"] for res in results for e in res])),
                    "seconds": time.time() - start,
                }
            )
            print(f"arm {arm} it {it:2d} R {rewards.mean():+.3f} best {rewards.max():+.3f} " + " ".join(f"{k} {policy.__dict__[k]:.3f}" for k in NAMES) + f" {time.time() - start:.0f}s", flush=True)
    return {"arm": arm, "lam_cal": lam_cal, "lam_dis": lam_dis, "policy": to_policy(mu, base).__dict__, "trace": trace}


def held_out_report(policy: Policy, label: str, stores: dict[str, FileWorldStore], held: dict[str, list[str]], splits: tuple[str, ...] = ()) -> dict:
    out = {}
    for name, store in stores.items():
        for mode in ("seq", "full"):
            ids = held[f"{name}_{mode}"]
            if not ids or (splits and f"{name}_{mode}" not in splits):
                continue
            row = run_condition(store, ids, policy, "disagreement", bootstrap=200)
            row["label"] = f"{label} / {name} / {mode}"
            out[f"{name}_{mode}"] = row
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="artifacts/onc/dev")
    parser.add_argument("--arm", default="all", help="all, grid, or a comma list of A, B, C, D or <lam_cal>/<lam_dis>")
    parser.add_argument("--held-out-splits", default="", help="comma list, for example dev_seq; empty means every split")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--held-out", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="artifacts/onc/train")
    args = parser.parse_args(argv)

    manifest = json.loads((Path(args.store) / "manifest.json").read_text())
    train_all, held_all = split_ids(manifest, held_out_fraction=args.held_out, seed=args.seed)
    train_ids = [w for w in train_all if w.endswith("-seq")]
    dev, toy = open_store(args.store), open_store("toy")
    held = {
        "dev_seq": [w for w in held_all if w.endswith("-seq")],
        "dev_full": [w for w in held_all if w.endswith("-full")],
        "toy_seq": world_ids(toy, "seq"),
        "toy_full": world_ids(toy, "full"),
    }
    stores = {"dev": dev, "toy": toy}
    arms = list(ARMS) if args.arm == "all" else (GRID if args.arm == "grid" else args.arm.split(","))
    splits = tuple(s for s in args.held_out_splits.split(",") if s)
    report = {"store": args.store, "train_worlds": len(train_ids), "held_out": {k: len(v) for k, v in held.items()}, "config": vars(args), "arms": {}}
    report["arms"]["untrained"] = {"arm": "untrained", "policy": Policy(seed=args.seed).__dict__, "trace": [], "held_out": held_out_report(Policy(seed=args.seed), "untrained", stores, held, splits)}
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        result = train_arm(arm, args.store, train_ids, iterations=args.iterations, group=args.group, batch=args.batch, sigma=args.sigma, lr=args.lr, seed=args.seed, workers=args.workers)
        result["held_out"] = held_out_report(Policy(**result["policy"]), arm, stores, held, splits)
        report["arms"][arm] = result
        stem.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float))
    stem.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))
    return 0


def markdown(report: dict) -> str:
    out = []
    for split in ("dev_seq", "toy_seq", "dev_full", "toy_full"):
        out += [f"## {split} (dev worlds are not benchmark results)", "", "| Arm | λ_cal | λ_dis | tau | stop | cap | sharp | DS | 95% | Find | Restraint | Strict | Leak | Cost | Acq gap | ECE P(sig) | ECE P(drv) | Brier held-out | ACI cov / commit | members | driver sets |", "|" + "---|" * 21]
        for arm, res in report["arms"].items():
            row = res["held_out"].get(split)
            if row is None:
                continue
            p = res["policy"]
            ho = row["held_out"]
            cov = f"{ho['aci_coverage']:.2f} / {ho['aci_committed']:.2f}" if ho.get("n") else "n/a"
            al = row["alignment"] or {}
            members = float(np.mean([len(w["members"]) for w in row["worlds"]]))
            sets = float(np.mean([len({tuple(m["drivers"]) for m in w["members"]}) for w in row["worlds"]]))
            out.append(
                f"| {arm} | {res.get('lam_cal', 0):.2f} | {res.get('lam_dis', 0):.2f} | {p['tau']:.2f} | {p['stop_threshold']:.3f} | {p['spend_cap']:.2f} | {p['sharpness']:.2f} | "
                f"{row['discovery_score']:.3f} | [{row['interval'][0]:+.2f}, {row['interval'][1]:+.2f}] | {row['find']:.2f} | {row['restraint']:+.2f} | {row['strict']:.2f} | {row['leak_rate']:.2f} | {row['mean_data_cost']:.0f} | {al.get('acquisition_gap', float('nan')):.2f} | "
                f"{row['p_signal'].get('ece', float('nan')):.3f} | {row['p_driver'].get('ece', float('nan')):.3f} | {ho.get('brier', float('nan')):.3f} | {cov} | {members:.1f} | {sets:.1f} |"
            )
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
