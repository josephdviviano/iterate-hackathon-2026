"""Selection among members: how much the vote loses to the best member, and
whether probes recover it.

Headroom: the oracle "any member right" bounds every combination rule. When
the oracle equals the best member, no weighting of the members can beat the
best one; the gain can only come from selecting it or from hypotheses that
no member holds. Probes as selection: observe the transition the survivors
disagree on most, drop the members that mispredicted it, and score the
survivors' vote on the transitions not yet observed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .calibrate import LEVELS
from .committee import Committee, Member, description_length
from .evaluate import load_runs
from .experiment import condition_dir
from .explore import Trace, simulate
from .loader import Transition, build_buffer, temporal_split
from .verify import canonical


def load_members(game: str, level: int, train_frac: float = 0.4, condition: str = "committee_devin"
                 ) -> tuple[list[Member], list[Transition], list[Transition]]:
    train, test = temporal_split(build_buffer(game), level, train_frac)
    cond = condition_dir(game, level, train_frac, condition)
    members = [Member(n, s, p, description_length(s))
               for n, m, s, p in load_runs(cond, train, test) if m["consistent"] and p]
    return members, train, test


def member_hits(members: list[Member], test: list[Transition]) -> list[list[bool]]:
    truth = [json.dumps(canonical(t.after_objs)) for t in test]
    return [[p is not None and json.dumps(canonical(p)) == t for p, t in zip(m.test_preds, truth)]
            for m in members]


def headroom(members: list[Member], test: list[Transition]) -> dict:
    hits = member_hits(members, test)
    n = len(test)
    accs = [sum(h) / n for h in hits]
    return {
        "k": len(members),
        "n_test": n,
        "mean_member": sum(accs) / len(accs),
        "best_member": max(accs),
        "vote": Committee(members).evaluate([t.after_objs for t in test])["vote_accuracy"],
        "oracle": sum(any(col) for col in zip(*hits)) / n,
    }


def _curve(trace: Trace) -> list[dict]:
    return [{"probes": i, "members": m, "vote_acc": round(1 - v, 3), "best_survivor_acc": round(1 - b, 3)}
            for i, (m, v, b) in enumerate(zip(trace.members_left, trace.vote_error_left, trace.best_error_left))]


def random_curve(members: list[Member], test: list[Transition], n_random: int = 20) -> list[dict]:
    """Mean over random probe orders, at each probe count, over the orders not yet falsified."""
    traces = [simulate(members, test, "random", 0.0, rng=random.Random(k)) for k in range(n_random)]
    out = []
    for i in range(max(len(t.members_left) for t in traces)):
        live = [t for t in traces if len(t.members_left) > i and t.members_left[i] > 0]
        if not live:
            break
        out.append({"probes": i, "n_orders": len(live),
                    "members": round(sum(t.members_left[i] for t in live) / len(live), 2),
                    "vote_acc": round(1 - sum(t.vote_error_left[i] for t in live) / len(live), 3),
                    "best_survivor_acc": round(1 - sum(t.best_error_left[i] for t in live) / len(live), 3)})
    return out


def run(levels=LEVELS, n_random: int = 20) -> dict:
    out = {}
    for game, level in levels:
        members, train, test = load_members(game, level)
        tr = simulate(members, test, "disagreement", 0.0, train)
        out[f"{game} L{level}"] = {"headroom": headroom(members, test),
                                   "disagreement": _curve(tr), "falsified_at": tr.falsified_at,
                                   "random": random_curve(members, test, n_random)}
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Oracle headroom and probes as selection, from stored runs.")
    parser.add_argument("--n-random", type=int, default=20)
    args = parser.parse_args(argv)
    out = run(n_random=args.n_random)
    Path("artifacts/selection.json").write_text(json.dumps(out, indent=1))
    print("| Level | K | n test | mean member | best member | vote | oracle any-right |")
    print("|---|---|---|---|---|---|---|")
    for lv, r in out.items():
        h = r["headroom"]
        print(f"| {lv} | {h['k']} | {h['n_test']} | {h['mean_member']:.2f} | {h['best_member']:.2f} "
              f"| {h['vote']:.2f} | {h['oracle']:.2f} |")
    print("\nprobes as selection (vote accuracy / best survivor accuracy on the unobserved transitions):")
    for lv, r in out.items():
        d = " -> ".join(f"n{c['probes']} K{c['members']} {c['vote_acc']}/{c['best_survivor_acc']}" for c in r["disagreement"])
        print(f"  {lv} disagreement: {d}  falsified at {r['falsified_at']}")
        rc = " -> ".join(f"n{c['probes']} K{c['members']} {c['vote_acc']}/{c['best_survivor_acc']} ({c['n_orders']})"
                         for c in r["random"][:6])
        print(f"  {lv} random:       {rc}")


if __name__ == "__main__":
    main()
