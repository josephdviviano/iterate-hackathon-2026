"""Open-loop rollouts: the world model the way play uses it.

Teacher forcing gives each program the observed state before every step. In
play the agent has only its own predictions between observations. Here each
member predicts the held-out trajectory from its own previous prediction,
with the real actions, and is resynchronised to the observed state only at
the first step and at a level change. Hidden state kept inside a program
persists across the level boundary. We report exact-match accuracy by
horizon (steps since the last resync), how many members are still exactly
right, and the committee's disagreement as the horizon grows.
"""

from __future__ import annotations

import json
from collections import Counter

from .evaluate import load_runs
from .experiment import condition_dir
from .loader import RESET, Transition, build_buffer, temporal_split
from .verify import canonical, run_program

HORIZON_BINS = ((1, 1, "1"), (2, 5, "2-5"), (6, 10, "6-10"), (11, 10**6, "11+"))


def multi_level_test(transitions: list[Transition], levels: list[int]) -> list[Transition]:
    return [t for t in transitions if t.level in levels and t.action_id != RESET and not t.level_advance]


def horizons(test: list[Transition]) -> list[int]:
    h, prev_level, out = 0, None, []
    for i, t in enumerate(test):
        if i == 0 or t.level != prev_level:
            h = 0
        h += 1
        out.append(h)
        prev_level = t.level
    return out


def rollout(game: str, level: int, train_frac: float, condition: str, test_levels: list[int] | None = None,
            test_level: int | None = None) -> dict:
    transitions = build_buffer(game)
    train, test = temporal_split(transitions, level, train_frac, test_level)
    if test_levels:
        test = multi_level_test(transitions, test_levels)
    cond = condition_dir(game, level, train_frac, condition, test_level)
    runs = load_runs(cond, train, test if not test_levels else None)
    sources = [(name, src) for name, m, src, _ in runs if m["consistent"]]
    truth = [json.dumps(canonical(t.after_objs)) for t in test]
    hs = horizons(test)
    closed, opened = [], []
    for name, src in sources:
        c = run_program(src, train, test).test_preds
        o = run_program(src, train, test, open_loop=True).test_preds
        closed.append([json.dumps(canonical(p)) if p is not None else "<error>" for p in c])
        opened.append([json.dumps(canonical(p)) if p is not None else "<error>" for p in o])

    def plurality(keys_by_member, i):
        return Counter(m[i] for m in keys_by_member).most_common(1)[0][0]

    def disagreement(keys_by_member, i):
        import math
        k = len(keys_by_member)
        cnt = Counter(m[i] for m in keys_by_member)
        return 0.0 if len(cnt) < 2 else -sum(n / k * math.log(n / k) for n in cnt.values()) / math.log(k)

    per_step = []
    for i in range(len(test)):
        per_step.append({
            "step": test[i].step, "level": test[i].level, "horizon": hs[i],
            "closed_vote_correct": plurality(closed, i) == truth[i],
            "open_vote_correct": plurality(opened, i) == truth[i],
            "open_members_correct": sum(m[i] == truth[i] for m in opened),
            "closed_members_correct": sum(m[i] == truth[i] for m in closed),
            "open_disagreement": round(disagreement(opened, i), 3),
            "closed_disagreement": round(disagreement(closed, i), 3),
        })
    bins = []
    for lo, hi, name in HORIZON_BINS:
        sel = [p for p in per_step if lo <= p["horizon"] <= hi]
        if not sel:
            continue
        n = len(sel)
        bins.append({"horizon": name, "n": n,
                     "closed_vote_acc": round(sum(p["closed_vote_correct"] for p in sel) / n, 3),
                     "open_vote_acc": round(sum(p["open_vote_correct"] for p in sel) / n, 3),
                     "open_members_correct_mean": round(sum(p["open_members_correct"] for p in sel) / n, 2),
                     "open_disagreement_mean": round(sum(p["open_disagreement"] for p in sel) / n, 3),
                     "closed_disagreement_mean": round(sum(p["closed_disagreement"] for p in sel) / n, 3)})
    # per member: steps until the first open-loop mismatch after a resync, averaged over segments
    first_miss = []
    for m in opened:
        segs, h = [], None
        for i in range(len(test)):
            if hs[i] == 1:
                if h is not None:
                    segs.append(h)
                h = None
            if h is None and m[i] != truth[i]:
                h = hs[i]
        if h is not None:
            segs.append(h)
        first_miss.append(round(sum(segs) / len(segs), 1) if segs else None)
    out = {"game": game, "level": level, "test_levels": test_levels or [test_level or level], "n_test": len(test),
           "n_members": len(sources), "bins": bins, "first_open_loop_miss_per_member": first_miss, "per_step": per_step}
    suffix = "_".join(str(l) for l in (test_levels or [])) or "same"
    (cond / f"rollout_{suffix}.json").write_text(json.dumps(out, indent=1))
    return out


def print_rollout(out: dict) -> None:
    print(f"{out['game']} trained L{out['level']}, tested on levels {out['test_levels']}: {out['n_test']} steps, "
          f"{out['n_members']} members")
    print("  horizon | n | vote acc closed -> open | members exactly right (open) | disagreement closed -> open")
    for b in out["bins"]:
        print(f"  {b['horizon']:>5} | {b['n']:3d} | {b['closed_vote_acc']:.2f} -> {b['open_vote_acc']:.2f} | "
              f"{b['open_members_correct_mean']:.2f} of {out['n_members']} | {b['closed_disagreement_mean']:.2f} -> {b['open_disagreement_mean']:.2f}")
    print("  steps until first open-loop miss, per member:", out["first_open_loop_miss_per_member"])


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Open-loop rollout evaluation of a stored committee.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--condition", default="committee_devin")
    parser.add_argument("--test-level", type=int, default=None)
    parser.add_argument("--test-levels", type=int, nargs="*", default=None, help="roll through several levels")
    args = parser.parse_args(argv)
    print_rollout(rollout(args.game, args.level, args.train_frac, args.condition, args.test_levels, args.test_level))


if __name__ == "__main__":
    main()
