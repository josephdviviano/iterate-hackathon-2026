"""The 90-second demo. Reads cached artifacts only; no live synthesis.

For one game and split it shows: what a single program gets wrong on held-out
transitions, what the committee gets right, where the committee says it does
not know, which rows the count-based ontology error cannot score, and which
transition it would probe next.
"""

from __future__ import annotations

import json
from pathlib import Path

from .committee import Committee, Member, description_length, row_uncertainty
from .evaluate import load_runs, summarize_single
from .experiment import condition_dir
from .explore import compare_strategies
from .loader import build_buffer, temporal_split


def _members(cond: Path, train, test) -> list[Member]:
    return [Member(name, src, preds, description_length(src))
            for name, m, src, preds in load_runs(cond, train, test) if m["consistent"] and preds]


def run_demo(game: str, level: int, train_frac: float, test_level: int | None,
             baseline: str = "baseline", committee: str = "committee", lam: float = 0.0, top: int = 3) -> None:
    transitions = build_buffer(game)
    train, test = temporal_split(transitions, level, train_frac, test_level)
    where = f"level {level}" + (f", tested on level {test_level}" if test_level else "")
    print(f"\n=== {game}: trained on the first {len(train)} transitions of {where}; {len(test)} held-out transitions\n")

    base_runs = load_runs(condition_dir(game, level, train_frac, baseline, test_level))
    s = summarize_single(base_runs)
    print(f"[1] Single program (OPINE-World style), {s['n_runs']} independent syntheses:")
    print(f"    exact replay on train: {s['n_consistent']}/{s['n_runs']}    held-out accuracy: "
          f"{', '.join(f'{a:.2f}' for a in s['test_accuracy_all'])}")

    members = _members(condition_dir(game, level, train_frac, committee, test_level), train, test)
    if not members:
        print("    (no consistent committee members cached for this split)")
        return
    com = Committee(members, lam=lam)
    ev = com.evaluate([t.after_objs for t in test])
    print(f"\n[2] Committee of {ev['n_members']} programs that all replay train exactly, "
          f"weighted by description length {ev['lengths']}:")
    print(f"    weights {ev['weights']}")
    print(f"    held-out accuracy: weighted vote {ev['vote_accuracy']:.2f}, simplest member "
          f"{ev['simplest_accuracy']:.2f}, mean member {ev['mean_member_accuracy']:.2f}")
    print(f"    distinct behaviours on held-out: {ev['n_distinct_behaviours']}")
    print(f"    does disagreement predict error?  AUROC = {ev['auroc_disagreement_vs_error']}")

    print("\n[3] Where the committee says it does not know (held-out transitions, by disagreement):")
    per = ev["per_transition"]
    order = sorted(range(len(per)), key=lambda i: -per[i]["disagreement"])
    for i in order[:top]:
        t = test[i]
        print(f"    step {t.step:4d} action {json.dumps(t.action):>36}  disagreement {per[i]['disagreement']:.2f}  "
              f"{per[i]['n_distinct']} candidate outcomes  vote {'right' if per[i]['correct'] else 'WRONG'}")
    quiet = [i for i in order if per[i]["disagreement"] == 0.0]
    if quiet:
        acc_q = sum(per[i]["correct"] for i in quiet) / len(quiet)
        print(f"    unanimous on {len(quiet)} transitions, accuracy there {acc_q:.2f}")

    rows = row_uncertainty(com, train, test)
    unseen = [r for r in rows if r["n_train"] == 0]
    print(f"\n[4] Effect rows (type | action | context) touched by held-out transitions: {len(rows)}, "
          f"of which {len(unseen)} never seen in train.")
    print("    Count-based ontology error is undefined there; the committee still scores them:")
    for r in (unseen or rows)[:top]:
        print(f"    {r['row']:<48} n_train={r['n_train']:3d}  eta_counts={r['eta_counts']}  "
              f"eta_committee={r['eta_committee']:.2f}  effects={r['committee_effects']}")

    print("\n[5] Exploration: observe the transition the committee disagrees on most, drop falsified members, repeat.")
    cmp = compare_strategies(members, train, test, lam=lam)
    for k in ("disagreement", "counts"):
        c = cmp[k]
        print(f"    {k:<13} probes to collapse: {c['probes_to_collapse']}   members left: {c['members_left']}"
              f"   falsified at: {c['falsified_at']}")
    r = cmp["random"]
    print(f"    random        mean probes to collapse: {r['mean_probes_to_collapse']} ({r['n_collapsed']} of "
          f"{r['n_runs']} orders collapsed, {r['n_falsified']} falsified, mean probes to falsify "
          f"{r['mean_probes_to_falsify']})")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Replay cached results for one split.")
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--test-level", type=int, default=None)
    parser.add_argument("--lam", type=float, default=0.0, help="MDL prior strength; 0 = equal weights")
    parser.add_argument("--baseline", default="baseline_devin", help="condition name of the single-program runs")
    parser.add_argument("--committee", default="committee_devin", help="condition name of the seeded runs")
    args = parser.parse_args(argv)
    run_demo(args.game, args.level, args.train_frac, args.test_level, baseline=args.baseline,
             committee=args.committee, lam=args.lam)


if __name__ == "__main__":
    main()
