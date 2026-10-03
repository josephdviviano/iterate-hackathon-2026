"""Single programs against the committee, and disagreement against error, on every level.

One row per level from the stored runs: the unseeded single programs of the
baseline condition, and the seeded committee's vote, member range, best member,
disagreement AUROC and the error rate of unanimous against split transitions.
"""

from __future__ import annotations

import json
from pathlib import Path

from .evaluate import evaluate_condition, pooled_auroc
from .experiment import condition_dir

LEVELS = (("ar25", 3), ("m0r0", 3), ("sk48", 2), ("ar25", 7), ("ls20", 3), ("ka59", 2), ("g50t", 1))


def split_stats(reliability: list[dict]) -> tuple[int, int, float | None, float | None]:
    """(unanimous n, split n, unanimous error, split error) from the uniform reliability bins."""
    una = next(r for r in reliability if r["bin"] == "unanimous")
    split = [r for r in reliability if r["bin"] != "unanimous" and r["n"]]
    n_split = sum(r["n"] for r in split)
    err_split = round(sum(r["n"] * r["error_rate"] for r in split) / n_split, 3) if n_split else None
    return una["n"], n_split, una["error_rate"], err_split


def level_row(game: str, level: int, train_frac: float, baseline: str, committee: str) -> dict:
    singles = sorted(m["test_accuracy"] for m in
                     (json.loads(p.read_text()) for p in condition_dir(game, level, train_frac, baseline).glob("run*/meta.json"))
                     if m["consistent"])
    e = evaluate_condition(game, level, train_frac, committee)
    c = e["committee"]
    n_una, n_split, err_una, err_split = split_stats(c["reliability_uniform"])
    members = sorted(c["member_accuracy"])
    return {"level": f"{game} L{level}", "n_test": e["n_test"], "singles": singles,
            "single_mean": round(sum(singles) / len(singles), 3) if singles else None,
            "n_members": c["n_members"], "vote": round(c["vote_accuracy"], 3), "member_mean": round(c["mean_member_accuracy"], 3),
            "member_min": round(members[0], 3), "member_max": round(members[-1], 3),
            "distinct": c["n_distinct_behaviours"], "auroc": None if c["auroc_uniform_disagreement_vs_error"] is None
            else round(c["auroc_uniform_disagreement_vs_error"], 3),
            "unanimous_n": n_una, "unanimous_err": err_una, "split_n": n_split, "split_err": err_split,
            "per_transition": [{"disagreement": r["uniform_disagreement"], "error": not r["correct"]} for r in c["per_transition"]]}


def table(rows: list[dict]) -> str:
    def f(x, nd=3):
        return "" if x is None else f"{x:.{nd}f}"
    lines = ["| Level | n test | Single programs | Committee vote | Members (K) | Best member | AUROC | Unanimous n (error) | Split n (error) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['level']} | {r['n_test']} | {', '.join(f(s, 2) for s in r['singles'])} (mean {f(r['single_mean'])}) | "
                     f"{f(r['vote'])} | {f(r['member_min'], 2)} to {f(r['member_max'], 2)} ({r['n_members']}, {r['distinct']} behaviours) | "
                     f"{f(r['member_max'])} | {f(r['auroc'], 2)} | {r['unanimous_n']} ({f(r['unanimous_err'], 2)}) | "
                     f"{r['split_n']} ({f(r['split_err'], 2)}) |")
    return "\n".join(lines)


def run(levels, train_frac: float, baseline: str, committee: str) -> dict:
    rows = [level_row(g, l, train_frac, baseline, committee) for g, l in levels]
    pooled = pooled_auroc(rows)
    una = sum(r["unanimous_n"] for r in rows)
    una_err = sum(r["unanimous_n"] * r["unanimous_err"] for r in rows) / una
    split = sum(r["split_n"] for r in rows)
    split_err = sum(r["split_n"] * r["split_err"] for r in rows if r["split_n"]) / split
    return {"rows": rows, "pooled_auroc": pooled,
            "pooled_unanimous": {"n": una, "error": round(una_err, 3)}, "pooled_split": {"n": split, "error": round(split_err, 3)},
            "mean_single": round(sum(r["single_mean"] for r in rows) / len(rows), 3),
            "mean_vote": round(sum(r["vote"] for r in rows) / len(rows), 3),
            "mean_best_member": round(sum(r["member_max"] for r in rows) / len(rows), 3),
            "vote_vs_single_mean": {"wins": sum(r["vote"] > r["single_mean"] for r in rows),
                                    "ties": sum(r["vote"] == r["single_mean"] for r in rows),
                                    "losses": sum(r["vote"] < r["single_mean"] for r in rows)}}


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Single against committee, disagreement against error, every level.")
    parser.add_argument("--levels", default=None, help="game:level pairs, comma separated; default is the seven levels")
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--baseline", default="baseline_opine_devin")
    parser.add_argument("--committee", default="committee_opine_devin")
    parser.add_argument("--out", default="artifacts/summary_opine.json")
    args = parser.parse_args(argv)
    levels = tuple((g, int(l)) for g, l in (p.split(":") for p in args.levels.split(","))) if args.levels else LEVELS
    out = run(levels, args.train_frac, args.baseline, args.committee)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(table(out["rows"]))
    print(f"\nmean over levels: single {out['mean_single']}, vote {out['mean_vote']}, best member {out['mean_best_member']}; "
          f"vote vs single mean: {out['vote_vs_single_mean']}")
    print(f"pooled AUROC (uniform disagreement -> error): {out['pooled_auroc']}; "
          f"unanimous {out['pooled_unanimous']}, split {out['pooled_split']}")


if __name__ == "__main__":
    main()
