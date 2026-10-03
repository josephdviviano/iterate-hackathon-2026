"""Evaluate stored runs: single programs as a baseline, consistent programs as a committee."""

from __future__ import annotations

import json
from pathlib import Path

from .committee import Committee, Member, description_length
from .experiment import condition_dir
from .loader import build_buffer, temporal_split
from .verify import run_program


def load_runs(cond: Path, train: list | None = None, test: list | None = None
              ) -> list[tuple[str, dict, str, list]]:
    """Stored runs of a condition. test_preds.json is not tracked in git; when it is missing
    and the split is given, the program is replayed to rebuild it."""
    runs = []
    for d in sorted(cond.glob("run*")):
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        source = (d / "program.py").read_text()
        preds_path = d / "test_preds.json"
        if preds_path.exists():
            preds = json.loads(preds_path.read_text())
        elif train is not None and test is not None:
            preds = run_program(source, train, test).test_preds
            preds_path.write_text(json.dumps(preds))
        else:
            preds = []
        runs.append((d.name, meta, source, preds))
    return runs


def summarize_single(runs) -> dict:
    accs = [m["test_accuracy"] for _, m, _, _ in runs if m["test_accuracy"] is not None]
    cons = [m["consistent"] for _, m, _, _ in runs]
    return {
        "n_runs": len(runs),
        "n_consistent": sum(cons),
        "test_accuracy_all": [round(a, 4) for a in accs],
        "mean_test_accuracy": sum(accs) / len(accs) if accs else None,
        "mean_test_accuracy_consistent": (
            sum(m["test_accuracy"] for _, m, _, _ in runs if m["consistent"]) / sum(cons) if sum(cons) else None),
        "mean_wall_s": sum(m["synth"].get("wall_s", 0) for _, m, _, _ in runs) / max(1, len(runs)),
    }


def evaluate_condition(game: str, level: int, train_frac: float, condition: str, lam: float = 0.01,
                       include_inconsistent: bool = False, test_level: int | None = None) -> dict:
    cond = condition_dir(game, level, train_frac, condition, test_level)
    train, test = temporal_split(build_buffer(game), level, train_frac, test_level)
    runs = load_runs(cond, train, test)
    out = {"game": game, "level": level, "train_frac": train_frac, "test_level": test_level, "condition": condition,
           "n_test": len(test), "single": summarize_single(runs)}
    members = [Member(name, src, preds, length=description_length(src))
               for name, m, src, preds in runs if (m["consistent"] or include_inconsistent) and preds]
    if members:
        after = [t.after_objs for t in test]
        out["committee"] = Committee(members, lam=lam).evaluate(after)
        out["lambda_sweep"] = [
            {"lam": l, "vote_accuracy": round(e["vote_accuracy"], 4), "auroc": e["auroc_disagreement_vs_error"],
             "max_weight": max(e["weights"])}
            for l in (0.0, 0.001, 0.003, 0.01, 0.03) for e in [Committee(members, lam=l).evaluate(after)]]
    (cond / "evaluation.json").write_text(json.dumps(out, indent=1))
    return out


def print_report(out: dict) -> None:
    s = out["single"]
    tl = f" test=L{out['test_level']}" if out.get("test_level") else ""
    print(f"{out['game']} L{out['level']} f{out['train_frac']}{tl} {out['condition']}  n_test={out['n_test']}")
    print(f"  single programs: {s['n_consistent']}/{s['n_runs']} consistent, "
          f"test acc all={s['mean_test_accuracy']}, consistent={s['mean_test_accuracy_consistent']}, "
          f"wall={s['mean_wall_s']:.0f}s")
    c = out.get("committee")
    if c:
        print(f"  committee of {c['n_members']}: vote acc={c['vote_accuracy']:.3f} "
              f"simplest acc={c['simplest_accuracy']:.3f} mean member acc={c['mean_member_accuracy']:.3f} "
              f"AUROC(disagreement->error)={c['auroc_disagreement_vs_error']} "
              f"distinct behaviours={c['n_distinct_behaviours']} weights={c['weights']}")
        print(f"  uniform-weight disagreement AUROC={c['auroc_uniform_disagreement_vs_error']}")
        print("  reliability (uniform): " + ", ".join(f"{r['bin']} n={r['n']} err={r['error_rate']}"
                                                       for r in c["reliability_uniform"]))
        print("  lambda sweep: " + "; ".join(f"lam={d['lam']}: vote={d['vote_accuracy']} auroc={d['auroc']} "
                                            f"maxw={d['max_weight']:.2f}" for d in out["lambda_sweep"]))


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--include-inconsistent", action="store_true")
    parser.add_argument("--test-level", type=int, default=None)
    args = parser.parse_args(argv)
    print_report(evaluate_condition(args.game, args.level, args.train_frac, args.condition, args.lam,
                                    args.include_inconsistent, args.test_level))


if __name__ == "__main__":
    main()
