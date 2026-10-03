"""Evaluate stored runs: single programs as a baseline, consistent programs as a committee."""

from __future__ import annotations

import json
from pathlib import Path

from .committee import Committee, Member, description_length
from .env import engine_source, extract_objects, returns_frame
from .experiment import condition_dir
from .loader import build_buffer, temporal_split
from .verify import run_program


def load_runs(cond: Path, train: list | None = None, test: list | None = None, game: str | None = None
              ) -> list[tuple[str, dict, str, list]]:
    """Stored runs of a condition. test_preds.json is not tracked in git; when it is missing
    and the split is given, the program is replayed in the run's mode to rebuild it, and in
    frame_out the object view too when the game is given."""
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
            mode = meta.get("mode", "objects")
            if returns_frame(mode) and game is None:
                raise ValueError(f"{d}: rebuilding frame_out predictions needs the game, for the object view")
            verdict = run_program(source, train, test, mode=mode, engine_src=engine_source(game, mode))
            preds = verdict.test_preds
            preds_path.write_text(json.dumps(preds))
            if verdict.test_objs is not None:
                (d / "test_objs.json").write_text(json.dumps(verdict.test_objs))
        else:
            preds = []
        runs.append((d.name, meta, source, preds))
    return runs


def members_from(cond: Path, train: list | None = None, test: list | None = None, game: str | None = None,
                 include_inconsistent: bool = False, need_preds: bool = True) -> list[Member]:
    """The stored runs as committee members, each with its mode and, in frame_out, the extractor's
    view of its predicted frames."""
    out = []
    for name, m, src, preds in load_runs(cond, train, test, game):
        if not (m["consistent"] or include_inconsistent) or (need_preds and not preds):
            continue
        mode = m.get("mode", "objects")
        objs_path = cond / name / "test_objs.json"
        objs = json.loads(objs_path.read_text()) if objs_path.exists() else None
        if objs is None and returns_frame(mode) and game is not None:
            objs = extract_objects(engine_source(game, mode), preds)
            objs_path.write_text(json.dumps(objs))
        out.append(Member(name, src, preds, description_length(src), mode=mode, test_objs=objs))
    return out


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


def evaluate_condition(game: str, level: int, train_frac: float, condition: str, lam: float = 0.0,
                       include_inconsistent: bool = False, test_level: int | None = None,
                       train_n: int | None = None, test_n: int | None = None) -> dict:
    cond = condition_dir(game, level, train_frac, condition, test_level, train_n, test_n)
    train, test = temporal_split(build_buffer(game), level, train_frac, test_level, train_n, test_n)
    runs = load_runs(cond, train, test, game)
    out = {"game": game, "level": level, "train_frac": train_frac, "test_level": test_level, "condition": condition,
           "n_test": len(test), "single": summarize_single(runs)}
    members = members_from(cond, train, test, game, include_inconsistent)
    if members:
        out["mode"] = members[0].mode
        after = Committee(members).truth(test)
        out["committee"] = Committee(members, lam=lam).evaluate(after)
        out["lambda_sweep"] = [
            {"lam": l, "vote_accuracy": round(e["vote_accuracy"], 4), "auroc": e["auroc_disagreement_vs_error"],
             "max_weight": max(e["weights"])}
            for l in (0.0, 0.001, 0.003, 0.01, 0.03) for e in [Committee(members, lam=l).evaluate(after)]]
    (cond / "evaluation.json").write_text(json.dumps(out, indent=1))
    return out


def member_curve(game: str, level: int, train_frac: float, condition: str, lam: float = 0.0,
                 test_level: int | None = None, start: int = 1) -> list[dict]:
    """Committee metrics as members are added in run order: how fast each construction order
    gains accuracy and calibration per synthesized program."""
    cond = condition_dir(game, level, train_frac, condition, test_level)
    train, test = temporal_split(build_buffer(game), level, train_frac, test_level)
    members = members_from(cond, train, test, game)
    after = Committee(members).truth(test)
    curve = []
    for k in range(start, len(members) + 1):
        e = Committee(members[:k], lam=lam).evaluate(after)
        unanimous = next(r for r in e["reliability_uniform"] if r["bin"] == "unanimous")
        split_n = len(test) - unanimous["n"]
        split_err = (sum((not p["correct"]) for p in e["per_transition"] if p["uniform_disagreement"] > 0)
                     / split_n if split_n else None)
        curve.append({"members": k, "vote_accuracy": round(e["vote_accuracy"], 4),
                      "auroc": e["auroc_uniform_disagreement_vs_error"],
                      "unanimous_n": unanimous["n"], "unanimous_error": unanimous["error_rate"],
                      "split_n": split_n, "split_error": round(split_err, 4) if split_err is not None else None,
                      "distinct": e["n_distinct_behaviours"], "mean_disagreement": round(e["mean_disagreement"], 4)})
    (cond / "member_curve.json").write_text(json.dumps(curve, indent=1))
    return curve


def print_curve(curve: list[dict], label: str) -> None:
    print(f"{label}: members | vote acc | AUROC | unanimous n (err) | split n (err) | distinct")
    for c in curve:
        print(f"  {c['members']:2d} | {c['vote_accuracy']:.3f} | {c['auroc'] if c['auroc'] is None else round(c['auroc'], 3)} "
              f"| {c['unanimous_n']:2d} ({c['unanimous_error']}) | {c['split_n']:2d} ({c['split_error']}) | {c['distinct']}")


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
    parser.add_argument("--lam", type=float, default=0.0, help="MDL prior strength; 0 = equal weights (default, see R14)")
    parser.add_argument("--include-inconsistent", action="store_true")
    parser.add_argument("--test-level", type=int, default=None)
    parser.add_argument("--curve", action="store_true", help="metrics as members are added in run order")
    parser.add_argument("--train-n", type=int, default=None)
    parser.add_argument("--test-n", type=int, default=None)
    args = parser.parse_args(argv)
    if args.curve:
        print_curve(member_curve(args.game, args.level, args.train_frac, args.condition, args.lam, args.test_level),
                    f"{args.game} L{args.level} {args.condition}")
        return
    print_report(evaluate_condition(args.game, args.level, args.train_frac, args.condition, args.lam,
                                    args.include_inconsistent, args.test_level, args.train_n, args.test_n))


if __name__ == "__main__":
    main()


def k_sweep(game: str, level: int, train_frac: float, condition: str, lam: float = 0.0,
            test_level: int | None = None, ks: tuple[int, ...] = (2, 4, 8), n_boot: int = 1000, seed: int = 0) -> dict:
    """Calibration against committee size: every subset of the stored members for each K (sampled
    when there are many), and a transition bootstrap of the full committee's AUROC."""
    import random
    from itertools import combinations

    from .committee import auroc

    cond = condition_dir(game, level, train_frac, condition, test_level)
    train, test = temporal_split(build_buffer(game), level, train_frac, test_level)
    members = members_from(cond, train, test, game)
    after = Committee(members).truth(test)
    rng = random.Random(seed)
    out: dict = {"n_members": len(members), "by_k": []}
    for k in ks:
        if k > len(members):
            continue
        subsets = list(combinations(range(len(members)), k))
        if len(subsets) > 60:
            subsets = rng.sample(subsets, 60)
        rows = []
        for sub in subsets:
            e = Committee([members[i] for i in sub], lam=lam).evaluate(after)
            un = next(r for r in e["reliability_uniform"] if r["bin"] == "unanimous")
            split = [p for p in e["per_transition"] if p["uniform_disagreement"] > 0]
            rows.append((e["auroc_uniform_disagreement_vs_error"], e["vote_accuracy"], un["error_rate"],
                         (sum(not p["correct"] for p in split) / len(split)) if split else None, len(split)))
        def mean_sd(vals):
            vals = [v for v in vals if v is not None]
            if not vals:
                return (None, None)
            m = sum(vals) / len(vals)
            return (round(m, 3), round((sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5, 3))
        out["by_k"].append({"k": k, "n_subsets": len(subsets),
                            "auroc": mean_sd([r[0] for r in rows]), "vote_accuracy": mean_sd([r[1] for r in rows]),
                            "unanimous_error": mean_sd([r[2] for r in rows]), "split_error": mean_sd([r[3] for r in rows]),
                            "split_n": mean_sd([r[4] for r in rows])})
    full = Committee(members, lam=lam).evaluate(after)
    d = [p["uniform_disagreement"] for p in full["per_transition"]]
    err = [not p["correct"] for p in full["per_transition"]]
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(d)) for _ in range(len(d))]
        a = auroc([d[i] for i in idx], [err[i] for i in idx])
        if a is not None:
            boots.append(a)
    boots.sort()
    out["auroc_full"] = full["auroc_uniform_disagreement_vs_error"]
    out["auroc_ci95"] = (round(boots[int(0.025 * len(boots))], 3), round(boots[int(0.975 * len(boots))], 3)) if boots else None
    out["per_transition"] = [{"disagreement": x, "error": e} for x, e in zip(d, err)]
    (cond / "k_sweep.json").write_text(json.dumps(out, indent=1))
    return out


def pooled_auroc(parts: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC over the transitions of several levels pooled, with a transition bootstrap."""
    import random

    from .committee import auroc

    rows = [r for p in parts for r in p["per_transition"]]
    d = [r["disagreement"] for r in rows]
    err = [r["error"] for r in rows]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(d)) for _ in range(len(d))]
        a = auroc([d[i] for i in idx], [err[i] for i in idx])
        if a is not None:
            boots.append(a)
    boots.sort()
    return {"n": len(rows), "auroc": round(auroc(d, err), 3),
            "ci95": (round(boots[int(0.025 * len(boots))], 3), round(boots[int(0.975 * len(boots))], 3))}
