"""Counterexample-guided resynthesis: the loop after the first falsifying probe.

Round 1 is the stored committee. Probing by disagreement (explore.simulate) ends
when one transition refutes every surviving member. Round 2 observes the probes:
they join the train set, each seed states the counterexample and what the refuted
programs predicted, and a new committee is synthesized. The report scores round 1,
round 2 and a passive round 2 (the next transitions in time instead of the
probes) on the transitions that none of them observed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .committee import Committee, Member, description_length
from .evaluate import load_runs
from .experiment import add_backend_args, backend_cfg, condition_dir, run_split
from .explore import simulate
from .loader import Transition, build_buffer, temporal_split
from .seeds import make_seeds
from .verify import canonical


def _key(pred: list[dict] | None) -> str:
    return json.dumps(canonical(pred)) if pred is not None else "<error>"


def members_of(cond: Path, train: list[Transition], test: list[Transition]) -> list[Member]:
    return [Member(name, src, preds, description_length(src))
            for name, m, src, preds in load_runs(cond, train, test) if m["consistent"] and preds]


def admission(cond: Path) -> tuple[int, int]:
    runs = load_runs(cond)
    return sum(m["consistent"] for _, m, _, _ in runs), len(runs)


def observed_probes(members: list[Member], test: list[Transition]) -> list[int]:
    """Test indices the disagreement order observes, up to and including the first probe
    that refutes every surviving member. Empty when no probe refutes all of them."""
    trace = simulate(members, test, "disagreement", 0.0)
    return trace.probes[:trace.falsified_at] if trace.falsified_at else []


def split_after_probes(train: list[Transition], test: list[Transition], probes: list[int]
                       ) -> tuple[list[Transition], list[Transition]]:
    chosen = set(probes)
    return train + [test[i] for i in probes], [t for i, t in enumerate(test) if i not in chosen]


def _brief(obj_json: str) -> str:
    o = json.loads(obj_json)
    return json.dumps({k: o[k] for k in ("name", "type", "x", "y", "w", "h", "layer") if k in o})


def _diff(pred: list[dict] | None, actual: list[dict], limit: int = 3) -> str:
    """Objects that differ between a prediction and the observed state, without pixels."""
    if pred is None:
        return "  no prediction (exception)"
    p, a = set(canonical(pred)), set(canonical(actual))
    lines = [f"  observed but not predicted: {_brief(m)}" for m in sorted(a - p)[:limit]]
    lines += [f"  predicted but not observed: {_brief(e)}" for e in sorted(p - a)[:limit]]
    return "\n".join(lines)


def counterexample_text(members: list[Member], test: list[Transition], probes: list[int]) -> str:
    """What the refuted programs predicted on the falsifying probe, grouped by prediction."""
    last = probes[-1]
    truth = [_key(test[i].after_objs) for i in probes[:-1]]
    alive = [m for m in members if all(_key(m.test_preds[i]) == t for i, t in zip(probes[:-1], truth))]
    groups: dict[str, tuple[int, list[dict] | None]] = {}
    for m in alive:
        pred = m.test_preds[last]
        n, _ = groups.get(_key(pred), (0, pred))
        groups[_key(pred)] = (n + 1, pred)
    t = test[last]
    lines = [f"Counterexample. The last transition in transitions.md (step {t.step}, action "
             f"{json.dumps(t.action)}) refuted every program of an earlier committee of {len(alive)} "
             f"programs. Each replayed all the other transitions exactly and predicted this one wrong:"]
    for n, pred in sorted(groups.values(), key=lambda g: -g[0]):
        lines.append(f"\n{n} program(s) predicted:\n{_diff(pred, t.after_objs)}")
    lines.append("\nFind the mechanic those programs missed. It must explain this transition and every "
                 "earlier one, and it must not be a special case keyed to this step.")
    return "\n".join(lines)


def probe_split_dir(game: str, level: int, train_frac: float, n_probes: int, condition: str) -> Path:
    base = condition_dir(game, level, train_frac, condition)
    return base.parent.parent / f"{base.parent.name}_probe{n_probes}" / condition


def round1(game: str, level: int, train_frac: float, source_condition: str):
    transitions = build_buffer(game)
    train, test = temporal_split(transitions, level, train_frac)
    members = members_of(condition_dir(game, level, train_frac, source_condition), train, test)
    return transitions, train, test, members, observed_probes(members, test)


def run_round2(game: str, level: int, train_frac: float, source_condition: str, condition: str,
               runs: int, cfg: dict, parallel: int, start: int, dry_run: bool) -> None:
    _, train, test, members, probes = round1(game, level, train_frac, source_condition)
    if not probes:
        raise SystemExit("no probe refutes every member of the stored committee")
    train2, test2 = split_after_probes(train, test, probes)
    text = counterexample_text(members, test, probes)
    seeds = [f"{s}\n\n{text}" for s in make_seeds(train2, runs)]
    base = probe_split_dir(game, level, train_frac, len(probes), condition)
    print(f"probes {probes} (steps {[test[i].step for i in probes]}); train {len(train2)}, held out {len(test2)}")
    print(f"artifacts: {base}\n\n{seeds[0]}\n")
    if dry_run:
        return
    base.parent.mkdir(parents=True, exist_ok=True)
    (base.parent / "split.json").write_text(json.dumps(
        {"source_condition": source_condition, "probes": probes, "steps": [test[i].step for i in probes]}, indent=1))
    run_split(train2, test2, base, seeds, cfg, f"{game} L{level} {condition}", start, parallel)


def restrict(members: list[Member], test: list[Transition], steps: set[int]) -> list[Member]:
    idx = [i for i, t in enumerate(test) if t.step in steps]
    return [Member(m.name, m.source, [m.test_preds[i] for i in idx], m.length) for m in members]


def score(members: list[Member], test: list[Transition]) -> dict:
    e = Committee(members).evaluate([t.after_objs for t in test])
    un = next(r for r in e["reliability_uniform"] if r["bin"] == "unanimous")
    split = [p for p in e["per_transition"] if p["uniform_disagreement"] > 0]
    return {"members": len(members), "vote": round(e["vote_accuracy"], 3),
            "mean_member": round(e["mean_member_accuracy"], 3), "best_member": round(max(e["member_accuracy"]), 3),
            "auroc": e["auroc_uniform_disagreement_vs_error"],
            "unanimous_n": un["n"], "unanimous_error": un["error_rate"], "split_n": len(split),
            "split_error": round(sum(not p["correct"] for p in split) / len(split), 3) if split else None,
            "distinct": e["n_distinct_behaviours"],
            "next_falsified_at": simulate(members, test, "disagreement", 0.0).falsified_at}


def report(game: str, level: int, train_frac: float, source_condition: str, condition: str,
           passive_condition: str) -> dict:
    transitions, train, test, r1, probes = round1(game, level, train_frac, source_condition)
    train2, test2 = split_after_probes(train, test, probes)
    base = probe_split_dir(game, level, train_frac, len(probes), condition)
    active = members_of(base, train2, test2)
    n_passive = len(train) + len(probes)
    ptrain, ptest = temporal_split(transitions, level, train_frac, train_n=n_passive)
    pcond = condition_dir(game, level, train_frac, passive_condition, train_n=n_passive)
    passive = members_of(pcond, ptrain, ptest) if pcond.exists() else []
    common = {t.step for t in test2} & ({t.step for t in ptest} if passive else {t.step for t in test2})
    arms = {"round1": (r1, test, condition_dir(game, level, train_frac, source_condition)),
            "active": (active, test2, base), "passive": (passive, ptest, pcond)}
    out = {"game": game, "level": level, "train_frac": train_frac, "probes": probes,
           "steps": [test[i].step for i in probes], "n_common": len(common), "arms": {}}
    for name, (members, ts, cond) in arms.items():
        if not members:
            continue
        kept = [t for t in ts if t.step in common]
        out["arms"][name] = {"admitted": admission(cond), **score(restrict(members, ts, common), kept)}
    (base.parent / "cegis_report.json").write_text(json.dumps(out, indent=1))
    return out


def print_report(out: dict) -> None:
    print(f"{out['game']} L{out['level']}: probes {out['probes']} (steps {out['steps']}); "
          f"{out['n_common']} transitions held out from every arm")
    cols = ["admitted", "members", "vote", "mean_member", "best_member", "auroc", "unanimous_n", "unanimous_error",
            "split_n", "split_error", "distinct", "next_falsified_at"]
    print("arm      | " + " | ".join(cols))
    for name, row in out["arms"].items():
        print(f"{name:8s} | " + " | ".join(str(row[c]) for c in cols))


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Resynthesize a committee after its first falsifying probe.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--source-condition", default="committee_devin", help="the round 1 committee")
    parser.add_argument("--condition", default="cegis_devin", help="the round 2 committee")
    parser.add_argument("--passive-condition", default="passive_devin",
                        help="committee on the next transitions in time, under L<level>_n<train+probes>")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="print the split and the seed, synthesize nothing")
    parser.add_argument("--report", action="store_true")
    add_backend_args(parser)
    args = parser.parse_args(argv)
    if args.report:
        print_report(report(args.game, args.level, args.train_frac, args.source_condition, args.condition,
                            args.passive_condition))
        return
    run_round2(args.game, args.level, args.train_frac, args.source_condition, args.condition, args.runs,
               backend_cfg(args), args.parallel, args.start, args.dry_run)


if __name__ == "__main__":
    main()
