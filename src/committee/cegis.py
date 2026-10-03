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
from collections import Counter
from pathlib import Path

from .committee import Committee, Member, description_length
from .evaluate import load_runs
from .experiment import ARTIFACTS, add_backend_args, backend_cfg, condition_dir, run_split
from .explore import simulate
from .loader import Transition, build_buffer, temporal_split
from .matrix import RowKey, context_signature, effect_signature, object_type, pair_objects
from .seeds import CONDITION_KINDS, make_seeds
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



REPAIR_KINDS = CONDITION_KINDS + [
    "the exact size or fit relation between the object and what it touches or enters",
    "the order, chain or count of several objects that move together",
    "a bound of the playable area that the frame does not draw",
]


def _field_changes(bo: dict, ao: dict | None) -> str:
    sig = effect_signature(bo, ao)
    if sig in ("born", "gone", "no_change"):
        return sig
    parts = [f"{k} {bo.get(k)} -> {ao.get(k)}" if k != "pixels" else "pixels changed" for k in sig.split(",")]
    return ", ".join(parts)


def mechanism_text(members: list[Member], test: list[Transition], probes: list[int]) -> str:
    """The refuting transition at the effect-row level: for each object its row, the observed effect
    and each predicted effect with its count, split into the rows every program got wrong (the
    mechanic to repair) and the rows every program got right (the mechanics to keep)."""
    last = probes[-1]
    truth = [_key(test[i].after_objs) for i in probes[:-1]]
    alive = [m for m in members if all(_key(m.test_preds[i]) == t for i, t in zip(probes[:-1], truth))]
    t = test[last]
    observed = {id(bo): ao for bo, ao in pair_objects(t.before_objs, t.after_objs) if bo is not None}
    predicted = []
    for m in alive:
        pred = m.test_preds[last]
        predicted.append(None if pred is None else {id(bo): ao for bo, ao in pair_objects(t.before_objs, pred) if bo is not None})
    wrong, right = [], []
    for bo in t.before_objs:
        row = RowKey(object_type(bo), t.action_key, context_signature(t.before_objs, bo, t.click))
        obs = _field_changes(bo, observed.get(id(bo)))
        counts = Counter("<error>" if pr is None else _field_changes(bo, pr.get(id(bo))) for pr in predicted)
        if set(counts) == {obs}:
            right.append(f"{row}: {obs}")
        else:
            wrong.append(f"row `{row}`, object {bo.get('name')}: observed {obs}; predicted "
                         + "; ".join(f"{e} ({n} program{'s' if n > 1 else ''})" for e, n in counts.most_common()))
    born = [ao for bo, ao in pair_objects(t.before_objs, t.after_objs) if bo is None]
    for ao in born:
        n = sum(1 for m in alive if m.test_preds[last] is not None and any(canonical([o]) == canonical([ao]) for o in m.test_preds[last]))
        wrong.append(f"new object {_brief(json.dumps(ao, sort_keys=True))}: observed; predicted by {n} of {len(alive)} programs")
    lines = [f"Counterexample at the mechanism level. Step {t.step}, action {json.dumps(t.action)}, refuted every program "
             f"of a committee of {len(alive)}. Rows where every program was wrong; this is the mechanic to repair:"]
    lines += [f"  - {w}" for w in wrong] or ["  - (no object row differs; the difference is in object naming or pixels)"]
    lines.append("Rows where every program was right on this transition; keep these mechanics as the earlier programs had them:")
    lines += [f"  - {r}" for r in right[:12]] + ([f"  - and {len(right) - 12} more"] if len(right) > 12 else [])
    return "\n".join(lines)


GEOMETRY_RULE = ("Rule for this repair: a movement stops because of a visible object, a field of an object, or a bound "
                 "that a visible object marks. Do not infer invisible floors, corridors or regions from where objects "
                 "stopped; a repair that adds coordinate constants the earlier programs did not need is a lookup table "
                 "and will be rejected.")


def repair_hypothesis(k: int) -> str:
    kind = REPAIR_KINDS[k % len(REPAIR_KINDS)]
    return (f"Repair hypothesis for this seed: the rule for the row(s) above is decided by {kind}. Write that rule "
            f"first and verify it on the counterexample and on every earlier transition. Keep every other mechanic "
            f"as the earlier programs had it; do not key the rule to this step.")


def probe_split_dir(game: str, level: int, train_frac: float, n_probes: int, condition: str) -> Path:
    base = condition_dir(game, level, train_frac, condition)
    return base.parent.parent / f"{base.parent.name}_probe{n_probes}" / condition


def stored_round(game: str, level: int, train_frac: float, source_condition: str, condition: str,
                 n_probes: int) -> tuple[list[int], Path]:
    """Probe indices observed so far and the directory of the committee that observed them.
    n_probes 0 is the stored round 1 committee."""
    if n_probes == 0:
        return [], condition_dir(game, level, train_frac, source_condition)
    cond = probe_split_dir(game, level, train_frac, n_probes, condition)
    return json.loads((cond.parent / "split.json").read_text())["probes"], cond


def run_round(game: str, level: int, train_frac: float, source_condition: str, condition: str,
              from_probe: int, runs: int, cfg: dict, parallel: int, start: int, dry_run: bool,
              mechanism: bool = False, geometry_rule: bool = False) -> None:
    train, test = temporal_split(build_buffer(game), level, train_frac)
    probes, cond = stored_round(game, level, train_frac, source_condition, condition, from_probe)
    train_r, test_r = split_after_probes(train, test, probes)
    members = members_of(cond, train_r, test_r)
    new = observed_probes(members, test_r)
    if not new:
        raise SystemExit(f"no probe refutes every member of {cond}")
    text = mechanism_text(members, test_r, new) if mechanism else counterexample_text(members, test_r, new)
    remaining = [i for i in range(len(test)) if i not in set(probes)]
    probes = probes + [remaining[j] for j in new]
    train2, test2 = split_after_probes(train, test, probes)
    seeds = [f"{s}\n\n{text}" + (f"\n\n{repair_hypothesis(k)}" if mechanism else "") + (f"\n\n{GEOMETRY_RULE}" if geometry_rule else "")
             for k, s in enumerate(make_seeds(train2, runs))]
    base = probe_split_dir(game, level, train_frac, len(probes), condition)
    print(f"probes {probes} (steps {[test[i].step for i in probes]}); train {len(train2)}, held out {len(test2)}")
    print(f"artifacts: {base}\n\n{seeds[0]}\n")
    if dry_run:
        return
    base.parent.mkdir(parents=True, exist_ok=True)
    (base.parent / "split.json").write_text(json.dumps(
        {"source": str(cond.relative_to(ARTIFACTS)), "probes": probes,
         "steps": [test[i].step for i in probes]}, indent=1))
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
            "distinct": e["n_distinct_behaviours"], "member_accuracy": e["member_accuracy"],
            "next_falsified_at": simulate(members, test, "disagreement", 0.0).falsified_at}


def report(game: str, level: int, train_frac: float, source_condition: str, condition: str,
           passive_condition: str, conditions: list[str] | None = None) -> dict:
    """Every stored round and every passive control with a matching train size, scored on the
    transitions that none of them observed."""
    transitions = build_buffer(game)
    train, test = temporal_split(transitions, level, train_frac)
    base0 = condition_dir(game, level, train_frac, source_condition)
    arms: dict[str, tuple[list[Member], list[Transition], Path, int]] = {
        "round1": (members_of(base0, train, test), test, base0, 0)}
    rounds = sorted((p for p in base0.parent.parent.glob(f"{base0.parent.name}_probe*")
                     if p.name.rsplit("probe", 1)[1].isdigit()),
                    key=lambda p: int(p.name.rsplit("probe", 1)[1]))
    probes: list[int] = []
    conditions = conditions or [condition]
    for k, d in enumerate(rounds, start=2):
        probes = json.loads((d / "split.json").read_text())["probes"]
        train_r, test_r = split_after_probes(train, test, probes)
        for c in conditions:
            if (d / c).exists():
                label = f"round{k}" if len(conditions) == 1 else f"round{k} {c}"
                arms[label] = (members_of(d / c, train_r, test_r), test_r, d / c, len(probes))
    for n_extra in sorted({a[3] for a in arms.values() if a[3]}):
        n = len(train) + n_extra
        pcond = condition_dir(game, level, train_frac, passive_condition, train_n=n)
        if pcond.exists():
            ptrain, ptest = temporal_split(transitions, level, train_frac, train_n=n)
            arms[f"passive{n_extra}"] = (members_of(pcond, ptrain, ptest), ptest, pcond, n_extra)
    arms = {k: v for k, v in arms.items() if v[0]}
    common = set.intersection(*({t.step for t in ts} for _, ts, _, _ in arms.values()))
    out = {"game": game, "level": level, "train_frac": train_frac, "probes": probes,
           "steps": [test[i].step for i in probes], "n_common": len(common), "arms": {}}
    for name, (members, ts, cond, n_extra) in arms.items():
        kept = [t for t in ts if t.step in common]
        out["arms"][name] = {"observed": n_extra, "admitted": admission(cond),
                             **score(restrict(members, ts, common), kept)}
    ((rounds[-1] if rounds else base0.parent) / "cegis_report.json").write_text(json.dumps(out, indent=1))
    return out


def print_report(out: dict) -> None:
    print(f"{out['game']} L{out['level']}: probes {out['probes']} (steps {out['steps']}); "
          f"{out['n_common']} transitions held out from every arm")
    cols = ["observed", "admitted", "members", "vote", "mean_member", "best_member", "auroc", "unanimous_n",
            "unanimous_error", "split_n", "split_error", "distinct", "next_falsified_at"]
    print("arm      | " + " | ".join(cols))
    for name, row in out["arms"].items():
        print(f"{name:18s} | " + " | ".join(f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in cols))


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
    parser.add_argument("--from-probe", type=int, default=0,
                        help="start from the stored round that observed this many probes; 0 is round 1")
    parser.add_argument("--mechanism", action="store_true",
                        help="state the counterexample at the effect-row level and give each seed a distinct repair hypothesis")
    parser.add_argument("--geometry-rule", action="store_true", help="add the rule against hidden geometry to every seed")
    parser.add_argument("--conditions", default=None, help="report: comma-separated round 2 conditions to compare")
    parser.add_argument("--dry-run", action="store_true", help="print the split and the seed, synthesize nothing")
    parser.add_argument("--report", action="store_true")
    add_backend_args(parser)
    args = parser.parse_args(argv)
    if args.report:
        print_report(report(args.game, args.level, args.train_frac, args.source_condition, args.condition,
                            args.passive_condition, args.conditions.split(",") if args.conditions else None))
        return
    run_round(args.game, args.level, args.train_frac, args.source_condition, args.condition, args.from_probe,
              args.runs, backend_cfg(args), args.parallel, args.start, args.dry_run, args.mechanism, args.geometry_rule)


if __name__ == "__main__":
    main()
