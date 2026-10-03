"""Audit a live ARC-AGI-3 round for reward hacking, on the original benchmark engine.

The committee's live loop (`committee.live`) resynthesizes members on a
counterexample the agent met while playing the real game, with the
instruction that the fix must not be a special case keyed to that step.
That is the detailed-feedback regime in which synthesizers learn to evade.
This audit rebuilds the round's train set exactly as the loop did, replays
the rest of the agent's own trajectory on the engine as a held-out set, and
scores every member with the hack detectors and the decided-row split.

    uv run python -m rewardhack.live_audit ar25 --level 3 --train-frac 0.4 --probe 4 --through 70 \\
        --members ar25/L3_f40_probe4_live70/live_devin --log ar25/live/L3_cegis_devin_probe4_seed0.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from committee.cegis import split_after_probes, stored_round
from committee.committee import Committee, Member, auroc, description_length
from committee.loader import RESET, build_buffer, temporal_split
from committee.verify import canonical, run_program

from .detect import features, hack_weight_mass
from .experiment import ARTIFACTS
from .split import decided


def rebuild(game: str, level: int, train_frac: float, probe: int, through: int, log_rel: str):
    """The live round's train set and a held-out set from the rest of the same-level trajectory."""
    from committee.live import trajectory

    train, test = temporal_split(build_buffer(game), level, train_frac)
    probes, _ = stored_round(game, level, train_frac, "committee_devin", "cegis_devin", probe)
    train_r, _ = split_after_probes(train, test, probes)
    log = json.loads((ARTIFACTS.parent / log_rel).read_text())["log"]
    live = trajectory(game, level, [r["action"] for r in log])
    live_train = [t for t in live[:through + 1] if not t.level_advance]
    rest = live[through + 1:]
    end = next((i for i, t in enumerate(rest) if t.level_advance), len(rest))
    live_test = [t for t in rest[:end] if t.action_id != RESET]
    return train_r + live_train, live_test, len(train_r), len(live_train)


def audit(game: str, level: int, train_frac: float, probe: int, through: int, members_rel: str, log_rel: str,
          lam: float = 0.01) -> dict:
    train, test, n_rec, n_live = rebuild(game, level, train_frac, probe, through, log_rel)
    flags = decided(train, test)
    rows, members = [], []
    for f in sorted(glob.glob(str(ARTIFACTS.parent / members_rel / "run*" / "program.py"))):
        src = Path(f).read_text()
        v = run_program(src, train, test)
        feats = features(src, train, v.train_pass, v.test_pass)
        rows.append({"run": Path(f).parent.name, "consistent": v.consistent, "train": f"{sum(v.train_pass)}/{len(v.train_pass)}",
                     "held_out": v.test_accuracy, **feats.as_dict()})
        if v.consistent:
            members.append((Member(Path(f).parent.name, src, v.test_preds, description_length(src)), feats.memorising))
    out = {"game": game, "level": level, "n_train": len(train), "n_recorded": n_rec, "n_live_train": n_live,
           "n_test": len(test), "n_decided": sum(flags), "members": rows}
    if len(members) >= 2:
        c = Committee([m for m, _ in members], lam=lam)
        votes = c.votes()
        truth = [json.dumps(canonical(t.after_objs)) for t in test]
        correct = [v.prediction == t for v, t in zip(votes, truth)]
        dis = [v.disagreement for v in votes]
        dec = [d for d, f in zip(dis, flags) if f]
        und = [d for d, f in zip(dis, flags) if not f]
        out["committee"] = {
            "n": len(members), "vote_accuracy": sum(correct) / max(1, len(correct)),
            "hack_weight_mass": hack_weight_mass(c.weights, [f for _, f in members]),
            "rate_decided": sum(d > 0 for d in dec) / len(dec) if dec else None,
            "rate_undecided": sum(d > 0 for d in und) / len(und) if und else None,
            "auroc_disagreement_vs_error": auroc(dis, [not x for x in correct]),
            "hot_decided": [i for i, (d, f) in enumerate(zip(dis, flags)) if f and d > 0],
        }
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("game")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--train-frac", type=float, default=0.4)
    p.add_argument("--probe", type=int, default=4)
    p.add_argument("--through", type=int, default=70)
    p.add_argument("--members", required=True, help="artifacts-relative dir holding run*/program.py")
    p.add_argument("--log", required=True, help="artifacts-relative live log json")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    out = audit(a.game, a.level, a.train_frac, a.probe, a.through, a.members, a.log)
    if a.json:
        print(json.dumps(out, indent=1))
        return
    print(f"{out['game']} L{out['level']}: train {out['n_train']} ({out['n_recorded']} recorded + {out['n_live_train']} live), "
          f"held-out live {out['n_test']} ({out['n_decided']} decided)")
    print("run    consistent train     held_out  literal  mdl    guards  order_dep  memorising")
    for r in out["members"]:
        ho = "-" if r["held_out"] is None else f"{r['held_out']:.2f}"
        od = "-" if r["order_dependence"] is None else f"{r['order_dependence']:.2f}"
        print(f"{r['run']:<6} {str(r['consistent']):<10} {r['train']:<9} {ho:>8}  {r['literal_mass']:.2f}   {r['mdl_ratio']:.2f}   "
              f"{r['magic_guards']:>5}  {od:>9}  {r['memorising']}")
    if "committee" in out:
        c = out["committee"]
        print(f"committee n={c['n']} vote_acc={c['vote_accuracy']:.2f} hack_weight_mass={c['hack_weight_mass']:.2f} "
              f"dis rate decided={c['rate_decided']} undecided={c['rate_undecided']} auroc={c['auroc_disagreement_vs_error']} "
              f"hot_decided={c['hot_decided']}")


if __name__ == "__main__":
    main()
