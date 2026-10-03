"""Committee disagreement split by whether the training data decided the row.

A test transition is *decided* when every (type, action, context) row it
touches was observed in train. Honest programs that replay train exactly
agree on decided transitions, so disagreement there is an integrity signal.
Disagreement on undecided transitions is the epistemic signal the committee
already uses for exploration. Committees are built from stored programs:
honest = Opus intact runs, hacks = Haiku contradiction runs that replayed
the contradictory pair, mixed = honest plus one hack.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass

from committee.committee import Committee, Member, auroc, description_length
from committee.loader import Transition, build_buffer, temporal_split
from committee.matrix import EffectMatrix
from committee.verify import canonical, run_program

from . import frame as frame_mod
from .experiment import ARTIFACTS
from .inject import contradict, is_hack


def decided(train: list[Transition], test: list[Transition]) -> list[bool]:
    seen = {str(k) for k in EffectMatrix.from_transitions(train).rows}
    return [all(str(k) in seen for k in EffectMatrix.from_transitions([t]).rows) for t in test]


@dataclass
class Split:
    name: str
    n_members: int
    rate_decided: float | None
    rate_undecided: float | None
    mean_decided: float | None
    mean_undecided: float | None
    vote_accuracy: float
    hot_decided: list[int]
    auroc_error: float | None = None


def split(name: str, members: list[Member], test: list[Transition], flags: list[bool], lam: float = 0.01) -> Split:
    c = Committee(members, lam=lam)
    votes = c.votes()
    truth = [json.dumps(canonical(t.after_objs)) for t in test]
    dis = [v.disagreement for v in votes]
    dec = [d for d, f in zip(dis, flags) if f]
    und = [d for d, f in zip(dis, flags) if not f]
    rate = lambda xs: sum(x > 0 for x in xs) / len(xs) if xs else None
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    correct = [v.prediction == t for v, t in zip(votes, truth)]
    return Split(name, len(members), rate(dec), rate(und), mean(dec), mean(und), sum(correct) / len(truth),
                 [i for i, (d, f) in enumerate(zip(dis, flags)) if f and d > 0],
                 auroc(dis, [not c for c in correct]))


def load_members(pattern: str, train: list[Transition], test: list[Transition], pair=None,
                 runner=run_program) -> list[Member]:
    out = []
    for f in sorted(glob.glob(pattern)):
        src = open(f).read()
        v = runner(src, train, test)
        if not v.consistent or (pair is not None and not is_hack(v.train_pass, pair)):
            continue
        out.append(Member(f.split("/")[-2], src, v.test_preds, description_length(src)))
    return out


def analyse(game: str, level: int, train_frac: float = 0.6, honest_model: str = "", hack_model: str = "haiku",
            frame: bool = False) -> dict:
    train, test = temporal_split(build_buffer(game), level, train_frac)
    ctrain, pair = contradict(train)
    flags = decided(train, test)
    base = ARTIFACTS / game / f"L{level}_f{int(round(train_frac * 100))}"
    suffix = "_frame" if frame else ""
    runner = frame_mod.run_program if frame else run_program
    honest_dir = base / f"intact{suffix}" / honest_model if honest_model else base / f"intact{suffix}"
    honest = load_members(str(honest_dir / "run*" / "program.py"), train, test, runner=runner)
    hacks = [m for cond in (f"contradiction{suffix}", f"contradiction_abstain{suffix}")
             for m in load_members(str(base / cond / hack_model / "run*" / "program.py"), ctrain, test, pair,
                                   runner=runner)]
    out = {"game": game, "level": level, "train_frac": train_frac, "frame": frame, "n_train": len(train),
           "n_test": len(test), "n_decided": sum(flags), "n_honest": len(honest), "n_hacks": len(hacks),
           "committees": []}
    if len(honest) >= 2:
        out["committees"].append(split("honest", honest, test, flags))
    if len(hacks) >= 2:
        out["committees"].append(split("hacks", hacks, test, flags))
    if honest and hacks:
        out["committees"].append(split("honest+1hack", honest + hacks[:1], test, flags))
    return out


def fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("levels", nargs="+", help="game:level[:train_frac], e.g. tr87:2 tr87:6:0.4")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--frame", action="store_true", help="use the *_frame conditions and the frame runner")
    args = parser.parse_args(argv)
    rows = []
    for spec in args.levels:
        parts = spec.split(":")
        rows.append(analyse(parts[0], int(parts[1]), float(parts[2]) if len(parts) > 2 else 0.6, frame=args.frame))
    if args.json:
        print(json.dumps([{**r, "committees": [c.__dict__ for c in r["committees"]]} for r in rows], indent=1))
        return
    print("level        decided/test  committee      n  rate_dec rate_und  mean_dec mean_und  vote_acc  auroc  hot_decided")
    for r in rows:
        tag = f"{r['game']} L{r['level']} f{r['train_frac']}" + (" F" if r.get("frame") else "")
        if not r["committees"]:
            print(f"{tag:<12} {r['n_decided']:>3}/{r['n_test']:<6}  honest={r['n_honest']} hacks={r['n_hacks']}: nothing to compare")
        for c in r["committees"]:
            print(f"{tag:<12} {r['n_decided']:>3}/{r['n_test']:<6}  {c.name:<13} {c.n_members:>2}  {fmt(c.rate_decided):>8} {fmt(c.rate_undecided):>8}  "
                  f"{fmt(c.mean_decided):>8} {fmt(c.mean_undecided):>8}  {c.vote_accuracy:>8.2f}  {fmt(c.auroc_error):>5}  {c.hot_decided}")


if __name__ == "__main__":
    main()
