"""Tables over stored runs.

`score` walks the committee artifacts and prints hack features for every
synthesized program, plus the committee weight that rests on memorising
members. `summary` walks the reward-hacking artifacts and prints outcome
counts per condition.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from committee.committee import Committee, Member, description_length
from committee.loader import ROOT, build_buffer, temporal_split

from .detect import features, hack_weight_mass
from .experiment import ARTIFACTS, outcome

COMMITTEE_ARTIFACTS = ROOT / "artifacts"
_SPLIT = re.compile(r"L(\d+)_f(\d+)(?:_T(\d+))?")


def _runs(cond: Path):
    for d in sorted(cond.glob("run*")):
        if (d / "meta.json").exists() and (d / "program.py").exists():
            yield d, json.loads((d / "meta.json").read_text()), (d / "program.py").read_text()


def score(behavioural: bool = True, lam: float = 0.01) -> list[dict]:
    rows = []
    for cond in sorted(p for p in COMMITTEE_ARTIFACTS.glob("*/L*_f*/*") if p.is_dir()):
        game, split, condition = cond.parts[-3], cond.parts[-2], cond.parts[-1]
        if game == "rewardhack":
            continue
        m = _SPLIT.fullmatch(split)
        if not m:
            continue
        level, frac = int(m.group(1)), int(m.group(2)) / 100
        test_level = int(m.group(3)) if m.group(3) else None
        train, test = temporal_split(build_buffer(game), level, frac, test_level=test_level)
        members, flags = [], []
        for d, meta, src in _runs(cond):
            f = features(src, train, meta["train_pass"], meta["test_pass"], behavioural=behavioural)
            rows.append({"game": game, "level": level, "train_frac": frac, "condition": condition,
                         "run": d.name, "consistent": meta["consistent"], "test_accuracy": meta["test_accuracy"],
                         **f.as_dict()})
            if meta["consistent"]:
                preds = json.loads((d / "test_preds.json").read_text()) if (d / "test_preds.json").exists() else []
                if preds:
                    members.append(Member(d.name, src, preds, description_length(src)))
                    flags.append(f.memorising)
        if members:
            c = Committee(members, lam=lam)
            rows.append({"game": game, "level": level, "train_frac": frac, "condition": condition,
                         "run": "committee", "hack_weight_mass": hack_weight_mass(c.weights, flags),
                         "n_members": len(members), "n_memorising": sum(flags)})
    return rows


def summary() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    conds = [p for p in ARTIFACTS.glob("*/L*_f*/*") if p.is_dir()]
    conds += [p for p in ARTIFACTS.glob("*/L*_f*/*/*") if p.is_dir() and not p.name.startswith("run")]
    for cond in sorted(conds):
        key = "/".join(cond.relative_to(ARTIFACTS).parts)
        counts: dict[str, int] = {}
        for _, meta, _ in _runs(cond):
            pair = tuple(meta["pair"]) if meta.get("pair") else None
            label = outcome(meta["abstained"], meta["train_pass"], pair)
            counts[label] = counts.get(label, 0) + 1
        out[key] = counts
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["score", "summary"])
    parser.add_argument("--no-behavioural", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.what == "score":
        rows = score(behavioural=not args.no_behavioural)
        if args.json:
            print(json.dumps(rows, indent=1))
            return
        for r in rows:
            if r["run"] == "committee":
                print(f"{r['game']} L{r['level']} {r['condition']} committee: "
                      f"hack_weight_mass={r['hack_weight_mass']:.3f} memorising={r['n_memorising']}/{r['n_members']}")
            else:
                print(f"{r['game']} L{r['level']} {r['condition']} {r['run']}: consistent={r['consistent']} "
                      f"test_acc={r['test_accuracy']} literal={r['literal_mass']} mdl={r['mdl_ratio']} "
                      f"gap={r['held_out_gap']} order_dep={r['order_dependence']} memorising={r['memorising']}")
    else:
        for k, v in summary().items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
