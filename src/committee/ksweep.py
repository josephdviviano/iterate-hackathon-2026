"""Committee size: accuracy and calibration against K, for seeded and unseeded members.

Each draw takes one random K-subset of a level's admitted members on every level and pools the
held-out transitions, so every K gets a pooled AUROC, vote accuracy and unanimous and split error,
with a 95 percent interval over draws.

    uv run python -m committee.ksweep
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .committee import Committee, auroc
from .evaluate import members_from
from .experiment import condition_dir
from .loader import build_buffer, temporal_split

LEVELS = (("ar25", 3), ("m0r0", 3), ("sk48", 2), ("ar25", 7), ("ls20", 3), ("ka59", 2), ("g50t", 1))
ARMS = {"seeded": ("committee_opine_devin", "committee16_opine_devin"),
        "unseeded": ("baseline_opine_devin", "unseeded_opine_devin")}


def level_members(game: str, level: int, conditions, train_frac: float = 0.4):
    train, test = temporal_split(build_buffer(game), level, train_frac)
    members = []
    for c in conditions:
        d = condition_dir(game, level, train_frac, c)
        if d.exists() and any(d.glob("run*/meta.json")):
            members += members_from(d, train, test, game)
    return members, test


def per_transition(members, test) -> list[tuple[float, bool, bool]]:
    """(disagreement, vote wrong, any member right) per held-out transition."""
    import json as _json

    from .verify import canonical

    com = Committee(members)
    truth = com.truth(test)
    e = com.evaluate(truth)
    keys = [_json.dumps(canonical(t)) for t in truth]
    return [(p["uniform_disagreement"], not p["correct"], any(m.keys[i] == keys[i] for m in members))
            for i, p in enumerate(e["per_transition"])]


def summarise(rows: list[tuple[float, bool, bool]]) -> dict:
    d = [r[0] for r in rows]
    wrong = [r[1] for r in rows]
    una = [w for x, w in zip(d, wrong) if x == 0]
    spl = [w for x, w in zip(d, wrong) if x > 0]
    return {"vote_accuracy": 1 - sum(wrong) / len(wrong), "oracle_accuracy": sum(r[2] for r in rows) / len(rows),
            "auroc": auroc(d, wrong), "unanimous_share": len(una) / len(rows),
            "unanimous_error": sum(una) / len(una) if una else None, "split_error": sum(spl) / len(spl) if spl else None}


def sweep(by_level: dict, ks, n_draws: int = 200, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for k in ks:
        if all(len(m) < k for m, _ in by_level.values()):
            continue
        draws = []
        for _ in range(n_draws):
            rows = []
            for members, test in by_level.values():  # a level with fewer admitted members uses all of them
                rows += per_transition(rng.sample(members, min(k, len(members))), test)
            draws.append(summarise(rows))
        stat = {}
        for key in draws[0]:
            vals = sorted(v[key] for v in draws if v[key] is not None)
            stat[key] = None if not vals else {"mean": sum(vals) / len(vals), "lo": vals[int(0.025 * len(vals))],
                                               "hi": vals[min(len(vals) - 1, int(0.975 * len(vals)))]}
        out.append({"k": k, "n_draws": n_draws, "levels_capped": [lv for lv, (m, _) in by_level.items() if len(m) < k], **stat})
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Committee size sweep, seeded and unseeded.")
    parser.add_argument("--ks", default="1,2,4,8,12,16")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--out", default="artifacts/ksweep_opine.json")
    args = parser.parse_args(argv)
    ks = [int(k) for k in args.ks.split(",")]
    result = {}
    for arm, conditions in ARMS.items():
        by_level = {f"{g} L{l}": level_members(g, l, conditions) for g, l in LEVELS}
        sizes = {lv: len(m) for lv, (m, _) in by_level.items()}
        result[arm] = {"members_per_level": sizes, "by_k": sweep(by_level, ks, args.draws)}
        print(f"\n{arm}: members per level {sizes}")
        for r in result[arm]["by_k"]:
            f = lambda key: "none" if r[key] is None else f"{r[key]['mean']:.3f} [{r[key]['lo']:.3f}, {r[key]['hi']:.3f}]"
            print(f"  K={r['k']:2d}  vote {f('vote_accuracy')}  oracle {f('oracle_accuracy')}  AUROC {f('auroc')}  "
                  f"unanimous share {f('unanimous_share')}  unanimous err {f('unanimous_error')}  split err {f('split_error')}")
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
