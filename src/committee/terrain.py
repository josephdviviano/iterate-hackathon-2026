"""Static terrain as objects.

OPINE-World's extractors list the object types that change and leave static
terrain out of the state. On some levels that terrain gates a mechanic (ka59: a
kicked block slides until it meets a wall the extractor never emits), so a
program over objects can only memorise where things stop. This module adds the
terrain back as constant objects: the non-background cells whose colour never
changes across the training frames, grouped by colour into 4-connected
components, appended to every state. They are identical in every state, so a
program passes them through; what it gains is the geometry.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

from .loader import Transition


def static_terrain(train: list[Transition]) -> list[dict]:
    frames = [t.before_grid for t in train] + [train[-1].after_grid]
    h, w = len(frames[0]), len(frames[0][0])
    bg = Counter(v for row in frames[0] for v in row).most_common(1)[0][0]
    boxes = [(int(o.get("x", 0)), int(o.get("y", 0)), int(o.get("w", 1)), int(o.get("h", 1)))
             for o in train[0].before_objs if int(o.get("w", 1)) * int(o.get("h", 1)) < 0.9 * h * w]

    def inside_object(x: int, y: int) -> bool:
        return any(bx <= x < bx + bw and by <= y < by + bh for bx, by, bw, bh in boxes)

    static = {(x, y): frames[0][y][x] for y in range(h) for x in range(w)
              if frames[0][y][x] != bg and not inside_object(x, y)
              and all(f[y][x] == frames[0][y][x] for f in frames)}
    seen: set[tuple[int, int]] = set()
    objects = []
    for start in sorted(static, key=lambda p: (p[1], p[0])):
        if start in seen:
            continue
        colour, comp, stack = static[start], [], [start]
        seen.add(start)
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in static and (nx, ny) not in seen and static[(nx, ny)] == colour:
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        if len(comp) < 4:
            continue
        xs, ys = [p[0] for p in comp], [p[1] for p in comp]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        cells = set(comp)
        pixels = [[colour if (x, y) in cells else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]
        objects.append({"name": f"terrain_{len(objects)}", "type": "terrain", "x": x0, "y": y0,
                        "w": x1 - x0 + 1, "h": y1 - y0 + 1, "layer": 0, "colour": colour, "pixels": pixels})
    return objects


def add_terrain(transitions: list[Transition], terrain: list[dict]) -> list[Transition]:
    extra = json.loads(json.dumps(terrain))
    return [replace(t, before_objs=t.before_objs + extra, after_objs=t.after_objs + extra) for t in transitions]


def strip_terrain(state: list[dict] | None) -> list[dict] | None:
    return None if state is None else [o for o in state if o.get("type") != "terrain"]


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .committee import Committee, Member, description_length
    from .evaluate import load_runs
    from .experiment import add_backend_args, backend_cfg, condition_dir, run_split
    from .loader import build_buffer, temporal_split
    from .seeds import make_seeds

    parser = argparse.ArgumentParser(description="Synthesize or evaluate a committee with static terrain objects.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--condition", default="terrain_devin")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true")
    add_backend_args(parser)
    args = parser.parse_args(argv)
    train, test = temporal_split(build_buffer(args.game), args.level, args.train_frac)
    terrain = static_terrain(train)
    train_t, test_t = add_terrain(train, terrain), add_terrain(test, terrain)
    cond = condition_dir(args.game, args.level, args.train_frac, args.condition)
    print(f"{args.game} L{args.level}: {len(terrain)} terrain objects, "
          f"{sum(sum(v >= 0 for row in o['pixels'] for v in row) for o in terrain)} cells, "
          f"colours {sorted({o['colour'] for o in terrain})}; train {len(train_t)}, test {len(test_t)}")
    if args.dry_run:
        for o in terrain[:12]:
            print("  ", {k: o[k] for k in ("name", "x", "y", "w", "h", "colour")})
        return
    if args.report:
        runs = load_runs(cond, train_t, test_t)
        members = [Member(n, s, [strip_terrain(p) for p in preds], description_length(s))
                   for n, m, s, preds in runs if m["consistent"] and preds]
        print(f"admitted {len(members)} of {len(runs)}")
        if members:
            e = Committee(members).evaluate([t.after_objs for t in test])
            un = next(r for r in e["reliability_uniform"] if r["bin"] == "unanimous")
            print(f"vote {e['vote_accuracy']:.3f}  members {sorted(round(a, 2) for a in e['member_accuracy'])}  "
                  f"AUROC {e['auroc_uniform_disagreement_vs_error']}  unanimous {un['n']} (error {un['error_rate']})  "
                  f"distinct {e['n_distinct_behaviours']}")
        return
    seeds = list(make_seeds(train_t, args.runs))
    run_split(train_t, test_t, cond, seeds, backend_cfg(args), f"{args.game} L{args.level} {args.condition}", 0, args.parallel)


if __name__ == "__main__":
    main()
