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


def _boxes(objs: list[dict], h: int, w: int) -> list[tuple[int, int, int, int]]:
    """Object bounding boxes, without frame-sized container objects."""
    return [(int(o.get("x", 0)), int(o.get("y", 0)), int(o.get("w", 1)), int(o.get("h", 1)))
            for o in objs if int(o.get("w", 1)) * int(o.get("h", 1)) < 0.9 * h * w]


def _covered(boxes, h: int, w: int) -> list[list[bool]]:
    cov = [[False] * w for _ in range(h)]
    for bx, by, bw, bh in boxes:
        for y in range(max(0, by), min(h, by + bh)):
            for x in range(max(0, bx), min(w, bx + bw)):
                cov[y][x] = True
    return cov


def static_terrain(train: list[Transition], min_cells: int = 4) -> list[dict]:
    """Terrain cells: non-background cells whose colour is the same in every training frame where no
    object covers them. A cell under an object in some frames still counts when it is consistent
    whenever visible; a cell never visible is left out."""
    frames = [(t.before_grid, t.before_objs) for t in train] + [(train[-1].after_grid, train[-1].after_objs)]
    h, w = len(frames[0][0]), len(frames[0][0][0])
    bg = Counter(v for row in frames[0][0] for v in row).most_common(1)[0][0]
    seen_colour: dict[tuple[int, int], int] = {}
    unstable: set[tuple[int, int]] = set()
    for grid, objs in frames:
        cov = _covered(_boxes(objs, h, w), h, w)
        for y in range(h):
            for x in range(w):
                if cov[y][x] or (x, y) in unstable:
                    continue
                c = grid[y][x]
                if (x, y) in seen_colour and seen_colour[(x, y)] != c:
                    unstable.add((x, y))
                    del seen_colour[(x, y)]
                elif (x, y) not in seen_colour:
                    seen_colour[(x, y)] = c
    static = {p: c for p, c in seen_colour.items() if c != bg}
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
        if len(comp) < min_cells:
            continue
        xs, ys = [p[0] for p in comp], [p[1] for p in comp]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        cells = set(comp)
        pixels = [[colour if (x, y) in cells else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]
        objects.append({"name": f"terrain_{len(objects)}", "type": "terrain", "x": x0, "y": y0,
                        "w": x1 - x0 + 1, "h": y1 - y0 + 1, "layer": 0, "colour": colour, "pixels": pixels})
    return objects


def completeness(transitions: list[Transition], terrain: list[dict]) -> dict:
    """Share of non-background cells, over every frame given, that lie inside an extracted object or a
    terrain object, and the residual by colour. The residual is what a program over objects cannot see."""
    tcells = {(o["x"] + i, o["y"] + j) for o in terrain for j, row in enumerate(o["pixels"]) for i, v in enumerate(row) if v >= 0}
    total = explained = 0
    residual: Counter = Counter()
    for t in transitions:
        for grid, objs in ((t.before_grid, t.before_objs), (t.after_grid, t.after_objs)):
            h, w = len(grid), len(grid[0])
            bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
            cov = _covered(_boxes(objs, h, w), h, w)
            for y in range(h):
                for x in range(w):
                    if grid[y][x] == bg:
                        continue
                    total += 1
                    if cov[y][x] or (x, y) in tcells:
                        explained += 1
                    else:
                        residual[grid[y][x]] += 1
    return {"non_background_cells": total, "explained_share": round(explained / max(1, total), 4),
            "residual_by_colour": dict(residual.most_common(6))}


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
    parser.add_argument("--coverage", action="store_true", help="completeness of objects plus terrain over train and test frames")
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
    if args.coverage:
        before = completeness(train + test, [])
        after = completeness(train + test, terrain)
        print(f"  objects only: explained {before['explained_share']:.3f}, residual {before['residual_by_colour']}")
        print(f"  objects + terrain: explained {after['explained_share']:.3f}, residual {after['residual_by_colour']}")
        return
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
