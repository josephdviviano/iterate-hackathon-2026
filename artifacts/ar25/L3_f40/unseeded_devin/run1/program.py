# Mechanics: all sprites are 3x3 cells. One of {mirror axis (wall), pieces} is selected (holes fill 0); actions
# 1-4 move it by 3 (axis only L/R; blocked by bounds/overlap), action 5 cycles axis -> pieces (x,y order) -> axis.
# Reflections (colour 4, holes fill 4 over empty) mirror every piece across the axis centre column.
# Frame is rendered by layer, then re-extracted: wall/piece/reflection = 4-connected visible parts, target = visible cells.
# Unconfirmed: cycle order past first piece, blocking rules, actions 6/7, wall name offset (= # of non-wall 0-components).
import copy

W = 63  # column 63 is covered by the counter HUD
LAYER = {"wall": 1, "target": 2, "reflection": 3, "player": 4}
COLOR = {"wall": 10, "target": 11, "reflection": 4, "player": 5}
DEFAULT_TAGS = {"wall": ["axis", "horizontal"], "target": ["goal", "yellow"],
                "reflection": ["mirror", "gray"], "player": ["movable", "black"]}
PREFIX = {"wall": "wall_h_", "target": "target_", "reflection": "reflection_", "player": "piece_"}


def obj_pixels(o):
    for i, col in enumerate(o.get("pixels", [])):
        for j, v in enumerate(col):
            if v >= 0:
                yield o["x"] + i, o["y"] + j, v


def obj_cells(o):
    cells = {}
    for x, y, v in obj_pixels(o):
        c = (x - x % 3, y - y % 3)
        cells.setdefault(c, {})[(x - c[0], y - c[1])] = v
    return cells


def neighbours(c, step):
    x, y = c
    return [(x + step, y), (x - step, y), (x, y + step), (x, y - step)]


def components(points, step):
    points, out = set(points), []
    while points:
        seed = points.pop()
        comp, stack = {seed}, [seed]
        while stack:
            for n in neighbours(stack.pop(), step):
                if n in points:
                    points.discard(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


# ---------------------------------------------------------------- parse
def parse(state):
    by_type = {}
    for o in state:
        by_type.setdefault(o["type"], []).append(o)
    walls = by_type.get("wall", [])
    axis = None
    if walls:
        ys = [y for o in walls for _, y, _ in obj_pixels(o)]
        axis = {"x": min(o["x"] for o in walls), "y0": min(ys) - min(ys) % 3, "y1": max(ys),
                "sel": any(v == 0 for o in walls for _, _, v in obj_pixels(o))}
    targets = [set(obj_cells(o)) for o in by_type.get("target", [])]
    target_px = {(x, y) for o in by_type.get("target", []) for x, y, _ in obj_pixels(o)}
    # piece cells: centre 0 = selected, centre empty over empty = unselected, over target = ambiguous
    kind = {}
    for o in by_type.get("player", []):
        for c, px in obj_cells(o).items():
            centre = px.get((1, 1))
            if centre == 0:
                kind[c] = "sel"
            elif (c[0] + 1, c[1] + 1) in target_px:
                kind[c] = "amb"
            else:
                kind[c] = "uns"
    sel_cells = {c for c, k in kind.items() if k == "sel"}
    stack = list(sel_cells)
    while stack:
        for n in neighbours(stack.pop(), 3):
            if kind.get(n) == "amb" and n not in sel_cells:
                sel_cells.add(n)
                stack.append(n)
    pieces = []
    if sel_cells:
        pieces.append({"cells": sel_cells, "sel": True})
    rest = [comp for comp in components(set(kind) - sel_cells, 3)]
    if not sel_cells and not (axis and axis["sel"]):
        for comp in rest:
            if all(kind[c] == "amb" for c in comp):
                rest.remove(comp)
                pieces.append({"cells": comp, "sel": True})
                break
    pieces += [{"cells": comp, "sel": False} for comp in rest]
    templates = {t: os_[0] for t, os_ in by_type.items()}
    others = [o for o in state if o["type"] not in LAYER]
    return axis, pieces, targets, templates, others


# ---------------------------------------------------------------- dynamics
def in_bounds(cells):
    return all(0 <= x and x + 2 < W and 0 <= y and y + 2 < 64 for x, y in cells)


def axis_cols(axis):
    return set(range(axis["x"], axis["x"] + 3)) if axis else set()


def step(axis, pieces, action):
    if action == 5:
        order = sorted(pieces, key=lambda p: (min(c[0] for c in p["cells"]), min(c[1] for c in p["cells"])))
        chain = ([axis] if axis else []) + order
        cur = next((i for i, s in enumerate(chain) if s["sel"]), None)
        if cur is not None and len(chain) > 1:
            chain[cur]["sel"] = False
            chain[(cur + 1) % len(chain)]["sel"] = True
        return
    d = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}.get(action)
    if d is None:
        return
    if axis and axis["sel"]:
        if d[1] != 0:
            return
        nx = axis["x"] + d[0]
        piece_x = {x + dx for p in pieces for x, _ in p["cells"] for dx in range(3)}
        if 0 <= nx and nx + 2 < W and not (set(range(nx, nx + 3)) & piece_x):
            axis["x"] = nx
        return
    for p in pieces:
        if not p["sel"]:
            continue
        moved = {(x + d[0], y + d[1]) for x, y in p["cells"]}
        others = set().union(*[q["cells"] for q in pieces if q is not p])
        hits_axis = any(x + dx in axis_cols(axis) for x, _ in moved for dx in range(3))
        if in_bounds(moved) and not (moved & others) and not hits_axis:
            p["cells"] = moved
        return


def reflect(axis, pieces):
    if not axis:
        return []
    c2 = 2 * (axis["x"] + 1)
    return [{"cells": {(c2 - x - 2, y) for x, y in p["cells"]}} for p in pieces]


# ---------------------------------------------------------------- render + extract
def render(axis, pieces, targets, refls):
    grid = {}  # (x,y) -> (type, sprite index, colour)
    sprites = []
    if axis:
        sprites.append(("wall", {(axis["x"], y) for y in range(axis["y0"], axis["y1"] + 1, 3)},
                        0 if axis["sel"] else None))
    sprites += [("target", t, "solid") for t in targets]
    sprites += [("reflection", r["cells"], 4) for r in refls]
    sprites += [("player", p["cells"], 0 if p["sel"] else None) for p in pieces]
    for idx, (typ, cells, hole) in sorted(enumerate(sprites), key=lambda e: LAYER[e[1][0]]):
        col = COLOR[typ]
        for cx, cy in cells:
            for dx in range(3):
                for dy in range(3):
                    x, y = cx + dx, cy + dy
                    if not (0 <= x < W and 0 <= y < 64):
                        continue
                    if (dx, dy) == (1, 1) and hole != "solid":
                        if hole is not None and (x, y) not in grid:
                            grid[(x, y)] = (typ, idx, hole)
                    elif typ != "wall" or y <= axis["y1"]:
                        grid[(x, y)] = (typ, idx, col)
    return grid


def make_obj(typ, name, pts, grid, templates):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for x, y in pts:
        pix[x - x0][y - y0] = grid[(x, y)][2]
    t = templates.get(typ, {})
    return {"name": name, "type": typ, "x": x0, "y": y0, "w": w, "h": h,
            "layer": t.get("layer", LAYER[typ]), "tags": list(t.get("tags", DEFAULT_TAGS[typ])),
            "pixels": pix}


def extract(grid, templates):
    out = []
    zero_comps = len(components([p for p, v in grid.items() if v[2] == 0 and v[0] != "wall"], 1))
    for typ in ("wall", "player", "reflection"):
        comps = components([p for p, v in grid.items() if v[0] == typ], 1)
        comps.sort(key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
        base = zero_comps if typ == "wall" else 0
        for i, c in enumerate(comps):
            out.append(make_obj(typ, PREFIX[typ] + str(base + i), c, grid, templates))
    by_sprite = {}
    for p, v in grid.items():
        if v[0] == "target":
            by_sprite.setdefault(v[1], set()).add(p)
    tl = sorted(by_sprite.values(), key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    for i, c in enumerate(tl):
        out.append(make_obj("target", PREFIX["target"] + str(i), c, grid, templates))
    return out


def transition_function(state, action):
    state = copy.deepcopy(state)
    if isinstance(action, dict):
        action = action.get("action_id")
    axis, pieces, targets, templates, others = parse(state)
    step(axis, pieces, action)
    grid = render(axis, pieces, targets, reflect(axis, pieces))
    return others + extract(grid, templates)
