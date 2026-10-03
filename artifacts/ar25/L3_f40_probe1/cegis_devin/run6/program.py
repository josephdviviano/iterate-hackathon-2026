# Mechanics: one 3x3-block mirror axis (wall row block A) + pieces (player block sets) + targets (static
# block sets) + reflections (piece block b -> row 2A-b, skipped on piece blocks, clipped to the board).
# ACTION5 cycles axis -> pieces (bbox order) -> axis; 1/2/3/4 move the selection by x-3/x+3/y-3/y+3 (the
# axis only on 1/2, passing UNDER pieces; pieces blocked by bounds/other pieces). Frame = layered render
# (piece4>reflection3>target2>wall1, first solid wins, else topmost hole fill) then re-extraction.
# Unconfirmed: piece->piece/axis order of ACTION5 after the first piece, ACTION6/7 (treated as no-ops).
import json

N = 21  # blocks per side (63 cells); row 63 holds the counter
_memo = {"out": None, "model": None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells_of(o):
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v >= 0:
                yield o["x"] + i, o["y"] + j, v


def frame(state):
    g = {}
    for o in state:
        for x, y, v in cells_of(o):
            g[(x, y)] = v
    return g


def parse(state):
    g = frame(state)
    walls = [o for o in state if o["type"] == "wall"]
    axis = max(walls, key=lambda o: o["h"])["x"] // 3
    axis_sel = any(v == 0 for o in walls for _, _, v in cells_of(o))
    targets = []
    for o in state:
        if o["type"] == "target":
            targets.append(frozenset((x // 3, y // 3) for x, y, _ in cells_of(o)))
    pieces, sel = [], None
    for o in state:
        if o["type"] != "player":
            continue
        blocks = {(x // 3, y // 3) for x, y, v in cells_of(o) if v == 5}
        def zero_c(b):
            return g.get((3 * b[0] + 1, 3 * b[1] + 1)) == 0 and not (axis_sel and b[0] == axis)
        def lower_c(b):
            return g.get((3 * b[0] + 1, 3 * b[1] + 1), 0) not in (0, 5) and (3 * b[0] + 1, 3 * b[1] + 1) in g
        seeds = {b for b in blocks if zero_c(b)}
        if seeds and not axis_sel:
            grp, stack = set(seeds), list(seeds)
            while stack:
                bx, by = stack.pop()
                for nb in ((bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)):
                    if nb in blocks and nb not in grp and (zero_c(nb) or lower_c(nb)):
                        grp.add(nb)
                        stack.append(nb)
            rest = blocks - grp
            pieces.append(frozenset(grp))
            sel = len(pieces) - 1
            for comp in block_comps(rest):
                pieces.append(comp)
        else:
            pieces.append(frozenset(blocks)) if len(block_comps(blocks)) <= 1 else pieces.extend(block_comps(blocks))
    if axis_sel:
        sel = None
    return {"axis": axis, "axis_sel": axis_sel, "pieces": pieces, "sel": sel, "targets": targets}


def block_comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        s = blocks.pop()
        comp, stack = {s}, [s]
        while stack:
            bx, by = stack.pop()
            for nb in ((bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)):
                if nb in blocks:
                    blocks.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        out.append(frozenset(comp))
    return out


def bbox_key(blocks):
    return (min(b[0] for b in blocks), min(b[1] for b in blocks))


def step(m, action):
    m = dict(m)
    pieces = list(m["pieces"])
    a = action["action_id"] if isinstance(action, dict) else action
    if a == 5:
        order = sorted(range(len(pieces)), key=lambda i: bbox_key(pieces[i]))
        if m["axis_sel"]:
            m["axis_sel"], m["sel"] = (False, order[0]) if order else (True, None)
        else:
            k = order.index(m["sel"])
            if k + 1 < len(order):
                m["sel"] = order[k + 1]
            else:
                m["axis_sel"], m["sel"] = True, None
    elif a in (1, 2, 3, 4):
        dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[a]
        if m["axis_sel"]:
            if dx and 0 <= m["axis"] + dx < N:
                m["axis"] += dx
        elif m["sel"] is not None:
            moved = frozenset((bx + dx, by + dy) for bx, by in pieces[m["sel"]])
            others = set().union(*[p for i, p in enumerate(pieces) if i != m["sel"]]) if len(pieces) > 1 else set()
            if all(0 <= bx < N and 0 <= by < N for bx, by in moved) and not (moved & others):
                pieces[m["sel"]] = moved
    m["pieces"] = pieces
    return m


def render(m):
    """Return {cell: (value, kind, owner)} for every visible cell."""
    A = m["axis"]
    occupied = set().union(*m["pieces"]) if m["pieces"] else set()
    refl = set()
    for p in m["pieces"]:
        for bx, by in p:
            r = 2 * A - bx
            if 0 <= r < N and (r, by) not in occupied:
                refl.add((r, by))
    # layers top-down: (kind, owner, blocks, ring colour, hole: ('solid', v) | ('fill', v))
    layers = []
    for i, p in enumerate(m["pieces"]):
        layers.append(("player", i, p, 5, ("fill", 0 if m["sel"] == i and not m["axis_sel"] else -1)))
    layers.append(("reflection", 0, refl, 4, ("fill", 4)))
    for i, t in enumerate(m["targets"]):
        layers.append(("target", i, t, 11, ("solid", 11)))
    wall = {(A, by) for by in range(N)}
    layers.append(("wall", 0, wall, 10, ("solid", 0) if m["axis_sel"] else ("fill", -1)))
    out = {}
    for bx in range(N):
        for by in range(N):
            stack = [L for L in layers if (bx, by) in L[2]]
            if not stack:
                continue
            for i in range(3):
                for j in range(3):
                    centre = (i, j) == (1, 1)
                    hit = None
                    for kind, own, _, col, hole in stack:
                        if not centre:
                            hit = (col, kind, own)
                            break
                        if hole[0] == "solid":
                            hit = (hole[1], kind, own)
                            break
                    if hit is None:
                        for kind, own, _, col, hole in stack:
                            if hole[1] >= 0:
                                hit = (hole[1], kind, own)
                                break
                    if hit is not None:
                        out[(3 * bx + i, 3 * by + j)] = hit
    return out


def comps(cells):
    cells, res = set(cells), []
    while cells:
        s = cells.pop()
        comp, stack = {s}, [s]
        while stack:
            x, y = stack.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    stack.append(n)
        res.append(comp)
    return res


def make_obj(name, typ, tags, layer, cells, vals):
    x0 = min(c[0] for c in cells); y0 = min(c[1] for c in cells)
    x1 = max(c[0] for c in cells); y1 = max(c[1] for c in cells)
    pix = [[vals[(x, y)] if (x, y) in cells else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {"name": name, "type": typ, "tags": tags, "layer": layer, "x": x0, "y": y0,
            "w": x1 - x0 + 1, "h": y1 - y0 + 1, "pixels": pix}


def extract(r, scenery):
    vals = {c: v[0] for c, v in r.items()}
    zeros = {c for c, v in r.items() if v[0] == 0}
    objs = list(scenery)
    spec = {"wall": ("wall_h", ["axis", "horizontal"], 1), "player": ("piece", ["movable", "black"], 4)}
    for kind, (prefix, tags, layer) in spec.items():
        own = {c for c, v in r.items() if v[1] == kind and v[0] != 0}
        cs = sorted(comps(own | zeros), key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
        for k, c in enumerate(cs):
            if c & own:
                objs.append(make_obj(f"{prefix}_{k}", kind, tags, layer, c, vals))
    refl = {c for c, v in r.items() if v[1] == "reflection"}
    cs = sorted(comps(refl), key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    for k, c in enumerate(cs):
        objs.append(make_obj(f"reflection_{k}", "reflection", ["mirror", "gray"], 3, c, vals))
    tg = {}
    for c, v in r.items():
        if v[1] == "target":
            tg.setdefault(v[2], set()).add(c)
    cs = sorted(tg.values(), key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    for k, c in enumerate(cs):
        objs.append(make_obj(f"target_{k}", "target", ["goal", "yellow"], 2, c, vals))
    return objs


def transition_function(state, action):
    if _memo["out"] is not None and canon(state) == _memo["out"]:
        m = _memo["model"]
    else:
        m = parse(state)
    m2 = step(m, action)
    scenery = [o for o in state if o["type"] not in ("wall", "player", "reflection", "target")]
    out = extract(render(m2), scenery)
    _memo["out"], _memo["model"] = canon(out), m2
    return out
