# Mechanics: a horizontal mirror axis (wall, 3 rows) and holed 3x3-block pieces; ACTION5 cycles the
# selection axis -> pieces (sorted) -> axis; the selection is shown by 0-filled holes. ACTION1-4 move
# the selection by 3 (axis only vertically). Each piece is mirrored across the axis centre as gray
# reflections. Frame is rendered by layer (wall<target<reflection<piece) and re-extracted: components
# per type, names sorted by (x,y), wall index offset by black pixels elsewhere. Unconfirmed: bounds/collisions, ACTION6/7.
import json

B = 3
N = 63
LAYER = {"wall": 1, "target": 2, "reflection": 3, "player": 4}
TAGS = {"wall": ["axis", "horizontal"], "target": ["goal", "yellow"],
        "reflection": ["mirror", "gray"], "player": ["movable", "black"]}
COLOR = {"wall": 10, "target": 11, "reflection": 4, "player": 5}
PREFIX = {"wall": "wall_h_", "target": "target_", "reflection": "reflection_", "player": "piece_"}
_MEM = {"key": None, "model": None, "hist": []}


def canon(state):
    return json.dumps(sorted(json.dumps(o, sort_keys=True) for o in state))


def cells_of(o):
    out = {}
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v >= 0:
                out[(o["x"] + i, o["y"] + j)] = v
    return out


def blk(r, c):
    return (r // B * B, c // B * B)


def block_cells(blocks):
    body, holes = set(), set()
    for (r, c) in blocks:
        for i in range(B):
            for j in range(B):
                (holes if (i, j) == (1, 1) else body).add((r + i, c + j))
    return body, holes


def components(blocks):
    blocks, comps = set(blocks), []
    while blocks:
        stack, comp = [blocks.pop()], set()
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for d in ((B, 0), (-B, 0), (0, B), (0, -B)):
                n = (r + d[0], c + d[1])
                if n in blocks:
                    blocks.remove(n)
                    stack.append(n)
        comps.append(frozenset(comp))
    return comps


def parse(state):
    """Stateless reading of the model from an observed frame."""
    grid, owner = {}, {}
    for o in sorted(state, key=lambda o: o.get("layer", 0)):
        for p, v in cells_of(o).items():
            grid[p], owner[p] = v, o["type"]
    walls = [o for o in state if o["type"] == "wall"]
    axis = min(o["x"] for o in walls) if walls else 30
    axis_sel = any(v == 0 for o in walls for p, v in cells_of(o).items())
    targets = []
    for o in state:
        if o["type"] == "target":
            targets.append(frozenset(blk(*p) for p, v in cells_of(o).items() if v == COLOR["target"]))
    pblocks = set()
    for o in state:
        if o["type"] == "player":
            pblocks |= {blk(*p) for p, v in cells_of(o).items() if v == COLOR["player"]}
    marked = {b for b in pblocks if grid.get((b[0] + 1, b[1] + 1)) == 0}
    covered = {b for b in pblocks if owner.get((b[0] + 1, b[1] + 1)) == "target"}
    pieces, sel = [], None
    for comp in components(pblocks):
        if comp & marked:
            chosen, stack = set(), list(comp & marked)
            while stack:
                b = stack.pop()
                if b in chosen:
                    continue
                chosen.add(b)
                for d in ((B, 0), (-B, 0), (0, B), (0, -B)):
                    n = (b[0] + d[0], b[1] + d[1])
                    if n in comp and (n in marked or n in covered):
                        stack.append(n)
            sel = len(pieces)
            pieces.append(frozenset(chosen))
            pieces.extend(components(comp - chosen))
        else:
            pieces.append(comp)
    if sel is None and not axis_sel and pieces:
        sel = 0
    return {"axis": axis, "sel": None if axis_sel else sel, "pieces": pieces, "targets": targets}


def order_key(blocks):
    return min(blocks)


def in_grid(blocks):
    return all(0 <= r and r + B <= N and 0 <= c and c + B <= N for r, c in blocks)


def hits_axis(blocks, axis):
    return any(axis <= r + i < axis + B for r, c in blocks for i in range(B))


def step(model, action):
    m = dict(model)
    m["pieces"] = list(model["pieces"])
    a = action.get("action_id") if isinstance(action, dict) else action
    moves = {1: (-B, 0), 2: (B, 0), 3: (0, -B), 4: (0, B)}
    if a == 5:
        order = sorted(range(len(m["pieces"])), key=lambda i: order_key(m["pieces"][i]))
        if m["sel"] is None:
            m["sel"] = order[0] if order else None
        else:
            k = order.index(m["sel"])
            m["sel"] = order[k + 1] if k + 1 < len(order) else None
    elif a == 6:
        p = (action.get("x"), action.get("y"))
        for i, pc in enumerate(m["pieces"]):
            if blk(*p) in pc:
                m["sel"] = i
                break
        else:
            if m["axis"] <= p[0] < m["axis"] + B:
                m["sel"] = None
    elif a in moves:
        dr, dc = moves[a]
        if m["sel"] is None:
            na = m["axis"] + dr
            if dc == 0 and 0 <= na <= N - B and not any(hits_axis(pc, na) for pc in m["pieces"]):
                m["axis"] = na
        else:
            moved = frozenset((r + dr, c + dc) for r, c in m["pieces"][m["sel"]])
            others = set().union(*[pc for i, pc in enumerate(m["pieces"]) if i != m["sel"]])
            if in_grid(moved) and not hits_axis(moved, m["axis"]) and not (moved & others):
                m["pieces"][m["sel"]] = moved
    return m


def render(m):
    """Paint sprites by layer; returns {cell: (type, sprite_id, value)}."""
    frame = {}

    def paint(kind, sid, body, holes, color, fill):
        for p in body:
            if 0 <= p[0] < N and 0 <= p[1] < N:
                frame[p] = (kind, sid, color)
        if fill >= 0:
            for p in holes:
                if 0 <= p[0] < N and 0 <= p[1] < N and p not in frame:
                    frame[p] = (kind, sid, fill)

    a = m["axis"]
    wall = {(a + i, c) for i in range(B) for c in range(N)}
    dots = {(a + 1, c) for c in range(1, N, B)}
    paint("wall", 0, wall - dots, dots, COLOR["wall"], 0 if m["sel"] is None else -1)
    for k, t in enumerate(m["targets"]):
        body, holes = block_cells(t)
        paint("target", k, body | holes, set(), COLOR["target"], -1)
    centre = 2 * a + 2
    for k, pc in enumerate(m["pieces"]):
        body, holes = block_cells(pc)
        paint("reflection", 0, {(centre - r, c) for r, c in body},
              {(centre - r, c) for r, c in holes}, COLOR["reflection"], COLOR["reflection"])
    for k, pc in enumerate(m["pieces"]):
        body, holes = block_cells(pc)
        paint("player", 0, body, holes, COLOR["player"], 0 if m["sel"] == k else -1)
    return frame


def cell_components(cells):
    cells, comps = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def make_obj(kind, cells, frame, templ):
    xs = [p[0] for p in cells]
    ys = [p[1] for p in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[frame[(x0 + i, y0 + j)][2] if (x0 + i, y0 + j) in cells else -1 for j in range(h)]
           for i in range(w)]
    t = templ.get(kind, {})
    return {"name": None, "type": kind, "x": x0, "y": y0, "w": w, "h": h,
            "layer": t.get("layer", LAYER[kind]), "tags": list(t.get("tags", TAGS[kind])), "pixels": pix}


def extract(m, before):
    frame = render(m)
    templ = {}
    for o in before:
        templ.setdefault(o["type"], o)
    out = [dict(o) for o in before if o["type"] not in LAYER]
    groups = {}
    for p, (kind, sid, v) in frame.items():
        groups.setdefault(kind, {}).setdefault(sid, set()).add(p)
    objs = {}
    for kind in LAYER:
        lst = []
        for sid, cells in groups.get(kind, {}).items():
            parts = [cells] if kind == "target" else cell_components(cells)
            lst += [make_obj(kind, part, frame, templ) for part in parts]
        lst.sort(key=lambda o: (o["x"], o["y"]))
        objs[kind] = lst
    black = sum(v == 0 for k in ("target", "reflection", "player") for o in objs[k]
                for row in o["pixels"] for v in row)
    for kind, lst in objs.items():
        off = black if kind == "wall" else 0
        for i, o in enumerate(lst):
            o["name"] = PREFIX[kind] + str(off + i)
        out += lst
    return out


def transition_function(state, action):
    key = canon(state)
    if _MEM["key"] == key and _MEM["model"] is not None:
        model, hist = _MEM["model"], _MEM["hist"]
    else:
        model, hist = parse(state), []
    a = action.get("action_id") if isinstance(action, dict) else action
    if a == 7:
        new, hist = (hist[-1], hist[:-1]) if hist else (model, hist)
    else:
        new = step(model, action)
        hist = hist + [model]
    after = extract(new, state)
    _MEM.update(key=canon(after), model=new, hist=hist[-50:])
    return after
