# Mechanics: horizontal mirror axis (wall row block A), 3x3-block pieces, yellow targets, gray reflections; 0 in hole centres = selection.
# ACTION5 cycles axis -> pieces (sorted by min block) -> axis; ACTION1/2 move selection up/down one block (axis: bounds only, passes
# under pieces); ACTION3/4 move a piece sideways (blocked by bounds/other pieces). Blocks above the axis reflect to row 2A-b (clipped,
# skipped on piece blocks). Layers wall<target<refl<piece, first solid else topmost fill, re-extract; hidden model gated on continuity.
# Unconfirmed: wrap from last piece back to axis, axis at board edges, piece cycle order after moves; hidden counter not needed (passes stateless).
import json

N = 21
_memo = {"key": None, "model": None}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _cells(o):
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v != -1:
                yield (o["x"] + i, o["y"] + j), v


def _comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        stack = [blocks.pop()]
        comp = set(stack)
        while stack:
            r, c = stack.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in blocks:
                    blocks.discard(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(state):
    walls = [o for o in state if o["type"] == "wall"]
    axis = walls[0]["x"] // 3 if walls else 16
    axis_sel = any(v == 0 for o in walls for _, v in _cells(o))
    owner = {}
    for k, o in enumerate(state):
        for cell, v in _cells(o):
            owner[cell] = k
    pieces, sel = [], -1
    for k, o in enumerate(state):
        if o["type"] != "player":
            continue
        vals = dict(_cells(o))
        blocks = {(r // 3, c // 3) for (r, c), v in vals.items() if v == 5}
        chosen, ambiguous = set(), set()
        if not axis_sel:
            for b in blocks:
                centre = (3 * b[0] + 1, 3 * b[1] + 1)
                if vals.get(centre) == 0:
                    chosen.add(b)
                elif centre in owner and owner[centre] != k:
                    ambiguous.add(b)
        grow = True
        while chosen and grow:
            grow = False
            for b in list(ambiguous):
                if any((b[0] + dr, b[1] + dc) in chosen for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    chosen.add(b)
                    ambiguous.discard(b)
                    grow = True
        if chosen:
            pieces.append((chosen, True))
        for comp in _comps(blocks - chosen):
            pieces.append((comp, False))
    pieces.sort(key=lambda p: min(p[0]))
    for i, (_, s) in enumerate(pieces):
        if s:
            sel = i
    targets = []
    for o in state:
        if o["type"] == "target":
            targets.append({(r // 3, c // 3) for (r, c), v in _cells(o) if v == 11})
    others = [o for o in state if o["type"] not in ("wall", "player", "reflection", "target")]
    return {"axis": axis, "sel": sel, "pieces": [p for p, _ in pieces], "targets": targets, "others": others}


def step(m, action):
    m = dict(m, pieces=[set(p) for p in m["pieces"]])
    aid = action.get("action_id") if isinstance(action, dict) else action
    if aid == 5:
        m["sel"] = m["sel"] + 1 if m["sel"] + 1 < len(m["pieces"]) else -1
        return m
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid, (0, 0))
    if (dr, dc) == (0, 0):
        return m
    if m["sel"] < 0:
        if dr and 0 <= m["axis"] + dr < N:
            m["axis"] += dr
        return m
    i = m["sel"]
    moved = {(r + dr, c + dc) for r, c in m["pieces"][i]}
    others = set().union(*[p for j, p in enumerate(m["pieces"]) if j != i]) if len(m["pieces"]) > 1 else set()
    if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
        m["pieces"][i] = moved
    return m


def _block_sprite(b, frame, fill, owner):
    r0, c0 = 3 * b[0], 3 * b[1]
    out = {}
    for i in range(3):
        for j in range(3):
            if (i, j) == (1, 1):
                out[(r0 + i, c0 + j)] = ("hole", fill, owner)
            else:
                out[(r0 + i, c0 + j)] = ("solid", frame, owner)
    return out


def render(m):
    A, sel = m["axis"], m["sel"]
    layers = {1: {}, 2: {}, 3: {}, 4: {}}
    for c in range(N):
        layers[1].update(_block_sprite((A, c), 10, 0 if sel < 0 else -1, "wall"))
    for k, t in enumerate(m["targets"]):
        for (r, c) in t:
            for i in range(3):
                for j in range(3):
                    layers[2].setdefault((3 * r + i, 3 * c + j), ("solid", 11, ("target", k)))
    occupied = set().union(*m["pieces"]) if m["pieces"] else set()
    for p in m["pieces"]:
        for (b, c) in p:
            rb = 2 * A - b
            if b < A and rb < N and (rb, c) not in occupied:
                layers[3].update(_block_sprite((rb, c), 4, 4, "refl"))
    for i, p in enumerate(m["pieces"]):
        for b in p:
            layers[4].update(_block_sprite(b, 5, 0 if i == sel else -1, "player"))
    frame = {}
    cells = set().union(*[set(l) for l in layers.values()])
    for cell in cells:
        stack = [layers[L][cell] for L in (4, 3, 2, 1) if cell in layers[L]]
        solid = [s for s in stack if s[0] == "solid"]
        if solid:
            frame[cell] = (solid[0][1], solid[0][2])
        else:
            fills = [s for s in stack if s[1] >= 0]
            if fills:
                frame[cell] = (fills[0][1], fills[0][2])
    return frame


def _obj(cells, frame, name, typ, tags, layer):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    x, y = min(rs), min(cs)
    w, h = max(rs) - x + 1, max(cs) - y + 1
    pix = [[-1] * h for _ in range(w)]
    for (r, c) in cells:
        pix[r - x][c - y] = frame[(r, c)][0]
    return {"name": name, "type": typ, "tags": list(tags), "layer": layer, "x": x, "y": y, "w": w, "h": h, "pixels": pix}


def _ranked(frame, kind, prefix, typ, tags, layer, with_zero):
    own = {c for c, (v, o) in frame.items() if o == kind and v != 0}
    pool = own | ({c for c, (v, o) in frame.items() if v == 0} if with_zero else set())
    comps = sorted(_comps(pool), key=lambda s: (min(r for r, _ in s), min(c for _, c in s)))
    return [_obj(s, frame, "%s_%d" % (prefix, i), typ, tags, layer) for i, s in enumerate(comps) if s & own]


def extract(m, frame):
    out = [dict(o) for o in m["others"]]
    out += _ranked(frame, "wall", "wall_h", "wall", ["axis", "horizontal"], 1, True)
    out += _ranked(frame, "player", "piece", "player", ["movable", "black"], 4, True)
    out += _ranked(frame, "refl", "reflection", "reflection", ["mirror", "gray"], 3, False)
    tobjs = []
    for k in range(len(m["targets"])):
        cells = {c for c, (v, o) in frame.items() if o == ("target", k)}
        if cells:
            tobjs.append(_obj(cells, frame, "", "target", ["goal", "yellow"], 2))
    tobjs.sort(key=lambda o: (o["x"], o["y"]))
    for i, o in enumerate(tobjs):
        o["name"] = "target_%d" % i
    return out + tobjs


def transition_function(state, action):
    key = _canon(state)
    model = _memo["model"] if _memo["key"] == key else parse(state)
    model = step(model, action)
    after = extract(model, render(model))
    _memo["key"], _memo["model"] = _canon(after), model
    return after
