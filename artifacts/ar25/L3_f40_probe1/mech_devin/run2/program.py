# Mechanics: single horizontal mirror axis (wall row, 3x3-block lattice 21x21) + holed 3x3 pieces; ACTION5 cycles
# selection axis -> pieces (largest first) -> axis; A1/A2 move selection x-3/x+3, A3/A4 move a piece y-3/y+3 (axis: no-op).
# Every piece block not on the axis row mirrors to row 2a-b (both directions; step-71 = pieces below/on axis reflect up),
# skipping off-board dests and dests covered by a piece. Layers wall<target<reflection<piece, holes transparent; frame is
# re-extracted (4-conn comps, 0-dots ranked into wall/player names). Unconfirmed: piece-vs-piece/edge blocking (never seen).
N = 21
WALL, TARGET, REFL, PIECE = 10, 11, 4, 5
_last = {"out": None, "model": None}


def _canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _cells(o):
    out = {}
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v >= 0:
                out[(o["x"] + i, o["y"] + j)] = v
    return out


def _blocks(o, colour=None):
    return {(r // 3, c // 3) for (r, c), v in _cells(o).items() if colour is None or v == colour}


def parse(state):
    walls = [o for o in state if o["type"] == "wall"]
    axis = walls[0]["x"] // 3 if walls else 0
    axis_sel = any(v == 0 for o in walls for v in _cells(o).values())
    targets = [_blocks(o) for o in state if o["type"] == "target"]
    shown = {}
    for o in state:
        if o["type"] in ("target", "reflection"):
            shown.update(_cells(o))
    pieces, sel = [], None
    for o in state:
        if o["type"] != "player":
            continue
        own = _cells(o)
        blocks = _blocks(o, PIECE)
        if axis_sel or not any(v == 0 for v in own.values()):
            pieces.append(blocks)
            continue
        centre = lambda b: (3 * b[0] + 1, 3 * b[1] + 1)
        chosen = {b for b in blocks if own.get(centre(b)) == 0}
        maybe = {b for b in blocks if centre(b) not in own and centre(b) in shown}
        grow = list(chosen)
        while grow:
            r, c = grow.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in maybe and n not in chosen:
                    chosen.add(n)
                    grow.append(n)
        sel = len(pieces)
        pieces.append(chosen)
        pieces.extend(components(blocks - chosen))
    scenery = [o for o in state if o["type"] not in ("wall", "target", "player", "reflection")]
    return {"axis": axis, "sel": "axis" if axis_sel or sel is None else sel,
            "pieces": pieces, "targets": targets, "scenery": scenery}


def cycle_order(pieces):
    return sorted(range(len(pieces)), key=lambda k: (-len(pieces[k]), min(pieces[k])))


def step(m, action):
    m = {"axis": m["axis"], "sel": m["sel"], "pieces": [set(p) for p in m["pieces"]],
         "targets": m["targets"], "scenery": m["scenery"]}
    if action == 5:
        order = ["axis"] + cycle_order(m["pieces"])
        m["sel"] = order[(order.index(m["sel"]) + 1) % len(order)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None:
        return m
    if m["sel"] == "axis":
        if d[1] == 0 and 0 <= m["axis"] + d[0] < N:
            m["axis"] += d[0]
        return m
    k = m["sel"]
    moved = {(r + d[0], c + d[1]) for r, c in m["pieces"][k]}
    others = set().union(*[p for i, p in enumerate(m["pieces"]) if i != k])
    if all(0 <= r < N and 0 <= c < N for r, c in moved) and not moved & others:
        m["pieces"][k] = moved
    return m


def reflections(m):
    a = m["axis"]
    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    out = set()
    for r, c in occ:
        if r == a:
            continue
        d = (2 * a - r, c)
        if 0 <= d[0] < N and d not in occ:
            out.add(d)
    return out


def sprites(m):
    """Low-to-high list of (kind, id, {cell: ('s'|'h', colour)})."""
    sp = []
    wall = {}
    for j in range(N):
        _block(wall, m["axis"], j, WALL, ("s", 0) if m["sel"] == "axis" else ("h", -1))
    sp.append(("wall", 0, wall))
    for t, blocks in enumerate(m["targets"]):
        cells = {}
        for r, c in blocks:
            _block(cells, r, c, TARGET, ("s", TARGET))
        sp.append(("target", t, cells))
    cells = {}
    for r, c in reflections(m):
        _block(cells, r, c, REFL, ("h", REFL))
    sp.append(("reflection", 0, cells))
    for k, blocks in enumerate(m["pieces"]):
        cells = {}
        for r, c in blocks:
            _block(cells, r, c, PIECE, ("h", 0 if m["sel"] == k else -1))
        sp.append(("player", k, cells))
    return sp


def _block(cells, r, c, colour, centre):
    for i in range(3):
        for j in range(3):
            cells[(3 * r + i, 3 * c + j)] = centre if (i, j) == (1, 1) else ("s", colour)


def composite(m):
    sp = sprites(m)
    frame, owner = {}, {}
    allcells = set().union(*[s[2] for s in sp])
    for cell in allcells:
        cover = [s for s in reversed(sp) if cell in s[2]]
        val = None
        for kind, sid, cells in cover:
            if cells[cell][0] == "s":
                val, own = cells[cell][1], (kind, sid)
                break
        if val is None:
            for kind, sid, cells in cover:
                if cells[cell][1] >= 0:
                    val, own = cells[cell][1], (kind, sid)
                    break
        if val is not None and val >= 0:
            frame[cell], owner[cell] = val, own
    return frame, owner


def components(cells):
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


def make_obj(name, typ, tags, layer, cells, frame):
    x0, y0 = min(r for r, _ in cells), min(c for _, c in cells)
    w, h = max(r for r, _ in cells) - x0 + 1, max(c for _, c in cells) - y0 + 1
    px = [[frame[(x0 + i, y0 + j)] if (x0 + i, y0 + j) in cells else -1 for j in range(h)] for i in range(w)]
    return {"name": name, "type": typ, "tags": tags, "x": x0, "y": y0, "w": w, "h": h, "layer": layer, "pixels": px}


def ranked(frame, colour, with_dots):
    pts = [p for p, v in frame.items() if v == colour or (with_dots and v == 0)]
    comps = sorted(components(pts), key=lambda s: (min(r for r, _ in s), min(c for _, c in s)))
    return [(i, comp) for i, comp in enumerate(comps) if any(frame[p] == colour for p in comp)]


def render(m):
    frame, owner = composite(m)
    out = [dict(o) for o in m["scenery"]]
    for i, comp in ranked(frame, WALL, True):
        out.append(make_obj("wall_h_%d" % i, "wall", ["axis", "horizontal"], 1, comp, frame))
    for i, comp in ranked(frame, PIECE, True):
        out.append(make_obj("piece_%d" % i, "player", ["movable", "black"], 4, comp, frame))
    for i, comp in ranked(frame, REFL, False):
        out.append(make_obj("reflection_%d" % i, "reflection", ["mirror", "gray"], 3, comp, frame))
    tcells = {}
    for p, (kind, sid) in owner.items():
        if kind == "target" and frame[p] == TARGET:
            tcells.setdefault(sid, set()).add(p)
    tl = sorted(tcells.values(), key=lambda s: (min(r for r, _ in s), min(c for _, c in s)))
    for i, cells in enumerate(tl):
        out.append(make_obj("target_%d" % i, "target", ["goal", "yellow"], 2, cells, frame))
    return out


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get("action_id")
    if _last["out"] is not None and _canon(state) == _last["out"]:
        model = _last["model"]
    else:
        model = parse(state)
    new = step(model, action)
    out = render(new)
    _last["out"], _last["model"] = _canon(out), new
    return out
