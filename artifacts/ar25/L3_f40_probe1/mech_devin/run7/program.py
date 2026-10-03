# Mechanics: 21x21 grid of 3x3 blocks (x=row). Horizontal axis row A (wall), holed pieces, solid targets.
# A5 cycles selection axis -> pieces (largest first) -> axis; A1/A2 move axis/piece by one block row,
# A3/A4 move the selected piece by a column; pieces stay on the 63x63 board and never overlap.
# Each piece block b off the axis mirrors to row 2A-b; mirrors past the undrawn board bound (row 20) or on a piece are dropped.
# Render layers wall<target<reflection<piece, holes see-through; re-extract comps; wall/piece ranks count 0-dot comps. Unconfirmed: A5 order beyond 2 pieces.
N = 21
DEF_TAGS = {"wall": ["axis", "horizontal"], "player": ["movable", "black"],
            "reflection": ["mirror", "gray"], "target": ["goal", "yellow"]}
LAYER = {"wall": 1, "target": 2, "reflection": 3, "player": 4}
_mem = {"out": None, "model": None}


def canon(s):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def owned(o):
    px = {}
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v >= 0:
                px[(o["x"] + i, o["y"] + j)] = v
    return px


def parse(state):
    walls = [o for o in state if o["type"] == "wall"]
    owner = {}
    for o in state:
        for p, v in owned(o).items():
            owner[p] = (o["type"], v)
    A = walls[0]["x"] // 3 if walls else 10
    axis_sel = any(v == 0 for o in walls for v in owned(o).values())
    pieces, sel = [], "axis" if axis_sel else None
    for o in state:
        if o["type"] != "player":
            continue
        px = owned(o)
        blocks = {(a // 3, b // 3) for (a, b), v in px.items() if a % 3 == 0 and b % 3 == 0 and v == 5}
        cls = {}
        for (bx, by) in blocks:
            c = (3 * bx + 1, 3 * by + 1)
            if px.get(c) == 0 and not (axis_sel and bx == A):
                cls[(bx, by)] = "s"
            elif c in owner and owner[c][0] != "player":
                cls[(bx, by)] = "a"
            else:
                cls[(bx, by)] = "u"
        selb = {b for b in blocks if cls[b] == "s"}
        grow = True
        while grow and selb:
            grow = False
            for b in blocks:
                if cls[b] == "a" and b not in selb and any((b[0] + dx, b[1] + dy) in selb for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    selb.add(b)
                    grow = True
        rest = blocks - selb
        if selb:
            sel = len(pieces)
            pieces.append(selb)
        while rest:
            seed = min(rest)
            comp, todo = {seed}, [seed]
            while todo:
                bx, by = todo.pop()
                for nb in ((bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)):
                    if nb in rest and nb not in comp:
                        comp.add(nb)
                        todo.append(nb)
            rest -= comp
            pieces.append(comp)
    targets = []
    for o in state:
        if o["type"] == "target":
            targets.append({(a // 3, b // 3) for (a, b) in owned(o)})
    if sel is None:
        sel = "axis"
    return {"A": A, "sel": sel, "pieces": pieces, "targets": targets}


def cycle_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(m, action):
    aid = action["action_id"] if isinstance(action, dict) else action
    m = {"A": m["A"], "sel": m["sel"], "pieces": [set(p) for p in m["pieces"]], "targets": m["targets"]}
    if aid == 5:
        order = cycle_order(m["pieces"])
        if m["sel"] == "axis":
            m["sel"] = order[0] if order else "axis"
        else:
            k = order.index(m["sel"])
            m["sel"] = order[k + 1] if k + 1 < len(order) else "axis"
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if d is None:
        return m
    if m["sel"] == "axis":
        if d[1] == 0 and 0 <= m["A"] + d[0] < N:
            m["A"] += d[0]
        return m
    i = m["sel"]
    moved = {(bx + d[0], by + d[1]) for bx, by in m["pieces"][i]}
    others = set().union(*[p for j, p in enumerate(m["pieces"]) if j != i]) if len(m["pieces"]) > 1 else set()
    if all(0 <= bx < N and 0 <= by < N for bx, by in moved) and not (moved & others):
        m["pieces"][i] = moved
    return m


def block_cells(bx, by, ring, centre):
    out = []
    for i in range(3):
        for j in range(3):
            out.append(((3 * bx + i, 3 * by + j), centre if (i, j) == (1, 1) else ("solid", ring)))
    return out


def render(m):
    A, sel = m["A"], m["sel"]
    stack = {}  # pixel -> list of (layer, sid, type, kind, value)
    sprites = []

    def add(sid, typ, cells):
        for p, (kind, v) in cells:
            stack.setdefault(p, []).append((LAYER[typ], sid, typ, kind, v))

    for by in range(N):
        add(("wall", 0), "wall", block_cells(A, by, 10, ("solid", 0) if sel == "axis" else ("hole", -1)))
    for k, t in enumerate(m["targets"]):
        for bx, by in t:
            add(("target", k), "target", block_cells(bx, by, 11, ("solid", 11)))
    occupied = set().union(*m["pieces"]) if m["pieces"] else set()
    for p in m["pieces"]:
        for bx, by in p:
            dx = 2 * A - bx
            if bx != A and 0 <= dx < N and (dx, by) not in occupied:
                add(("reflection", 0), "reflection", block_cells(dx, by, 4, ("hole", 4)))
    for k, p in enumerate(m["pieces"]):
        for bx, by in p:
            add(("player", k), "player", block_cells(bx, by, 5, ("hole", 0 if sel == k else -1)))
    pix = {}  # pixel -> (sid, type, colour)
    for p, entries in stack.items():
        entries.sort(key=lambda e: -e[0])
        top = entries[0]
        solid = next((e for e in entries if e[3] == "solid"), None)
        if solid is not None:
            if solid is top or solid[4] != 0:
                pix[p] = (solid[1], solid[2], solid[4])
            else:
                pix[p] = (top[1], top[2], 0)
        else:
            fill = next((e for e in entries if e[4] >= 0), None)
            if fill is not None:
                pix[p] = (fill[1], fill[2], fill[4])
    return pix


def comps(cells):
    cells, out = set(cells), []
    while cells:
        seed = min(cells)
        comp, todo = {seed}, [seed]
        cells.discard(seed)
        while todo:
            a, b = todo.pop()
            for nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if nb in cells:
                    cells.discard(nb)
                    comp.add(nb)
                    todo.append(nb)
        out.append(comp)
    return out


def make_obj(name, typ, tags, cells, pix):
    xs = [a for a, _ in cells]
    ys = [b for _, b in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    grid = [[-1] * h for _ in range(w)]
    for (a, b) in cells:
        grid[a - x0][b - y0] = pix[(a, b)][2]
    return {"name": name, "type": typ, "tags": list(tags), "x": x0, "y": y0, "w": w, "h": h,
            "layer": LAYER[typ], "pixels": grid}


def extract(pix, tags, prefix):
    out = []
    zeros = {p for p, v in pix.items() if v[2] == 0}
    bytype = {}
    for p, v in pix.items():
        bytype.setdefault(v[1], set()).add(p)
    for typ in ("wall", "player"):
        own = bytype.get(typ, set())
        cs = sorted(comps(own | zeros), key=lambda c: (min(a for a, _ in c), min(b for _, b in c)))
        for r, c in enumerate(cs):
            mine = c & own
            if mine:
                out.append(make_obj("%s%d" % (prefix[typ], r), typ, tags[typ], mine, pix))
    own = bytype.get("reflection", set())
    cs = sorted(comps(own), key=lambda c: (min(a for a, _ in c), min(b for _, b in c)))
    for r, c in enumerate(cs):
        out.append(make_obj("reflection_%d" % r, "reflection", tags["reflection"], c, pix))
    groups = {}
    for p, v in pix.items():
        if v[1] == "target":
            groups.setdefault(v[0], set()).add(p)
    cs = sorted(groups.values(), key=lambda c: (min(a for a, _ in c), min(b for _, b in c)))
    for r, c in enumerate(cs):
        out.append(make_obj("target_%d" % r, "target", tags["target"], c, pix))
    return out


def transition_function(state, action):
    tags = dict(DEF_TAGS)
    prefix = {"wall": "wall_h_", "player": "piece_"}
    for o in state:
        if o["type"] in tags:
            tags[o["type"]] = list(o.get("tags", tags[o["type"]]))
    if _mem["out"] is not None and canon(state) == _mem["out"]:
        model = _mem["model"]
    else:
        model = parse(state)
    model = step(model, action)
    out = [dict(o) for o in state if o["type"] not in LAYER]
    out += extract(render(model), tags, prefix)
    _mem["out"], _mem["model"] = canon(out), model
    return out
