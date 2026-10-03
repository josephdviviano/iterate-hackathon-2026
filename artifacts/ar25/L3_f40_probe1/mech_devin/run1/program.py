# Mechanics: mirror-axis puzzle on a 3x3-cell block grid (x = row). A horizontal wall axis (row block ax),
# holed pieces (player), solid targets and gray reflections. ACTION5 cycles selection axis -> pieces -> axis;
# selected axis moves by A1/A2 (rows -3/+3, may overlap pieces); selected piece moves A1-A4, blocked by board/pieces.
# Reflection of piece block b = 2*ax-b (skip axis row, piece-occupied and off-board blocks). Render layers + re-extract:
# 4-conn comps per type ranked by bbox (x,y); wall/piece ranks also count 0-dot comps; targets per sprite, re-ranked. Unconfirmed: A5 order after a piece, A7.
import json

N = 21  # board blocks per side (cells 0..62)
_mem = {"out": None, "model": None}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _cells(o):
    p = o.get("pixels") or []
    for i, row in enumerate(p):
        for j, v in enumerate(row):
            yield o["x"] + i, o["y"] + j, v


def _comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        c = set(st)
        while st:
            a, b = st.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if n in blocks:
                    blocks.remove(n); c.add(n); st.append(n)
        out.append(c)
    return out


def parse(state):
    m = {"ax": None, "sel": None, "pieces": [], "targets": [], "tags": {}, "static": []}
    tvals = {}
    for o in state:
        m["tags"].setdefault(o["type"], o.get("tags"))
        if o["type"] == "wall":
            m["ax"] = o["x"] // 3
            if any(v == 0 for _, _, v in _cells(o)):
                m["sel"] = "wall"
        elif o["type"] == "target":
            bl = {(x // 3, y // 3) for x, y, v in _cells(o) if v >= 0}
            m["targets"].append((o["name"], bl))
            for x, y, v in _cells(o):
                if v >= 0:
                    tvals[(x, y)] = v
        elif o["type"] not in ("player", "reflection"):
            m["static"].append(o)
    selp = None
    for o in state:
        if o["type"] != "player":
            continue
        p = o["pixels"]
        zero, amb, empty = set(), set(), set()
        for i in range(0, o["w"], 3):
            for j in range(0, o["h"], 3):
                if p[i][j] != 5:
                    continue
                b = ((o["x"] + i) // 3, (o["y"] + j) // 3)
                c = p[i + 1][j + 1]
                if c == 0 and m["sel"] != "wall":
                    zero.add(b)
                elif (o["x"] + i + 1, o["y"] + j + 1) in tvals:
                    amb.add(b)
                else:
                    empty.add(b)
        if zero:
            grow = set(zero)
            for comp in _comps(amb):
                if any((a + da, b + db) in zero for a, b in comp
                       for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    grow |= comp
            selp = grow
            m["pieces"].append(grow)
            m["pieces"].extend(_comps((amb | empty) - grow))
        else:
            m["pieces"].append(amb | empty)
    m["pieces"] = order(m["pieces"])
    if selp is not None:
        m["sel"] = m["pieces"].index(selp)
    return m


def order(pieces):
    return sorted(pieces, key=lambda s: (-len(s), min(s)))


def step(m, action):
    a = action["action_id"] if isinstance(action, dict) else action
    sel = m["sel"]
    if a == 5:
        if sel == "wall":
            m["sel"] = 0 if m["pieces"] else "wall"
        elif isinstance(sel, int):
            m["sel"] = sel + 1 if sel + 1 < len(m["pieces"]) else "wall"
        return
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(a)
    if d is None:
        return
    if sel == "wall":
        if d[1] == 0 and 0 <= m["ax"] + d[0] < N:
            m["ax"] += d[0]
    elif isinstance(sel, int):
        nb = {(x + d[0], y + d[1]) for x, y in m["pieces"][sel]}
        others = set().union(*[p for k, p in enumerate(m["pieces"]) if k != sel])
        if all(0 <= x < N and 0 <= y < N for x, y in nb) and not (nb & others):
            m["pieces"][sel] = nb


def reflections(m):
    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    ax, out = m["ax"], set()
    for p in m["pieces"]:
        for x, y in p:
            if x == ax:
                continue
            r = (2 * ax - x, y)
            if 0 <= r[0] < N and r not in occ:
                out.add(r)
    return out


def render(m):
    # per cell: list of (owner_type, owner_id, kind, value) from top layer down
    stack = {}

    def put(bl, otype, oid, ring, fill, solid=False):
        for bx, by in bl:
            for i in range(3):
                for j in range(3):
                    c = (3 * bx + i, 3 * by + j)
                    hole = (i, j) == (1, 1) and not solid
                    stack.setdefault(c, []).append((otype, oid, "hole" if hole else "solid", fill if hole else ring))
    selp = m["sel"]
    for k, p in enumerate(m["pieces"]):
        put(p, "player", None, 5, 0 if selp == k else -1)
    put(reflections(m), "reflection", None, 4, 4)
    for name, bl in m["targets"]:
        put(bl, "target", name, 11, 11, solid=True)
    if m["ax"] is not None:
        put({(m["ax"], by) for by in range(N)}, "wall", None, 10, 0 if selp == "wall" else -1)
    grid = {}
    for c, lst in stack.items():
        solid = next((e for e in lst if e[2] == "solid"), None)
        top = lst[0]
        if top[2] == "solid":
            grid[c] = (top[0], top[1], top[3])
        elif solid is not None:
            grid[c] = (solid[0], solid[1], solid[3])
        else:
            f = next((e[3] for e in lst if e[3] >= 0), -1)
            if f >= 0:
                grid[c] = (top[0], top[1], f)
    return grid


def _obj(cells, grid, name, otype, tags, layer):
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for x, y in cells:
        pix[x - x0][y - y0] = grid[(x, y)][2]
    return {"name": name, "type": otype, "tags": tags, "x": x0, "y": y0, "w": w, "h": h,
            "layer": layer, "pixels": pix}


def _cell_comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        c = set(st)
        while st:
            a, b = st.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if n in cells:
                    cells.remove(n); c.add(n); st.append(n)
        out.append(c)
    return out


def extract(m, grid):
    out = [dict(o) for o in m["static"]]
    zeros = {c for c, g in grid.items() if g[2] == 0}
    tg = m["tags"]
    for otype, layer, use_zero in (("wall", 1, True), ("player", 4, True), ("reflection", 3, False)):
        own = {c for c, g in grid.items() if g[0] == otype}
        base = own | zeros if use_zero else own
        comps = sorted(_cell_comps(base), key=lambda c: (min(x for x, _ in c), min(y for _, y in c)))
        for k, comp in enumerate(comps):
            cells = comp & own
            if not cells:
                continue
            o = _obj(cells, grid, "", otype, None, layer)
            if otype == "wall":
                o["name"] = "wall_%s_%d" % ("v" if o["w"] > o["h"] else "h", k)
                o["tags"] = tg.get("wall") or ["axis", "horizontal"]
            elif otype == "player":
                o["name"] = "piece_%d" % k
                o["tags"] = tg.get("player") or ["movable", "black"]
            else:
                o["name"] = "reflection_%d" % k
                o["tags"] = tg.get("reflection") or ["mirror", "gray"]
            out.append(o)
    tobjs = []
    for name, _ in m["targets"]:
        cells = {c for c, g in grid.items() if g[0] == "target" and g[1] == name}
        if cells:
            tobjs.append(_obj(cells, grid, name, "target", tg.get("target") or ["goal", "yellow"], 2))
    for k, o in enumerate(sorted(tobjs, key=lambda o: (o["x"], o["y"]))):
        o["name"] = "target_%d" % k
        out.append(o)
    return out


def _copy(m):
    return {**m, "pieces": [set(p) for p in m["pieces"]], "targets": list(m["targets"])}


def transition_function(state, action):
    if _mem["out"] is not None and _canon(state) == _mem["out"]:
        m = _copy(_mem["model"])
    else:
        m = parse(state)
    step(m, action)
    out = extract(m, render(m))
    _mem["out"], _mem["model"] = _canon(out), _copy(m)
    return out
