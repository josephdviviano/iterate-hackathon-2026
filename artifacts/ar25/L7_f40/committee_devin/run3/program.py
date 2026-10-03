# Mechanics: two mirror axes (H band rows r..r+2, V column c..c+2, on a 3-cell grid) and holed 3x3-block pieces; ACTION5 cycles H->V->piece0->piece1;
# A1/A2 move H axis or selected piece x-/+3, A3/A4 move V axis or selected piece y-/+3 (clipped to the 63x63 board). Reflections are derived from pieces:
# across V = visible gray (4); across H and across both = invisible occluders (hide targets/walls, counted in reflection index). Layers wall1<target2<refl3<piece4;
# hole centres are transparent, else fill (selected piece 0, visible refl 4); selected wall centres are solid 0. Re-extract 4-conn comps; targets merge by bbox gap<=3.
# Unconfirmed: piece-vs-piece/axis blocking, visible-vs-invisible reflection overlap order, extractor ranking by bbox (x,y); names offset by 0-dots of the other type.
N = 63
_memo = {"out": None, "order": None}


def _canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _cells(o):
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v >= 0:
                yield o["x"] + i, o["y"] + j, v


def parse(state):
    walls = [c for o in state if o["type"] == "wall" for c in _cells(o)]
    rc, cc = [0] * 21, [0] * 21
    for x, y, v in walls:
        rc[x // 3] += 1
        cc[y // 3] += 1
    r0 = 3 * max(range(21), key=lambda k: rc[k])
    c0 = 3 * max(range(21), key=lambda k: cc[k])
    in_h = lambda x, y: r0 <= x < r0 + 3
    in_v = lambda x, y: c0 <= y < c0 + 3
    pieces, sel = [], None
    for o in state:
        if o["type"] != "player":
            continue
        cs = list(_cells(o))
        blocks = frozenset((x // 3, y // 3) for x, y, v in cs)
        pieces.append(blocks)
        if any(v == 0 and not in_h(x, y) and not in_v(x, y) for x, y, v in cs):
            sel = blocks
    zeros = [(x, y) for o in state for x, y, v in _cells(o) if v == 0]
    if sel is not None:
        selection = sel
    elif any(in_h(x, y) and not in_v(x, y) for x, y in zeros):
        selection = "H"
    elif any(in_v(x, y) for x, y in zeros):
        selection = "V"
    else:
        selection = "H"
    targets = set()
    for o in state:
        if o["type"] == "target":
            for x, y, v in _cells(o):
                targets.add((x // 3, y // 3))
    pieces.sort(key=lambda b: (min(bx for bx, _ in b), min(by for _, by in b)))
    return {"r0": r0, "c0": c0, "pieces": pieces, "sel": selection, "targets": targets}


def step(m, action):
    m = dict(m)
    pieces = list(m["pieces"])
    sel = m["sel"]
    if action == 5:
        cyc = ["H", "V"] + pieces
        i = cyc.index(sel) if sel in cyc else 0
        m["sel"] = cyc[(i + 1) % len(cyc)]
        return m
    if action not in (1, 2, 3, 4):
        return m
    dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    if sel == "H":
        if dx and 0 <= m["r0"] + 3 * dx <= N - 3:
            m["r0"] += 3 * dx
    elif sel == "V":
        if dy and 0 <= m["c0"] + 3 * dy <= N - 3:
            m["c0"] += 3 * dy
    else:
        k = pieces.index(sel)
        nb = frozenset((bx + dx, by + dy) for bx, by in sel)
        others = set().union(*[p for j, p in enumerate(pieces) if j != k]) if len(pieces) > 1 else set()
        if all(0 <= bx < 21 and 0 <= by < 21 for bx, by in nb) and not (nb & others):
            pieces[k] = nb
            m["pieces"] = pieces
            m["sel"] = nb
    return m


def _block_cells(bx, by):
    for i in range(3):
        for j in range(3):
            yield 3 * bx + i, 3 * by + j, (i == 1 and j == 1)


def render(m):
    r0, c0, sel = m["r0"], m["c0"], m["sel"]
    hc, vc = r0 + 1, c0 + 1
    stacks = {}  # cell -> list of (layer, prio, kind, solid, fill)

    def put(x, y, layer, prio, kind, solid, fill):
        if 0 <= x < N and 0 <= y < N:
            stacks.setdefault((x, y), []).append((layer, prio, kind, solid, fill))

    for axis in ("H", "V"):
        selected = sel == axis
        for a in range(N):
            for b in range(3):
                x, y = (r0 + b, a) if axis == "H" else (a, c0 + b)
                centre = (b == 1 and a % 3 == 1)
                if centre and selected:
                    put(x, y, 1, 0, "wall", True, 0)
                else:
                    put(x, y, 1, 0, "wall", not centre, 10 if not centre else -1)
    for bx, by in m["targets"]:
        for x, y, c in _block_cells(bx, by):
            put(x, y, 2, 0, "target", True, 11)
    for p in m["pieces"]:
        for bx, by in p:
            for x, y, c in _block_cells(bx, by):
                put(x, 2 * vc - y, 3, 1, "refl", not c, 4)
                put(2 * hc - x, y, 3, 0, "inv", not c, -1)
                put(2 * hc - x, 2 * vc - y, 3, 0, "inv", not c, -1)
        is_sel = p == sel
        for bx, by in p:
            for x, y, c in _block_cells(bx, by):
                put(x, y, 4, 0, "piece", not c, 0 if (c and is_sel) else (5 if not c else -1))
    grid = {}
    for cell, st in stacks.items():
        st.sort(key=lambda t: (-t[0], -t[1]))
        hit = next((t for t in st if t[3]), None)
        if hit is not None:
            grid[cell] = (hit[2], hit[4])
        else:
            grid[cell] = (st[0][2], st[0][4])
    for (x, y), (k, v) in list(grid.items()):
        if v == 0:
            left = grid.get((x, y - 1))
            if left is not None and left[1] >= 0 and left[0] != "inv":
                grid[(x, y)] = (left[0], 0)
    return grid


def _components(cells):
    cells = set(cells)
    out = []
    while cells:
        s = cells.pop()
        comp, todo = [s], [s]
        while todo:
            x, y = todo.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    comp.append(n)
                    todo.append(n)
        out.append(comp)
    return out


def _bbox(comp):
    xs = [c[0] for c in comp]
    ys = [c[1] for c in comp]
    return min(xs), min(ys), max(xs), max(ys)


def _obj(name, typ, comp, grid, layer, tags):
    x0, y0, x1, y1 = _bbox(comp)
    cs = set(comp)
    px = [[grid[(x, y)][1] if (x, y) in cs else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {"name": name, "type": typ, "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
            "layer": layer, "tags": tags, "pixels": px}


def _rank(items, extra):
    keys = sorted([(k, 1) for k in extra] + [(k, 0) for k in items])
    return {k: i for i, (k, f) in enumerate(keys) if f == 0}


def extract(grid, counter):
    by = {}
    for cell, (k, v) in grid.items():
        if v >= 0 or k == "inv":
            by.setdefault(k, []).append(cell)
    out = [counter] if counter else []
    zero = {k: [c for c in by.get(k, []) if grid[c][1] == 0] for k in ("wall", "piece")}
    walls = _components(by.get("wall", []))
    wk = {id(c): _bbox(c)[:2] for c in walls}
    rk = _rank(list(wk.values()), zero["piece"])
    for c in walls:
        x0, y0, x1, y1 = _bbox(c)
        ori = "vertical" if x1 - x0 >= y1 - y0 else "horizontal"
        out.append(_obj("wall_%s_%d" % (ori[0], rk[wk[id(c)]]), "wall", c, grid, 1, ["axis", ori]))
    pcs = _components(by.get("piece", []))
    rk = _rank([_bbox(c)[:2] for c in pcs], zero["wall"])
    for c in pcs:
        out.append(_obj("piece_%d" % rk[_bbox(c)[:2]], "player", c, grid, 4, ["movable", "black"]))
    refl = _components(by.get("refl", []))
    inv = _components(by.get("inv", []))
    rk = _rank([_bbox(c)[:2] for c in refl], [_bbox(c)[:2] for c in inv])
    for c in refl:
        out.append(_obj("reflection_%d" % rk[_bbox(c)[:2]], "reflection", c, grid, 3, ["mirror", "gray"]))
    clusters = [[c, _bbox(c)] for c in _components(by.get("target", []))]
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i][1], clusters[j][1]
                gx = max(b[0] - a[2], a[0] - b[2]) - 1
                gy = max(b[1] - a[3], a[1] - b[3]) - 1
                if gx <= 3 and gy <= 3:
                    comp = clusters[i][0] + clusters[j][0]
                    clusters[i] = [comp, _bbox(comp)]
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break
    clusters.sort(key=lambda t: t[1][:2])
    for i, (c, bb) in enumerate(clusters):
        out.append(_obj("target_%d" % i, "target", c, grid, 2, ["goal", "yellow"]))
    return out


def transition_function(state, action):
    import copy
    aid = action.get("action_id") if isinstance(action, dict) else action
    m = parse(state)
    if _memo["out"] is not None and _memo["out"] == _canon(state) and _memo["order"]:
        order = _memo["order"]
        if sorted(map(sorted, order)) == sorted(map(sorted, m["pieces"])):
            m["pieces"] = list(order)
    counter = next((copy.deepcopy(o) for o in state if o["type"] == "counter"), None)
    m2 = step(m, aid)
    out = extract(render(m2), counter)
    _memo["out"] = _canon(out)
    _memo["order"] = list(m2["pieces"])
    return out
