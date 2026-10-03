# Mechanics: two mirror axes (H = wall block row ah, V = wall block col av) on a 21x21 grid of 3x3 blocks; pieces (players).
# ACTION5 cycles selection H -> V -> pieces (by min block) -> H; A1/A2 move H or the selected piece by one block row,
# A3/A4 move V or the selected piece by one block column. Pieces mirror across V as visible gray reflections; mirrors
# across H / both are invisible occluders. Frame is re-rendered by layers and re-extracted (4-conn comps; wall/player
# ranks include all 0 pixels; targets merge if bbox gap<=3). Unconfirmed: piece-piece blocking, the A5 piece order.
N = 21
LAYER = {"wall": 1, "target": 2, "reflection": 3, "player": 4}
_memo = {"out": None, "model": None}


def _cells(o):
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o["x"] + i, o["y"] + j, v


def parse(state):
    val, typ = {}, {}
    pieces, tblocks, counter = [], set(), []
    for o in state:
        if o["type"] == "counter" or "pixels" not in o:
            counter.append(o)
            continue
        cs = list(_cells(o))
        for r, c, v in cs:
            val[(r, c)], typ[(r, c)] = v, o["type"]
        if o["type"] == "player":
            pieces.append({(r // 3, c // 3) for r, c, v in cs})
        elif o["type"] == "target":
            tblocks |= {(r // 3, c // 3) for r, c, v in cs}
    wb = {(r // 3, c // 3) for (r, c), t in typ.items() if t == "wall"}
    rows = [sum(1 for b in wb if b[0] == R) for R in range(N)]
    cols = [sum(1 for b in wb if b[1] == C) for C in range(N)]
    ah, av = rows.index(max(rows)), cols.index(max(cols))
    pieces.sort(key=min)
    occ = set().union(*pieces) if pieces else set()

    def zero(R, C):
        return (R, C) not in occ and val.get((3 * R + 1, 3 * C + 1)) == 0
    sel = None
    if any(zero(ah, C) for C in range(N) if C != av):
        sel = "H"
    elif any(zero(R, av) for R in range(N) if R != ah):
        sel = "V"
    else:
        for k, p in enumerate(pieces):
            if any(val.get((3 * R + 1, 3 * C + 1)) == 0 and typ.get((3 * R + 1, 3 * C + 1)) == "player"
                   for R, C in p):
                sel = k
        if sel is None:
            sel = "H"
    return {"ah": ah, "av": av, "pieces": pieces, "sel": sel, "targets": tblocks, "counter": counter}


def step(m, action):
    m = dict(m, pieces=[set(p) for p in m["pieces"]])
    a = action["action_id"] if isinstance(action, dict) else action
    sel, np_ = m["sel"], len(m["pieces"])
    if a == 5:
        order = ["H", "V"] + list(range(np_))
        m["sel"] = order[(order.index(sel) + 1) % len(order)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(a)
    if d is None:
        return m
    if sel == "H":
        if d[0] and 0 <= m["ah"] + d[0] < N:
            m["ah"] += d[0]
    elif sel == "V":
        if d[1] and 0 <= m["av"] + d[1] < N:
            m["av"] += d[1]
    else:
        moved = {(R + d[0], C + d[1]) for R, C in m["pieces"][sel]}
        others = set().union(set(), *[p for k, p in enumerate(m["pieces"]) if k != sel])
        if all(0 <= R < N and 0 <= C < N for R, C in moved) and not (moved & others):
            m["pieces"][sel] = moved
    return m


def render(m):
    ah, av, sel = m["ah"], m["av"], m["sel"]
    stacks = {}  # block -> list of sprites top-down: (kind, frame value, hole_solid, hole value)

    def add(b, spr):
        if 0 <= b[0] < N and 0 <= b[1] < N:
            stacks.setdefault(b, []).append(spr)
    occ = set().union(set(), *m["pieces"])
    for k, p in enumerate(m["pieces"]):
        for b in p:
            add(b, ("player", 5, False, 0 if sel == k else -1))
    vis, inv = set(), set()
    for p in m["pieces"]:
        for R, C in p:
            vis.add((R, 2 * av - C))
            inv.add((2 * ah - R, C))
            inv.add((2 * ah - R, 2 * av - C))
    vis -= occ
    inv -= occ | vis
    for b in vis:
        add(b, ("reflection", 4, False, 4))
    for b in inv:
        add(b, ("refl_inv", -1, False, -1))
    for b in m["targets"]:
        add(b, ("target", 11, True, 11))
    for R in range(N):
        on = sel == "V" or (R == ah and sel == "H")
        add((R, av), ("wall", 10, on, 0 if on else -1))
    for C in range(N):
        if C != av:
            add((ah, C), ("wall", 10, sel == "H", 0 if sel == "H" else -1))
    grid = {}
    for (R, C), st in stacks.items():
        for i in range(3):
            for j in range(3):
                cell = (3 * R + i, 3 * C + j)
                centre = i == 1 and j == 1
                hit = None
                for kind, fv, hs, hv in st:
                    if not centre:
                        hit = (kind, fv)
                        break
                    if hs:
                        hit = (kind, hv)
                        break
                if hit is None and centre:
                    hit = (st[0][0], st[0][3])
                grid[cell] = hit
    return grid


def _comps(cells):
    cells, out = set(cells), []
    while cells:
        s = cells.pop()
        comp, todo = [s], [s]
        while todo:
            r, c = todo.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells:
                    cells.remove(n)
                    comp.append(n)
                    todo.append(n)
        out.append(comp)
    return out


def _bbox(cs):
    rs, cs2 = [r for r, c in cs], [c for r, c in cs]
    return min(rs), min(cs2), max(rs), max(cs2)


def _obj(name, typ, tags, cells, grid):
    r0, c0, r1, c1 = _bbox(cells)
    pix = [[-1] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
    for r, c in cells:
        pix[r - r0][c - c0] = grid[(r, c)][1]
    return {"h": c1 - c0 + 1, "layer": LAYER[typ], "name": name, "pixels": pix, "tags": tags,
            "type": typ, "w": r1 - r0 + 1, "x": r0, "y": c0}


def extract(grid, counter):
    out = [dict(o) for o in counter]
    zeros = {p for p, (k, v) in grid.items() if v == 0}
    for typ in ("wall", "player"):
        own = {p for p, (k, v) in grid.items() if k == typ and v not in (-1, 0)}
        comps = sorted(_comps(own | zeros), key=_bbox)
        for i, comp in enumerate(comps):
            if not any(p in own for p in comp):
                continue
            if typ == "player":
                out.append(_obj("piece_%d" % i, typ, ["movable", "black"], comp, grid))
                continue
            r0, c0, r1, c1 = _bbox(comp)
            v = r1 - r0 >= c1 - c0
            out.append(_obj("wall_%s_%d" % ("v" if v else "h", i), typ,
                            ["axis", "vertical" if v else "horizontal"], comp, grid))
    refl = {p for p, (k, v) in grid.items() if k in ("reflection", "refl_inv")}
    comps = sorted(_comps(refl), key=_bbox)
    for i, comp in enumerate(comps):
        vis = [p for p in comp if grid[p][0] == "reflection" and grid[p][1] != -1]
        if vis:
            out.append(_obj("reflection_%d" % i, "reflection", ["mirror", "gray"], vis, grid))
    groups = [(_bbox(c), c) for c in _comps(p for p, (k, v) in grid.items() if k == "target")]
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                (ar0, ac0, ar1, ac1), (br0, bc0, br1, bc1) = groups[a][0], groups[b][0]
                if max(br0 - ar1, ar0 - br1) <= 4 and max(bc0 - ac1, ac0 - bc1) <= 4:
                    cs = groups[a][1] + groups[b][1]
                    groups[a] = (_bbox(cs), cs)
                    del groups[b]
                    merged = True
                    break
            if merged:
                break
    for i, (bb, cs) in enumerate(sorted(groups, key=lambda g: g[0])):
        out.append(_obj("target_%d" % i, "target", ["goal", "yellow"], cs, grid))
    return out


def _canon(s):
    return sorted(repr(sorted(o.items())) for o in s)


def transition_function(state, action):
    m = parse(state)
    if _memo["out"] is not None and _canon(state) == _memo["out"]:
        m = _memo["model"]
    m2 = step(m, action)
    out = extract(render(m2), m2["counter"])
    _memo["out"], _memo["model"] = _canon(out), m2
    return out
