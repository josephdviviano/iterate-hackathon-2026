# Mechanics: two wall axes on a 21x21 grid of 3x3 blocks (h = row ax, moved by A1/A2; v = column ay, A3/A4);
# A5 cycles selection h -> v -> pieces (bbox order) -> h; a selected piece moves 1 block per A1-4 (board-bounded).
# Pieces mirror across the v axis into solid gray blocks; h- and diagonal mirrors, and the whole v-mirror of a piece
# touching the h row, are invisible: they erase wall and mask targets (masked target = centre pixel only, top layer).
# Re-extract (0-dots shift wall/piece ranks). Unconfirmed: axis bounds, A1/A2 under v / A3/A4 under h = no-op, A5 >2 pieces.
N = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_last = {"out": None}


def _canon(state):
    return sorted(repr(sorted(o.items())) for o in state)


def _pixmap(state):
    g = {}
    for o in sorted(state, key=lambda o: o.get("layer", 0)):
        p = o.get("pixels")
        if not p:
            continue
        for i, row in enumerate(p):
            for j, v in enumerate(row):
                if v != -1:
                    g[(o["x"] + i, o["y"] + j)] = (v, o["type"])
    return g


def _block(g, a, b):
    return [g.get((3 * a + i, 3 * b + j)) for i in range(3) for j in range(3)]


def _comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            a, b = st.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    st.append(n)
        out.append(comp)
    return out


def parse(state):
    g = _pixmap(state)
    wall, pcs, tgt, zero = set(), set(), set(), set()
    for a in range(N):
        for b in range(N):
            bl = _block(g, a, b)
            if any(c and c[1] == "wall" and c[0] == 10 for c in bl):
                wall.add((a, b))
            if any(c and c[1] == "player" and c[0] == 5 for c in bl):
                pcs.add((a, b))
            if bl[4] and bl[4][1] == "target":
                tgt.add((a, b))
            if bl[4] and bl[4][0] == 0:
                zero.add((a, b, bl[4][1]))
    ax = max(range(N), key=lambda a: sum((a, b) in wall for b in range(N)))
    ay = max(range(N), key=lambda b: sum((a, b) in wall for a in range(N)))
    pieces = sorted(_comps(pcs), key=lambda c: (min(c), min(b for _, b in c)))
    pieces = [sorted(c) for c in pieces]
    sel = None
    for a, b, t in sorted(zero, key=lambda z: z[2] != "wall"):
        if sel is not None:
            break
        if t == "wall" and a == ax and b != ay:
            sel = "h"
        elif t == "wall" and b == ay and a != ax:
            sel = "v"
        elif (a, b) in pcs and a != ax and b != ay:
            sel = next(k for k, c in enumerate(pieces) if (a, b) in c)
    if sel is None:
        sel = "v" if any(t == "wall" for _, _, t in zero) else "h"
    return {"ax": ax, "ay": ay, "sel": sel, "pieces": pieces, "targets": tgt}


def _order(pieces):
    return sorted(range(len(pieces)), key=lambda k: (min(a for a, _ in pieces[k]), min(b for a, b in pieces[k] if a == min(x for x, _ in pieces[k]))))


def step(m, action):
    sel = m["sel"]
    if action == 5:
        order = _order(m["pieces"])
        if sel == "h":
            m["sel"] = "v"
        elif sel == "v":
            m["sel"] = order[0] if order else "h"
        else:
            i = order.index(sel)
            m["sel"] = order[i + 1] if i + 1 < len(order) else "h"
        return m
    if action not in DIRS:
        return m
    da, db = DIRS[action]
    if sel == "h":
        if da and 0 <= m["ax"] + da < N:
            m["ax"] += da
    elif sel == "v":
        if db and 0 <= m["ay"] + db < N:
            m["ay"] += db
    else:
        new = [(a + da, b + db) for a, b in m["pieces"][sel]]
        others = set(c for k, p in enumerate(m["pieces"]) if k != sel for c in p)
        if all(0 <= a < N and 0 <= b < N and (a, b) not in others for a, b in new):
            m["pieces"][sel] = new
    return m


def mirrors(m):
    ax, ay = m["ax"], m["ay"]
    occ = set(c for p in m["pieces"] for c in p)
    vis, inv = set(), set()
    on = lambda c: 0 <= c[0] < N and 0 <= c[1] < N and c not in occ
    for p in m["pieces"]:
        touch = any(a == ax for a, _ in p)
        for a, b in p:
            if b != ay and on((a, 2 * ay - b)):
                (inv if touch else vis).add((a, 2 * ay - b))
            if a != ax:
                if on((2 * ax - a, b)):
                    inv.add((2 * ax - a, b))
                if b != ay and on((2 * ax - a, 2 * ay - b)):
                    inv.add((2 * ax - a, 2 * ay - b))
    return vis, inv - vis, occ


def render(m):
    ax, ay, sel = m["ax"], m["ay"], m["sel"]
    vis, inv, occ = mirrors(m)
    g = {}

    def put(a, b, v, owner, centre=True, ring=True):
        for i in range(3):
            for j in range(3):
                if (i == 1 and j == 1 and centre) or ((i, j) != (1, 1) and ring):
                    if v[i][j] is not None:
                        g[(3 * a + i, 3 * b + j)] = (v[i][j], owner)
    ringv = lambda r, c: [[r, r, r], [r, c, r], [r, r, r]]
    for a in range(N):
        for b in range(N):
            if (a == ax or b == ay) and (a, b) not in inv:
                dot = (sel == "h" and a == ax) or (sel == "v" and b == ay)
                put(a, b, ringv(10, 0 if dot else None), ("wall", 0))
    for c in vis:
        put(c[0], c[1], ringv(4, 4), ("reflection", 0))
    for k, p in enumerate(m["pieces"]):
        for a, b in p:
            under = g.get((3 * a + 1, 3 * b + 1), (None, ("",)))
            dot = sel == k or (under[0] == 0 and under[1][0] == "wall")
            put(a, b, ringv(5, 0 if dot else None), ("player", k))
    mask = vis | inv | occ
    for a, b in m["targets"]:
        put(a, b, ringv(11, 11), ("target", 0), ring=(a, b) not in mask)
    return g, inv


def _bbox(px):
    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    return min(xs), min(ys), max(xs), max(ys)


def _obj(name, typ, px, g, layer, tags):
    x0, y0, x1, y1 = _bbox(px)
    pix = [[-1] * (y1 - y0 + 1) for _ in range(x1 - x0 + 1)]
    for (x, y) in px:
        pix[x - x0][y - y0] = g[(x, y)][0]
    return {"name": name, "type": typ, "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
            "layer": layer, "tags": tags, "pixels": pix}


def _ranked(units, keep):
    units = sorted(units, key=lambda c: _bbox(c)[:2])
    return [(i, c) for i, c in enumerate(units) if keep(c)]


def extract(g, inv):
    out = []
    zeros = set(p for p, v in g.items() if v[0] == 0)
    for typ, pre, layer, tags in (("wall", "wall", 1, None), ("player", "piece", 4, ["movable", "black"])):
        own = set(p for p, v in g.items() if v[1][0] == typ)
        for i, c in _ranked(_comps(own | zeros), lambda c: c & own):
            px = c & own
            o = _obj("", typ, px, {p: (g[p][0],) for p in px}, layer, tags)
            if typ == "wall":
                v = o["w"] >= o["h"]
                o["name"] = "wall_%s_%d" % ("v" if v else "h", i)
                o["tags"] = ["axis", "vertical" if v else "horizontal"]
            else:
                o["name"] = "piece_%d" % i
            out.append(o)
    own = set(p for p, v in g.items() if v[1][0] == "reflection")
    hid = set((3 * a + i, 3 * b + j) for a, b in inv for i in range(3) for j in range(3))
    for i, c in _ranked(_comps(own | hid), lambda c: c & own):
        out.append(_obj("reflection_%d" % i, "reflection", c & own, g, 3, ["mirror", "gray"]))
    own = set(p for p, v in g.items() if v[1][0] == "target")
    groups = [_bbox(c) + (c,) for c in _comps(own)]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                A, B = groups[i], groups[j]
                if B[0] - A[2] <= 4 and A[0] - B[2] <= 4 and B[1] - A[3] <= 4 and A[1] - B[3] <= 4:
                    c = A[4] | B[4]
                    groups[i] = _bbox(c) + (c,)
                    groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    for i, c in _ranked([gr[4] for gr in groups], lambda c: True):
        out.append(_obj("target_%d" % i, "target", c, g, 2, ["goal", "yellow"]))
    return out


def transition_function(state, action):
    aid = action["action_id"] if isinstance(action, dict) else action
    m = parse(state)
    if _last["out"] is not None and _last["out"][0] == _canon(state):
        m["sel"] = _last["out"][1]["sel"] if _last["out"][1]["sel"] in ("h", "v") else m["sel"]
    m = step(m, aid)
    g, inv = render(m)
    out = [dict(o) for o in state if o["type"] == "counter"] + extract(g, inv)
    _last["out"] = (_canon(out), m)
    return out
