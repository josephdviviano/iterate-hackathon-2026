# Mechanics: two mirror axes (H wall row, V wall column) + 3x3-block pieces on a 21x21 block board; ACTION5 cycles
# selection H -> V -> pieces (by min block) -> H; A1/A2 move H or a piece by -/+3 rows, A3/A4 move V or a piece by -/+3 cols
# (bounds + other pieces block pieces). Each piece is mirrored across V (visible gray), across H and both (invisible,
# occluding); NEW: a piece lying on the H axis row has an invisible V mirror too. Frame = layered composite, re-extracted
# into 4-conn components. Unconfirmed: selection order beyond observed, visibility rule for pieces above H.
N = 21
SEL0, WALL, PIECE, REFL, TGT = 0, 10, 5, 4, 11
_memo = {"out": None, "model": None}


def _canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _cells(state):
    g = {}
    for o in state:
        if "pixels" not in o:
            continue
        for i, row in enumerate(o["pixels"]):
            for j, v in enumerate(row):
                if v >= 0:
                    g[(o["x"] + i, o["y"] + j)] = (v, o["type"])
    return g


def parse(state):
    g = _cells(state)
    wall_blocks = {(r // 3, c // 3) for (r, c), (v, t) in g.items() if t == "wall" and v == WALL}
    rows, cols = {}, {}
    for a, b in wall_blocks:
        rows[a] = rows.get(a, 0) + 1
        cols[b] = cols.get(b, 0) + 1
    ah = max(rows, key=lambda k: (rows[k], -k)) if rows else 0
    av = max(cols, key=lambda k: (cols[k], -k)) if cols else 0
    pblocks = {(r // 3, c // 3) for (r, c), (v, t) in g.items()
               if t == "player" and v == PIECE and (r % 3, c % 3) != (1, 1)}
    targets = {(r // 3, c // 3) for (r, c), (v, t) in g.items() if v == TGT}
    pieces, seen = [], set()
    for b in sorted(pblocks):
        if b in seen:
            continue
        comp, stack = set(), [b]
        seen.add(b)
        while stack:
            a, c = stack.pop()
            comp.add((a, c))
            for n in ((a + 1, c), (a - 1, c), (a, c + 1), (a, c - 1)):
                if n in pblocks and n not in seen:
                    seen.add(n)
                    stack.append(n)
        pieces.append(comp)
    zero = {(r // 3, c // 3) for (r, c), (v, t) in g.items() if v == SEL0 and (r % 3, c % 3) == (1, 1)}
    sel = None
    free = zero - pblocks
    if any(a == ah and b != av for a, b in free):
        sel = "H"
    elif any(b == av and a != ah for a, b in free):
        sel = "V"
    else:
        for i, p in enumerate(pieces):
            if p & zero:
                sel = i
                break
        if sel is None and free:
            sel = "H"
    return {"ah": ah, "av": av, "pieces": [frozenset(p) for p in pieces], "sel": sel, "targets": targets}


def _order(pieces):
    return sorted(range(len(pieces)), key=lambda i: min(pieces[i]))


def step(m, action):
    m = dict(m)
    pieces = list(m["pieces"])
    sel = m["sel"]
    aid = action["action_id"] if isinstance(action, dict) else action
    if aid == 5:
        cyc = ["H", "V"] + _order(pieces)
        m["sel"] = cyc[(cyc.index(sel) + 1) % len(cyc)] if sel in cyc else "H"
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if d is None:
        return m
    if sel == "H":
        if d[0] and 0 <= m["ah"] + d[0] < N:
            m["ah"] += d[0]
    elif sel == "V":
        if d[1] and 0 <= m["av"] + d[1] < N:
            m["av"] += d[1]
    elif isinstance(sel, int):
        moved = frozenset((a + d[0], b + d[1]) for a, b in pieces[sel])
        others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
        if all(0 <= a < N and 0 <= b < N for a, b in moved) and not (moved & others):
            pieces[sel] = moved
    m["pieces"] = pieces
    return m


def render(m):
    """Per block: list of sprites top-down as (kind, ring_colour, centre_solid, centre_fill)."""
    ah, av, pieces, sel = m["ah"], m["av"], m["pieces"], m["sel"]
    stacks = {}

    def add(blk, spr):
        if 0 <= blk[0] < N and 0 <= blk[1] < N:
            stacks.setdefault(blk, []).append(spr)
    occupied = set().union(*pieces) if pieces else set()
    for i, p in enumerate(pieces):
        for b in p:
            add(b, ("piece", PIECE, None, SEL0 if sel == i else None))
    vis, inv = [], []
    for p in pieces:
        v_visible = all(a != ah for a, _ in p)
        for a, b in p:
            (vis if v_visible else inv).append((a, 2 * av - b))
            inv.append((2 * ah - a, b))
            inv.append((2 * ah - a, 2 * av - b))
    for b in vis:
        if b not in occupied:
            add(b, ("vref", REFL, None, REFL))
    for b in inv:
        if b not in occupied:
            add(b, ("iref", -1, None, -1))
    for b in m["targets"]:
        add(b, ("target", TGT, TGT, None))
    for b in range(N):
        add((b, av), ("wall", WALL, SEL0 if sel == "V" else None, None))
    for b in range(N):
        add((ah, b), ("wall", WALL, SEL0 if sel == "H" else None, None))
    frame = {}
    for (a, b), st in stacks.items():
        for i in range(3):
            for j in range(3):
                cell = (3 * a + i, 3 * b + j)
                if (i, j) != (1, 1):
                    k, col = st[0][0], st[0][1]
                else:
                    solid = [s for s in st if s[2] is not None]
                    if solid:
                        k, col = solid[0][0], solid[0][2]
                    else:
                        k, col = st[0][0], st[0][3] if st[0][3] is not None else -1
                frame[cell] = (col, k)
    return frame


def _comps(cells):
    cells, out = set(cells), []
    while cells:
        s = cells.pop()
        comp, stack = {s}, [s]
        while stack:
            r, c = stack.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


def _bbox(cells):
    xs = [r for r, _ in cells]
    ys = [c for _, c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def _obj(name, typ, tags, layer, cells, colours):
    x0, y0, x1, y1 = _bbox(cells)
    px = [[colours[(x, y)] if (x, y) in cells else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {"name": name, "type": typ, "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
            "layer": layer, "tags": tags, "pixels": px}


def extract(frame):
    col = {p: v[0] for p, v in frame.items()}
    zeros = {p for p, v in frame.items() if v[0] == SEL0}
    out = []
    for kind, typ, layer in (("wall", "wall", 1), ("piece", "player", 4)):
        own = {p for p, v in frame.items() if v[1] == kind and v[0] != SEL0 and v[0] >= 0}
        comps = sorted(_comps(own | zeros), key=lambda c: _bbox(c)[:2])
        for i, c in enumerate(comps):
            if not (c & own):
                continue
            x0, y0, x1, y1 = _bbox(c)
            if typ == "wall":
                v = (x1 - x0) >= (y1 - y0)
                o = _obj("wall_%s_%d" % ("v" if v else "h", i), "wall",
                         ["axis", "vertical" if v else "horizontal"], 1, c, col)
            else:
                o = _obj("piece_%d" % i, "player", ["movable", "black"], 4, c, col)
            out.append(o)
    refl = {p for p, v in frame.items() if v[1] in ("vref", "iref")}
    comps = sorted(_comps(refl), key=lambda c: _bbox(c)[:2])
    for i, c in enumerate(comps):
        visc = {p for p in c if col[p] == REFL}
        if visc:
            out.append(_obj("reflection_%d" % i, "reflection", ["mirror", "gray"], 3, visc, col))
    groups = [set(c) for c in _comps({p for p, v in frame.items() if v[0] == TGT})]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = _bbox(groups[i]), _bbox(groups[j])
                dx = max(a[0] - b[2], b[0] - a[2])
                dy = max(a[1] - b[3], b[1] - a[3])
                if max(dx, dy) <= 4:
                    groups[i] |= groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    for i, c in enumerate(sorted(groups, key=lambda c: _bbox(c)[:2])):
        out.append(_obj("target_%d" % i, "target", ["goal", "yellow"], 2, c, col))
    return out


def transition_function(state, action):
    if _memo["out"] is not None and _canon(state) == _memo["out"]:
        model = _memo["model"]
    else:
        model = parse(state)
    new = step(model, action)
    out = extract(render(new)) + [dict(o) for o in state if "pixels" not in o]
    _memo["out"], _memo["model"] = _canon(out), new
    return out
