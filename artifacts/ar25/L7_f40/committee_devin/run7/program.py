# Mechanics: two-axis mirror puzzle on a 21x21 grid of 3x3 blocks. Model = {H axis block row, V axis block col,
# pieces (8-conn block sprites), target blocks, selection}. ACTION5 cycles H -> V -> pieces (by min block) -> H;
# A1/A2 move selected H axis or piece by -/+3 rows, A3/A4 the V axis or piece by -/+3 cols (bounded, no piece overlap).
# Reflections: across V visible gray; across H and both axes invisible occluders. Render layers piece>refl>target>wall, re-extract.
# Hypotheses unconfirmed: piece-vs-axis blocking, piece cycle order after moves, ACTION6/7 (treated as no-ops).
import json

N, B = 63, 21
WALL, TGT, REFL, PLAY = 10, 11, 4, 5
INV = -9
_memo = {"key": None, "model": None}


def _canon(state):
    return json.dumps(sorted(json.dumps(o, sort_keys=True) for o in state))


def _cells(o):
    out = {}
    for i, row in enumerate(o.get("pixels", [])):
        for j, v in enumerate(row):
            if v >= 0:
                out[(o["x"] + i, o["y"] + j)] = v
    return out


def _blocks(cells):
    return {(r // 3, c // 3) for r, c in cells}


def _comps(nodes, diag=False):
    nodes, seen, out = set(nodes), set(), []
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for n in sorted(nodes):
        if n in seen:
            continue
        stack, comp = [n], set()
        seen.add(n)
        while stack:
            a = stack.pop()
            comp.add(a)
            for dr, dc in steps:
                b = (a[0] + dr, a[1] + dc)
                if b in nodes and b not in seen:
                    seen.add(b)
                    stack.append(b)
        out.append(comp)
    return out


def parse(state):
    wall, play, tgt, zeros = {}, {}, {}, set()
    for o in state:
        cs = _cells(o)
        for k, v in cs.items():
            if v == 0:
                zeros.add(k)
        if o["type"] == "wall":
            wall.update(cs)
        elif o["type"] == "player":
            play.update(cs)
        elif o["type"] == "target":
            tgt.update(cs)
    hr = max(range(B), key=lambda b: sum(1 for r, c in wall if r // 3 == b))
    vc = max(range(B), key=lambda b: sum(1 for r, c in wall if c // 3 == b))
    pieces = [frozenset(p) for p in _comps(_blocks(play), diag=True)]
    pieces.sort(key=min)
    pblocks = set().union(*pieces) if pieces else set()
    sel = None
    zb = {(r // 3, c // 3) for r, c in zeros if r % 3 == 1 and c % 3 == 1}
    if any(b[0] == hr and b[1] != vc and b not in pblocks for b in zb):
        sel = "H"
    elif any(b[1] == vc and b[0] != hr and b not in pblocks for b in zb):
        sel = "V"
    else:
        for i, p in enumerate(pieces):
            if zb & p:
                sel = i
                break
    if sel is None:
        sel = "H"
    return {"hr": hr, "vc": vc, "pieces": pieces, "targets": frozenset(_blocks(tgt)), "sel": sel}


def _inside(blocks):
    return all(0 <= r < B and 0 <= c < B for r, c in blocks)


def step(m, action):
    m = dict(m)
    m["pieces"] = list(m["pieces"])
    sel = m["sel"]
    if action == 5:
        order = ["H", "V"] + list(range(len(m["pieces"])))
        m["sel"] = order[(order.index(sel) + 1) % len(order)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None:
        return m
    if sel == "H":
        if d[0] and 0 <= m["hr"] + d[0] < B:
            m["hr"] += d[0]
    elif sel == "V":
        if d[1] and 0 <= m["vc"] + d[1] < B:
            m["vc"] += d[1]
    else:
        moved = frozenset((r + d[0], c + d[1]) for r, c in m["pieces"][sel])
        others = set().union(*[p for i, p in enumerate(m["pieces"]) if i != sel]) if len(m["pieces"]) > 1 else set()
        if _inside(moved) and not (moved & others):
            m["pieces"][sel] = moved
    return m


def _ring(b, solid, hole):
    r0, c0 = 3 * b[0], 3 * b[1]
    out = {}
    for i in range(3):
        for j in range(3):
            out[(r0 + i, c0 + j)] = hole if (i == 1 and j == 1) else solid
    return out


def render(m):
    hr, vc, sel = m["hr"], m["vc"], m["sel"]
    pall = set().union(*m["pieces"]) if m["pieces"] else set()
    layers = []
    lay = {}
    for i, p in enumerate(m["pieces"]):
        for b in p:
            lay.update(_ring(b, ("s", PLAY, "player"), ("h", 0 if sel == i else -1, "player")))
    layers.append(lay)
    vis = {(r, 2 * vc - c) for r, c in pall}
    inv = {(2 * hr - r, c) for r, c in pall} | {(2 * hr - r, 2 * vc - c) for r, c in pall}
    lay = {}
    for b in vis:
        if 0 <= b[0] < B and 0 <= b[1] < B and b not in pall:
            lay.update(_ring(b, ("s", REFL, "reflection"), ("h", REFL, "reflection")))
    layers.append(lay)
    lay = {}
    for b in inv:
        if 0 <= b[0] < B and 0 <= b[1] < B and b not in pall:
            lay.update(_ring(b, ("s", INV, "reflection"), ("h", -1, "reflection")))
    layers.append(lay)
    lay = {}
    for b in m["targets"]:
        lay.update(_ring(b, ("s", TGT, "target"), ("s", TGT, "target")))
    layers.append(lay)
    for axis, blocks in (("V", [(r, vc) for r in range(B)]), ("H", [(hr, c) for c in range(B)])):
        lay = {}
        for b in blocks:
            lay.update(_ring(b, ("s", WALL, "wall"), ("s", 0, "wall") if sel == axis else ("h", -1, "wall")))
        layers.append(lay)
    grid = {}
    for r in range(N):
        for c in range(N):
            hole = None
            for lay in layers:
                e = lay.get((r, c))
                if e is None:
                    continue
                if e[0] == "s":
                    grid[(r, c)] = (e[1], e[2])
                    break
                if hole is None:
                    hole = e
            else:
                if hole is not None and hole[1] >= 0:
                    grid[(r, c)] = (hole[1], hole[2])
    return grid


def _obj(name, typ, tags, layer, cells, grid):
    xs = [r for r, c in cells]
    ys = [c for r, c in cells]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x + 1, max(ys) - y + 1
    pix = [[-1] * h for _ in range(w)]
    for r, c in cells:
        pix[r - x][c - y] = grid[(r, c)][0]
    return {"name": name, "type": typ, "tags": tags, "layer": layer, "x": x, "y": y, "w": w, "h": h, "pixels": pix}


def _ranked(comps, keep):
    comps = sorted(comps, key=lambda s: (min(r for r, c in s), min(c for r, c in s)))
    return [(i, s) for i, s in enumerate(comps) if keep(s)]


def extract(grid):
    out = []
    zeros = {k for k, v in grid.items() if v[0] == 0}
    for typ, prefix, layer in (("wall", "wall", 1), ("player", "piece", 4)):
        own = {k for k, v in grid.items() if v[1] == typ and v[0] != 0}
        for i, s in _ranked(_comps(own | zeros), lambda s: bool(s & own)):
            o = _obj("", typ, [], layer, s, grid)
            if typ == "wall":
                hz = o["h"] > o["w"]
                o["name"] = "wall_%s_%d" % ("h" if hz else "v", i)
                o["tags"] = ["axis", "horizontal" if hz else "vertical"]
            else:
                o["name"] = "piece_%d" % i
                o["tags"] = ["movable", "black"]
            out.append(o)
    refl = {k for k, v in grid.items() if v[1] == "reflection" and v[0] != 0}
    for i, s in _ranked(_comps(refl), lambda s: any(grid[k][0] == REFL for k in s)):
        vs = {k for k in s if grid[k][0] == REFL}
        out.append(_obj("reflection_%d" % i, "reflection", ["mirror", "gray"], 3, vs, grid))
    tc = [set(s) for s in _comps({k for k, v in grid.items() if v[1] == "target"})]
    merged = True
    while merged:
        merged = False
        for a in range(len(tc)):
            for b in range(a + 1, len(tc)):
                if _gap(tc[a], tc[b]) <= 3:
                    tc[a] |= tc.pop(b)
                    merged = True
                    break
            if merged:
                break
    for i, s in _ranked(tc, lambda s: True):
        out.append(_obj("target_%d" % i, "target", ["goal", "yellow"], 2, s, grid))
    return out


def _gap(a, b):
    def bb(s):
        return min(r for r, c in s), max(r for r, c in s), min(c for r, c in s), max(c for r, c in s)
    a0, a1, a2, a3 = bb(a)
    b0, b1, b2, b3 = bb(b)
    return max(b0 - a1 - 1, a0 - b1 - 1, b2 - a3 - 1, a2 - b3 - 1)


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get("action_id")
    key = _canon(state)
    if _memo["key"] == key and _memo["model"] is not None:
        model = _memo["model"]
    else:
        model = parse(state)
    model = step(model, action)
    out = [dict(o) for o in state if o["type"] not in ("wall", "player", "reflection", "target")]
    out += extract(render(model))
    _memo["key"], _memo["model"] = _canon(out), model
    return out
