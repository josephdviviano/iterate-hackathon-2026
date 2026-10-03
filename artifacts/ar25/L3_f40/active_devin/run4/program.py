# Mechanics: mirror puzzle on a 21x21 grid of 3x3 cells (63x63 px, x=row). A horizontal axis (wall, 3 rows,
# dotted centre row) mirrors every piece block (bx,by) to (2a-bx,by) as gray reflections. ACTION5 cycles the
# selection axis -> pieces (by bbox x,y) -> axis; ACTION1-4 move the selection by one cell (axis: rows only).
# Moves are blocked off-board or when a piece would overlap the axis row/another piece (hypothesis: unconfirmed
# for the piece-into-axis case). Holes are see-through over lower layers, else filled (selected 0, reflection 4).
import json

N = 21
META = {"wall": (["axis", "horizontal"], 1), "target": (["goal", "yellow"], 2),
        "reflection": (["mirror", "gray"], 3), "player": (["movable", "black"], 4)}
_last = {"canon": None, "model": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def blocks_of(o):
    out = set()
    for i, row in enumerate(o["pixels"]):
        for j, v in enumerate(row):
            if v != -1:
                out.add(((o["x"] + i) // 3, (o["y"] + j) // 3))
    return out


def pix(o, r, c):
    i, j = r - o["x"], c - o["y"]
    if 0 <= i < o["w"] and 0 <= j < o["h"]:
        return o["pixels"][i][j]
    return -1


def comps(bs):
    bs, out = set(bs), []
    while bs:
        st = [bs.pop()]; cur = set(st)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in bs:
                    bs.discard(n); cur.add(n); st.append(n)
        out.append(cur)
    return out


def parse(state):
    walls = [o for o in state if o["type"] == "wall"]
    a = min(o["x"] for o in walls) // 3 if walls else N // 2
    axis_sel = any(v == 0 for o in walls for r in o["pixels"] for v in r)
    targets = [blocks_of(o) for o in state if o["type"] == "target"]
    tcells = set().union(*targets) if targets else set()
    pieces, sel = [], None
    for o in state:
        if o["type"] != "player":
            continue
        bs = blocks_of(o)
        zero = {b for b in bs if pix(o, b[0] * 3 + 1, b[1] * 3 + 1) == 0}
        if zero and zero != bs:
            s = set(zero); grow = True
            while grow:
                grow = False
                for b in bs - s:
                    if b in tcells and any((b[0] + dx, b[1] + dy) in s for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        s.add(b); grow = True
            pieces.append(s); sel = s
            pieces.extend(comps(bs - s))
        else:
            pieces.append(bs)
            if zero:
                sel = bs
    pieces = [frozenset(p) for p in pieces]
    if axis_sel:
        si = -1
    elif sel is not None:
        si = pieces.index(frozenset(sel))
    else:
        si = -1
    return {"a": a, "targets": targets, "pieces": pieces, "sel": si}


def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (min(b[0] for b in pieces[i]), min(b[1] for b in pieces[i])))


def step(m, action):
    m = dict(m); m["pieces"] = list(m["pieces"])
    if action == 5:
        od = order(m["pieces"])
        if m["sel"] == -1:
            m["sel"] = od[0] if od else -1
        else:
            k = od.index(m["sel"])
            m["sel"] = od[k + 1] if k + 1 < len(od) else -1
        return m
    if action not in (1, 2, 3, 4):
        return m
    dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    if m["sel"] == -1:
        na = m["a"] + dx
        if dy or not 0 <= na < N or any(b[0] == na for p in m["pieces"] for b in p):
            return m
        m["a"] = na
        return m
    p = frozenset((x + dx, y + dy) for x, y in m["pieces"][m["sel"]])
    others = set().union(*[q for i, q in enumerate(m["pieces"]) if i != m["sel"]]) if len(m["pieces"]) > 1 else set()
    if any(not (0 <= x < N and 0 <= y < N) or x == m["a"] or (x, y) in others for x, y in p):
        return m
    m["pieces"][m["sel"]] = p
    return m


def render(m):
    g = {}  # (r,c) -> (kind, owner, colour)
    a = m["a"]
    def put(r, c, kind, owner, col, hole_fill):
        if hole_fill is not None:
            if (r, c) in g:
                return
            if hole_fill == -1:
                return
            col = hole_fill
        g[(r, c)] = (kind, owner, col)
    for c in range(N * 3):
        for i in range(3):
            hole = i == 1 and c % 3 == 1
            put(a * 3 + i, c, "wall", 0, 10, (0 if m["sel"] == -1 else -1) if hole else None)
    for t, bs in enumerate(m["targets"]):
        for bx, by in bs:
            for i in range(3):
                for j in range(3):
                    put(bx * 3 + i, by * 3 + j, "target", t, 11, None)
    refl = set()
    for p in m["pieces"]:
        for bx, by in p:
            rx = 2 * a - bx
            if 0 <= rx < N:
                refl.add((rx, by))
    for bx, by in refl:
        for i in range(3):
            for j in range(3):
                put(bx * 3 + i, by * 3 + j, "reflection", 0, 4, 4 if (i, j) == (1, 1) else None)
    for k, p in enumerate(m["pieces"]):
        for bx, by in p:
            for i in range(3):
                for j in range(3):
                    hf = (0 if k == m["sel"] else -1) if (i, j) == (1, 1) else None
                    put(bx * 3 + i, by * 3 + j, "player", 0, 5, hf)
    return g


def make(kind, cells, g):
    xs = [r for r, c in cells]; ys = [c for r, c in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    px = [[-1] * h for _ in range(w)]
    for r, c in cells:
        px[r - x0][c - y0] = g[(r, c)][2]
    tags, layer = META[kind]
    return {"type": kind, "tags": list(tags), "layer": layer, "x": x0, "y": y0, "w": w, "h": h, "pixels": px}


def extract(g, counter):
    out = list(counter)
    zeros = sum(1 for v in g.values() if v[0] != "wall" and v[2] == 0)
    for kind, prefix in (("wall", "wall_h_"), ("player", "piece_"), ("reflection", "reflection_"), ("target", "target_")):
        cells = [k for k, v in g.items() if v[0] == kind]
        if kind == "target":
            groups = {}
            for k in cells:
                groups.setdefault(g[k][1], set()).add(k)
            groups = list(groups.values())
        else:
            left, groups = set(cells), []
            while left:
                st = [left.pop()]; cur = set(st)
                while st:
                    r, c = st.pop()
                    for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if n in left:
                            left.discard(n); cur.add(n); st.append(n)
                groups.append(cur)
        objs = sorted((make(kind, s, g) for s in groups), key=lambda o: (o["x"], o["y"]))
        off = zeros if kind == "wall" else 0
        for i, o in enumerate(objs):
            o["name"] = prefix + str(i + off)
            out.append(o)
    return out


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get("action_id")
    c = canon(state)
    if _last["canon"] is not None and c == _last["canon"]:
        m = _last["model"]
    else:
        m = parse(state)
    m2 = step(m, action)
    out = extract(render(m2), [o for o in state if o["type"] == "counter"])
    _last["canon"], _last["model"] = canon(out), m2
    return out
