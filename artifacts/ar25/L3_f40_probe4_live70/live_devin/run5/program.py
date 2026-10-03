# Mechanics: single horizontal mirror axis (wall, 3x3-block grid, x=row) + holed pieces + gray reflections + targets.
# A1-A4 move the selection (axis: rows only, bounds; piece: bounds, no overlap), A5 cycles axis -> pieces by
# (size desc, min block) -> axis, A7 undoes the last position change (selection kept, free). Reflection of piece block (r,c) is (2A-r,c).
# Budget counter: first 64 non-A7 actions free, then h=128-n; once spent, the HUD segment ranks as one extra player comp (+1).
# Render layers wall<target<reflection<piece, holes fill 0 if selected; re-extract by colour. Unconfirmed: phantom rank vs 0-only comps.
import json

N = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_mem = {"out": None, "model": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def blk_cells(b):
    return [(3 * b[0] + i, 3 * b[1] + j) for i in range(3) for j in range(3)]


def centre(b):
    return (3 * b[0] + 1, 3 * b[1] + 1)


def comps(blocks, adj_ok=None):
    blocks, out = set(blocks), []
    while blocks:
        s = blocks.pop()
        comp, stack = {s}, [s]
        while stack:
            r, c = stack.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + d[0], c + d[1])
                if n in blocks:
                    blocks.discard(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


# ---------- parse ----------
def read_grid(state):
    g = {}
    for o in state:
        for i, row in enumerate(o.get("pixels") or []):
            for j, v in enumerate(row):
                if v >= 0:
                    g[(o["x"] + i, o["y"] + j)] = (v, o["type"], o["name"])
    return g


def parse(state):
    g = read_grid(state)
    wall = [(k, v) for k, v in g.items() if v[1] == "wall"]
    A = next(k[0] for k, v in wall if v[0] == 10) // 3
    axis_sel = any(v[0] == 0 for k, v in wall)
    pblocks = {(r // 3, c // 3) for (r, c), v in g.items()
               if v[1] == "player" and v[0] == 5 and r % 3 == 0 and c % 3 == 0}
    tnames = sorted({v[2] for v in g.values() if v[1] == "target"})
    targets = [frozenset((r // 3, c // 3) for (r, c), v in g.items() if v[2] == t) for t in tnames]

    def cls(b):
        v = g.get(centre(b))
        if v is None:
            return "unsel"
        if v[1] == "player" and v[0] == 0 and not axis_sel:
            return "sel"
        if v[1] == "player":
            return "unsel"
        return "amb"

    pieces, sel = [], None
    for comp in comps(pblocks):
        s = {b for b in comp if cls(b) == "sel"}
        if s:
            grow = True
            while grow:
                grow = False
                for b in comp - s:
                    if cls(b) == "amb" and any((b[0] + d[0], b[1] + d[1]) in s
                                               for d in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        s.add(b)
                        grow = True
            sel = len(pieces)
            pieces.append(frozenset(s))
            pieces.extend(frozenset(p) for p in comps(comp - s))
        else:
            pieces.append(frozenset(comp))
    if axis_sel:
        sel = "axis"
    elif sel is None:
        for i, p in enumerate(pieces):
            if all(cls(b) == "amb" for b in p):
                sel = i
                break
    cnt = next(o for o in state if o["type"] == "counter")
    n = 128 - cnt["h"] if cnt["h"] < 64 else 0
    return {"A": A, "pieces": pieces, "sel": sel, "targets": targets, "n": n, "hist": []}


# ---------- step ----------
def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(m, action):
    aid = action["action_id"] if isinstance(action, dict) else action
    A, pieces, sel = m["A"], list(m["pieces"]), m["sel"]
    if aid in DIRS:
        dr, dc = DIRS[aid]
        if sel == "axis":
            if dr and 0 <= A + dr < N:
                m["hist"].append((A, list(pieces)))
                m["A"] = A + dr
        elif sel is not None:
            moved = frozenset((r + dr, c + dc) for r, c in pieces[sel])
            others = set().union(*[p for i, p in enumerate(pieces) if i != sel])
            if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
                m["hist"].append((A, list(pieces)))
                pieces[sel] = moved
                m["pieces"] = pieces
    elif aid == 5:
        seq = ["axis"] + order(pieces)
        m["sel"] = seq[(seq.index(sel) + 1) % len(seq)] if sel in seq else "axis"
    elif aid == 7 and m["hist"]:
        m["A"], m["pieces"] = m["hist"].pop()
    if aid != 7:
        m["n"] += 1
    return m


# ---------- render ----------
def render(m):
    A, pieces, sel = m["A"], m["pieces"], m["sel"]
    occupied = set().union(*pieces) if pieces else set()
    refl = {(2 * A - r, c) for p in pieces for r, c in p if r != A}
    refl = {b for b in refl if 0 <= b[0] < N and b not in occupied}
    tblk = {}
    for t, bs in enumerate(m["targets"]):
        for b in bs:
            tblk.setdefault(b, t)
    sprites = []  # top-down: (blocks, colour, fill, owner)
    for i, p in enumerate(pieces):
        sprites.append((p, 5, 0 if sel == i else -1, None))
    sprites.append((refl, 4, 4, None))
    for t, bs in enumerate(m["targets"]):
        sprites.append((bs, 11, "solid", t))
    sprites.append(({(A, c) for c in range(N)}, 10, 0 if sel == "axis" else -1, None))
    g, own = {}, {}
    for br in range(N):
        for bc in range(N):
            b = (br, bc)
            cov = [s for s in sprites if b in s[0]]
            if not cov:
                continue
            for cell in blk_cells(b):
                if cell != centre(b):
                    g[cell], own[cell] = cov[0][1], cov[0][3]
                    continue
                solid = [s for s in cov if s[2] == "solid"]
                if solid:
                    g[cell], own[cell] = solid[0][1], solid[0][3]
                    continue
                fills = [s[2] for s in cov if s[2] != "solid" and s[2] >= 0]
                if fills:
                    g[cell], own[cell] = fills[0], None
    return g, own


def cell_comps(cells):
    cells, out = set(cells), []
    while cells:
        s = cells.pop()
        comp, stack = {s}, [s]
        while stack:
            r, c = stack.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


def bbox(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def make_obj(name, typ, tags, layer, cells, g):
    r0, c0, r1, c1 = bbox(cells)
    pix = [[g[(r, c)] if (r, c) in cells else -1 for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
    return {"name": name, "type": typ, "tags": tags, "x": r0, "y": c0,
            "w": r1 - r0 + 1, "h": c1 - c0 + 1, "layer": layer, "pixels": pix}


def extract(m):
    g, own = render(m)
    out = []
    h = max(0, min(64, 128 - m["n"]))
    out.append({"name": "counter", "type": "counter", "tags": ["hud", "budget"],
                "x": 63, "y": 64 - h, "w": 1, "h": h, "layer": 5})
    specs = [("wall_h_", "wall", ["axis", "horizontal"], 1, 10, 0),
             ("piece_", "player", ["movable", "black"], 4, 5, 1 if h < 64 else 0),
             ("reflection_", "reflection", ["mirror", "gray"], 3, 4, 0)]
    for prefix, typ, tags, layer, col, off in specs:
        cs = cell_comps(k for k, v in g.items() if v == col or (v == 0 and col != 4))
        cs.sort(key=lambda c: bbox(c)[:2])
        for i, c in enumerate(cs):
            if any(g[k] == col for k in c):
                out.append(make_obj(prefix + str(i + off), typ, tags, layer, c, g))
    tc = [set(k for k, o in own.items() if o == t and g[k] == 11) for t in range(len(m["targets"]))]
    tc = sorted((c for c in tc if c), key=lambda c: bbox(c)[:2])
    for i, c in enumerate(tc):
        out.append(make_obj("target_" + str(i), "target", ["goal", "yellow"], 2, c, g))
    return out


def transition_function(state, action):
    if _mem["out"] is not None and canon(state) == _mem["out"]:
        m = _mem["model"]
    else:
        m = parse(state)
    m = step(m, action)
    out = extract(m)
    _mem["out"], _mem["model"] = canon(out), m
    return out
