# Mechanics: one horizontal mirror axis (wall row band, 3x3-cell blocks) + holed 3x3-block pieces;
# ACTION5 cycles selection axis -> pieces (largest first, tie min block) -> axis; A1/A2 move the selected
# axis or piece up/down a block, A3/A4 move a selected piece left/right (axis ignores them); every piece
# block reflects to row 2A-r (gray, clipped to the 21x21 board). Frame re-rendered by layer
# (piece>reflection>target>wall; hole = first solid below, else top sprite's fill) + re-extracted. Unconfirmed: piece collisions.
import json

N = 21
TAGS = {"wall": ["axis", "horizontal"], "player": ["movable", "black"],
        "reflection": ["mirror", "gray"], "target": ["goal", "yellow"]}
LAYER = {"wall": 1, "target": 2, "reflection": 3, "player": 4}
_mem = {"out": None, "model": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells(o):
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o["x"] + i, o["y"] + j, v


def comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in blocks:
                    blocks.discard(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    return out


# ---------- parse ----------
def parse(state):
    grid, owner = {}, {}
    for o in state:
        for r, c, v in cells(o):
            grid[(r, c)] = v
            owner[(r, c)] = o["type"]
    walls = [o for o in state if o["type"] == "wall"]
    axis = walls[0]["x"] // 3 if walls else None
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    pieces, sel = [], None
    for o in sorted((o for o in state if o["type"] == "player"), key=lambda o: o["name"]):
        blocks = {(r // 3, c // 3) for r, c, v in cells(o) if v == 5 and r % 3 == 0 and c % 3 == 0}
        centre = {b: grid.get((3 * b[0] + 1, 3 * b[1] + 1), -1) for b in blocks}
        zero = {b for b in blocks if centre[b] == 0}
        amb = {b for b in blocks if centre[b] not in (0, -1)}
        if zero:
            selb = set()
            for comp in comps(zero | amb):
                if comp & zero:
                    selb |= comp
            sel = len(pieces)
            pieces.append(selb)
            blocks = blocks - selb
        for comp in comps(blocks):
            pieces.append(comp)
    if not axis_sel and sel is None:
        for k, p in enumerate(pieces):
            if all(grid.get((3 * r + 1, 3 * c + 1), -1) not in (0, -1) for r, c in p):
                sel = k
                break
    if axis_sel:
        sel = "axis"
    targets = []
    for o in sorted((o for o in state if o["type"] == "target"), key=lambda o: o["name"]):
        targets.append({(r // 3, c // 3) for r, c, _ in cells(o)})
    others = [o for o in state if o["type"] not in LAYER]
    return {"axis": axis, "sel": sel, "pieces": pieces, "targets": targets, "others": others}


# ---------- step (per-type rules) ----------
def piece_order(pieces):
    return sorted(range(len(pieces)), key=lambda k: (-len(pieces[k]), min(pieces[k])))


def in_board(blocks):
    return all(0 <= r < N and 0 <= c < N for r, c in blocks)


def step(m, action):
    m = dict(m)
    m["pieces"] = [set(p) for p in m["pieces"]]
    aid = action["action_id"] if isinstance(action, dict) else action
    sel = m["sel"]
    if aid == 5:
        order = piece_order(m["pieces"])
        if not order:
            m["sel"] = "axis"
        elif sel == "axis" or sel is None:
            m["sel"] = order[0]
        else:
            i = order.index(sel)
            m["sel"] = order[i + 1] if i + 1 < len(order) else "axis"
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if d is None or sel is None:
        return m
    if sel == "axis":
        if d[1] == 0 and 0 <= m["axis"] + d[0] < N:
            m["axis"] += d[0]
        return m
    moved = {(r + d[0], c + d[1]) for r, c in m["pieces"][sel]}
    if in_board(moved):
        m["pieces"][sel] = moved
    return m


# ---------- render ----------
def render(m):
    A, sel = m["axis"], m["sel"]
    stacks = {}  # block -> list of (type, idx, ring, centre_solid, fill) top-down

    def put(b, entry):
        stacks.setdefault(b, []).append(entry)

    for k, p in enumerate(m["pieces"]):
        for b in p:
            put(b, ("player", k, 5, None, 0 if sel == k else -1))
    refl = set()
    for p in m["pieces"]:
        for r, c in p:
            rr = 2 * A - r
            if r != A and 0 <= rr < N:
                refl.add((rr, c))
    for b in refl:
        put(b, ("reflection", 0, 4, None, 4))
    for k, t in enumerate(m["targets"]):
        for b in t:
            put(b, ("target", k, 11, 11, None))
    if A is not None:
        for c in range(N):
            put((A, c), ("wall", 0, 10, None, 0 if sel == "axis" else -1))
    grid = {}
    for (br, bc), st in stacks.items():
        top = st[0]
        for i in range(3):
            for j in range(3):
                if (i, j) != (1, 1):
                    grid[(3 * br + i, 3 * bc + j)] = (top[2], top[0], top[1])
        val = None
        for e in st:
            if e[3] is not None:
                val = (e[3], e[0], e[1])
                break
        if val is None and top[4] >= 0:
            val = (top[4], top[0], top[1])
        if val is not None:
            grid[(3 * br + 1, 3 * bc + 1)] = val
    return grid


# ---------- extract ----------
def make_obj(typ, name, pts):
    r0 = min(r for r, _, _ in pts)
    c0 = min(c for _, c, _ in pts)
    w = max(r for r, _, _ in pts) - r0 + 1
    h = max(c for _, c, _ in pts) - c0 + 1
    px = [[-1] * h for _ in range(w)]
    for r, c, v in pts:
        px[r - r0][c - c0] = v
    return {"name": name, "type": typ, "tags": list(TAGS[typ]), "x": r0, "y": c0,
            "w": w, "h": h, "layer": LAYER[typ], "pixels": px}


def cell_comps(cellset):
    return comps(cellset)


def extract(grid, m):
    out = []
    zeros = {p for p, v in grid.items() if v[0] == 0}
    for typ, col, prefix in (("wall", 10, "wall_h_"), ("player", 5, "piece_")):
        own = {p for p, v in grid.items() if v[0] == col}
        cs = sorted(cell_comps(own | zeros), key=lambda s: (min(r for r, _ in s), min(c for _, c in s)))
        for i, s in enumerate(cs):
            if s & own:
                out.append(make_obj(typ, prefix + str(i), [(r, c, grid[(r, c)][0]) for r, c in s]))
    own = {p for p, v in grid.items() if v[0] == 4}
    cs = sorted(cell_comps(own), key=lambda s: (min(r for r, _ in s), min(c for _, c in s)))
    for i, s in enumerate(cs):
        out.append(make_obj("reflection", "reflection_" + str(i), [(r, c, 4) for r, c in s]))
    tg = {}
    for p, v in grid.items():
        if v[1] == "target":
            tg.setdefault(v[2], []).append((p[0], p[1], v[0]))
    ts = sorted(tg.values(), key=lambda s: (min(r for r, _, _ in s), min(c for _, c, _ in s)))
    for i, s in enumerate(ts):
        out.append(make_obj("target", "target_" + str(i), s))
    return out + [dict(o) for o in m["others"]]


def transition_function(state, action):
    if _mem["out"] is not None and canon(state) == _mem["out"]:
        model = _mem["model"]
    else:
        model = parse(state)
    nm = step(model, action)
    out = extract(render(nm), nm)
    _mem["out"], _mem["model"] = canon(out), nm
    return out
