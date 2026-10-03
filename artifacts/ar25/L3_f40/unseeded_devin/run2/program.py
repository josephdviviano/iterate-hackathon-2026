# Mechanics: 21x21 grid of 3x3 cells (obj x=row, y=col). ACTION5 cycles selection wall->pieces->wall;
# ACTION1-4 move the selected piece one cell (up/down/left/right) or the mirror wall along its normal.
# Each piece cell is reflected across the mirror axis into a solid gray reflection (clipped to the board).
# Render layers wall<target<reflection<piece; cell centres are holes (selected: black 0) that let targets show.
# Extraction: walls/pieces/reflections are 4-connected components, targets per sprite, names sorted by (x,y).
# Unconfirmed: blocking rules (bounds, piece overlap, crossing the mirror), selection order, win/reset.
N = 21
P = 3 * N


def is_center(r, c):
    return r % 3 == 1 and c % 3 == 1


def cells_of(o, pred):
    out = set()
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if pred(v, o["x"] + i, o["y"] + j):
                out.add(((o["x"] + i) // 3, (o["y"] + j) // 3))
    return out


def parse(state):
    walls = [o for o in state if o["type"] == "wall"]
    vertical = any("vertical" in o.get("tags", []) for o in walls)
    axis = (min(o["y"] for o in walls) if vertical else min(o["x"] for o in walls)) // 3 if walls else 10
    wall_sel = any(v == 0 for o in walls for row in o.get("pixels") or [] for v in row)
    targets = [cells_of(o, lambda v, r, c: v >= 0) for o in sorted(state, key=lambda o: (o["x"], o["y"])) if o["type"] == "target"]
    tcells = set().union(*targets) if targets else set()
    pieces, sel = [], None
    for o in sorted((o for o in state if o["type"] == "player"), key=lambda o: (o["x"], o["y"])):
        allc = cells_of(o, lambda v, r, c: v == 5 and not is_center(r, c))
        S = cells_of(o, lambda v, r, c: v == 0 and is_center(r, c)) & allc
        U = {cl for cl in allc - S if cl not in tcells}
        groups = [g for g in (S, U) if g]
        if not groups:
            groups = [set(allc)]
        rest = allc - set().union(*groups)
        while rest:
            moved = False
            for g in groups:
                for cl in list(rest):
                    if any((cl[0] + a, cl[1] + b) in g for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        g.add(cl); rest.discard(cl); moved = True
            if not moved:
                groups[0] |= rest; rest = set()
        for g in groups:
            if g is S and S:
                sel = len(pieces)
            pieces.append(g)
    if not wall_sel and sel is None and pieces:
        cand = [k for k, g in enumerate(pieces) if g <= tcells]
        sel = cand[0] if cand else None
    return {"vertical": vertical, "axis": axis, "sel": "wall" if wall_sel or sel is None else sel,
            "targets": targets, "pieces": pieces,
            "other": [o for o in state if o["type"] not in ("wall", "target", "player", "reflection")]}


def mirror_cell(m, cl):
    r, c = cl
    return (r, 2 * m["axis"] - c) if m["vertical"] else (2 * m["axis"] - r, c)


def on_axis(m, cl):
    return (cl[1] if m["vertical"] else cl[0]) == m["axis"]


def inside(cl):
    return 0 <= cl[0] < N and 0 <= cl[1] < N


DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def step(m, action):
    if action == 5:
        order = ["wall"] + list(range(len(m["pieces"])))
        m["sel"] = order[(order.index(m["sel"]) + 1) % len(order)]
        return
    if action not in DIRS:
        return
    dr, dc = DIRS[action]
    if m["sel"] == "wall":
        d = dc if m["vertical"] else dr
        if d == 0:
            return
        na = m["axis"] + d
        if not 0 <= na < N:
            return
        trial = dict(m, axis=na)
        if any(on_axis(trial, cl) for g in m["pieces"] for cl in g):
            return
        m["axis"] = na
        return
    k = m["sel"]
    moved = {(r + dr, c + dc) for r, c in m["pieces"][k]}
    others = set().union(set(), *(g for j, g in enumerate(m["pieces"]) if j != k))
    if all(inside(cl) and not on_axis(m, cl) for cl in moved) and not (moved & others):
        m["pieces"][k] = moved


def render(m):
    own = [[None] * P for _ in range(P)]
    val = [[-1] * P for _ in range(P)]
    for i in range(N):
        cl = (i, m["axis"]) if m["vertical"] else (m["axis"], i)
        for a in range(3):
            for b in range(3):
                r, c = 3 * cl[0] + a, 3 * cl[1] + b
                if is_center(r, c):
                    if m["sel"] == "wall":
                        own[r][c], val[r][c] = ("wall", 0), 0
                else:
                    own[r][c], val[r][c] = ("wall", 0), 10
    for t, cells in enumerate(m["targets"]):
        for R, C in cells:
            for a in range(3):
                for b in range(3):
                    own[3 * R + a][3 * C + b], val[3 * R + a][3 * C + b] = ("target", t), 11
    refl = {mirror_cell(m, cl) for g in m["pieces"] for cl in g}
    for R, C in (cl for cl in refl if inside(cl)):
        for a in range(3):
            for b in range(3):
                r, c = 3 * R + a, 3 * C + b
                if is_center(r, c) and own[r][c] and own[r][c][0] == "target":
                    continue
                own[r][c], val[r][c] = ("reflection", 0), 4
    for k, g in enumerate(m["pieces"]):
        for R, C in g:
            for a in range(3):
                for b in range(3):
                    r, c = 3 * R + a, 3 * C + b
                    if not is_center(r, c):
                        own[r][c], val[r][c] = ("player", 0), 5
                    elif own[r][c] is None and m["sel"] == k:
                        own[r][c], val[r][c] = ("player", 0), 0
    return own, val


def make_obj(pix, val, name, typ, tags, layer):
    rs = [p[0] for p in pix]; cs = [p[1] for p in pix]
    x, y = min(rs), min(cs)
    w, h = max(rs) - x + 1, max(cs) - y + 1
    grid = [[-1] * h for _ in range(w)]
    for r, c in pix:
        grid[r - x][c - y] = val[r][c]
    return {"name": name, "type": typ, "tags": list(tags), "x": x, "y": y, "w": w, "h": h, "layer": layer, "pixels": grid}


def components(own, key):
    seen, comps = set(), []
    for r in range(P):
        for c in range(P):
            if own[r][c] == key and (r, c) not in seen:
                comp, stack = [], [(r, c)]
                seen.add((r, c))
                while stack:
                    a, b = stack.pop(); comp.append((a, b))
                    for p in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= p[0] < P and 0 <= p[1] < P and p not in seen and own[p[0]][p[1]] == key:
                            seen.add(p); stack.append(p)
                comps.append(comp)
    return comps


def extract(m):
    own, val = render(m)
    out = [dict(o) for o in m["other"]]
    zeros = sum(1 for r in range(P) for c in range(P) if val[r][c] == 0 and own[r][c] != ("wall", 0))
    specs = [("wall", "wall_v_" if m["vertical"] else "wall_h_", ["axis", "vertical" if m["vertical"] else "horizontal"], 1, zeros),
             ("player", "piece_", ["movable", "black"], 4, 0),
             ("reflection", "reflection_", ["mirror", "gray"], 3, 0)]
    for typ, prefix, tags, layer, off in specs:
        objs = [make_obj(cp, val, "", typ, tags, layer) for cp in components(own, (typ, 0))]
        for i, o in enumerate(sorted(objs, key=lambda o: (o["x"], o["y"]))):
            o["name"] = prefix + str(off + i); out.append(o)
    tobjs = []
    for t in range(len(m["targets"])):
        pix = [(r, c) for r in range(P) for c in range(P) if own[r][c] == ("target", t)]
        if pix:
            tobjs.append(make_obj(pix, val, "", "target", ["goal", "yellow"], 2))
    for i, o in enumerate(sorted(tobjs, key=lambda o: (o["x"], o["y"]))):
        o["name"] = "target_" + str(i); out.append(o)
    return out


def canon(s):
    return sorted(repr(sorted(o.items())) for o in s)


_memo = {"state": None, "model": None}


def copy_model(m):
    return dict(m, pieces=[set(g) for g in m["pieces"]], targets=[set(t) for t in m["targets"]])


def transition_function(state, action):
    aid = action.get("action_id") if isinstance(action, dict) else action
    if _memo["state"] is not None and canon(state) == _memo["state"]:
        m = copy_model(_memo["model"])
        m["other"] = [o for o in state if o["type"] not in ("wall", "target", "player", "reflection")]
    else:
        m = parse(state)
    step(m, aid)
    out = extract(m)
    _memo["state"], _memo["model"] = canon(out), copy_model(m)
    return out
