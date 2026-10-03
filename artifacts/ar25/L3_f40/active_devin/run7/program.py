# Mechanics: mirror puzzle on a 63x63 board of 3x3 cells (x=row, pixels x-major). A horizontal
# axis (wall, 3 rows, dotted centre row) mirrors every piece block row b to 2a-b as gray reflections.
# ACTION5 cycles selection axis -> pieces (by bbox x,y) -> axis; ACTION1-4 move it by one cell (axis rows only).
# Moves are blocked off-board, onto the axis row, or onto another piece (axis: onto a piece row). Frame is rendered
# by layer (wall<target<reflection<piece; holes transparent) and re-extracted. Unconfirmed: clicks/ACTION7 are no-ops.
import json

N = 21
DEFAULTS = {"wall": (1, ["axis", "horizontal"]), "target": (2, ["goal", "yellow"]),
            "reflection": (3, ["mirror", "gray"]), "player": (4, ["movable", "black"])}
_memo = {"out": None, "model": None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells(o):
    out = {}
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v >= 0:
                out[(o["x"] + i, o["y"] + j)] = v
    return out


def blocks_of(pix):
    return {(r // 3, c // 3) for (r, c) in pix}


def adj_components(blocks):
    blocks, comps = set(blocks), []
    while blocks:
        stack, comp = [blocks.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (b[0] + d[0], b[1] + d[1])
                if nb in blocks:
                    blocks.remove(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def bkey(blocks):
    return (min(b[0] for b in blocks), min(b[1] for b in blocks))


# ---------- parse observed state into a model ----------
def parse(state):
    walls = [o for o in state if o["type"] == "wall"]
    targets = sorted([o for o in state if o["type"] == "target"], key=lambda o: (o["x"], o["y"]))
    pieces = [o for o in state if o["type"] == "player"]
    axis = min(o["x"] for o in walls) // 3 if walls else 10
    axis_sel = any(0 in row for o in walls for row in o.get("pixels") or [])
    lower = {}
    for o in state:
        if o["type"] in ("wall", "target", "reflection"):
            lower.update(cells(o))
    tblocks = [blocks_of({p for p, v in cells(o).items() if v == 11}) for o in targets]
    logical, sel = [], None
    for o in pieces:
        pix = cells(o)
        S, U, A = set(), set(), set()
        for b in blocks_of(pix):
            ctr = (3 * b[0] + 1, 3 * b[1] + 1)
            if pix.get(ctr) == 0:
                S.add(b)
            elif ctr in lower:
                A.add(b)
            else:
                U.add(b)
        if S:
            grown, frontier = set(S), list(S)
            while frontier:
                b = frontier.pop()
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (b[0] + d[0], b[1] + d[1])
                    if nb in A and nb not in grown:
                        grown.add(nb)
                        frontier.append(nb)
            logical.append(frozenset(grown))
            sel = logical[-1]
            rest = (U | A) - grown
        else:
            rest = U | A
        for comp in adj_components(rest):
            logical.append(frozenset(comp))
    if axis_sel:
        sel = "axis"
    elif sel is None and logical:
        cand = [p for p in logical if all((3 * b[0] + 1, 3 * b[1] + 1) in lower for b in p)]
        sel = (cand or logical)[0]
    return {"axis": axis, "pieces": logical, "sel": sel, "targets": tblocks}


# ---------- per-type update rules ----------
def on_board(blocks):
    return all(0 <= r < N and 0 <= c < N for r, c in blocks)


def piece_blocked(model, piece, moved):
    others = set().union(*[p for p in model["pieces"] if p != piece]) if len(model["pieces"]) > 1 else set()
    return (not on_board(moved) or any(r == model["axis"] for r, _ in moved) or bool(moved & others))


def axis_blocked(model, row):
    return not (0 <= row < N) or any(r == row for p in model["pieces"] for r, _ in p)


def step(model, action):
    m = dict(model)
    m["pieces"] = list(model["pieces"])
    if action == 5:
        order = sorted(m["pieces"], key=bkey)
        if m["sel"] == "axis":
            m["sel"] = order[0] if order else "axis"
        else:
            k = order.index(m["sel"])
            m["sel"] = order[k + 1] if k + 1 < len(order) else "axis"
        return m
    if action not in (1, 2, 3, 4):
        return m
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    if m["sel"] == "axis":
        if dc == 0 and not axis_blocked(m, m["axis"] + dr):
            m["axis"] += dr
        return m
    piece = m["sel"]
    moved = frozenset((r + dr, c + dc) for r, c in piece)
    if not piece_blocked(m, piece, moved):
        m["pieces"][m["pieces"].index(piece)] = moved
        m["sel"] = moved
    return m


# ---------- render + re-extract ----------
def render(model):
    top = {}
    a = model["axis"]
    for r in range(3 * a, 3 * a + 3):
        for c in range(3 * N):
            if r == 3 * a + 1 and c % 3 == 1:
                if model["sel"] == "axis":
                    top[(r, c)] = (0, "wall")
            else:
                top[(r, c)] = (10, "wall")
    for t, tb in enumerate(model["targets"]):
        for br, bc in tb:
            for r in range(3 * br, 3 * br + 3):
                for c in range(3 * bc, 3 * bc + 3):
                    top[(r, c)] = (11, ("target", t))
    refl = {(2 * a - r, c) for p in model["pieces"] for r, c in p}
    refl = {b for b in refl if on_board([b])}

    def draw(blocks, colour, kind, hole):
        for br, bc in blocks:
            for r in range(3 * br, 3 * br + 3):
                for c in range(3 * bc, 3 * bc + 3):
                    if (r, c) == (3 * br + 1, 3 * bc + 1):
                        if (r, c) not in top and hole is not None:
                            top[(r, c)] = (hole, kind)
                    else:
                        top[(r, c)] = (colour, kind)
    draw(refl, 4, "reflection", 4)
    for p in model["pieces"]:
        sel = p == model["sel"]
        for br, bc in p:
            ctr = (3 * br + 1, 3 * bc + 1)
            for r in range(3 * br, 3 * br + 3):
                for c in range(3 * bc, 3 * bc + 3):
                    if (r, c) != ctr:
                        top[(r, c)] = (5, "piece")
            if ctr not in top and sel:
                top[ctr] = (0, "piece")
    return top


def pix_components(pixels):
    pixels, comps = set(pixels), []
    while pixels:
        stack, comp = [pixels.pop()], set()
        while stack:
            p = stack.pop()
            comp.add(p)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (p[0] + d[0], p[1] + d[1])
                if q in pixels:
                    pixels.remove(q)
                    stack.append(q)
        comps.append(comp)
    return comps


def make_obj(typ, pts, top, meta):
    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
    x1, y1 = max(p[0] for p in pts), max(p[1] for p in pts)
    pix = [[top[(r, c)][0] if (r, c) in pts else -1 for c in range(y0, y1 + 1)] for r in range(x0, x1 + 1)]
    layer, tags = meta[typ]
    return {"type": typ, "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
            "layer": layer, "tags": list(tags), "pixels": pix}


def extract(model, top, meta, passthrough):
    out = [dict(o) for o in passthrough]
    groups = {"wall": [], "piece": [], "reflection": []}
    tpix = {}
    for p, (v, kind) in top.items():
        if isinstance(kind, tuple):
            tpix.setdefault(kind[1], set()).add(p)
        else:
            groups[kind].append(p)
    objs = {}
    for kind, typ in (("wall", "wall"), ("piece", "player"), ("reflection", "reflection")):
        objs[typ] = [make_obj(typ, c, top, meta) for c in pix_components(groups[kind])]
    objs["target"] = [make_obj("target", tpix[t], top, meta) for t in sorted(tpix)]
    for typ, prefix in (("player", "piece"), ("reflection", "reflection"), ("target", "target")):
        for i, o in enumerate(sorted(objs[typ], key=lambda o: (o["x"], o["y"]))):
            o["name"] = "%s_%d" % (prefix, i)
            out.append(o)
    zeros = [(p, None) for p, (v, kind) in top.items() if v == 0 and kind != "wall"]
    ranked = sorted([((o["x"], o["y"]), o) for o in objs["wall"]] + zeros, key=lambda e: e[0])
    for i, (_, o) in enumerate(ranked):
        if o is not None:
            o["name"] = "wall_h_%d" % i
            out.append(o)
    return out


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get("action_id")
    meta = dict(DEFAULTS)
    for o in state:
        if o["type"] in meta:
            meta[o["type"]] = (o["layer"], o["tags"])
    passthrough = [o for o in state if o["type"] not in meta]
    if _memo["out"] is not None and canon(state) == _memo["out"]:
        model = _memo["model"]
    else:
        model = parse(state)
    new = step(model, action)
    out = extract(new, render(new), meta, passthrough)
    _memo["out"], _memo["model"] = canon(out), new
    return out
