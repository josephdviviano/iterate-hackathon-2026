# Mechanics: single horizontal mirror axis (wall row band) + holed 3x3-block pieces + gray reflections + yellow targets; x=row, y=col.
# ACTION1-4 move the selection (axis: rows only; piece: by one block, blocked by bounds/other pieces); ACTION5 cycles
# axis -> pieces in INTRINSIC order (largest piece first, not by position; step 64 refutes positional order) -> axis.
# Selection is shown as 0 hole fills; reflections at block row 2A-r for piece blocks off the axis, skipping piece blocks.
# Layered render (piece>refl>target>wall) + re-extraction; unconfirmed: piece-vs-piece blocking, size-tie order.
import copy

N = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_last = {"out": None, "model": None}


def canon(s):
    return sorted(repr(sorted((k, repr(v)) for k, v in o.items())) for o in s)


def cells_of(o):
    out = {}
    for i, row in enumerate(o.get("pixels", [])):
        for j, v in enumerate(row):
            if v != -1:
                out[(o["x"] + i, o["y"] + j)] = v
    return out


def block_comps(blocks):
    blocks, comps = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + d[0], c + d[1])
                if n in blocks:
                    blocks.remove(n)
                    comp.add(n)
                    st.append(n)
        comps.append(comp)
    return comps


def order_pieces(pieces):
    return sorted(pieces, key=lambda p: (-len(p), min(p)))


def parse(state):
    allc = {}
    for o in state:
        for k, v in cells_of(o).items():
            allc[k] = (v, o["type"])
    axis, axis_sel = 0, False
    for o in state:
        if o["type"] == "wall":
            for (r, c), v in cells_of(o).items():
                if v == 10:
                    axis = r // 3
                if v == 0:
                    axis_sel = True
    pblocks = set()
    for o in state:
        if o["type"] == "player":
            pc = cells_of(o)
            for (r, c), v in pc.items():
                if v == 5 and r % 3 == 0 and c % 3 == 0:
                    pblocks.add((r // 3, c // 3))
    cls = {}
    for b in pblocks:
        k = (3 * b[0] + 1, 3 * b[1] + 1)
        if k not in allc:
            cls[b] = "empty"
        elif allc[k][0] == 0 and not axis_sel:
            cls[b] = "sel"
        elif allc[k][0] == 0:
            cls[b] = "empty"
        else:
            cls[b] = "amb"
    pieces, selp = [], None
    for comp in block_comps(pblocks):
        seeds = [b for b in comp if cls[b] == "sel"]
        if not seeds:
            pieces.append(comp)
            continue
        sel, st = set(seeds), list(seeds)
        while st:
            r, c = st.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + d[0], c + d[1])
                if n in comp and n not in sel and cls[n] != "empty":
                    sel.add(n)
                    st.append(n)
        selp = frozenset(sel)
        pieces.append(sel)
        pieces.extend(block_comps(comp - sel))
    pieces = [frozenset(p) for p in order_pieces(pieces)]
    if axis_sel:
        sel = "axis"
    elif selp is not None:
        sel = pieces.index(selp)
    else:
        sel = None
    targets = []
    for o in state:
        if o["type"] == "target":
            targets.append(frozenset((r // 3, c // 3) for (r, c) in cells_of(o)))
    others = [copy.deepcopy(o) for o in state if o["type"] not in ("wall", "player", "target", "reflection")]
    return {"axis": axis, "sel": sel, "pieces": pieces, "targets": targets, "others": others}


def step(m, action):
    m = dict(m)
    m["pieces"] = list(m["pieces"])
    if action == 5:
        order = ["axis"] + list(range(len(m["pieces"])))
        m["sel"] = "axis" if m["sel"] not in order else order[(order.index(m["sel"]) + 1) % len(order)]
    elif action in DIRS:
        dr, dc = DIRS[action]
        if m["sel"] == "axis":
            if dc == 0 and 0 <= m["axis"] + dr < N:
                m["axis"] += dr
        elif isinstance(m["sel"], int):
            i = m["sel"]
            nb = frozenset((r + dr, c + dc) for r, c in m["pieces"][i])
            occ = set().union(*[p for j, p in enumerate(m["pieces"]) if j != i])
            if all(0 <= r < N and 0 <= c < N for r, c in nb) and not (nb & occ):
                m["pieces"][i] = nb
    return m


def render(m):
    A, sel = m["axis"], m["sel"]
    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    refl = set()
    for p in m["pieces"]:
        for r, c in p:
            rr = 2 * A - r
            if r != A and 0 <= rr < N and (rr, c) not in occ:
                refl.add((rr, c))
    # sprites top-down: (blocks, ring colour, centre solid colour or None, centre fill, owner)
    sprites = []
    for i, p in enumerate(m["pieces"]):
        sprites.append((p, 5, None, 0 if sel == i else None, ("p",)))
    sprites.append((refl, 4, None, 4, ("r",)))
    for t, blocks in enumerate(m["targets"]):
        sprites.append((blocks, 11, 11, None, ("t", t)))
    sprites.append((set((A, c) for c in range(N)), 10, None, 0 if sel == "axis" else None, ("w",)))
    frame = {}
    for br in range(N):
        for bc in range(N):
            stack = [s for s in sprites if (br, bc) in s[0]]
            if not stack:
                continue
            for i in range(3):
                for j in range(3):
                    centre = i == 1 and j == 1
                    val = None
                    for blocks, ring, solid, fill, own in stack:
                        v = solid if centre else ring
                        if v is not None:
                            val = (v, own)
                            break
                    if val is None:
                        for blocks, ring, solid, fill, own in stack:
                            if fill is not None:
                                val = (fill, own)
                                break
                    if val is not None:
                        frame[(3 * br + i, 3 * bc + j)] = val
    return frame


def make_obj(name, typ, tags, layer, cells, colour):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    x, y = min(rs), min(cs)
    w, h = max(rs) - x + 1, max(cs) - y + 1
    px = [[-1] * h for _ in range(w)]
    for (r, c) in cells:
        px[r - x][c - y] = colour[(r, c)]
    return {"name": name, "type": typ, "tags": list(tags), "layer": layer,
            "x": x, "y": y, "w": w, "h": h, "pixels": px}


def pixel_comps(cellset):
    cellset, comps = set(cellset), []
    while cellset:
        st = [cellset.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + d[0], c + d[1])
                if n in cellset:
                    cellset.remove(n)
                    comp.add(n)
                    st.append(n)
        comps.append(comp)
    return sorted(comps, key=lambda cp: (min(r for r, c in cp), min(c for r, c in cp if r == min(r for r, c in cp))))


def bbox_key(cp):
    return (min(r for r, c in cp), min(c for r, c in cp))


def extract(m, frame):
    colour = {k: v[0] for k, v in frame.items()}
    out = [copy.deepcopy(o) for o in m["others"]]
    zeros = {k for k, v in colour.items() if v == 0}
    specs = [(10, "wall_h_", "wall", ["axis", "horizontal"], 1), (5, "piece_", "player", ["movable", "black"], 4)]
    for col, pre, typ, tags, layer in specs:
        own = {k for k, v in colour.items() if v == col}
        comps = sorted(pixel_comps(own | zeros), key=bbox_key)
        for rank, cp in enumerate(comps):
            if cp & own:
                out.append(make_obj(pre + str(rank), typ, tags, layer, cp, colour))
    comps = sorted(pixel_comps({k for k, v in colour.items() if v == 4}), key=bbox_key)
    for rank, cp in enumerate(comps):
        out.append(make_obj("reflection_" + str(rank), "reflection", ["mirror", "gray"], 3, cp, colour))
    groups = {}
    for k, (v, own) in frame.items():
        if own[0] == "t" and v == 11:
            groups.setdefault(own[1], set()).add(k)
    tg = sorted(groups.values(), key=bbox_key)
    for rank, cp in enumerate(tg):
        out.append(make_obj("target_" + str(rank), "target", ["goal", "yellow"], 2, cp, colour))
    return out


def transition_function(state, action):
    if _last["out"] is not None and canon(state) == _last["out"]:
        model = _last["model"]
    else:
        model = parse(state)
    aid = action.get("action_id") if isinstance(action, dict) else action
    model = step(model, aid)
    out = extract(model, render(model))
    _last["out"] = canon(out)
    _last["model"] = model
    return out
