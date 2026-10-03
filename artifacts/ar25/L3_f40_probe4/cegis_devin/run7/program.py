# Mechanics: single mirror axis (wall row of 3x3 blocks) + movable pieces (block sets) + targets + gray reflections.
# ACTION5 cycles selection axis -> pieces (largest first, tie min block) -> axis; none selected -> axis.
# A1/A2 move the selected axis or piece up/down one block, A3/A4 move a piece left/right; pieces are blocked by
# bounds and other pieces only (axis passes under pieces). Every piece block off the axis reflects to (2A-r, c).
# Frame = layered composite (piece>refl>target>wall), re-extracted by colour; 0 dots count in wall/piece ranks.
N = 21
COL = {"wall": 10, "player": 5, "reflection": 4, "target": 11}
_last = {"out": None, "model": None}


def _canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _cells(o):
    for i, row in enumerate(o.get("pixels", [])):
        for j, v in enumerate(row):
            if v != -1:
                yield o["x"] + i, o["y"] + j, v


def _comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        s = blocks.pop(); comp, st = {s}, [s]
        while st:
            r, c = st.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in blocks:
                    blocks.discard(n); comp.add(n); st.append(n)
        out.append(comp)
    return out


def parse(state):
    walls = [o for o in state if o["type"] == "wall"]
    axis = walls[0]["x"] // 3 if walls else 0
    axis_sel = any(v == 0 for o in walls for _, _, v in _cells(o))
    other = {}
    for o in state:
        if o["type"] != "player":
            for r, c, v in _cells(o):
                other[(r, c)] = v
    targets = []
    for o in sorted((o for o in state if o["type"] == "target"), key=lambda o: o["name"]):
        targets.append({(r // 3, c // 3) for r, c, _ in _cells(o)})
    pix = {}
    for o in state:
        if o["type"] == "player":
            for r, c, v in _cells(o):
                pix[(r, c)] = v
    blocks = {(r // 3, c // 3) for (r, c), v in pix.items() if v == 5 and r % 3 == 0 and c % 3 == 0}
    pieces, sel = [], None
    for comp in _comps(blocks):
        cls = {}
        for b in comp:
            ctr = (3 * b[0] + 1, 3 * b[1] + 1)
            if pix.get(ctr) == 0 and not axis_sel:
                cls[b] = "z"
            elif ctr in other:
                cls[b] = "c"
            else:
                cls[b] = "e"
        zs = [b for b in comp if cls[b] == "z"]
        if zs:
            selb = set()
            for sub in _comps([b for b in comp if cls[b] != "e"]):
                if sub & set(zs):
                    selb |= sub
            rest = comp - selb
            sel = len(pieces)
            pieces.append(selb)
            pieces.extend(_comps(rest))
        else:
            pieces.append(comp)
    if axis_sel:
        sel = "axis"
    elif sel is None:
        for i, p in enumerate(pieces):
            if all((3 * r + 1, 3 * c + 1) in other for r, c in p):
                sel = i
                break
    counter = [o for o in state if o["type"] == "counter"]
    return {"axis": axis, "sel": sel, "pieces": pieces, "targets": targets, "counter": counter}


def order(m):
    idx = sorted(range(len(m["pieces"])), key=lambda i: (-len(m["pieces"][i]), min(m["pieces"][i])))
    return ["axis"] + idx


def step(m, action):
    m = {"axis": m["axis"], "sel": m["sel"], "pieces": [set(p) for p in m["pieces"]],
         "targets": m["targets"], "counter": m["counter"]}
    if action == 5:
        cyc = order(m)
        m["sel"] = "axis" if m["sel"] not in cyc else cyc[(cyc.index(m["sel"]) + 1) % len(cyc)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None or m["sel"] is None:
        return m
    if m["sel"] == "axis":
        if d[1] == 0 and 0 <= m["axis"] + d[0] < N:
            m["axis"] += d[0]
        return m
    i = m["sel"]
    moved = {(r + d[0], c + d[1]) for r, c in m["pieces"][i]}
    occ = set().union(*[p for j, p in enumerate(m["pieces"]) if j != i]) if len(m["pieces"]) > 1 else set()
    if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & occ):
        m["pieces"][i] = moved
    return m


def render(m):
    A = m["axis"]
    stacks = {}  # cell -> list of (layer, kind, solid, fill, owner) ; higher layer first after sort

    def put(b, layer, kind, ring, fill, owner):
        r0, c0 = 3 * b[0], 3 * b[1]
        for i in range(3):
            for j in range(3):
                cell = (r0 + i, c0 + j)
                if i == 1 and j == 1:
                    ent = (layer, kind, None, fill, owner)
                else:
                    ent = (layer, kind, ring, None, owner)
                stacks.setdefault(cell, []).append(ent)

    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    for c in range(N):
        put((A, c), 1, "wall", 10, 0 if m["sel"] == "axis" else -1, "wall")
    for t, blocks in enumerate(m["targets"]):
        for b in blocks:
            r0, c0 = 3 * b[0], 3 * b[1]
            for i in range(3):
                for j in range(3):
                    stacks.setdefault((r0 + i, c0 + j), []).append((2, "target", 11, None, ("t", t)))
    for p in m["pieces"]:
        for r, c in p:
            rb = 2 * A - r
            if r != A and 0 <= rb < N and (rb, c) not in occ:
                put((rb, c), 3, "reflection", 4, 4, "refl")
    for i, p in enumerate(m["pieces"]):
        for b in p:
            put(b, 4, "player", 5, 0 if m["sel"] == i else -1, "player")
    frame = {}
    for cell, st in stacks.items():
        st.sort(key=lambda e: -e[0])
        hit = next((e for e in st if e[2] is not None), None)
        if hit:
            frame[cell] = (hit[2], hit[4])
        else:
            f = next((e for e in st if e[3] is not None and e[3] >= 0), None)
            if f:
                frame[cell] = (f[3], f[4])
    return frame


def _obj(name, typ, tags, layer, cells, frame):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    x, y, w, h = min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
    px = [[-1] * h for _ in range(w)]
    for r, c in cells:
        px[r - x][c - y] = frame[(r, c)][0]
    return {"name": name, "type": typ, "tags": tags, "x": x, "y": y, "w": w, "h": h,
            "layer": layer, "pixels": px}


def _cellcomps(cells):
    cells, out = set(cells), []
    while cells:
        s = cells.pop(); comp, st = {s}, [s]
        while st:
            r, c = st.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells:
                    cells.discard(n); comp.add(n); st.append(n)
        out.append(comp)
    return sorted(out, key=lambda k: (min(r for r, _ in k), min(c for r, c in k if r == min(r for r, _ in k))))


def _bboxkey(k):
    return (min(r for r, _ in k), min(c for _, c in k))


def extract(m, frame):
    out = [dict(o) for o in m["counter"]]
    zeros = {c for c, v in frame.items() if v[0] == 0}
    for typ, prefix, tags, layer in (("wall", "wall_h_", ["axis", "horizontal"], 1),
                                     ("player", "piece_", ["movable", "black"], 4)):
        own = {c for c, v in frame.items() if v[0] == COL[typ]}
        comps = sorted(_cellcomps(own | zeros), key=_bboxkey)
        for k, comp in enumerate(comps):
            if comp & own:
                out.append(_obj(prefix + str(k), typ, tags, layer, comp, frame))
    refl = {c for c, v in frame.items() if v[0] == 4}
    for k, comp in enumerate(sorted(_cellcomps(refl), key=_bboxkey)):
        out.append(_obj("reflection_" + str(k), "reflection", ["mirror", "gray"], 3, comp, frame))
    tg = []
    for t in range(len(m["targets"])):
        cells = {c for c, v in frame.items() if v[1] == ("t", t)}
        if cells:
            tg.append(cells)
    for k, comp in enumerate(sorted(tg, key=_bboxkey)):
        out.append(_obj("target_" + str(k), "target", ["goal", "yellow"], 2, comp, frame))
    return out


def transition_function(state, action):
    aid = action["action_id"] if isinstance(action, dict) else action
    if _last["out"] is not None and _canon(state) == _last["out"]:
        model = _last["model"]
    else:
        model = parse(state)
    nm = step(model, aid)
    out = extract(nm, render(nm))
    _last["out"], _last["model"] = _canon(out), nm
    return out
