# Mechanics: mirror-axis game on a 21x21 grid of 3x3 blocks (obj x=row, y=col, pixels x-major).
# Sprites: wall axis (full block row A), pieces (block sets, layer 4), targets (static), reflections
# (each piece block b not on the axis -> row 2A-b, both sides, clipped, skipped on piece blocks).
# A1/A2/A3/A4 = row-1/row+1/col-1/col+1 of the selection (axis ignores A3/A4); A5 cycles axis->pieces.
# Render layers + re-extract 4-conn comps; names rank by bbox incl. 0-only comps. Hypothesis: piece cycle order.
N = 21
_memo = {"out": None, "model": None}


def _canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _cells(o):
    for i, row in enumerate(o.get("pixels", [])):
        for j, v in enumerate(row):
            if v != -1:
                yield o["x"] + i, o["y"] + j, v


def _adj(b):
    r, c = b
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _groups(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]; g = set(st)
        while st:
            for n in _adj(st.pop()):
                if n in blocks:
                    blocks.discard(n); g.add(n); st.append(n)
        out.append(g)
    return out


def parse(state):
    walls = [o for o in state if o["type"] == "wall"]
    axis = max(set(o["x"] // 3 for o in walls), key=lambda r: sum(o["h"] for o in walls if o["x"] // 3 == r))
    axis_sel = any(v == 0 for o in walls for _, _, v in _cells(o))
    targets = []
    for o in state:
        if o["type"] == "target":
            targets.append(set((x // 3, y // 3) for x, y, _ in _cells(o)))
    pieces, sel = [], None
    seen = set((x, y) for o in state for x, y, _ in _cells(o))
    for o in state:
        if o["type"] != "player":
            continue
        pix = {(x, y): v for x, y, v in _cells(o)}
        blocks = set((x // 3, y // 3) for (x, y), v in pix.items() if v == 5 and x % 3 != 1 and y % 3 != 1
                     and (x // 3 * 3, y // 3 * 3) in pix and pix[(x // 3 * 3, y // 3 * 3)] == 5)
        centre = {b: pix.get((b[0] * 3 + 1, b[1] * 3 + 1), 99 if (b[0] * 3 + 1, b[1] * 3 + 1) in seen else -1)
                  for b in blocks}
        if axis_sel:
            pieces.extend(_groups(blocks)); continue
        zero = set(b for b in blocks if centre[b] == 0)
        if not zero:
            pieces.extend(_groups(blocks)); continue
        selb, st = set(zero), list(zero)
        while st:
            for n in _adj(st.pop()):
                if n in blocks and n not in selb and centre[n] != -1:
                    selb.add(n); st.append(n)
        pieces.append(selb); sel = len(pieces) - 1
        pieces.extend(_groups(blocks - selb))
    order = sorted(range(len(pieces)), key=lambda i: min(pieces[i]))
    pieces = [pieces[i] for i in order]
    selection = "axis" if axis_sel else (order.index(sel) if sel is not None else None)
    return {"axis": axis, "sel": selection, "pieces": pieces, "targets": targets}


DELTA = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def step(m, action):
    m = {"axis": m["axis"], "sel": m["sel"], "pieces": [set(p) for p in m["pieces"]], "targets": m["targets"]}
    if isinstance(action, dict):
        return m
    if action == 5:
        if m["sel"] == "axis":
            m["sel"] = 0 if m["pieces"] else "axis"
        elif m["sel"] is None or m["sel"] + 1 >= len(m["pieces"]):
            m["sel"] = "axis"
        else:
            m["sel"] += 1
    elif action in DELTA:
        dr, dc = DELTA[action]
        if m["sel"] == "axis":
            if dr and 0 <= m["axis"] + dr < N:
                m["axis"] += dr
        elif m["sel"] is not None:
            p = m["pieces"][m["sel"]]
            new = set((r + dr, c + dc) for r, c in p)
            others = set().union(*[q for i, q in enumerate(m["pieces"]) if i != m["sel"]])
            if all(0 <= r < N and 0 <= c < N for r, c in new) and not (new & others):
                m["pieces"][m["sel"]] = new
    return m


def render(m):
    A = m["axis"]
    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    refl = set()
    for r, c in occ:
        rr = 2 * A - r
        if r != A and 0 <= rr < N and (rr, c) not in occ:
            refl.add((rr, c))
    stack = {}  # cell -> list of (layer, owner, solid_value or None, fill)

    def put(b, layer, owner, colour, centre_solid, fill):
        for i in range(3):
            for j in range(3):
                cell = (b[0] * 3 + i, b[1] * 3 + j)
                if i == 1 and j == 1:
                    item = (layer, owner, colour if centre_solid is True else centre_solid, fill)
                else:
                    item = (layer, owner, colour, None)
                stack.setdefault(cell, []).append(item)
    for c in range(N):
        put((A, c), 1, ("wall",), 10, 0 if m["sel"] == "axis" else None, None)
    for k, t in enumerate(m["targets"]):
        for b in t:
            put(b, 2, ("target", k), 11, True, None)
    for b in refl:
        put(b, 3, ("reflection",), 4, None, 4)
    for k, p in enumerate(m["pieces"]):
        for b in p:
            put(b, 4, ("player",), 5, None, 0 if m["sel"] == k else None)
    comp = {}
    for cell, items in stack.items():
        items.sort(key=lambda it: -it[0])
        val = None
        for layer, owner, solid, fill in items:
            if solid is not None:
                val = (solid, owner); break
        if val is None and items[0][3] is not None:
            val = (items[0][3], items[0][1])
        if val is not None:
            comp[cell] = val
    return comp


def _obj(name, typ, tags, layer, cells, comp):
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for (x, y) in cells:
        pix[x - x0][y - y0] = comp[(x, y)][0]
    return {"name": name, "type": typ, "tags": list(tags), "layer": layer, "x": x0, "y": y0, "w": w, "h": h,
            "pixels": pix}


def _components(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]; g = {st[0]}
        while st:
            for n in _adj(st.pop()):
                if n in cells:
                    cells.discard(n); g.add(n); st.append(n)
        out.append(g)
    return out


def extract(comp, scenery):
    out = list(scenery)
    zeros = set(c for c, (v, _) in comp.items() if v == 0)
    for typ, prefix, tags, layer in (("wall", "wall_h_", ["axis", "horizontal"], 1),
                                     ("player", "piece_", ["movable", "black"], 4)):
        own = set(c for c, (v, o) in comp.items() if o[0] == typ and v != 0)
        comps = sorted(_components(own | zeros), key=lambda g: (min(x for x, _ in g), min(y for _, y in g)))
        for i, g in enumerate(comps):
            if g & own:
                out.append(_obj(prefix + str(i), typ, tags, layer, g, comp))
    own = set(c for c, (v, o) in comp.items() if o[0] == "reflection")
    comps = sorted(_components(own), key=lambda g: (min(x for x, _ in g), min(y for _, y in g)))
    for i, g in enumerate(comps):
        out.append(_obj("reflection_" + str(i), "reflection", ["mirror", "gray"], 3, g, comp))
    tg = {}
    for c, (v, o) in comp.items():
        if o[0] == "target":
            tg.setdefault(o[1], set()).add(c)
    comps = sorted(tg.values(), key=lambda g: (min(x for x, _ in g), min(y for _, y in g)))
    for i, g in enumerate(comps):
        out.append(_obj("target_" + str(i), "target", ["goal", "yellow"], 2, g, comp))
    return out


def transition_function(state, action):
    scenery = [dict(o) for o in state if o["type"] not in ("wall", "player", "reflection", "target")]
    if _memo["out"] is not None and _canon(state) == _memo["out"]:
        model = _memo["model"]
    else:
        model = parse(state)
    model = step(model, action)
    out = extract(render(model), scenery)
    _memo["out"], _memo["model"] = _canon(out), model
    return out
