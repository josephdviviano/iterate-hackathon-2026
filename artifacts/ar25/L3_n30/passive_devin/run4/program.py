# Mechanics: mirror-axis game, 3x3 block grid, pixels x-major (x=row). Model={axis row, piece block sets, selection, target sprites}.
# A5 cycles axis->pieces (by min block)->axis; A1/A2 move selection x-/+3, A3/A4 move a piece y-/+3 (no-op on axis).
# Pieces blocked by bounds/axis band/other pieces; axis only by bounds (passes under pieces: from a prior run, unobserved here).
# Reflections mirror piece blocks across the axis centre (rows<63, not on piece blocks). Render wall<target<refl<piece, holes
# show lower solids else fill; re-extract 4-conn, ranks incl. foreign 0s. 'player gone' under A1 = pieces merging, not hidden state.
N = 63
_memo = {"out": None, "model": None}


def _canon(state):
    return sorted(repr(sorted(o.items())) for o in state)


def _cells(o):
    for i, row in enumerate(o.get("pixels", [])):
        for j, v in enumerate(row):
            yield o["x"] + i, o["y"] + j, v


def parse(state):
    walls = [o for o in state if o["type"] == "wall"]
    players = [o for o in state if o["type"] == "player"]
    targets = [o for o in state if o["type"] == "target"]
    tgt_cells = {}
    tsprites = []
    for o in sorted(targets, key=lambda o: o["name"]):
        blocks = set()
        for x, y, v in _cells(o):
            if v != -1:
                blocks.add((x // 3, y // 3))
                tgt_cells[(x, y)] = v
        tsprites.append(blocks)
    a = min(o["x"] for o in walls) if walls else None
    axis_sel = any(v == 0 for o in walls for _, _, v in _cells(o))
    pieces = []
    for o in players:
        pix = {(x, y): v for x, y, v in _cells(o)}
        blocks = {}
        for (x, y), v in pix.items():
            if v == 5 and x % 3 != 1 or v == 5 and y % 3 != 1:
                b = (x // 3, y // 3)
                c = (b[0] * 3 + 1, b[1] * 3 + 1)
                cv = pix.get(c, -1)
                blocks[b] = "sel" if cv == 0 else ("amb" if c in tgt_cells else "unsel")
        pieces.extend(_split(blocks))
    pieces.sort(key=lambda p: min(p[0]))
    sel = "axis" if axis_sel else None
    if sel is None:
        for i, (bl, s) in enumerate(pieces):
            if s:
                sel = i
        if sel is None:
            sel = "axis" if not pieces else 0
    return {"a": a, "pieces": [set(p[0]) for p in pieces], "sel": sel, "targets": tsprites}


def _comps(blocks):
    left, out = set(blocks), []
    while left:
        st = [left.pop()]
        comp = set(st)
        while st:
            bx, by = st.pop()
            for nb in ((bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)):
                if nb in left:
                    left.discard(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    return out


def _split(blocks):
    sel = {b for b, k in blocks.items() if k == "sel"}
    res = []
    for comp in _comps(set(blocks)):
        s = comp & sel
        if not s:
            res.append((comp, False))
            continue
        grown, st = set(s), list(s)
        while st:
            bx, by = st.pop()
            for nb in ((bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)):
                if nb in comp and nb not in grown and blocks[nb] in ("sel", "amb"):
                    grown.add(nb)
                    st.append(nb)
        res.append((grown, True))
        for c in _comps(comp - grown):
            res.append((c, False))
    return res


def step(m, action):
    m = {"a": m["a"], "pieces": [set(p) for p in m["pieces"]], "sel": m["sel"],
         "targets": m["targets"]}
    aid = action["action_id"] if isinstance(action, dict) else action
    if aid == 5:
        if m["sel"] == "axis":
            m["sel"] = 0 if m["pieces"] else "axis"
        else:
            m["sel"] = m["sel"] + 1 if m["sel"] + 1 < len(m["pieces"]) else "axis"
        return m
    if aid not in (1, 2, 3, 4):
        return m
    dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[aid]
    if m["sel"] == "axis":
        if dy or m["a"] is None:
            return m
        nb = m["a"] // 3 + dx
        if 0 <= nb < N // 3:
            m["a"] = nb * 3
        return m
    i = m["sel"]
    moved = {(bx + dx, by + dy) for bx, by in m["pieces"][i]}
    others = set().union(*[p for j, p in enumerate(m["pieces"]) if j != i]) if len(m["pieces"]) > 1 else set()
    ok = all(0 <= bx < N // 3 and 0 <= by < N // 3 for bx, by in moved)
    ok = ok and not (moved & others) and (m["a"] is None or all(bx != m["a"] // 3 for bx, _ in moved))
    if ok:
        m["pieces"][i] = moved
    return m


def _block_cells(b):
    for i in range(3):
        for j in range(3):
            yield (b[0] * 3 + i, b[1] * 3 + j), (i == 1 and j == 1)


def render(m):
    layers = {}  # cell -> list of (layer, type, sprite_id, solid, colour)

    def put(cell, layer, typ, sid, solid, col):
        x, y = cell
        if 0 <= x < N and 0 <= y < N:
            layers.setdefault(cell, []).append((layer, typ, sid, solid, col))

    a = m["a"]
    if a is not None:
        for y in range(N):
            for i in range(3):
                hole = i == 1 and y % 3 == 1
                put((a + i, y), 1, "wall", 0, not hole, 0 if hole and m["sel"] == "axis" else (None if hole else 10))
    for t, blocks in enumerate(m["targets"]):
        for b in blocks:
            for c, _ in _block_cells(b):
                put(c, 2, "target", t, True, 11)
    pblocks = set().union(*m["pieces"]) if m["pieces"] else set()
    if a is not None:
        ab = a // 3
        rblocks = {(2 * ab - bx, by) for bx, by in pblocks} - pblocks
        for b in rblocks:
            for c, hole in _block_cells(b):
                put(c, 3, "reflection", 0, not hole, 4)
    for k, p in enumerate(m["pieces"]):
        for b in p:
            for c, hole in _block_cells(b):
                put(c, 4, "player", k, not hole, (0 if m["sel"] == k else None) if hole else 5)
    owner = {}
    for cell, st in layers.items():
        st.sort(key=lambda e: -e[0])
        top = next((e for e in st if e[3]), None)
        if top is None:
            top = st[0]
        if top[4] is not None:
            owner[cell] = (top[1], top[2], top[4])
    return owner


def _cc(cells):
    left, out = set(cells), []
    while left:
        st = [left.pop()]
        comp = set(st)
        while st:
            x, y = st.pop()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in left:
                    left.discard(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    return out


def _obj(cells, colours, name, typ, layer, tags):
    x0 = min(c[0] for c in cells); y0 = min(c[1] for c in cells)
    x1 = max(c[0] for c in cells); y1 = max(c[1] for c in cells)
    pix = [[-1] * (y1 - y0 + 1) for _ in range(x1 - x0 + 1)]
    for c in cells:
        pix[c[0] - x0][c[1] - y0] = colours[c]
    return {"name": name, "type": typ, "tags": list(tags), "x": x0, "y": y0,
            "w": x1 - x0 + 1, "h": y1 - y0 + 1, "layer": layer, "pixels": pix}


def _ranked(comps, foreign_zeros):
    items = [((min(c[0] for c in comp), min(c[1] for c in comp)), 0, k) for k, comp in enumerate(comps)]
    items += [(z, 1, None) for z in foreign_zeros]
    items.sort(key=lambda e: (e[0], e[1]))
    return {e[2]: r for r, e in enumerate(items) if e[2] is not None}


def extract(owner, extra):
    col = {c: v[2] for c, v in owner.items()}
    by = {}
    for c, (typ, sid, v) in owner.items():
        by.setdefault(typ, {}).setdefault(sid, set()).add(c)
    out = list(extra)
    zeros = {t: [c for c, v in owner.items() if v[2] == 0 and v[0] == t] for t in ("wall", "player")}
    spec = {"wall": ("wall_h_", 1, ["axis", "horizontal"], "player"),
            "player": ("piece_", 4, ["movable", "black"], "wall"),
            "reflection": ("reflection_", 3, ["mirror", "gray"], None)}
    for typ, (prefix, layer, tags, other) in spec.items():
        cells = set().union(*by.get(typ, {}).values()) if typ in by else set()
        comps = _cc(cells)
        rank = _ranked(comps, zeros[other] if other else [])
        for k, comp in enumerate(comps):
            out.append(_obj(comp, col, prefix + str(rank[k]), typ, layer, tags))
    tcomps = [s for _, s in sorted(by.get("target", {}).items())]
    tcomps.sort(key=lambda s: (min(c[0] for c in s), min(c[1] for c in s)))
    for k, s in enumerate(tcomps):
        out.append(_obj(s, col, "target_" + str(k), "target", 2, ["goal", "yellow"]))
    return out


def transition_function(state, action):
    if _memo["out"] is not None and _canon(state) == _memo["out"]:
        model = _memo["model"]
    else:
        model = parse(state)
    extra = [dict(o) for o in state if o["type"] not in ("wall", "player", "reflection", "target")]
    nm = step(model, action)
    out = extract(render(nm), extra)
    _memo["out"] = _canon(out)
    _memo["model"] = nm
    return out
