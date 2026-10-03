# Mechanics: one horizontal mirror axis (wall row band, layer 1) + movable 3x3-block pieces (layer 4); every
# piece block off the axis row casts a gray reflection at block row 2A-r (clipped, skipped on piece blocks).
# ACTION1-4 move the selected object (axis: rows only, bounds; piece: bounds, no overlap of other pieces).
# ACTION5 cycles axis -> pieces by size desc (tie: min block) -> axis. Selection = 0-filled holes. Frame is
# re-rendered by layers and re-extracted (4-conn colour comps; wall/player ranks count all 0 dots; targets per sprite).
N = 21
WALL, TARGET, REFL, PIECE = 1, 2, 3, 4
WTAGS, PTAGS, RTAGS = ["axis", "horizontal"], ["movable", "black"], ["mirror", "gray"]


def cells(o):
    for i, row in enumerate(o.get("pixels") or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o["x"] + i, o["y"] + j, v


def block_comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        stack, comp = [blocks.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (b[0] + d[0], b[1] + d[1])
                if n in blocks:
                    blocks.remove(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(state):
    frame, others = {}, []
    walls, players, targets = [], [], []
    for o in state:
        t = o["type"]
        if t == "wall":
            walls.append(o)
        elif t == "player":
            players.append(o)
        elif t == "target":
            targets.append(o)
        elif t != "reflection":
            others.append(o)
        for x, y, v in cells(o):
            frame[(x, y)] = (v, t)
    axis = None
    axis_sel = False
    for o in walls:
        for x, y, v in cells(o):
            if v == 10 and axis is None:
                axis = x // 3
            if v == 0:
                axis_sel = True
    tsprites = []
    for o in targets:
        tsprites.append({(x // 3, y // 3) for x, y, v in cells(o)})
    pblocks = set()
    for o in players:
        for x, y, v in cells(o):
            if v == 5 and x % 3 == 0 and y % 3 == 0:
                pblocks.add((x // 3, y // 3))
    # centre class: 'sel' (0), 'amb' (shows another object), 'empty'
    cls = {}
    for b in pblocks:
        c = frame.get((3 * b[0] + 1, 3 * b[1] + 1))
        if c is None:
            cls[b] = "empty"
        elif c[0] == 0 and not axis_sel:
            cls[b] = "sel"
        elif c[0] == 0:
            cls[b] = "empty"
        else:
            cls[b] = "amb"
    pieces, sel = [], ("axis" if axis_sel else None)
    for comp in block_comps(pblocks):
        seed = {b for b in comp if cls[b] == "sel"}
        if seed:
            grown, changed = set(seed), True
            while changed:
                changed = False
                for b in comp:
                    if b not in grown and cls[b] == "amb" and any(
                            (b[0] + d[0], b[1] + d[1]) in grown for d in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        grown.add(b)
                        changed = True
            selp = frozenset(grown)
            pieces.append(selp)
            sel = selp
            for rest in block_comps(comp - grown):
                pieces.append(frozenset(rest))
        else:
            pieces.append(frozenset(comp))
    if sel is None:
        covered = [p for p in pieces if all(cls[b] == "amb" for b in p)]
        if len(covered) == 1:
            sel = covered[0]
    return {"axis": axis, "sel": sel, "pieces": pieces, "targets": tsprites, "others": others}


def piece_order(pieces):
    return sorted(pieces, key=lambda p: (-len(p), min(p)))


def step(m, action):
    aid = action["action_id"] if isinstance(action, dict) else action
    sel, pieces = m["sel"], list(m["pieces"])
    if aid == 5:
        order = ["axis"] + piece_order(pieces)
        if sel is None or sel not in order:
            m["sel"] = "axis"
        else:
            m["sel"] = order[(order.index(sel) + 1) % len(order)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if d is None or sel is None:
        return m
    if sel == "axis":
        if d[1] == 0 and m["axis"] is not None and 0 <= m["axis"] + d[0] < N:
            m["axis"] += d[0]
        return m
    moved = frozenset((r + d[0], c + d[1]) for r, c in sel)
    occupied = set().union(*[p for p in pieces if p != sel]) if len(pieces) > 1 else set()
    if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & occupied):
        m["pieces"] = [moved if p == sel else p for p in pieces]
        m["sel"] = moved
    return m


def render(m):
    A, sel = m["axis"], m["sel"]
    stacks = {}  # block -> list of (layer, owner, ring colour, solid centre?, hole fill)

    def add(b, layer, owner, ring, solid, fill):
        stacks.setdefault(b, []).append((layer, owner, ring, solid, fill))
    if A is not None:
        for c in range(N):
            add((A, c), WALL, "wall", 10, False, 0 if sel == "axis" else -1)
    for k, t in enumerate(m["targets"]):
        for b in t:
            add(b, TARGET, ("target", k), 11, True, 11)
    allp = set().union(*m["pieces"]) if m["pieces"] else set()
    refl = set()
    if A is not None:
        for r, c in allp:
            rr = 2 * A - r
            if r != A and 0 <= rr < N and (rr, c) not in allp:
                refl.add((rr, c))
    for b in refl:
        add(b, REFL, "refl", 4, False, 4)
    for p in m["pieces"]:
        for b in p:
            add(b, PIECE, "piece", 5, False, 0 if p == sel else -1)
    pix = {}
    for (r, c), st in stacks.items():
        st.sort(key=lambda s: -s[0])
        top = st[0]
        solid = next((s for s in st if s[3]), None)
        fill = next((s for s in st if s[4] >= 0), None)
        for i in range(3):
            for j in range(3):
                pos = (3 * r + i, 3 * c + j)
                if i == 1 and j == 1:
                    if solid:
                        pix[pos] = (solid[2], solid[1])
                    elif fill:
                        pix[pos] = (fill[4], fill[1])
                else:
                    pix[pos] = (top[2], top[1])
    return pix


def comps(cellset):
    cellset, out = set(cellset), []
    while cellset:
        stack, comp = [cellset.pop()], set()
        while stack:
            p = stack.pop()
            comp.add(p)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (p[0] + d[0], p[1] + d[1])
                if n in cellset:
                    cellset.remove(n)
                    stack.append(n)
        out.append(comp)
    return out


def make(name, typ, layer, tags, comp, pix):
    x0 = min(p[0] for p in comp)
    y0 = min(p[1] for p in comp)
    w = max(p[0] for p in comp) - x0 + 1
    h = max(p[1] for p in comp) - y0 + 1
    pixels = [[pix[(x0 + i, y0 + j)][0] if (x0 + i, y0 + j) in comp else -1 for j in range(h)] for i in range(w)]
    return {"name": name, "type": typ, "x": x0, "y": y0, "w": w, "h": h, "layer": layer,
            "tags": list(tags), "pixels": pixels}


def bbox_key(comp):
    return (min(p[0] for p in comp), min(p[1] for p in comp))


def extract(pix, others):
    out = [dict(o) for o in others]
    zeros = {p for p, v in pix.items() if v[0] == 0}
    for colour, prefix, typ, layer, tags in ((10, "wall_h_", "wall", WALL, WTAGS),
                                             (5, "piece_", "player", PIECE, PTAGS)):
        own = {p for p, v in pix.items() if v[0] == colour}
        for k, comp in enumerate(sorted(comps(own | zeros), key=bbox_key)):
            if comp & own:
                out.append(make(prefix + str(k), typ, layer, tags, comp, pix))
    refl = {p for p, v in pix.items() if v[0] == 4}
    for k, comp in enumerate(sorted(comps(refl), key=bbox_key)):
        out.append(make("reflection_" + str(k), "reflection", REFL, RTAGS, comp, pix))
    tcells = {}
    for p, v in pix.items():
        if v[0] == 11 and isinstance(v[1], tuple):
            tcells.setdefault(v[1][1], set()).add(p)
    for k, comp in enumerate(sorted(tcells.values(), key=bbox_key)):
        out.append(make("target_" + str(k), "target", TARGET, ["goal", "yellow"], comp, pix))
    return out


def transition_function(state, action):
    m = parse(state)
    m = step(m, action)
    return extract(render(m), m["others"])
