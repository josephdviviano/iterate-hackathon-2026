# Mechanics: one horizontal mirror axis (wall row block A) + 3x3-block pieces, static 3x3-block targets.
# ACTION5 cycles selection axis -> pieces (by min block) -> axis; ACTION1/2 move selection x-/+3, ACTION3/4 y-/+3
# (axis ignores 3/4); moves blocked by board bounds (blocks 0..20) and piece overlap. Reflections: block b -> 2A-b,
# skipped on piece blocks / off board. Render layers wall1<target2<reflection3<piece4 (holes: first solid below, else
# fill 0 if selected / 4 for reflections) and re-extract; wall/player names rank among comps + other 0 pixels. Unconfirmed: piece cycle order.
import json

N = 21
WALL, PIECE, REFL = 10, 5, 4
_memo = {"out": None, "model": None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells_of(o):
    out = {}
    for i, row in enumerate(o.get("pixels", [])):
        for j, v in enumerate(row):
            if v != -1:
                out[(o["x"] + i, o["y"] + j)] = v
    return out


def parse(state):
    walls = [o for o in state if o["type"] == "wall"]
    players = [o for o in state if o["type"] == "player"]
    targets = [o for o in state if o["type"] == "target"]
    lower = {}
    for o in state:
        if o["type"] in ("target", "reflection"):
            lower.update(cells_of(o))
    axis = walls[0]["x"] // 3 if walls else 0
    wall_sel = any(v == 0 for o in walls for v in cells_of(o).values())
    tblocks = []
    for o in targets:
        c = cells_of(o)
        col = next(iter(c.values()), 11)
        tblocks.append((frozenset((x // 3, y // 3) for (x, y) in c), col))
    pieces, sel = [], None
    for o in players:
        c = cells_of(o)
        blocks = {(x // 3, y // 3) for (x, y), v in c.items() if v == PIECE}
        state_of = {}
        for b in blocks:
            ctr = (b[0] * 3 + 1, b[1] * 3 + 1)
            if c.get(ctr) == 0:
                state_of[b] = "sel"
            elif ctr in lower:
                state_of[b] = "amb"
            else:
                state_of[b] = "uns"
        selb = {b for b in blocks if state_of[b] == "sel"}
        grow = list(selb)
        while grow:
            bx, by = grow.pop()
            for nb in ((bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)):
                if nb in blocks and nb not in selb and state_of[nb] == "amb":
                    selb.add(nb)
                    grow.append(nb)
        if selb:
            pieces.append(frozenset(selb))
            sel = frozenset(selb)
        rest = blocks - selb
        while rest:
            seed = rest.pop()
            comp, st = {seed}, [seed]
            while st:
                bx, by = st.pop()
                for nb in ((bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)):
                    if nb in rest:
                        rest.discard(nb)
                        comp.add(nb)
                        st.append(nb)
            pieces.append(frozenset(comp))
    pieces.sort(key=lambda p: min(p))
    if wall_sel or sel is None:
        s = "axis"
    else:
        s = pieces.index(sel)
    return {"axis": axis, "pieces": pieces, "sel": s, "targets": tblocks}


def step(m, action):
    m = dict(m)
    pieces = list(m["pieces"])
    if action == 5:
        if m["sel"] == "axis":
            m["sel"] = 0 if pieces else "axis"
        else:
            m["sel"] = m["sel"] + 1 if m["sel"] + 1 < len(pieces) else "axis"
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action if isinstance(action, int) else 0)
    if d is None:
        return m
    if m["sel"] == "axis":
        if d[0] and 0 <= m["axis"] + d[0] < N:
            m["axis"] += d[0]
        return m
    k = m["sel"]
    moved = frozenset((x + d[0], y + d[1]) for x, y in pieces[k])
    others = set().union(*[p for i, p in enumerate(pieces) if i != k]) if len(pieces) > 1 else set()
    if all(0 <= x < N and 0 <= y < N for x, y in moved) and not (moved & others):
        pieces[k] = moved
    m["pieces"] = pieces
    return m


def render(m):
    stacks = {}

    def put(b, layer, owner, frame, fill, solid_centre=False, sid=None):
        for i in range(3):
            for j in range(3):
                cell = (b[0] * 3 + i, b[1] * 3 + j)
                centre = i == 1 and j == 1
                if centre and not solid_centre:
                    stacks.setdefault(cell, []).append((layer, owner, None, fill, sid))
                else:
                    stacks.setdefault(cell, []).append((layer, owner, frame, None, sid))

    A = m["axis"]
    for by in range(N):
        put((A, by), 1, "wall", WALL, 0 if m["sel"] == "axis" else -1)
    for t, (blocks, col) in enumerate(m["targets"]):
        for b in blocks:
            put(b, 2, "target", col, None, True, t)
    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    for p in m["pieces"]:
        for bx, by in p:
            rb = (2 * A - bx, by)
            if 0 <= rb[0] < N and rb not in occ:
                put(rb, 3, "reflection", REFL, REFL)
    for k, p in enumerate(m["pieces"]):
        for b in p:
            put(b, 4, "player", PIECE, 0 if m["sel"] == k else -1)
    disp = {}
    for cell, st in stacks.items():
        st.sort(key=lambda e: -e[0])
        hit = next((e for e in st if e[2] is not None), None)
        if hit:
            disp[cell] = (hit[2], hit[1], hit[4])
            continue
        hit = next((e for e in st if e[3] is not None and e[3] >= 0), None)
        if hit:
            disp[cell] = (hit[3], hit[1], hit[4])
    return disp


def comps(cells):
    cells = set(cells)
    out = []
    while cells:
        seed = cells.pop()
        comp, st = {seed}, [seed]
        while st:
            x, y = st.pop()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in cells:
                    cells.discard(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    return out


def make_obj(name, typ, tags, layer, cells, disp):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for (x, y) in cells:
        pix[x - x0][y - y0] = disp[(x, y)][0]
    return {"name": name, "type": typ, "tags": tags, "layer": layer,
            "x": x0, "y": y0, "w": w, "h": h, "pixels": pix}


def bbox_key(c):
    return (min(x for x, _ in c), min(y for _, y in c))


def extract(disp, scenery):
    out = [dict(o) for o in scenery]
    zeros = {c for c, v in disp.items() if v[0] == 0}
    for typ, prefix, tags, layer in (("wall", "wall_h_", ["axis", "horizontal"], 1),
                                     ("player", "piece_", ["movable", "black"], 4)):
        own = {c for c, v in disp.items() if v[1] == typ}
        cs = sorted(comps(own | zeros), key=bbox_key)
        for r, c in enumerate(cs):
            if c & own:
                out.append(make_obj(prefix + str(r), typ, tags, layer, c, disp))
    refl = {c for c, v in disp.items() if v[1] == "reflection"}
    for r, c in enumerate(sorted(comps(refl), key=bbox_key)):
        out.append(make_obj("reflection_" + str(r), "reflection", ["mirror", "gray"], 3, c, disp))
    tg = {}
    for c, v in disp.items():
        if v[1] == "target":
            tg.setdefault(v[2], set()).add(c)
    for r, c in enumerate(sorted(tg.values(), key=bbox_key)):
        out.append(make_obj("target_" + str(r), "target", ["goal", "yellow"], 2, c, disp))
    return out


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get("action_id")
    if _memo["out"] is not None and canon(state) == _memo["out"]:
        model = _memo["model"]
    else:
        model = parse(state)
    scenery = [o for o in state if o["type"] not in ("wall", "player", "reflection", "target")]
    new = step(model, action)
    out = extract(render(new), scenery)
    _memo["out"], _memo["model"] = canon(out), new
    return out
