# Mechanics: two mirror axes (H wall row ah, V wall column av, in 3x3 blocks on a 63x63 board) + movable pieces.
# ACTION5 cycles selection H -> V -> pieces (by bbox row,col) -> H; selection shows as 0 hole centres.
# A1/A2 move the selected H axis or piece by one block row, A3/A4 the V axis or piece by one block column (bounds, no piece overlap).
# Pieces mirror across V as visible gray; across H / both as invisible occluders (whole piece invisible if it touches row ah).
# Frame is re-rendered by layer stacks and re-extracted (comps over own cells + 0 cells); hidden sprite/selection state is continuity-gated.
import json

NB = 21
N = 63
LAST = {"canon": None, "model": None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells(o):
    for i, row in enumerate(o.get("pixels", [])):
        for j, v in enumerate(row):
            if v != -1:
                yield (o["x"] + i, o["y"] + j), v


def comps(items, nb):
    items = set(items)
    out, seen = [], set()
    for s in sorted(items):
        if s in seen:
            continue
        seen.add(s)
        st, comp = [s], []
        while st:
            p = st.pop()
            comp.append(p)
            for q in nb(p):
                if q in items and q not in seen:
                    seen.add(q)
                    st.append(q)
        out.append(comp)
    return out


def nb4(p):
    r, c = p
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def parse(state):
    wall, play, targ = {}, {}, {}
    for o in state:
        d = {"wall": wall, "player": play, "target": targ}.get(o["type"])
        if d is not None:
            for p, v in cells(o):
                d[p] = v
    wb = {(r // 3, c // 3) for (r, c) in wall}
    ah = max(range(NB), key=lambda b: sum(1 for (br, _) in wb if br == b))
    av = max(range(NB), key=lambda b: sum(1 for (_, bc) in wb if bc == b))
    pb = {(r // 3, c // 3) for (r, c), v in play.items() if v == 5}
    sprites = [frozenset(s) for s in comps(pb, nb4)]
    wz = lambda br, bc: wall.get((3 * br + 1, 3 * bc + 1)) == 0
    sel = None
    if any(wz(ah, bc) for bc in range(NB) if bc != av):
        sel = "H"
    elif any(wz(br, av) for br in range(NB) if br != ah):
        sel = "V"
    else:
        for s in sprites:
            if any(play.get((3 * br + 1, 3 * bc + 1)) == 0 for br, bc in s):
                sel = s
    tb = {(r // 3, c // 3) for (r, c), v in targ.items() if v == 11}
    return {"ah": ah, "av": av, "sprites": sprites, "sel": sel, "targets": tb}


def order(sprites):
    return sorted(sprites, key=lambda s: (min(b[0] for b in s), min(b[1] for b in s)))


def inb(b):
    return 0 <= b[0] < NB and 0 <= b[1] < NB


def step(m, action):
    m = dict(m)
    sel = m["sel"]
    if action == 5:
        cyc = ["H", "V"] + order(m["sprites"])
        i = cyc.index(sel) if sel in cyc else -1
        m["sel"] = cyc[(i + 1) % len(cyc)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None:
        return m
    if sel == "H" and d[0]:
        if 0 <= m["ah"] + d[0] < NB:
            m["ah"] += d[0]
    elif sel == "V" and d[1]:
        if 0 <= m["av"] + d[1] < NB:
            m["av"] += d[1]
    elif isinstance(sel, frozenset):
        moved = frozenset((br + d[0], bc + d[1]) for br, bc in sel)
        others = set().union(*[s for s in m["sprites"] if s != sel]) if len(m["sprites"]) > 1 else set()
        if all(inb(b) for b in moved) and not (moved & others):
            m["sprites"] = [moved if s == sel else s for s in m["sprites"]]
            m["sel"] = moved
    return m


def render(m):
    ah, av, sel = m["ah"], m["av"], m["sel"]
    stacks = {}
    add = lambda b, spr: stacks.setdefault(b, []).append(spr)
    occ = set()
    for s in m["sprites"]:
        occ |= s
    vis, inv = set(), set()
    for s in m["sprites"]:
        on_h = any(br == ah for br, _ in s)
        for br, bc in s:
            add((br, bc), ("P", 5, None, 0 if s == sel else None))
            (inv if on_h else vis).add((br, 2 * av - bc))
            inv.add((2 * ah - br, bc))
            inv.add((2 * ah - br, 2 * av - bc))
    for b in sorted(vis - occ):
        if inb(b):
            add(b, ("R", 4, None, 4))
    for b in sorted(inv - occ):
        if inb(b):
            add(b, ("I", -1, None, None))
    for b in m["targets"]:
        add(b, ("T", 11, 11, None))
    for br in range(NB):
        add((br, av), ("V", 10, 0 if sel == "V" else None, None))
    for bc in range(NB):
        add((ah, bc), ("H", 10, 0 if sel == "H" else None, None))
    grid = {}
    for (br, bc), st in stacks.items():
        for i in range(3):
            for j in range(3):
                p = (3 * br + i, 3 * bc + j)
                if (i, j) != (1, 1):
                    grid[p] = (st[0][1], st[0][0])
                    continue
                solid = next((x for x in st if x[2] is not None), None)
                if solid:
                    grid[p] = (solid[2], solid[0])
                elif st[0][3] is not None:
                    grid[p] = (st[0][3], st[0][0])
                elif st[0][0] == "I":
                    grid[p] = (-1, "I")
    return grid


def bbox(cs):
    rs = [p[0] for p in cs]
    cs2 = [p[1] for p in cs]
    return min(rs), min(cs2), max(rs), max(cs2)


def mkobj(name, typ, layer, tags, pts, grid):
    r0, c0, r1, c1 = bbox(pts)
    px = [[-1] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
    for p in pts:
        px[p[0] - r0][p[1] - c0] = grid[p][0]
    return {"name": name, "type": typ, "x": r0, "y": c0, "w": r1 - r0 + 1, "h": c1 - c0 + 1,
            "layer": layer, "tags": tags, "pixels": px}


def ranked(cs):
    return sorted(cs, key=lambda c: bbox(c)[:2])


def extract(grid):
    out = []
    zeros = {p for p, (v, _) in grid.items() if v == 0}
    wallc = {p for p, (v, o) in grid.items() if o in ("H", "V") and v == 10}
    for i, c in enumerate(ranked(comps(wallc | zeros, nb4))):
        if any(p in wallc for p in c):
            r0, c0, r1, c1 = bbox(c)
            hz = (c1 - c0) > (r1 - r0)
            out.append(mkobj("wall_%s_%d" % ("h" if hz else "v", i), "wall", 1,
                             ["axis", "horizontal" if hz else "vertical"], c, grid))
    pc = {p for p, (v, o) in grid.items() if o == "P" and v == 5}
    for i, c in enumerate(ranked(comps(pc | zeros, nb4))):
        if any(p in pc for p in c):
            out.append(mkobj("piece_%d" % i, "player", 4, ["movable", "black"], c, grid))
    rc = {p for p, (v, o) in grid.items() if o in ("R", "I")}
    for i, c in enumerate(ranked(comps(rc, nb4))):
        v = [p for p in c if grid[p][0] == 4]
        if v:
            out.append(mkobj("reflection_%d" % i, "reflection", 3, ["mirror", "gray"], v, grid))
    tc = {p for p, (v, o) in grid.items() if o == "T" and v == 11}
    groups = [list(c) for c in comps(tc, nb4)]
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                A, B = bbox(groups[a]), bbox(groups[b])
                gr = max(0, B[0] - A[2], A[0] - B[2])
                gc = max(0, B[1] - A[3], A[1] - B[3])
                if max(gr, gc) <= 4:
                    groups[a] += groups.pop(b)
                    merged = True
                    break
            if merged:
                break
    for i, c in enumerate(ranked(groups)):
        out.append(mkobj("target_%d" % i, "target", 2, ["goal", "yellow"], c, grid))
    return out


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get("action_id")
    c = canon(state)
    m = LAST["model"] if LAST["canon"] == c and LAST["model"] else parse(state)
    m = step(m, action)
    out = extract(render(m))
    out += [dict(o) for o in state if o["type"] not in ("wall", "player", "reflection", "target")]
    LAST["canon"], LAST["model"] = canon(out), m
    return out
