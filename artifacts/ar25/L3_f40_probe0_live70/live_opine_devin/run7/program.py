# Mechanics: 21x21 grid of 3x3 blocks; one horizontal axis (wall ring 10), targets (solid 11), pieces (ring 5, holed),
# reflections (ring 4) = every off-axis piece block mirrored to row 2A-r unless off-board or on a piece block.
# A1/A2 move the selection up/down, A3/A4 left/right (axis ignores A3/A4); pieces blocked by board edge and other
# pieces only. A5 cycles axis -> pieces by size desc -> axis. HUD column 63: n effective actions -> rows <n are 12,
# rows < n-64 turn 5 (64-cell free reserve then budget drains). Hypothesis unconfirmed: size ties broken by (row,col).
B = 21
WALL, PIECE, REFL, TARGET, BG, SPENT, TICK, IDLE = 10, 5, 4, 11, 9, 5, 12, 11


def cell(frame, r, c, dy, dx):
    return frame[3 * r + dy][3 * c + dx]


def parse(frame):
    corner = {}
    targets = set()
    for r in range(B):
        for c in range(B):
            corner[(r, c)] = cell(frame, r, c, 0, 0)
            if any(cell(frame, r, c, dy, dx) == TARGET for dy in range(3) for dx in range(3)):
                targets.add((r, c))
    counts = [sum(1 for c in range(B) if corner[(r, c)] == WALL) for r in range(B)]
    axis = max(range(B), key=lambda r: counts[r])
    axis_sel = any(corner[(axis, c)] == WALL and cell(frame, axis, c, 1, 1) == 0 for c in range(B))
    pblocks = {k for k, v in corner.items() if v == PIECE}
    centre = {k: cell(frame, k[0], k[1], 1, 1) for k in pblocks}
    sel = set()
    if not axis_sel:
        stack = [k for k in pblocks if centre[k] == 0]
        sel = set(stack)
        while stack:
            r, c = stack.pop()
            for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if n in pblocks and n not in sel and centre[n] != BG:
                    sel.add(n)
                    stack.append(n)
    pieces = []
    if sel:
        pieces.append(frozenset(sel))
    rest = pblocks - sel
    while rest:
        seed = min(rest)
        comp, stack = {seed}, [seed]
        while stack:
            r, c = stack.pop()
            for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if n in rest and n not in comp:
                    comp.add(n)
                    stack.append(n)
        rest -= comp
        pieces.append(frozenset(comp))
    selected = -1 if axis_sel else (0 if sel else -1)
    return {"axis": axis, "targets": targets, "pieces": pieces, "sel": selected}


def order_key(p):
    return (-len(p), min(p))


def step(m, action):
    pieces = list(m["pieces"])
    axis, sel = m["axis"], m["sel"]
    if action == 5:
        ordered = sorted(range(len(pieces)), key=lambda i: order_key(pieces[i]))
        cycle = [-1] + ordered
        sel = cycle[(cycle.index(sel) + 1) % len(cycle)] if sel in cycle else -1
    elif action in (1, 2, 3, 4):
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
        if sel == -1:
            if dr and 0 <= axis + dr < B:
                axis += dr
        else:
            moved = frozenset((r + dr, c + dc) for r, c in pieces[sel])
            others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
            if all(0 <= r < B and 0 <= c < B for r, c in moved) and not (moved & others):
                pieces[sel] = moved
    return {"axis": axis, "targets": m["targets"], "pieces": pieces, "sel": sel}


def same(a, b):
    return (a["axis"] == b["axis"] and a["sel"] == b["sel"]
            and [set(p) for p in a["pieces"]] == [set(p) for p in b["pieces"]])


def reflections(m):
    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    out = set()
    a = m["axis"]
    for p in m["pieces"]:
        for r, c in p:
            d = 2 * a - r
            if r != a and 0 <= d < B and (d, c) not in occ:
                out.add((d, c))
    return out


def render(m, frame):
    out = [list(row) for row in frame]
    refl = reflections(m)
    owner = {}
    for i, p in enumerate(m["pieces"]):
        for k in p:
            owner[k] = i
    for r in range(B):
        for c in range(B):
            k = (r, c)
            # layers top-down: (ring colour, centre solid?, centre fill)
            layers = []
            if k in owner:
                layers.append((PIECE, False, 0 if owner[k] == m["sel"] else None))
            if k in refl:
                layers.append((REFL, False, REFL))
            if k in m["targets"]:
                layers.append((TARGET, True, TARGET))
            if r == m["axis"]:
                layers.append((WALL, m["sel"] == -1, 0 if m["sel"] == -1 else None))
            ring = layers[0][0] if layers else BG
            solid = [f for _, s, f in layers if s]
            fills = [f for _, s, f in layers if f is not None]
            ctr = solid[0] if solid else (fills[0] if fills else BG)
            for dy in range(3):
                for dx in range(3):
                    out[3 * r + dy][3 * c + dx] = ctr if (dy, dx) == (1, 1) else ring
    return out


def hud_count(frame):
    col = [frame[y][63] for y in range(64)]
    return col.count(TICK) + 2 * sum(1 for v in col if v == SPENT)


def draw_hud(out, n):
    for y in range(64):
        out[y][63] = SPENT if y < n - 64 else (TICK if y < n else IDLE)


def transition_function(state, action, frame):
    aid = action.get("action_id") if isinstance(action, dict) else action
    m = parse(frame)
    m2 = step(m, aid)
    if same(m, m2):
        return [[int(v) for v in row] for row in frame]
    out = render(m2, frame)
    draw_hud(out, hud_count(frame) + 1)
    return [[int(v) for v in row] for row in out]

