# Mechanics: 21x21 grid of 3x3 blocks; axis = horizontal band (ring 10), pieces ring 5, targets solid 11, bg 9.
# Selection: axis (dot 0) or one piece (hole 0). A1/A2 move selection one block row, A3/A4 move a selected piece
# one column (axis: no-op); pieces are blocked by the board edge / other pieces. A5 cycles axis -> pieces by size desc.
# Every piece block off the axis reflects to row 2A-r (ring 4, hole 4) unless off-board or under a piece.
# HUD col 63: one more 11->12 from the top per effective action. Unconfirmed: whether the axis is blocked by pieces.
N = 21
BG, AXIS, PIECE, REFL, TGT = 9, 10, 5, 4, 11


def parse(frame):
    corner = {}
    centre = {}
    for r in range(N):
        for c in range(N):
            corner[(c, r)] = frame[3 * r][3 * c]
            centre[(c, r)] = frame[3 * r + 1][3 * c + 1]
    targets = {b for b in corner if corner[b] == TGT or centre[b] == TGT}
    rows = [sum(1 for c in range(N) if corner[(c, r)] == AXIS) for r in range(N)]
    axis = max(range(N), key=lambda r: rows[r])
    axis_sel = any(corner[(c, axis)] == AXIS and centre[(c, axis)] == 0 for c in range(N))
    pblocks = {b for b in corner if corner[b] == PIECE}
    sel = set()
    if not axis_sel:
        stack = [b for b in pblocks if centre[b] == 0]
        sel = set(stack)
        while stack:
            c, r = stack.pop()
            for nb in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
                if nb in pblocks and nb not in sel and centre[nb] != BG:
                    sel.add(nb)
                    stack.append(nb)
    pieces = [frozenset(sel)] if sel else []
    rest = pblocks - sel
    while rest:
        stack = [rest.pop()]
        comp = set(stack)
        while stack:
            c, r = stack.pop()
            for nb in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
                if nb in rest:
                    rest.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        pieces.append(frozenset(comp))
    selected = None if axis_sel else (0 if sel else None)
    return {"axis": axis, "targets": targets, "pieces": pieces, "sel": selected}


def cycle_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min((r, c) for c, r in pieces[i])))


def on_board(blocks):
    return all(0 <= c < N and 0 <= r < N for c, r in blocks)


def step(m, action):
    m = dict(m, pieces=list(m["pieces"]))
    sel = m["sel"]
    if action == 5:
        order = cycle_order(m["pieces"])
        if sel is None:
            m["sel"] = order[0] if order else None
        else:
            k = order.index(sel)
            m["sel"] = order[k + 1] if k + 1 < len(order) else None
        return m, m["sel"] != sel
    d = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(action)
    if d is None:
        return m, False
    dx, dy = d
    if sel is None:
        if dx:
            return m, False
        a = m["axis"] + dy
        if not 0 <= a < N:
            return m, False
        m["axis"] = a
        return m, True
    moved = frozenset((c + dx, r + dy) for c, r in m["pieces"][sel])
    others = set().union(*[p for i, p in enumerate(m["pieces"]) if i != sel]) if len(m["pieces"]) > 1 else set()
    if not on_board(moved) or moved & others:
        return m, False
    m["pieces"][sel] = moved
    return m, True


def reflections(m):
    a = m["axis"]
    occupied = set().union(*m["pieces"]) if m["pieces"] else set()
    out = set()
    for p in m["pieces"]:
        for c, r in p:
            if r == a:
                continue
            dst = (c, 2 * a - r)
            if 0 <= dst[1] < N and dst not in occupied:
                out.add(dst)
    return out


def render(m, frame):
    out = [list(row) for row in frame]
    refl = reflections(m)
    owner = {}
    for i, p in enumerate(m["pieces"]):
        for b in p:
            owner[b] = i
    for r in range(N):
        for c in range(N):
            b = (c, r)
            layers = []  # (ring, fill or None, solid) top-down
            if b in owner:
                layers.append((PIECE, 0 if m["sel"] == owner[b] else None, False))
            if b in refl:
                layers.append((REFL, REFL, False))
            if b in m["targets"]:
                layers.append((TGT, TGT, True))
            if r == m["axis"]:
                layers.append((AXIS, 0 if m["sel"] is None else None, False))
            ring = layers[0][0] if layers else BG
            if b in m["targets"]:
                mid = TGT
            else:
                mid = next((f for _, f, _ in layers if f is not None), BG)
            for y in range(3):
                for x in range(3):
                    out[3 * r + y][3 * c + x] = mid if (x, y) == (1, 1) else ring
    return out


def tick_hud(out):
    for y in range(63):
        if out[y][63] != 12:
            out[y][63] = 12
            return


def transition_function(state, action, frame):
    a = action.get("action_id") if isinstance(action, dict) else action
    m = parse(frame)
    m2, changed = step(m, a)
    if not changed:
        return [list(row) for row in frame]
    out = render(m2, frame)
    tick_hud(out)
    return [[int(v) for v in row] for row in out]

