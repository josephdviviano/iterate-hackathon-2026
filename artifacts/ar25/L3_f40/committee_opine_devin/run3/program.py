# Mechanics: 21x21 board of 3x3 blocks (bg 9); horizontal axis band (ring 10, dots 0 when selected),
# solid targets 11, pieces ring 5 (centre 0 when selected), reflections ring 4/centre 4 at row 2A-r.
# A1/A2 move the selection one block row (axis or piece), A3/A4 move a selected piece one column
# (axis selected -> no-op); pieces are blocked by the board edge and other pieces. A5 cycles
# axis -> pieces by size desc -> axis. HUD col 63: one 11->12 per effective action. Unconfirmed: axis vs piece-row blocking.
N = 21
BG, AXIS, TGT, PIECE, REFL, SEL = 9, 10, 11, 5, 4, 0


def block(frame, bx, by):
    return [frame[3 * by + j][3 * bx + i] for j in range(3) for i in range(3)]


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        c = {st[0]}
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.discard(n)
                    c.add(n)
                    st.append(n)
        out.append(c)
    return out


def parse(frame):
    corner = {(bx, by): frame[3 * by][3 * bx] for bx in range(N) for by in range(N)}
    centre = {(bx, by): frame[3 * by + 1][3 * bx + 1] for bx in range(N) for by in range(N)}
    rows = [sum(1 for bx in range(N) if corner[(bx, by)] == AXIS) for by in range(N)]
    axis = max(range(N), key=lambda r: rows[r])
    targets = {b for b in corner if TGT in block(frame, *b)}
    pcells = {b for b, c in corner.items() if c == PIECE}
    axis_sel = any(corner[(bx, axis)] == AXIS and centre[(bx, axis)] == SEL for bx in range(N))
    pieces, sel = [], None
    if not axis_sel:
        seeds = {b for b in pcells if centre[b] == SEL}
        grown, st = set(seeds), list(seeds)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in pcells and n not in grown and centre[n] != BG:
                    grown.add(n)
                    st.append(n)
        if grown:
            for g in comps(grown):
                pieces.append(g)
            sel = 0
            pcells = pcells - grown
    pieces += comps(pcells)
    return {"axis": axis, "targets": targets, "pieces": pieces, "sel": "axis" if axis_sel else sel}


def size_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(m, action):
    a = action["action_id"] if isinstance(action, dict) else action
    sel, pieces = m["sel"], m["pieces"]
    if a == 5:
        order = ["axis"] + size_order(pieces)
        cur = order.index(sel) if sel in order else 0
        m["sel"] = order[(cur + 1) % len(order)]
        return m["sel"] != sel
    d = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(a)
    if d is None or sel is None:
        return False
    if sel == "axis":
        if d[0]:
            return False
        na = m["axis"] + d[1]
        if not 0 <= na < N:
            return False
        m["axis"] = na
        return True
    others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
    moved = {(x + d[0], y + d[1]) for x, y in pieces[sel]}
    if any(not (0 <= x < N and 0 <= y < N) for x, y in moved) or moved & others:
        return False
    pieces[sel] = moved
    return True


def render(m, frame):
    out = [row[:] for row in frame]
    A = m["axis"]
    pcell, selcells = set(), set()
    for i, p in enumerate(m["pieces"]):
        pcell |= p
        if m["sel"] == i:
            selcells |= p
    refl = set()
    for x, y in pcell:
        if y != A:
            r = 2 * A - y
            if 0 <= r < N and (x, r) not in pcell:
                refl.add((x, r))
    for bx in range(N):
        for by in range(N):
            b = (bx, by)
            is_t, is_a = b in m["targets"], by == A
            if b in pcell:
                ring = PIECE
            elif b in refl:
                ring = REFL
            elif is_t:
                ring = TGT
            elif is_a:
                ring = AXIS
            else:
                ring = BG
            if is_t:
                c = TGT
            elif b in selcells:
                c = SEL
            elif b in refl:
                c = REFL
            elif is_a and m["sel"] == "axis":
                c = SEL
            else:
                c = BG
            for j in range(3):
                for i in range(3):
                    out[3 * by + j][3 * bx + i] = ring
            out[3 * by + 1][3 * bx + 1] = c
    return out


def tick_hud(out):
    for y in range(63):
        if out[y][63] == TGT:
            out[y][63] = 12
            return


def transition_function(state, action, frame):
    m = parse(frame)
    changed = step(m, action)
    out = render(m, frame)
    if changed:
        tick_hud(out)
    return [[int(v) for v in row] for row in out]
