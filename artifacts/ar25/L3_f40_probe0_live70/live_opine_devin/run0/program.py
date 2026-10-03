# Mechanics: 21x21 grid of 3x3 blocks; one horizontal mirror axis (ring 10), pieces (ring 5), targets (solid 11).
# Selection = axis (0 dots) or a piece (0 centres); A1/A2 move selection one block row, A3/A4 one block col
# (axis: rows only), blocked by board edge / other pieces; A5 cycles axis -> pieces by size desc -> axis.
# Reflections (solid 4) of piece blocks at row 2A-r, skipping on-axis sources, off-board and piece-covered dests.
# HUD col 63: n effective actions -> rows < n-64 are 5, rows < n are 12, else 11. Unconfirmed: piece tie order.
N = 21
BG, WALL, PIECE, TARGET, REFL, SEL = 9, 10, 5, 11, 4, 0


def blk(frame, r, c):
    return [frame[3 * r + i][3 * c + j] for i in range(3) for j in range(3)]


def comps(cells, link):
    left, out = set(cells), []
    while left:
        seed = min(left)
        stack, comp = [seed], {seed}
        left.discard(seed)
        while stack:
            r, c = stack.pop()
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in left and link(q):
                    left.discard(q)
                    comp.add(q)
                    stack.append(q)
        out.append(comp)
    return out


def parse(frame):
    targets, pieceblk, centre, wallrow = set(), set(), {}, {}
    for r in range(N):
        for c in range(N):
            b = blk(frame, r, c)
            if TARGET in b:
                targets.add((r, c))
            centre[(r, c)] = b[4]
            if b[0] == PIECE:
                pieceblk.add((r, c))
            if b[0] == WALL:
                wallrow[r] = wallrow.get(r, 0) + 1
    axis = max(wallrow, key=lambda r: (wallrow[r], -r)) if wallrow else None
    axis_sel = axis is not None and any(
        blk(frame, axis, c)[0] == WALL and centre[(axis, c)] == SEL for c in range(N))
    pieces, sel = [], None
    zero = {p for p in pieceblk if centre[p] == SEL}
    rest = set(pieceblk)
    if zero:
        grown = set()
        for comp in comps(zero | (pieceblk - {p for p in pieceblk if centre[p] == BG}), lambda q: True):
            if comp & zero:
                grown |= comp
        rest -= grown
        if not axis_sel:
            sel = frozenset(grown)
            pieces.append(sel)
    for comp in comps(rest, lambda q: True):
        pieces.append(frozenset(comp))
    return {"axis": axis, "axis_sel": axis_sel, "targets": targets, "pieces": pieces, "sel": sel}


def order(pieces):
    return sorted(pieces, key=lambda p: (-len(p), min(p)))


def step(m, action):
    pieces, sel = list(m["pieces"]), m["sel"]
    if action == 5:
        o = order(pieces)
        if m["axis_sel"] or sel is None:
            if o:
                return dict(m, axis_sel=False, sel=o[0]), True
            return m, False
        i = o.index(sel)
        if i + 1 < len(o):
            return dict(m, sel=o[i + 1]), True
        return dict(m, axis_sel=m["axis"] is not None, sel=None), m["axis"] is not None
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None:
        return m, False
    dr, dc = d
    if m["axis_sel"]:
        if dc or m["axis"] is None:
            return m, False
        a = m["axis"] + dr
        if not 0 <= a < N:
            return m, False
        return dict(m, axis=a), True
    if sel is None:
        return m, False
    moved = frozenset((r + dr, c + dc) for r, c in sel)
    if any(not (0 <= r < N and 0 <= c < N) for r, c in moved):
        return m, False
    others = set().union(*[p for p in pieces if p != sel]) if len(pieces) > 1 else set()
    if moved & others:
        return m, False
    pieces = [moved if p == sel else p for p in pieces]
    return dict(m, pieces=pieces, sel=moved), True


def reflections(m):
    a = m["axis"]
    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    out = set()
    if a is None:
        return out
    for r, c in occ:
        if r == a:
            continue
        q = (2 * a - r, c)
        if 0 <= q[0] < N and q not in occ:
            out.add(q)
    return out


def render(m, frame):
    out = [list(row) for row in frame]
    refl = reflections(m)
    selset = m["sel"] or frozenset()
    occ = set().union(*m["pieces"]) if m["pieces"] else set()
    for r in range(N):
        for c in range(N):
            q = (r, c)
            wall = r == m["axis"]
            if q in occ:
                ring = PIECE
            elif q in refl:
                ring = REFL
            elif q in m["targets"]:
                ring = TARGET
            elif wall:
                ring = WALL
            else:
                ring = BG
            if q in m["targets"]:
                cen = TARGET
            elif wall and m["axis_sel"]:
                cen = SEL
            elif q in selset:
                cen = SEL
            elif q in refl:
                cen = REFL
            else:
                cen = BG
            for i in range(3):
                for j in range(3):
                    out[3 * r + i][3 * c + j] = cen if (i, j) == (1, 1) else ring
    return out


def hud_count(frame):
    col = [frame[y][63] for y in range(64)]
    if TARGET in col:
        return col.count(12)
    return 64 + col.count(PIECE)


def draw_hud(out, n):
    for y in range(64):
        out[y][63] = PIECE if y < n - 64 else (12 if y < n else TARGET)


def transition_function(state, action, frame):
    aid = action.get("action_id") if isinstance(action, dict) else action
    m = parse(frame)
    m2, changed = step(m, aid)
    out = render(m2, frame)
    n = hud_count(frame) + (1 if changed else 0)
    draw_hud(out, n)
    return [[int(v) for v in row] for row in out]
