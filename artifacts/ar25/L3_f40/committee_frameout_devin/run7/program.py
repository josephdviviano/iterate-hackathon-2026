# Mechanics: 21x21 board of 3x3 blocks (bg 9, axis band 10, targets 11 solid, reflections 4, pieces 5 holed).
# Selection is the axis (dots 0) or one piece (centres 0). A1/A2 move the selection one block row (axis: board
# bounds only); A3/A4 move a selected piece one block column (blocked by edge/other pieces; axis A3/A4 no-op).
# A5 cycles axis -> pieces by size desc -> axis. Reflection of piece block row r = 2A-r (skip axis row, off-board,
# piece cells). HUD col 63: one more 12 from the top per effective action. Hidden-state hypothesis unconfirmed: none needed.
B, N = 3, 21
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def block(frame, c, r):
    return [[frame[r * B + j][c * B + i] for i in range(B)] for j in range(B)]


def parse(frame):
    walls, targets, pblocks, centre = set(), set(), set(), {}
    for r in range(N):
        for c in range(N):
            b = block(frame, c, r)
            ring, ctr = b[0][0], b[1][1]
            centre[(c, r)] = ctr
            if ring == 11 or ctr == 11:
                targets.add((c, r))
            if ring == 10:
                walls.add((c, r))
            elif ring == 5:
                pblocks.add((c, r))
    rows = {}
    for c, r in walls:
        rows[r] = rows.get(r, 0) + 1
    axis = max(rows, key=lambda k: rows[k]) if rows else None
    axis_sel = any(centre[p] == 0 for p in walls)
    sel_seed = {p for p in pblocks if centre[p] == 0}
    pieces, sel = [], None
    if sel_seed:
        grown, stack = set(sel_seed), list(sel_seed)
        while stack:
            c, r = stack.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (c + d[0], r + d[1])
                if q in pblocks and q not in grown and centre[q] != 9:
                    grown.add(q)
                    stack.append(q)
        pieces.append(grown)
        sel = 0
    rest = pblocks - set().union(*pieces) if pieces else set(pblocks)
    while rest:
        seed = min(rest)
        comp, stack = {seed}, [seed]
        while stack:
            c, r = stack.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (c + d[0], r + d[1])
                if q in rest and q not in comp:
                    comp.add(q)
                    stack.append(q)
        rest -= comp
        pieces.append(comp)
    if axis_sel:
        sel = 'axis'
    return {'axis': axis, 'targets': targets, 'pieces': pieces, 'sel': sel}


def piece_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min((r, c) for c, r in pieces[i])))


def step(m, action):
    if not isinstance(action, int):
        return False
    sel, pieces = m['sel'], m['pieces']
    if action == 5:
        order = piece_order(pieces)
        if sel == 'axis' or sel is None:
            m['sel'] = order[0] if order else 'axis'
        else:
            k = order.index(sel)
            m['sel'] = order[k + 1] if k + 1 < len(order) else ('axis' if m['axis'] is not None else order[0])
        return m['sel'] != sel
    if action not in DIRS:
        return False
    dx, dy = DIRS[action]
    if sel == 'axis':
        if dx or m['axis'] is None:
            return False
        a = m['axis'] + dy
        if not 0 <= a < N:
            return False
        m['axis'] = a
        return True
    if sel is None:
        return False
    others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
    moved = {(c + dx, r + dy) for c, r in pieces[sel]}
    if any(not (0 <= c < N and 0 <= r < N) or (c, r) in others for c, r in moved):
        return False
    pieces[sel] = moved
    return True


def reflections(m):
    a = m['axis']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    if a is None:
        return out
    for c, r in occ:
        if r == a:
            continue
        q = (c, 2 * a - r)
        if 0 <= q[1] < N and q not in occ:
            out.add(q)
    return out


def render(m, frame):
    out = [row[:] for row in frame]
    refl = reflections(m)
    sel_cells = m['pieces'][m['sel']] if isinstance(m['sel'], int) else set()
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    for r in range(N):
        for c in range(N):
            p = (c, r)
            wall = m['axis'] == r
            if p in occ:
                ring = 5
            elif p in refl:
                ring = 4
            elif p in m['targets']:
                ring = 11
            elif wall:
                ring = 10
            else:
                ring = 9
            if p in m['targets']:
                ctr = 11
            elif p in sel_cells:
                ctr = 0
            elif p in occ:
                ctr = 9
            elif p in refl:
                ctr = 4
            elif wall and m['sel'] == 'axis':
                ctr = 0
            else:
                ctr = 9
            for j in range(B):
                for i in range(B):
                    out[r * B + j][c * B + i] = ctr if (i, j) == (1, 1) else ring
    return out


def tick_hud(out):
    for y in range(N * B):
        if out[y][63] != 12:
            out[y][63] = 12
            return


def transition_function(state, action, frame):
    m = parse(frame)
    changed = step(m, action)
    out = render(m, frame)
    if changed:
        tick_hud(out)
    return out
