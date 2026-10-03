# Mechanics: 21x21 board of 3x3 blocks (bg 9); one horizontal axis row (ring 10, centre 0 when
# selected), solid targets 11, pieces ring 5 (centre 0 when selected, transparent otherwise),
# reflections ring 4 / centre 4 at row 2A-r of every off-axis piece block (skip off-board/piece cells).
# A1/A2 move selection one block up/down, A3/A4 move a selected piece left/right (blocked by edge or
# other pieces; axis ignores A3/A4), A5 cycles axis -> pieces by size desc -> axis. Column 63 HUD: one
# 11 -> 12 from the top per effective action. Unconfirmed: whether the axis is blocked by piece rows.
N = 21
AXIS, TARGET, PIECE, REFL, BG, DOT, HUD_OFF, HUD_ON = 10, 11, 5, 4, 9, 0, 11, 12


def ring(frame, c, r):
    return frame[3 * r][3 * c]


def centre(frame, c, r):
    return frame[3 * r + 1][3 * c + 1]


def comps(cells):
    cells, out = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            c, r = stack.pop()
            comp.add((c, r))
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c + d[0], r + d[1])
                if n in cells:
                    cells.discard(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(frame):
    axis_count = [0] * N
    targets, piece_cells, dots = set(), set(), set()
    axis_sel = False
    for r in range(N):
        for c in range(N):
            rg, ce = ring(frame, c, r), centre(frame, c, r)
            if rg == AXIS:
                axis_count[r] += 1
                if ce == DOT:
                    axis_sel = True
            if rg == TARGET or ce == TARGET:
                targets.add((c, r))
            if rg == PIECE:
                piece_cells.add((c, r))
                if ce == DOT:
                    dots.add((c, r))
    axis = max(range(N), key=lambda r: axis_count[r])
    pieces, sel = [], None
    if dots:
        grown, stack = set(dots), list(dots)
        while stack:
            c, r = stack.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c + d[0], r + d[1])
                if n in piece_cells and n not in grown and centre(frame, *n) != BG:
                    grown.add(n)
                    stack.append(n)
        pieces.append(grown)
        sel = 0
        piece_cells -= grown
    pieces += comps(piece_cells)
    if axis_sel:
        sel = 'axis'
    return {'axis': axis, 'targets': targets, 'pieces': pieces, 'sel': sel}


def piece_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min((r, c) for c, r in pieces[i])))


def step(m, action):
    """Return True when the action changed the model."""
    sel = m['sel']
    if action == 5:
        order = piece_order(m['pieces'])
        if sel == 'axis' or sel is None:
            m['sel'] = order[0] if order else 'axis'
        else:
            k = order.index(sel)
            m['sel'] = order[k + 1] if k + 1 < len(order) else 'axis'
        return m['sel'] != sel
    dc, dr = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(action, (0, 0))
    if (dc, dr) == (0, 0) or sel is None:
        return False
    if sel == 'axis':
        if dr == 0:
            return False
        a = m['axis'] + dr
        if not 0 <= a < N:
            return False
        m['axis'] = a
        return True
    others = set().union(*[p for i, p in enumerate(m['pieces']) if i != sel]) if len(m['pieces']) > 1 else set()
    moved = {(c + dc, r + dr) for c, r in m['pieces'][sel]}
    if any(not (0 <= c < N and 0 <= r < N) or (c, r) in others for c, r in moved):
        return False
    m['pieces'][sel] = moved
    return True


def reflections(m):
    a = m['axis']
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    for c, r in occupied:
        if r == a:
            continue
        d = (c, 2 * a - r)
        if 0 <= d[1] < N and d not in occupied:
            out.add(d)
    return out


def render(m, frame):
    out = [row[:] for row in frame]
    refl = reflections(m)
    sel_cells = m['pieces'][m['sel']] if isinstance(m['sel'], int) else set()
    all_pieces = set().union(*m['pieces']) if m['pieces'] else set()
    for r in range(N):
        for c in range(N):
            # layers top-down: (ring colour, centre solid?, centre fill)
            layers = []
            if (c, r) in all_pieces:
                layers.append((PIECE, None, DOT if (c, r) in sel_cells else None))
            if (c, r) in refl:
                layers.append((REFL, None, REFL))
            if (c, r) in m['targets']:
                layers.append((TARGET, TARGET, TARGET))
            if r == m['axis']:
                layers.append((AXIS, DOT if m['sel'] == 'axis' else None, None))
            rg = layers[0][0] if layers else BG
            ce = next((s for _, s, _ in layers if s is not None), None)
            if ce is None:
                ce = next((f for _, _, f in layers if f is not None), BG)
            for y in range(3):
                for x in range(3):
                    out[3 * r + y][3 * c + x] = ce if (x, y) == (1, 1) else rg
    return out


def tick_hud(out):
    for y in range(63):
        if out[y][63] == HUD_OFF:
            out[y][63] = HUD_ON
            return


def transition_function(state, action, frame):
    a = action if isinstance(action, int) else 6
    m = parse(frame)
    changed = step(m, a)
    out = render(m, frame)
    if changed:
        tick_hud(out)
    return out
