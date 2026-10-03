# Mechanics: 21x21 board of 3x3 blocks (bg 9); horizontal axis band (ring 10, dot 0 when selected),
# solid targets 11, pieces ring 5 (centre 0 selected, target 11 shows through holes), reflections
# ring 4 at block row 2A-r (skip on-axis sources, off-board and piece-occupied dests). A1/A2 move the
# selection one block row, A3/A4 move a selected piece one column (blocked by board edge / other pieces);
# A5 cycles axis -> pieces by size desc -> axis. HUD col 63: one 11->12 per state-changing action.
# Unconfirmed: whether pieces may enter the axis row (assumed allowed); clicks/A7 are treated as no-ops.
N = 21
BG, AXIS, TARGET, REFL, PIECE, DOT = 9, 10, 11, 4, 5, 0
HUD_ON = 12


def corner(frame, c, r):
    return frame[3 * r][3 * c]


def centre(frame, c, r):
    return frame[3 * r + 1][3 * c + 1]


def components(cells, grow_ok=None):
    cells = set(cells)
    comps = []
    while cells:
        start = min(cells)
        comp, stack = {start}, [start]
        cells.discard(start)
        while stack:
            c, r = stack.pop()
            for n in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def grow(seeds, pool, ok):
    got, stack = set(seeds), list(seeds)
    while stack:
        c, r = stack.pop()
        for n in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
            if n in pool and n not in got and ok(n):
                got.add(n)
                stack.append(n)
    return got


def parse(frame):
    counts = [sum(1 for c in range(N) if corner(frame, c, r) == AXIS) for r in range(N)]
    axis = max(range(N), key=lambda r: counts[r])
    targets = {(c, r) for r in range(N) for c in range(N)
               if TARGET in (corner(frame, c, r), centre(frame, c, r))}
    pblocks = {(c, r) for r in range(N) for c in range(N) if corner(frame, c, r) == PIECE}
    axis_sel = any(corner(frame, c, axis) == AXIS and centre(frame, c, axis) == DOT for c in range(N))
    sel_piece = set()
    if not axis_sel:
        seeds = {b for b in pblocks if centre(frame, *b) == DOT}
        sel_piece = grow(seeds, pblocks, lambda b: centre(frame, *b) != BG) if seeds else set()
    pieces = ([frozenset(sel_piece)] if sel_piece else []) + \
        [frozenset(p) for p in components(pblocks - sel_piece)]
    if axis_sel:
        sel = 'axis'
    elif sel_piece:
        sel = 0
    else:
        sel = None
    return {'axis': axis, 'targets': targets, 'pieces': pieces, 'sel': sel}


def piece_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min((r, c) for c, r in pieces[i])))


def move_piece(m, i, dc, dr):
    moved = frozenset((c + dc, r + dr) for c, r in m['pieces'][i])
    others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i]) if len(m['pieces']) > 1 else set()
    if any(not (0 <= c < N and 0 <= r < N) for c, r in moved) or moved & others:
        return False
    m['pieces'][i] = moved
    return True


def step(m, action):
    sel = m['sel']
    if not isinstance(action, int):
        return False
    if action == 5:
        order = piece_order(m['pieces'])
        if sel == 'axis' or sel is None:
            new = order[0] if order else 'axis'
        else:
            k = order.index(sel)
            new = order[k + 1] if k + 1 < len(order) else 'axis'
        if new == sel:
            return False
        m['sel'] = new
        return True
    moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    if action not in moves or sel is None:
        return False
    dc, dr = moves[action]
    if sel == 'axis':
        if dc or not 0 <= m['axis'] + dr < N:
            return False
        m['axis'] += dr
        return True
    return move_piece(m, sel, dc, dr)


def reflections(m):
    a = m['axis']
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    for p in m['pieces']:
        for c, r in p:
            if r == a:
                continue
            d = (c, 2 * a - r)
            if 0 <= d[1] < N and d not in occupied:
                out.add(d)
    return out


def paint(out, c, r, ring, mid):
    for dy in range(3):
        for dx in range(3):
            out[3 * r + dy][3 * c + dx] = ring
    out[3 * r + 1][3 * c + 1] = mid


def render(m, frame):
    out = [list(row) for row in frame]
    a, sel, tg = m['axis'], m['sel'], m['targets']
    refl = reflections(m)
    owner = {}
    for i, p in enumerate(m['pieces']):
        for b in p:
            owner[b] = i
    for r in range(N):
        for c in range(N):
            b = (c, r)
            axis_dot = DOT if (r == a and sel == 'axis') else BG
            if b in owner:
                mid = TARGET if b in tg else (DOT if owner[b] == sel else axis_dot)
                paint(out, c, r, PIECE, mid)
            elif b in refl:
                paint(out, c, r, REFL, TARGET if b in tg else REFL)
            elif b in tg:
                paint(out, c, r, TARGET, TARGET)
            elif r == a:
                paint(out, c, r, AXIS, axis_dot)
            else:
                paint(out, c, r, BG, BG)
    return out


def tick_hud(out):
    for y in range(63):
        if out[y][63] != HUD_ON:
            out[y][63] = HUD_ON
            return


def transition_function(state, action, frame):
    m = parse(frame)
    changed = step(m, action)
    out = render(m, frame)
    if changed:
        tick_hud(out)
    return [[int(v) for v in row] for row in out]
