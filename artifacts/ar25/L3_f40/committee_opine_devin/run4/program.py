# Mechanics: 63x63 board of 21x21 3x3 blocks; full-width axis band (10), solid targets (11),
# holed pieces (5), reflections (4) of every off-axis piece block at row 2A-r unless a piece is there.
# A1/A2 move the selection one block row, A3/A4 one block column (axis: rows only, unblocked; pieces:
# blocked by board edge / other pieces); A5 cycles axis -> pieces by size desc -> axis.
# HUD col 63 gains one 12 per state-changing action. Hypothesis: 'player gone' = touching pieces merging.
N = 21
BG, AXIS, TARGET, REFL, PIECE, SEL, TICK, HUDC = 9, 10, 11, 4, 5, 0, 12, 11
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def block(frame, r, c):
    return [frame[3 * r + i][3 * c + j] for i in range(3) for j in range(3)]


def parse(frame):
    cells = {(r, c): block(frame, r, c) for r in range(N) for c in range(N)}
    row_count = [sum(cells[(r, c)][0] == AXIS for c in range(N)) for r in range(N)]
    axis = max(range(N), key=lambda r: row_count[r])
    axis_sel = any(cells[(axis, c)][0] == AXIS and cells[(axis, c)][4] == SEL for c in range(N))
    targets = {k for k, b in cells.items() if TARGET in b}
    pblocks = {k for k, b in cells.items() if b[0] == PIECE}
    sel_piece = set()
    stack = [k for k in pblocks if cells[k][4] == SEL]
    while stack:
        k = stack.pop()
        if k in sel_piece:
            continue
        sel_piece.add(k)
        for dr, dc in DIRS.values():
            n = (k[0] + dr, k[1] + dc)
            if n in pblocks and n not in sel_piece and cells[n][4] != BG:
                stack.append(n)
    pieces = [sel_piece] if sel_piece else []
    rest = pblocks - sel_piece
    while rest:
        comp, stack = set(), [next(iter(rest))]
        while stack:
            k = stack.pop()
            if k in comp or k not in rest:
                continue
            comp.add(k)
            stack += [(k[0] + dr, k[1] + dc) for dr, dc in DIRS.values()]
        rest -= comp
        pieces.append(comp)
    sel = 'axis' if axis_sel else (0 if sel_piece else None)
    hud = sum(frame[y][63] == TICK for y in range(63))
    return {'axis': axis, 'pieces': pieces, 'sel': sel, 'targets': targets, 'hud': hud}


def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(m, action):
    """Mutates model m; returns True when the state changed."""
    if isinstance(action, int) and action == 5:
        seq = ['axis'] + order(m['pieces'])
        cur = m['sel']
        m['sel'] = seq[(seq.index(cur) + 1) % len(seq)] if cur in seq else 'axis'
        return m['sel'] != cur
    if not isinstance(action, int) or action not in DIRS:
        return False
    dr, dc = DIRS[action]
    if m['sel'] == 'axis':
        a = m['axis'] + dr
        if dc or not 0 <= a < N:
            return False
        m['axis'] = a
        return True
    if m['sel'] is None:
        return False
    i = m['sel']
    moved = {(r + dr, c + dc) for r, c in m['pieces'][i]}
    others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i]) if len(m['pieces']) > 1 else set()
    if any(not (0 <= r < N and 0 <= c < N) for r, c in moved) or moved & others:
        return False
    m['pieces'][i] = moved
    return True


def reflections(m):
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    A = m['axis']
    out = set()
    for r, c in occ:
        d = 2 * A - r
        if r != A and 0 <= d < N and (d, c) not in occ:
            out.add((d, c))
    return out


def render(m, frame):
    out = [list(row) for row in frame]
    refl = reflections(m)
    sel_blocks = m['pieces'][m['sel']] if isinstance(m['sel'], int) else set()
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    for r in range(N):
        for c in range(N):
            k = (r, c)
            layers = []  # top-down: (ring, solid centre, fill centre)
            if k in occ:
                layers.append((PIECE, None, SEL if k in sel_blocks else None))
            if k in refl:
                layers.append((REFL, None, REFL))
            if k in m['targets']:
                layers.append((TARGET, TARGET, None))
            if r == m['axis']:
                layers.append((AXIS, SEL if m['sel'] == 'axis' else None, None))
            ring = layers[0][0] if layers else BG
            centre = next((s for _, s, _ in layers if s is not None), None)
            if centre is None:
                centre = next((f for _, _, f in layers if f is not None), BG)
            for i in range(3):
                for j in range(3):
                    out[3 * r + i][3 * c + j] = centre if (i, j) == (1, 1) else ring
    for y in range(63):
        out[y][63] = TICK if y < m['hud'] else HUDC
    return out


def transition_function(state, action, frame):
    m = parse(frame)
    if step(m, action):
        m['hud'] += 1
    return [[int(v) for v in row] for row in render(m, frame)]
