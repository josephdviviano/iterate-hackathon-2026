# Mechanics: 21x21 board of 3x3 blocks; horizontal axis row (ring 10), pieces (ring 5), targets (11), reflections (4).
# A1/A2 move the selected item one block row, A3/A4 move a selected piece one column (axis: no-op); pieces are
# blocked by board edges and other pieces, the axis only by edges (it passes under pieces). A5 cycles the
# selection axis -> pieces by size desc -> axis. Every piece block off the axis mirrors to row 2A-r unless off-board
# or piece-covered. HUD col 63 gains one 12 per effective action. Unconfirmed: piece-vs-target occlusion splits.
N = 21
BG = 9


def parse(frame):
    corner = lambda r, c: frame[3 * r][3 * c]
    centre = lambda r, c: frame[3 * r + 1][3 * c + 1]
    targets = set()
    for r in range(N):
        for c in range(N):
            if any(frame[3 * r + i][3 * c + j] == 11 for i in range(3) for j in range(3)):
                targets.add((r, c))
    counts = {}
    for r in range(N):
        counts[r] = sum(1 for c in range(N) if corner(r, c) == 10)
    axis = max(range(N), key=lambda r: counts[r])
    axis_sel = any(corner(axis, c) == 10 and centre(axis, c) == 0 for c in range(N))
    pblocks = {(r, c) for r in range(N) for c in range(N) if corner(r, c) == 5}
    sel = set()
    if not axis_sel:
        sel = {b for b in pblocks if centre(*b) == 0}
        stack = list(sel)
        while stack:
            r, c = stack.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in pblocks and nb not in sel and centre(*nb) != BG:
                    sel.add(nb)
                    stack.append(nb)
    pieces = []
    if sel:
        pieces.append(frozenset(sel))
    rest = pblocks - sel
    while rest:
        start = rest.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in rest:
                    rest.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        pieces.append(frozenset(comp))
    if axis_sel:
        selected = 'axis'
    elif sel:
        selected = 0
    else:
        selected = None
    return {'axis': axis, 'pieces': pieces, 'sel': selected, 'targets': targets}


def order_key(p):
    return (-len(p), min(p))


def piece_fits(model, idx, blocks):
    for r, c in blocks:
        if not (0 <= r < N and 0 <= c < N):
            return False
    others = set()
    for j, p in enumerate(model['pieces']):
        if j != idx:
            others |= p
    return not (others & set(blocks))


def step(model, action):
    pieces = list(model['pieces'])
    axis, sel = model['axis'], model['sel']
    if action == 5:
        order = ['axis'] + sorted(range(len(pieces)), key=lambda i: order_key(pieces[i]))
        if sel in order:
            sel = order[(order.index(sel) + 1) % len(order)]
        else:
            sel = 'axis'
    elif action in (1, 2, 3, 4):
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
        if sel == 'axis':
            if dc == 0 and 0 <= axis + dr < N:
                axis += dr
        elif sel is not None:
            moved = frozenset((r + dr, c + dc) for r, c in pieces[sel])
            if piece_fits(model, sel, moved):
                pieces[sel] = moved
    return {'axis': axis, 'pieces': pieces, 'sel': sel, 'targets': model['targets']}


def same(a, b):
    return (a['axis'] == b['axis'] and a['sel'] == b['sel']
            and [set(p) for p in a['pieces']] == [set(p) for p in b['pieces']])


def reflections(model):
    A = model['axis']
    covered = set()
    for p in model['pieces']:
        covered |= p
    out = set()
    for r, c in covered:
        if r == A:
            continue
        d = (2 * A - r, c)
        if 0 <= d[0] < N and d not in covered:
            out.add(d)
    return out


def render(model, frame):
    out = [list(row) for row in frame]
    refl = reflections(model)
    owner = {}
    for i, p in enumerate(model['pieces']):
        for b in p:
            owner[b] = i
    for r in range(N):
        for c in range(N):
            stack = []  # (ring, solid_centre, fill_centre) top-down
            if (r, c) in owner:
                stack.append((5, None, 0 if model['sel'] == owner[(r, c)] else None))
            if (r, c) in refl:
                stack.append((4, None, 4))
            if (r, c) in model['targets']:
                stack.append((11, 11, None))
            if r == model['axis']:
                stack.append((10, 0 if model['sel'] == 'axis' else None, None))
            ring = stack[0][0] if stack else BG
            cen = next((s for _, s, _ in stack if s is not None), None)
            if cen is None:
                cen = next((f for _, _, f in stack if f is not None), BG)
            for i in range(3):
                for j in range(3):
                    out[3 * r + i][3 * c + j] = ring
            out[3 * r + 1][3 * c + 1] = cen
    return out


def hud(frame, out, changed):
    k = 0
    while k < 63 and frame[k][63] == 12:
        k += 1
    if changed and k < 63:
        out[k][63] = 12


def transition_function(state, action, frame):
    if isinstance(action, dict):
        return [list(row) for row in frame]
    model = parse(frame)
    new = step(model, action)
    out = render(new, frame)
    hud(frame, out, not same(model, new))
    return [[int(v) for v in row] for row in out]
