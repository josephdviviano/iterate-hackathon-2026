# Mechanics: 21x21 board of 3x3 blocks read from the frame; one horizontal wall axis (ring 10) at block row A,
# pieces (ring 5), targets (solid 11), reflections (ring 4, fill 4) of every piece block r != A at row 2A-r.
# A1/A2 move the selected axis (bounds only, passes under pieces) or the selected piece (A3/A4 too; blocked by
# board edge and other pieces); A5 cycles axis -> pieces by size desc -> axis. Selected entity shows 0 centres.
# HUD: column 63 fills 11->12 top-down, one cell per action that changed the board. Unconfirmed: blocked-move tick, A6/A7.
N, BG, WALL, PIECE, REFL, TARG, SEL = 21, 9, 10, 5, 4, 11, 0
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_memo = {}


def blocks_with(frame, pred):
    return {(r, c) for r in range(N) for c in range(N) if pred(frame, r, c)}


def corner(f, r, c):
    return f[3 * r][3 * c]


def centre(f, r, c):
    return f[3 * r + 1][3 * c + 1]


def has_colour(f, r, c, col):
    return any(f[3 * r + i][3 * c + j] == col for i in range(3) for j in range(3))


def comps(cells):
    cells, out = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            p = stack.pop()
            comp.add(p)
            for dr, dc in DIRS.values():
                q = (p[0] + dr, p[1] + dc)
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
        out.append(frozenset(comp))
    return out


def parse(f):
    walls = blocks_with(f, lambda f, r, c: corner(f, r, c) == WALL)
    rows = {}
    for r, c in walls:
        rows[r] = rows.get(r, 0) + 1
    axis = max(rows, key=lambda r: (rows[r], -r)) if rows else 0
    targets = blocks_with(f, lambda f, r, c: has_colour(f, r, c, TARG))
    pblocks = blocks_with(f, lambda f, r, c: corner(f, r, c) == PIECE)
    wall_sel = any(centre(f, r, c) == SEL for r, c in walls if r == axis)
    pieces, sel = [], ('axis' if wall_sel else None)
    for comp in comps(pblocks):
        dots = {p for p in comp if centre(f, *p) == SEL} if not wall_sel else set()
        if not dots:
            pieces.append(comp)
            continue
        grown, changed = set(dots), True
        while changed:
            changed = False
            for p in comp - grown:
                if centre(f, *p) == TARG and any((p[0] + dr, p[1] + dc) in grown for dr, dc in DIRS.values()):
                    grown.add(p)
                    changed = True
        sel = len(pieces)
        pieces.append(frozenset(grown))
        pieces.extend(comps(comp - grown))
    return {'axis': axis, 'targets': targets, 'pieces': pieces, 'sel': sel}


def size_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i], key=lambda p: (p[1], p[0]))))


def step(st, action):
    pieces, sel, axis = list(st['pieces']), st['sel'], st['axis']
    if action == 5:
        order = size_order(pieces)
        if sel == 'axis':
            sel = order[0] if order else 'axis'
        elif sel is None:
            sel = 'axis'
        else:
            k = order.index(sel)
            sel = order[k + 1] if k + 1 < len(order) else 'axis'
    elif action in DIRS:
        dr, dc = DIRS[action]
        if sel == 'axis':
            if dc == 0 and 0 <= axis + dr < N:
                axis += dr
        elif sel is not None:
            moved = frozenset((r + dr, c + dc) for r, c in pieces[sel])
            others = set().union(*[p for i, p in enumerate(pieces) if i != sel])
            if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
                pieces[sel] = moved
    return {'axis': axis, 'targets': st['targets'], 'pieces': pieces, 'sel': sel}


def render(st, f):
    out = [row[:] for row in f]
    axis, sel, targets = st['axis'], st['sel'], st['targets']
    occupied = {}
    for i, p in enumerate(st['pieces']):
        for b in p:
            occupied[b] = i
    refl = set()
    for r, c in occupied:
        if r != axis and 0 <= 2 * axis - r < N and (2 * axis - r, c) not in occupied:
            refl.add((2 * axis - r, c))
    for r in range(N):
        for c in range(N):
            b = (r, c)
            stack = []  # (ring colour, hole fill or None, solid centre?) top-down
            if b in occupied:
                stack.append((PIECE, SEL if sel == occupied[b] else None, False))
            if b in refl:
                stack.append((REFL, REFL, False))
            if b in targets:
                stack.append((TARG, TARG, True))
            if r == axis:
                stack.append((WALL, SEL if sel == 'axis' else None, False))
            ring = stack[0][0] if stack else BG
            mid = BG
            solid = [s for s in stack if s[2]]
            if solid:
                mid = solid[0][1]
            else:
                fills = [s[1] for s in stack if s[1] is not None]
                if fills:
                    mid = fills[0]
            for i in range(3):
                for j in range(3):
                    out[3 * r + i][3 * c + j] = mid if (i, j) == (1, 1) else ring
    return out


def tick(before, after):
    board_changed = any(before[y][:63] != after[y][:63] for y in range(63))
    if board_changed:
        for y in range(63):
            if after[y][63] != 12:
                after[y][63] = 12
                break
    return after


def transition_function(state, action, frame):
    st = _memo.get('state') if _memo.get('frame') == frame else None
    if st is None:
        st = parse(frame)
    if isinstance(action, dict):
        action = action.get('action_id')
    new = step(st, action)
    out = tick(frame, render(new, frame))
    _memo['frame'], _memo['state'] = [row[:] for row in out], new
    return out
