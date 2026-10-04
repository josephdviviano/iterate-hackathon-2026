# Mechanics: 21x21 grid of 3x3 blocks; H axis (wall row) + V axis (wall col), pieces (ring 5), targets (11).
# Selection = H axis / V axis / one piece, shown as 0 hole centres; A5 cycles H -> V -> pieces (min block) -> H.
# A1/A2 move selected H axis or piece up/down, A3/A4 move V axis or piece left/right; pieces blocked by edges/pieces.
# Each piece block mirrors across V, H and both into solid gray 4 blocks (skip on-axis sources, off-board, piece cells).
# HUD col 63: one more 12 per effective action. Hypothesis unconfirmed: axes pass freely under pieces/targets.
N = 21
WALL, PIECE, REFL, TARGET, BG, DOT, HUDON = 10, 5, 4, 11, 9, 0, 12


def block_px(frame, r, c):
    return [frame[3 * r + i][3 * c + j] for i in range(3) for j in range(3)]


def parse(frame):
    corner = [[frame[3 * r][3 * c] for c in range(N)] for r in range(N)]
    centre = [[frame[3 * r + 1][3 * c + 1] for c in range(N)] for r in range(N)]
    targets = set()
    for r in range(N):
        for c in range(N):
            if TARGET in block_px(frame, r, c):
                targets.add((r, c))
    ah = max(range(N), key=lambda r: sum(corner[r][c] == WALL for c in range(N)))
    av = max(range(N), key=lambda c: sum(corner[r][c] == WALL for r in range(N)))
    hsel = any(corner[ah][c] == WALL and centre[ah][c] == DOT for c in range(N) if c != av)
    vsel = any(corner[r][av] == WALL and centre[r][av] == DOT for r in range(N) if r != ah)
    cells = {(r, c) for r in range(N) for c in range(N) if corner[r][c] == PIECE}
    pieces = []
    seen = set()
    for cell in sorted(cells):
        if cell in seen:
            continue
        comp, stack = set(), [cell]
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        pieces.append(comp)
    pieces.sort(key=min)
    if hsel:
        sel = 'H'
    elif vsel:
        sel = 'V'
    else:
        sel = None
        for i, p in enumerate(pieces):
            if any(centre[r][c] == DOT for r, c in p):
                sel = i
                break
    return {'ah': ah, 'av': av, 'sel': sel, 'pieces': pieces, 'targets': targets}


def step(m, action):
    sel, pieces = m['sel'], m['pieces']
    if action == 5:
        order = ['H', 'V'] + list(range(len(pieces)))
        i = order.index(sel) if sel in order else -1
        m['sel'] = order[(i + 1) % len(order)]
        return m['sel'] != sel
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action if isinstance(action, int) else None)
    if d is None or sel is None:
        return False
    dr, dc = d
    if sel == 'H':
        nr = m['ah'] + dr
        if dc or not 0 <= nr < N:
            return False
        m['ah'] = nr
        return True
    if sel == 'V':
        nc = m['av'] + dc
        if dr or not 0 <= nc < N:
            return False
        m['av'] = nc
        return True
    others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
    moved = {(r + dr, c + dc) for r, c in pieces[sel]}
    if any(not (0 <= r < N and 0 <= c < N) or (r, c) in others for r, c in moved):
        return False
    pieces[sel] = moved
    return True


def reflections(m):
    ah, av = m['ah'], m['av']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    for r, c in occ:
        cand = []
        if c != av:
            cand.append((r, 2 * av - c))
        if r != ah:
            cand.append((2 * ah - r, c))
        if r != ah and c != av:
            cand.append((2 * ah - r, 2 * av - c))
        for rr, cc in cand:
            if 0 <= rr < N and 0 <= cc < N and (rr, cc) not in occ:
                out.add((rr, cc))
    return out


def render(m, frame):
    out = [row[:] for row in frame]
    ah, av, sel = m['ah'], m['av'], m['sel']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    selp = m['pieces'][sel] if isinstance(sel, int) else set()
    refl = reflections(m)
    for r in range(N):
        for c in range(N):
            b = (r, c)
            wall = r == ah or c == av
            if b in occ:
                ring = PIECE
            elif b in refl:
                ring = REFL
            elif b in m['targets']:
                ring = TARGET
            elif wall:
                ring = WALL
            else:
                ring = BG
            axis_dot = (sel == 'H' and r == ah) or (sel == 'V' and c == av)
            if b in m['targets']:
                cen = TARGET
            elif axis_dot:
                cen = DOT
            elif b in selp:
                cen = DOT
            elif b in refl:
                cen = REFL
            else:
                cen = BG
            for i in range(3):
                for j in range(3):
                    out[3 * r + i][3 * c + j] = cen if (i == 1 and j == 1) else ring
    return out


def transition_function(state, action, frame):
    m = parse(frame)
    changed = step(m, action)
    if not changed:
        return [row[:] for row in frame]
    out = render(m, frame)
    for y in range(64):
        if out[y][63] != HUDON:
            out[y][63] = HUDON
            break
    return out
