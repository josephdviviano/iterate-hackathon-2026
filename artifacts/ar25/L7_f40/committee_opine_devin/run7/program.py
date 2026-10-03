# Mechanics: 21x21 board of 3x3 blocks (bg 9); wall axes = block row H and block col V (ring 10), pieces ring 5,
# targets solid 11, mirrors of every piece block across V, across H and across both drawn as solid 4 (clipped,
# skipped on piece cells). A5 cycles selection H -> V -> pieces (reading order) -> H; A1/A2 move H or piece up/down,
# A3/A4 move V or piece left/right; pieces blocked by edges and other pieces. HUD col 63 adds one 12 per effective action.
# Hypothesis: no hidden state needed (stateless); the frame-gap reset between steps is just a new before frame.
N = 21
BG, WALL, PIECE, TARGET, REFL, DOT, HUDON = 9, 10, 5, 11, 4, 0, 12


def ring_cells(R, C):
    return [(3 * R + dy, 3 * C + dx) for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)]


def block_corner(frame, R, C):
    return frame[3 * R][3 * C]


def block_centre(frame, R, C):
    return frame[3 * R + 1][3 * C + 1]


def components(cells):
    cells = set(cells)
    comps = []
    while cells:
        start = min(cells)
        stack, comp = [start], {start}
        cells.discard(start)
        while stack:
            r, c = stack.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells:
                    cells.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def parse(frame):
    walls, pieces, targets = set(), set(), set()
    for R in range(N):
        for C in range(N):
            corner, centre = block_corner(frame, R, C), block_centre(frame, R, C)
            if corner == WALL:
                walls.add((R, C))
            if corner == PIECE:
                pieces.add((R, C))
            if corner == TARGET or centre == TARGET:
                targets.add((R, C))
    rows = [sum(1 for (r, c) in walls if r == R) for R in range(N)]
    cols = [sum(1 for (r, c) in walls if c == C) for C in range(N)]
    ah = max(range(N), key=lambda R: rows[R])
    av = max(range(N), key=lambda C: cols[C])
    h_dots = sum(1 for C in range(N) if C != av and block_centre(frame, ah, C) == DOT
                 and block_corner(frame, ah, C) == WALL)
    v_dots = sum(1 for R in range(N) if R != ah and block_centre(frame, R, av) == DOT
                 and block_corner(frame, R, av) == WALL)
    comps = split_pieces(frame, pieces)
    sel = None
    if h_dots or v_dots:
        sel = 'H' if h_dots >= v_dots else 'V'
    else:
        for i, comp in enumerate(comps):
            if any(block_centre(frame, r, c) == DOT for r, c in comp):
                sel = i
                break
    return {'ah': ah, 'av': av, 'pieces': comps, 'targets': targets, 'sel': sel}


def split_pieces(frame, pieces):
    out = []
    for comp in components(pieces):
        seeds = {b for b in comp if block_centre(frame, *b) == DOT}
        if seeds and len(seeds) < len(comp):
            grown = set(seeds)
            stack = list(seeds)
            while stack:
                r, c = stack.pop()
                for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if nb in comp and nb not in grown and block_centre(frame, *nb) != BG:
                        grown.add(nb)
                        stack.append(nb)
            rest = comp - grown
            out.append(grown)
            out.extend(components(rest))
        else:
            out.append(comp)
    out.sort(key=lambda p: min(p))
    return out


def on_board(r, c):
    return 0 <= r < N and 0 <= c < N


DELTAS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def step(m, action):
    m = dict(m)
    m['pieces'] = [set(p) for p in m['pieces']]
    sel = m['sel']
    if action == 5:
        order = ['H', 'V'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else 'H'
        return m
    if action not in DELTAS or sel is None:
        return m
    dr, dc = DELTAS[action]
    if sel == 'H':
        if dr and on_board(m['ah'] + dr, 0):
            m['ah'] += dr
    elif sel == 'V':
        if dc and on_board(0, m['av'] + dc):
            m['av'] += dc
    else:
        piece = m['pieces'][sel]
        others = set().union(*[p for i, p in enumerate(m['pieces']) if i != sel])
        moved = {(r + dr, c + dc) for r, c in piece}
        if all(on_board(*b) and b not in others for b in moved):
            m['pieces'][sel] = moved
    return m


def mirrors(m):
    ah, av = m['ah'], m['av']
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    for r, c in occupied:
        cand = []
        if c != av:
            cand.append((r, 2 * av - c))
        if r != ah:
            cand.append((2 * ah - r, c))
        if c != av and r != ah:
            cand.append((2 * ah - r, 2 * av - c))
        for b in cand:
            if on_board(*b) and b not in occupied:
                out.add(b)
    return out


def render(m, frame):
    out = [list(row) for row in frame]
    ah, av, sel = m['ah'], m['av'], m['sel']
    refl = mirrors(m)
    sel_cells = m['pieces'][sel] if isinstance(sel, int) else set()
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    for R in range(N):
        for C in range(N):
            b = (R, C)
            is_wall = R == ah or C == av
            wall_dot = is_wall and ((sel == 'H' and R == ah) or (sel == 'V' and C == av))
            if b in occupied:
                ring = PIECE
            elif b in refl:
                ring = REFL
            elif b in m['targets']:
                ring = TARGET
            elif is_wall:
                ring = WALL
            else:
                ring = BG
            if b in m['targets']:
                centre = TARGET
            elif wall_dot:
                centre = DOT
            elif b in sel_cells:
                centre = DOT
            elif b in refl and b not in occupied:
                centre = REFL
            else:
                centre = BG
            for y, x in ring_cells(R, C):
                out[y][x] = ring
            out[3 * R + 1][3 * C + 1] = centre
    return out


def same_model(a, b):
    return (a['ah'], a['av'], a['sel'], sorted(map(sorted, a['pieces']))) == \
        (b['ah'], b['av'], b['sel'], sorted(map(sorted, b['pieces'])))


def tick_hud(out, frame):
    for y in range(64):
        if frame[y][63] != HUDON:
            out[y][63] = HUDON
            return


def transition_function(state, action, frame):
    if isinstance(action, dict):
        action = action.get('action_id')
    m = parse(frame)
    m2 = step(m, action)
    out = render(m2, frame)
    if not same_model(m, m2):
        tick_hud(out, frame)
    return [[int(v) for v in row] for row in out]
