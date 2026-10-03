# Mechanics: 21x21 board of 3x3 blocks; H axis (wall row, colour 10) and V axis (wall column); pieces (colour 5).
# A5 cycles selection H -> V -> pieces (top-left reading order) -> H; A1/A2 move H or a piece up/down, A3/A4 move V or a
# piece left/right; axes pass under pieces, pieces are blocked by board edges and other pieces.  Every piece block is
# mirrored across H, V and both (solid colour 4 rings, clipped to the board, not drawn on pieces).  Holes show target 11, a
# selected axis dot 0, else the top sprite fill.  HUD col 63 = one colour-12 cell per effective action.  Unconfirmed: piece
# touching the H row hides its V mirror (invisible mask), A6/A7 no-ops, piece-vs-axis blocking.
N, S = 21, 3
WALL, PIECE, REFL, TARGET, BG, DOT, HUD_ON = 10, 5, 4, 11, 9, 0, 12


def block(frame, r, c):
    return [frame[S * r + i][S * c + j] for i in range(S) for j in range(S)]


def ring_colour(p):
    ring = p[:4] + p[5:]
    return ring[0] if len(set(ring)) == 1 else None


def parse(frame):
    walls, pieces, targets, centres = {}, set(), set(), {}
    for r in range(N):
        for c in range(N):
            p = block(frame, r, c)
            k, ce = ring_colour(p), p[4]
            centres[(r, c)] = ce
            if ce == TARGET:
                targets.add((r, c))
            if k == WALL:
                walls[(r, c)] = ce
            elif k == PIECE:
                pieces.add((r, c))
    rows, cols = {}, {}
    for (r, c) in walls:
        rows[r] = rows.get(r, 0) + 1
        cols[c] = cols.get(c, 0) + 1
    ah = max(rows, key=lambda r: (rows[r], -r)) if rows else 0
    av = max(cols, key=lambda c: (cols[c], -c)) if cols else 0
    comps = components(pieces)
    hd = sum(1 for (r, c), ce in walls.items() if ce == DOT and r == ah and c != av)
    vd = sum(1 for (r, c), ce in walls.items() if ce == DOT and c == av and r != ah)
    sel = ('H' if hd >= vd else 'V') if hd or vd else None
    zero = {b for b in pieces if centres[b] == DOT} if sel is None else set()
    if zero:
        grown, todo = set(zero), list(zero)
        while todo:
            r, c = todo.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + d[0], c + d[1])
                if nb in pieces and nb not in grown and centres[nb] != BG:
                    grown.add(nb)
                    todo.append(nb)
        comps = [cp - grown for cp in comps if cp - grown] + [grown]
        sel = grown
    elif sel is None and walls.get((ah, av)) == DOT:
        sel = 'H'
    comps.sort(key=lambda cp: min(cp))
    if isinstance(sel, set):
        sel = next(i for i, cp in enumerate(comps) if cp == sel)
    return {'ah': ah, 'av': av, 'pieces': comps, 'targets': targets, 'sel': sel}


def components(cells):
    out, seen = [], set()
    for s in sorted(cells):
        if s in seen:
            continue
        comp, todo = {s}, [s]
        seen.add(s)
        while todo:
            r, c = todo.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + d[0], c + d[1])
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    comp.add(nb)
                    todo.append(nb)
        out.append(comp)
    return out


DELTA = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def on_board(r, c):
    return 0 <= r < N and 0 <= c < N


def step(m, action):
    sel = m['sel']
    if action == 5:
        order = ['H', 'V'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else 'H'
        return True
    if not isinstance(action, int) or action not in DELTA or sel is None:
        return False
    dr, dc = DELTA[action]
    if sel == 'H':
        if dc or not 0 <= m['ah'] + dr < N:
            return False
        m['ah'] += dr
        return True
    if sel == 'V':
        if dr or not 0 <= m['av'] + dc < N:
            return False
        m['av'] += dc
        return True
    others = set().union(*[p for i, p in enumerate(m['pieces']) if i != sel]) if len(m['pieces']) > 1 else set()
    moved = {(r + dr, c + dc) for r, c in m['pieces'][sel]}
    if any(not on_board(*b) or b in others for b in moved):
        return False
    m['pieces'][sel] = moved
    return True


def mirrors(m):
    ah, av = m['ah'], m['av']
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    vis, hid = set(), set()
    for p in m['pieces']:
        touches_h = any(r == ah for r, c in p)
        for r, c in p:
            for (a, b), hidden in (((r, 2 * av - c), touches_h), ((2 * ah - r, c), False), ((2 * ah - r, 2 * av - c), False)):
                if on_board(a, b) and (a, b) not in occupied and (a, b) != (r, c):
                    (hid if hidden else vis).add((a, b))
    return vis, hid - vis


def render(m, frame, n_actions):
    out = [row[:] for row in frame]
    ah, av, sel = m['ah'], m['av'], m['sel']
    vis, hid = mirrors(m)
    owner = {}
    for i, p in enumerate(m['pieces']):
        for b in p:
            owner[b] = i
    for r in range(N):
        for c in range(N):
            b = (r, c)
            wall = (r == ah or c == av) and b not in hid
            dot = (r == ah and sel == 'H') or (c == av and sel == 'V')
            if b in owner:
                ring, fill = PIECE, (DOT if owner[b] == sel else BG)
            elif b in vis:
                ring, fill = REFL, REFL
            elif b in m['targets'] and b not in hid:
                ring, fill = TARGET, TARGET
            elif wall:
                ring, fill = WALL, BG
            else:
                ring, fill = BG, BG
            if b in m['targets']:
                centre = TARGET
            elif wall and dot:
                centre = DOT
            else:
                centre = fill
            for i in range(S):
                for j in range(S):
                    out[S * r + i][S * c + j] = centre if (i, j) == (1, 1) else ring
    for y in range(64):
        out[y][63] = HUD_ON if y < n_actions else TARGET
    for x in range(63):
        out[63][x] = PIECE
    return out


def transition_function(state, action, frame):
    m = parse(frame)
    n = 0
    while n < 64 and frame[n][63] == HUD_ON:
        n += 1
    changed = step(m, action)
    return render(m, frame, n + 1 if changed else n)
