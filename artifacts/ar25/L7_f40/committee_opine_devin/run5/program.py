# Mechanics: 21x21 grid of 3x3 blocks (bg 9). Walls 10 = one H axis row + one V axis col; targets 11 (solid);
# pieces 5 (holed); mirrors 4 = each piece block reflected across V, across H, and across both (clip board, skip piece cells).
# Selection (0 dot): H axis, V axis or one piece; A1/A2 move H or piece up/down, A3/A4 move V or piece left/right,
# pieces blocked by board edge / other pieces, axes pass under pieces; A5 cycles H -> V -> pieces by min block -> H.
# HUD col 63: one 12 per effective action from the top. Unconfirmed: piece on an axis row hiding its mirror (unseen).
N = 21
BG, WALL, TGT, REFL, PIECE, SEL, HUD_ON, HUD_OFF = 9, 10, 11, 4, 5, 0, 12, 11


def ring(f, r, c):
    return f[3 * r][3 * c]


def centre(f, r, c):
    return f[3 * r + 1][3 * c + 1]


def components(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in cells:
                    cells.discard(q)
                    comp.add(q)
                    st.append(q)
        out.append(comp)
    return out


def parse(f):
    walls = {(r, c) for r in range(N) for c in range(N) if ring(f, r, c) == WALL}
    ah = max(range(N), key=lambda r: sum(1 for c in range(N) if ring(f, r, c) == WALL))
    av = max(range(N), key=lambda c: sum(1 for r in range(N) if ring(f, r, c) == WALL))
    targets = {(r, c) for r in range(N) for c in range(N)
               if ring(f, r, c) == TGT or centre(f, r, c) == TGT}
    pblocks = {(r, c) for r in range(N) for c in range(N) if ring(f, r, c) == PIECE}
    hdots = sum(1 for c in range(N) if c != av and centre(f, ah, c) == SEL and (ah, c) in walls)
    vdots = sum(1 for r in range(N) if r != ah and centre(f, r, av) == SEL and (r, av) in walls)
    sel = None
    if hdots or vdots:
        sel = 'H' if hdots >= vdots else 'V'
    pieces = []
    for comp in components(pblocks):
        zeros = {b for b in comp if centre(f, *b) == SEL}
        if sel is None and zeros:
            grown = set(zeros)
            st = list(zeros)
            while st:
                r, c = st.pop()
                for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if q in comp and q not in grown and centre(f, *q) != BG:
                        grown.add(q)
                        st.append(q)
            rest = comp - grown
            pieces.append(frozenset(grown))
            sel = len(pieces) - 1
            pieces.extend(frozenset(x) for x in components(rest))
        else:
            pieces.append(frozenset(comp))
    hud = sum(1 for y in range(64) if f[y][63] == HUD_ON)
    return {'ah': ah, 'av': av, 'targets': targets, 'pieces': pieces, 'sel': sel, 'hud': hud}


def ordered(pieces):
    return sorted(pieces, key=lambda p: min(p))


def canon(m):
    ps = ordered(m['pieces'])
    s = m['sel']
    if isinstance(s, int):
        s = ('P', ps.index(m['pieces'][s]))
    return (m['ah'], m['av'], frozenset(m['targets']), tuple(ps), s)


def step(m, action):
    m = dict(m)
    ps = ordered(m['pieces'])
    s = m['sel']
    if isinstance(s, int):
        s = ps.index(m['pieces'][s])
    m['pieces'] = ps
    m['sel'] = s
    if action == 5:
        if s == 'H':
            m['sel'] = 'V'
        elif s == 'V':
            m['sel'] = 0 if ps else 'H'
        elif isinstance(s, int):
            m['sel'] = s + 1 if s + 1 < len(ps) else 'H'
        else:
            m['sel'] = 'H'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None or s is None:
        return m
    dr, dc = d
    if s == 'H':
        if dr and 0 <= m['ah'] + dr < N:
            m['ah'] += dr
    elif s == 'V':
        if dc and 0 <= m['av'] + dc < N:
            m['av'] += dc
    else:
        p = ps[s]
        moved = frozenset((r + dr, c + dc) for r, c in p)
        others = set().union(*[q for i, q in enumerate(ps) if i != s]) if len(ps) > 1 else set()
        if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
            ps = list(ps)
            ps[s] = moved
            m['pieces'] = ps
    return m


def mirrors(m):
    ah, av = m['ah'], m['av']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    for r, c in occ:
        for q in ((r, 2 * av - c), (2 * ah - r, c), (2 * ah - r, 2 * av - c)):
            if 0 <= q[0] < N and 0 <= q[1] < N and q not in occ:
                out.add(q)
    return out


def render(m, frame):
    out = [list(row) for row in frame]
    ah, av, s = m['ah'], m['av'], m['sel']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    selp = m['pieces'][s] if isinstance(s, int) else set()
    refl = mirrors(m)
    for r in range(N):
        for c in range(N):
            b = (r, c)
            wall = r == ah or c == av
            seldot = (s == 'H' and r == ah) or (s == 'V' and c == av)
            if b in occ:
                rc = PIECE
            elif b in refl:
                rc = REFL
            elif b in m['targets']:
                rc = TGT
            elif wall:
                rc = WALL
            else:
                rc = BG
            if b in m['targets']:
                cc = TGT
            elif b not in occ and b not in refl and wall:
                cc = SEL if seldot else BG
            elif seldot and wall:
                cc = SEL
            elif b in selp:
                cc = SEL
            elif b in occ:
                cc = BG
            elif b in refl:
                cc = REFL
            else:
                cc = BG
            for y in range(3):
                for x in range(3):
                    out[3 * r + y][3 * c + x] = rc
            out[3 * r + 1][3 * c + 1] = cc
    for y in range(63):
        out[y][63] = HUD_ON if y < m['hud'] else HUD_OFF
    return out


def transition_function(state, action, frame):
    a = action.get('action_id') if isinstance(action, dict) else action
    m = parse(frame)
    nm = step(m, a)
    if canon(nm) != canon(m):
        nm['hud'] = m['hud'] + 1
    return render(nm, frame)
