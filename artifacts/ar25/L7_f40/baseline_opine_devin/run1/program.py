# Two-axis mirror game on a 21x21 grid of 3x3 blocks, parsed from frame colours (wall 10, target 11, piece 5, mirror 4, bg 9).
# A5 cycles H band -> V strip -> pieces (min col desc, min row desc) -> H; A1/A2 move H or the selected piece, A3/A4 move V
# or the piece; pieces blocked by board edge / other pieces, axes only by the edge. Every piece block mirrors across V, H and
# both (solid-ring 4, hole shows target); selected axis/piece shows 0 centres. HUD col 63 gains one 12 per effective action.
# Unconfirmed: A5 order with >2 pieces, mirrors hidden when a piece touches an axis (never observed), A6/A7 (treated as no-ops).
BG, WALL, TGT, PIECE, REFL, DOT, HUD_ON = 9, 10, 11, 5, 4, 0, 12
N = 21
_memo = {}


def cells(f, by, bx):
    return [f[by * 3 + j][bx * 3 + i] for j in range(3) for i in range(3)]


def comps(blocks):
    left, out = set(blocks), []
    while left:
        st = [left.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + d[0], c + d[1])
                if n in left:
                    left.discard(n)
                    comp.add(n)
                    st.append(n)
        out.append(frozenset(comp))
    return out


def order(pieces):
    return sorted(pieces, key=lambda p: (min(c for _, c in p), min(r for r, _ in p)), reverse=True)


def parse(f):
    blk = {(r, c): cells(f, r, c) for r in range(N) for c in range(N)}
    wall_rows = [sum(WALL in blk[(r, c)] for c in range(N)) for r in range(N)]
    wall_cols = [sum(WALL in blk[(r, c)] for r in range(N)) for c in range(N)]
    H = max(range(N), key=lambda r: wall_rows[r])
    V = max(range(N), key=lambda c: wall_cols[c])
    targets = {k for k, v in blk.items() if TGT in v}
    pblocks = {k for k, v in blk.items() if PIECE in v[:4] + v[5:]}
    pieces = order(comps(pblocks))

    def dotted(k):
        v = blk[k]
        return v[4] == DOT and v[0] == WALL and k not in pblocks
    sel = None
    if any(dotted((H, c)) for c in range(N) if c != V):
        sel = 'H'
    elif any(dotted((r, V)) for r in range(N) if r != H):
        sel = 'V'
    else:
        for i, p in enumerate(pieces):
            if any(blk[k][4] == DOT for k in p):
                sel = i
                break
    return {'H': H, 'V': V, 'targets': frozenset(targets), 'pieces': pieces, 'sel': sel}


def mirrors(m):
    H, V = m['H'], m['V']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    on = lambda k: 0 <= k[0] < N and 0 <= k[1] < N and k not in occ
    vis = set()
    for p in m['pieces']:
        for r, c in p:
            for k, ok in (((r, 2 * V - c), c != V), ((2 * H - r, c), r != H), ((2 * H - r, 2 * V - c), r != H and c != V)):
                if ok and on(k):
                    vis.add(k)
    return vis


def render(m, f):
    g = [row[:] for row in f]
    vis = mirrors(m)
    sel = m['sel']
    psel = m['pieces'][sel] if isinstance(sel, int) and sel < len(m['pieces']) else set()
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    for r in range(N):
        for c in range(N):
            k = (r, c)
            stack = []  # (ring, solid_centre, fill)
            if k in occ:
                stack.append((PIECE, None, DOT if k in psel else None))
            if k in vis:
                stack.append((REFL, None, REFL))
            if k in m['targets']:
                stack.append((TGT, TGT, None))
            if c == m['V']:
                stack.append((WALL, DOT if sel == 'V' else None, None))
            if r == m['H']:
                stack.append((WALL, DOT if sel == 'H' else None, None))
            ring = stack[0][0] if stack else BG
            centre = next((s[1] for s in stack if s[1] is not None), None)
            if centre is None:
                centre = stack[0][2] if stack and stack[0][2] is not None else BG
            for j in range(3):
                for i in range(3):
                    g[r * 3 + j][c * 3 + i] = centre if (i, j) == (1, 1) else ring
    return g


def step(m, action):
    m = dict(m)
    sel = m['sel']
    if action == 5:
        n = len(m['pieces'])
        if sel == 'H':
            m['sel'] = 'V'
        elif sel == 'V':
            m['sel'] = 0 if n else 'H'
        elif isinstance(sel, int):
            m['sel'] = sel + 1 if sel + 1 < n else 'H'
        else:
            m['sel'] = 'H'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action) if isinstance(action, int) else None
    if d is None or sel is None:
        return m
    if sel == 'H':
        if d[0] and 0 <= m['H'] + d[0] < N:
            m['H'] += d[0]
    elif sel == 'V':
        if d[1] and 0 <= m['V'] + d[1] < N:
            m['V'] += d[1]
    else:
        p = m['pieces'][sel]
        others = set().union(*[q for i, q in enumerate(m['pieces']) if i != sel]) if len(m['pieces']) > 1 else set()
        moved = frozenset((r + d[0], c + d[1]) for r, c in p)
        if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
            pieces = list(m['pieces'])
            pieces[sel] = moved
            m['pieces'] = pieces
    return m


def transition_function(state, action, frame):
    global _memo
    f = [list(map(int, row)) for row in frame]
    m = _memo.get('model') if _memo.get('frame') == f else None
    if m is None:
        m = parse(f)
    m2 = step(m, action)
    g = render(m2, f)
    if m2 != m:
        n = sum(1 for y in range(64) if f[y][63] == HUD_ON)
        if n < 64:
            g[n][63] = HUD_ON
    _memo = {'frame': g, 'model': m2}
    return g
