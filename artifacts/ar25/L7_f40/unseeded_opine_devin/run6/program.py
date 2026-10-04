# Mechanics: 64x64 frame = 21x21 grid of 3x3 blocks (board 63x63), col 63 = HUD, row 63 static.
# Walls: H axis (block row, ring 10) and V axis (block col); selected axis shows 0 centres.
# A5 cycles H -> V -> pieces (sorted by min block) -> H; A1/A2 move H or piece by a row,
# A3/A4 move V or piece by a col; pieces blocked by edge/other pieces; axes only by edge.
# Each piece block mirrors (gray 4) across V, H and both; HUD: one 11->12 per effective action.
N = 21
BG, WALL, TGT, REFL, PIECE, HUD_ON = 9, 10, 11, 4, 5, 12


def blocks(frame):
    ring, cen = {}, {}
    for r in range(N):
        for c in range(N):
            ring[(r, c)] = frame[3 * r][3 * c]
            cen[(r, c)] = frame[3 * r + 1][3 * c + 1]
    return ring, cen


def comps(cells):
    cells, out = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            p = stack.pop()
            comp.add(p)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (p[0] + d[0], p[1] + d[1])
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
        out.append(comp)
    return out


def parse(frame):
    ring, cen = blocks(frame)
    rows = [sum(1 for c in range(N) if ring[(r, c)] == WALL) for r in range(N)]
    cols = [sum(1 for r in range(N) if ring[(r, c)] == WALL) for c in range(N)]
    H, V = rows.index(max(rows)), cols.index(max(cols))
    targets = {k for k in ring if ring[k] == TGT or cen[k] == TGT}
    pblocks = {k for k in ring if ring[k] == PIECE}
    sel = None
    if any(ring[(H, c)] == WALL and cen[(H, c)] == 0 for c in range(N) if c != V):
        sel = 'H'
    elif any(ring[(r, V)] == WALL and cen[(r, V)] == 0 for r in range(N) if r != H):
        sel = 'V'
    pieces = []
    if sel is None:
        seeds = {k for k in pblocks if cen[k] == 0}
        grown, stack = set(seeds), list(seeds)
        while stack:
            p = stack.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (p[0] + d[0], p[1] + d[1])
                if q in pblocks and q not in grown and cen[q] != BG:
                    grown.add(q)
                    stack.append(q)
        pieces = [grown] if grown else []
        pieces += comps(pblocks - grown)
    else:
        pieces = comps(pblocks)
    pieces = [frozenset(p) for p in pieces]
    pieces.sort(key=min)
    if sel is None:
        sel = next((i for i, p in enumerate(pieces) if p & grown), None) if pieces else None
        if sel is None:
            sel = 'H'
    return {'H': H, 'V': V, 'targets': targets, 'pieces': pieces, 'sel': sel}


def step(m, action):
    m = dict(m)
    pieces = list(m['pieces'])
    sel = m['sel']
    if action == 5:
        order = ['H', 'V'] + list(range(len(pieces)))
        m['sel'] = order[(order.index(sel) + 1) % len(order)]
        return m
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action, (0, 0))
    if (dr, dc) == (0, 0):
        return m
    if sel == 'H':
        if dr and 0 <= m['H'] + dr < N:
            m['H'] += dr
    elif sel == 'V':
        if dc and 0 <= m['V'] + dc < N:
            m['V'] += dc
    else:
        moved = frozenset((r + dr, c + dc) for r, c in pieces[sel])
        others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
        if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
            pieces[sel] = moved
            m['pieces'] = pieces
    return m


def reflections(m):
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    H, V = m['H'], m['V']
    out = set()
    for r, c in occ:
        for d in ((r, 2 * V - c), (2 * H - r, c), (2 * H - r, 2 * V - c)):
            if 0 <= d[0] < N and 0 <= d[1] < N and d not in occ:
                out.add(d)
    return out


def render(m, frame):
    out = [list(row) for row in frame]
    H, V, sel = m['H'], m['V'], m['sel']
    refl = reflections(m)
    pcell = {}
    for i, p in enumerate(m['pieces']):
        for b in p:
            pcell[b] = i
    for r in range(N):
        for c in range(N):
            k = (r, c)
            wall = r == H or c == V
            wsel = wall and ((sel == 'H' and r == H) or (sel == 'V' and c == V))
            if k in pcell:
                ring = PIECE
            elif k in refl:
                ring = REFL
            elif k in m['targets']:
                ring = TGT
            elif wall:
                ring = WALL
            else:
                ring = BG
            if k in m['targets']:
                cen = TGT
            elif wsel:
                cen = 0
            elif k in pcell:
                cen = 0 if pcell[k] == sel else BG
            elif k in refl:
                cen = REFL
            else:
                cen = BG
            for i in range(3):
                for j in range(3):
                    out[3 * r + i][3 * c + j] = cen if (i, j) == (1, 1) else ring
    return out


def same(a, b):
    return a['H'] == b['H'] and a['V'] == b['V'] and a['sel'] == b['sel'] and \
        sorted(map(sorted, a['pieces'])) == sorted(map(sorted, b['pieces']))


def transition_function(state, action, frame):
    if isinstance(action, dict):
        action = action.get('action_id', 0)
    m = parse(frame)
    m2 = step(m, action)
    out = render(m2, frame)
    if not same(m, m2):
        for y in range(63):
            if out[y][63] != HUD_ON:
                out[y][63] = HUD_ON
                break
    return out
