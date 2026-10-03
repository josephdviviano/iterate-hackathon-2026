# Mechanics: two mirror axes (row band R, column strip C, 10-coloured 3x3 holed blocks) + 3x3-block pieces (5).
# ACTION5 cycles selection C -> R -> pieces (bbox x,y order) -> C; selection shows as 0 centre fills.
# A1/A2 move C or the selected piece in x (-3/+3), A3/A4 move R or the piece in y; pieces blocked by bounds/pieces.
# Reflections: across R visible gray (4); across C / both = invisible erasers; a piece occupying the C column gets
# its R-reflection invisible too (seen once, step 268; piece-on-R-row analogue unconfirmed). Render layers + re-extract.
N = 63
B = 21
RING = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]


def to_grid(state):
    g = [[-1] * N for _ in range(N)]
    for o in sorted(state, key=lambda o: o.get('layer', 0)):
        P = o.get('pixels')
        if not P:
            continue
        for i, col in enumerate(P):
            for j, v in enumerate(col):
                if v != -1 and 0 <= o['y'] + j < N and 0 <= o['x'] + i < N:
                    g[o['y'] + j][o['x'] + i] = v
    return g


def block_cells(g, br, bc):
    return [g[3 * br + i][3 * bc + j] for i in range(3) for j in range(3)]


def comps4(cells):
    cells = set(cells)
    out = []
    while cells:
        s = cells.pop()
        st, c = [s], [s]
        while st:
            r, q = st.pop()
            for n in ((r + 1, q), (r - 1, q), (r, q + 1), (r, q - 1)):
                if n in cells:
                    cells.remove(n)
                    st.append(n)
                    c.append(n)
        out.append(c)
    return out


def parse(state):
    g = to_grid(state)
    wall_rows = [0] * B
    wall_cols = [0] * B
    targets = set()
    pblocks = set()
    for br in range(B):
        for bc in range(B):
            cs = block_cells(g, br, bc)
            if 10 in cs:
                wall_rows[br] += 1
                wall_cols[bc] += 1
            if 11 in cs:
                targets.add((br, bc))
            if 5 in cs:
                pblocks.add((br, bc))
    R = max(range(B), key=lambda r: wall_rows[r])
    C = max(range(B), key=lambda c: wall_cols[c])
    pieces = [frozenset(c) for c in comps4(pblocks)]
    pieces.sort(key=lambda p: min((c, r) for r, c in p))
    zero = {(br, bc) for br in range(B) for bc in range(B) if g[3 * br + 1][3 * bc + 1] == 0}
    off = zero - pblocks
    sel = None
    if any(b[0] == R and b[1] != C for b in off):
        sel = 'R'
    elif any(b[1] == C for b in off):
        sel = 'C'
    else:
        for k, p in enumerate(pieces):
            if p & zero:
                sel = k
                break
    return dict(R=R, C=C, pieces=pieces, sel=sel, targets=targets)


def free(blocks, others):
    return all(0 <= r < B and 0 <= c < B for r, c in blocks) and not (blocks & others)


def step(m, action):
    sel = m['sel']
    pieces = list(m['pieces'])
    if action == 5:
        order = ['C', 'R'] + list(range(len(pieces)))
        i = order.index(sel) if sel in order else -1
        m['sel'] = order[(i + 1) % len(order)]
        return m
    d = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(action)
    if d is None:
        return m
    dr, dc = d
    if sel == 'C':
        if dc and 0 <= m['C'] + dc < B:
            m['C'] += dc
    elif sel == 'R':
        if dr and 0 <= m['R'] + dr < B:
            m['R'] += dr
    elif isinstance(sel, int):
        p = pieces[sel]
        moved = frozenset((r + dr, c + dc) for r, c in p)
        others = set().union(*[q for k, q in enumerate(pieces) if k != sel]) if len(pieces) > 1 else set()
        if free(moved, others):
            pieces[sel] = moved
    m['pieces'] = pieces
    return m


def render(m):
    R, C, sel = m['R'], m['C'], m['sel']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    stacks = {}

    def add(b, spr):
        if 0 <= b[0] < B and 0 <= b[1] < B:
            stacks.setdefault(b, []).append(spr)
    # sprite = (layer, kind, ring colour, centre solid?, centre colour)
    for k, p in enumerate(m['pieces']):
        for b in p:
            add(b, (4, 'P', 5, False, 0 if sel == k else -1))
    vis, inv = set(), set()
    for p in m['pieces']:
        on_c = any(c == C for _, c in p)
        for r, c in p:
            for t, visible in (((2 * R - r, c), not on_c), ((r, 2 * C - c), False), ((2 * R - r, 2 * C - c), False)):
                if t in occ or t == (r, c):
                    continue
                (vis if visible else inv).add(t)
    for b in vis:
        add(b, (3.5, 'V', 4, False, 4))
    for b in inv - vis:
        add(b, (3, 'I', -1, False, -1))
    for b in m['targets']:
        add(b, (2, 'T', 11, True, 11))
    for i in range(B):
        add((i, C), (1.5, 'W', 10, sel == 'C', 0 if sel == 'C' else -1))
        add((R, i), (1, 'W', 10, sel == 'R', 0 if sel == 'R' else -1))
    g = [[-1] * N for _ in range(N)]
    kind = [[None] * N for _ in range(N)]
    for (br, bc), st in stacks.items():
        st.sort(key=lambda s: -s[0])
        top = st[0]
        for i, j in RING:
            g[3 * br + i][3 * bc + j] = top[2]
            kind[3 * br + i][3 * bc + j] = top[1]
        ctr = next((s for s in st if s[3]), None)
        if ctr is None:
            g[3 * br + 1][3 * bc + 1] = top[4]
            kind[3 * br + 1][3 * bc + 1] = top[1]
        else:
            g[3 * br + 1][3 * bc + 1] = ctr[4]
            kind[3 * br + 1][3 * bc + 1] = ctr[1]
    return g, kind


def make_obj(name, typ, cells, g, layer, tags):
    ys = [c[0] for c in cells]
    xs = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for r, c in cells:
        pix[c - x0][r - y0] = g[r][c]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': layer,
            'tags': tags, 'pixels': pix}


def bbox_key(c):
    return (min(q for _, q in c), min(r for r, _ in c))


def extract(g, kind, extra):
    cells = {}
    for r in range(N):
        for c in range(N):
            cells.setdefault(g[r][c], set()).add((r, c))
    zeros = cells.get(0, set())
    out = list(extra)
    for colour, typ in ((10, 'wall'), (5, 'player')):
        own = cells.get(colour, set())
        cs = sorted(comps4(own | zeros), key=bbox_key)
        for k, c in enumerate(cs):
            if not any(p in own for p in c):
                continue
            o = make_obj('', typ, c, g, 1 if typ == 'wall' else 4, None)
            if typ == 'wall':
                v = o['w'] >= o['h']
                o['name'] = ('wall_v_%d' if v else 'wall_h_%d') % k
                o['tags'] = ['axis', 'vertical' if v else 'horizontal']
            else:
                o['name'] = 'piece_%d' % k
                o['tags'] = ['movable', 'black']
            out.append(o)
    rcells = {(r, c) for r in range(N) for c in range(N) if kind[r][c] in ('V', 'I')}
    vcells = cells.get(4, set())
    cs = sorted(comps4(rcells | vcells), key=bbox_key)
    for k, c in enumerate(cs):
        vc = [p for p in c if p in vcells]
        if vc:
            out.append(make_obj('reflection_%d' % k, 'reflection', vc, g, 3, ['mirror', 'gray']))
    groups = [list(c) for c in comps4(cells.get(11, set()))]
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if gap(groups[a], groups[b]) <= 4:
                    groups[a] += groups.pop(b)
                    merged = True
                    break
            if merged:
                break
    for k, c in enumerate(sorted(groups, key=bbox_key)):
        out.append(make_obj('target_%d' % k, 'target', c, g, 2, ['goal', 'yellow']))
    return out


def gap(a, b):
    ar = [p[0] for p in a]; ac = [p[1] for p in a]
    br = [p[0] for p in b]; bc = [p[1] for p in b]
    dr = max(0, min(br) - max(ar), min(ar) - max(br))
    dc = max(0, min(bc) - max(ac), min(ac) - max(bc))
    return max(dr, dc)


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get('action_id')
    extra = [dict(o) for o in state if o['type'] not in ('wall', 'player', 'reflection', 'target')]
    m = parse(state)
    m = step(m, action)
    g, kind = render(m)
    return extract(g, kind, extra)
