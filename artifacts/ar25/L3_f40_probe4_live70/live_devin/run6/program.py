# Mechanics: 21x21 grid of 3x3 blocks; wall = horizontal mirror axis row A, pieces = holed blocks, targets = solid blocks.
# A1-A4 move the selection (axis: rows only) one block, blocked by board edges / other pieces; A5 cycles axis -> pieces
# (largest first) -> axis; A7 undoes the last move (free). Every piece block b reflects to row 2A-b (gray, holed).
# Frame = layered composite (piece>reflection>target>wall), centre = first solid else top fill; objects re-extracted.
# Budget counter: 64 free actions, then shrinks 1/action; spent bar = 1 extra player comp (names +1). Hidden: action count.
N = 21
TAGS = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'], 'target': ['goal', 'yellow'],
        'reflection': ['mirror', 'gray'], 'counter': ['hud', 'budget']}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
_mem = {'out': None, 'model': None}


def canon(s):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def frame(state):
    g, own = {}, {}
    for o in sorted(state, key=lambda o: o['layer']):
        for i, row in enumerate(o.get('pixels', [])):
            for j, v in enumerate(row):
                if v != -1:
                    g[(o['x'] + i, o['y'] + j)] = v
                    own[(o['x'] + i, o['y'] + j)] = o
    return g, own


def comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]; c = set(st)
        while st:
            r, k = st.pop()
            for nb in ((r + 1, k), (r - 1, k), (r, k + 1), (r, k - 1)):
                if nb in blocks:
                    blocks.discard(nb); c.add(nb); st.append(nb)
        out.append(c)
    return out


def parse(state):
    g, own = frame(state)
    walls = [o for o in state if o['type'] == 'wall']
    A = walls[0]['x'] // 3 if walls else 0
    axis_sel = any(v == 0 for o in walls for row in o['pixels'] for v in row)
    targets = {}
    for o in state:
        if o['type'] == 'target':
            bl = set()
            for i, row in enumerate(o['pixels']):
                for j, v in enumerate(row):
                    if v != -1:
                        bl.add(((o['x'] + i) // 3, (o['y'] + j) // 3))
            targets[o['name']] = bl
    pb = set()
    for (x, y), v in g.items():
        if v == 5 and x % 3 == 0 and y % 3 == 0 and own[(x, y)]['type'] == 'player':
            pb.add((x // 3, y // 3))

    def centre(b):
        p = (3 * b[0] + 1, 3 * b[1] + 1)
        if p not in g:
            return 'empty'
        if g[p] == 0 and not axis_sel:
            return 'sel'
        return 'empty' if own[p]['type'] == 'player' else 'other'
    pieces, sel = [], None
    zero = {b for b in pb if centre(b) == 'sel'}
    if zero:
        S = set(zero); grow = True
        while grow:
            grow = False
            for b in pb - S:
                if centre(b) == 'other' and any((b[0] + d[0], b[1] + d[1]) in S for d in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    S.add(b); grow = True
        pieces = [S] + comps(pb - S)
        sel = 0
    else:
        pieces = comps(pb)
        if axis_sel:
            sel = 'axis'
        else:
            for i, p in enumerate(pieces):
                if all(centre(b) == 'other' for b in p):
                    sel = i
                    break
    cnt = [o for o in state if o['type'] == 'counter']
    h = cnt[0]['h'] if cnt else 64
    n = 128 - h if h < 64 else 0
    return {'A': A, 'pieces': pieces, 'sel': sel, 'targets': targets, 'n': n, 'hist': []}


def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(m, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    m = dict(m)
    m['pieces'] = [set(p) for p in m['pieces']]
    m['hist'] = list(m['hist'])
    if aid == 7:
        if m['hist']:
            m['A'], m['pieces'] = m['hist'].pop()
            m['pieces'] = [set(p) for p in m['pieces']]
        return m
    m['n'] += 1
    if aid == 5:
        ordr = order(m['pieces'])
        if m['sel'] == 'axis':
            m['sel'] = ordr[0] if ordr else 'axis'
        elif m['sel'] is None:
            m['sel'] = 'axis'
        else:
            k = ordr.index(m['sel'])
            m['sel'] = ordr[k + 1] if k + 1 < len(ordr) else 'axis'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if d is None or m['sel'] is None:
        return m
    snap = (m['A'], [set(p) for p in m['pieces']])
    if m['sel'] == 'axis':
        if d[1] == 0 and 0 <= m['A'] + d[0] < N:
            m['A'] += d[0]
            m['hist'].append(snap)
        return m
    i = m['sel']
    moved = {(r + d[0], c + d[1]) for r, c in m['pieces'][i]}
    others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i]) if len(m['pieces']) > 1 else set()
    if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
        m['pieces'][i] = moved
        m['hist'].append(snap)
    return m


def render(m):
    stacks = {}  # block -> list of sprites top-down: (type, id, ring, solid_centre, fill)
    def add(b, spr):
        stacks.setdefault(b, []).append(spr)
    for i, p in enumerate(m['pieces']):
        for b in p:
            add(b, ('player', i, 5, None, 0 if m['sel'] == i else -1))
    A = m['A']
    for i, p in enumerate(m['pieces']):
        for r, c in p:
            if 0 <= 2 * A - r < N:
                add((2 * A - r, c), ('reflection', 0, 4, None, 4))
    for name, bl in m['targets'].items():
        for b in bl:
            add(b, ('target', name, 11, 11, None))
    for c in range(N):
        add((A, c), ('wall', 0, 10, 0 if m['sel'] == 'axis' else None, -1))
    g, own = {}, {}
    for (r, c), st in stacks.items():
        top = st[0]
        for i in range(3):
            for j in range(3):
                p = (3 * r + i, 3 * c + j)
                if (i, j) != (1, 1):
                    g[p], own[p] = top[2], top
                else:
                    sol = [s for s in st if s[3] is not None]
                    if sol:
                        g[p], own[p] = sol[0][3], sol[0]
                    elif top[4] != -1:
                        g[p], own[p] = top[4], top
    return g, own


def cells_to_obj(cells, g, typ, name):
    xs = [p[0] for p in cells]; ys = [p[1] for p in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for (x, y) in cells:
        pix[x - x0][y - y0] = g[(x, y)]
    return {'name': name, 'type': typ, 'tags': list(TAGS[typ]), 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': LAYER[typ], 'pixels': pix}


def pixel_comps(cells):
    return comps(cells)


def extract(m, g, own):
    out = []
    zeros = {p for p, v in g.items() if v == 0}
    h = max(0, min(64, 128 - m['n']))
    for typ, prefix, off in (('wall', 'wall_h_', 0), ('player', 'piece_', 1 if h < 64 else 0)):
        mine = {p for p, s in own.items() if s[0] == typ}
        cs = sorted(pixel_comps(mine | zeros), key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
        for k, c in enumerate(cs):
            if c & mine:
                out.append(cells_to_obj(c, g, typ, prefix + str(k + off)))
    mine = {p for p, s in own.items() if s[0] == 'reflection'}
    cs = sorted(pixel_comps(mine), key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    for k, c in enumerate(cs):
        out.append(cells_to_obj(c, g, 'reflection', 'reflection_' + str(k)))
    tg = {}
    for p, s in own.items():
        if s[0] == 'target':
            tg.setdefault(s[1], set()).add(p)
    cs = sorted(tg.values(), key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    for k, c in enumerate(cs):
        out.append(cells_to_obj(c, g, 'target', 'target_' + str(k)))
    out.append({'name': 'counter', 'type': 'counter', 'tags': list(TAGS['counter']), 'x': 63, 'y': 64 - h,
                'w': 1, 'h': h, 'layer': 5})
    return out


def transition_function(state, action):
    if _mem['out'] is not None and canon(state) == _mem['out']:
        m = _mem['model']
    else:
        m = parse(state)
    m = step(m, action)
    out = extract(m, *render(m))
    _mem['out'], _mem['model'] = canon(out), m
    return out
