# Mechanics: 21x21 lattice of 3x3 blocks; h-axis wall row ax (A1/A2 move), v-axis wall column ay (A3/A4 move);
# A5 cycles selection h -> v -> pieces (bbox order) -> h; selected piece moves 1 block per A1-A4 within board.
# Each piece block mirrors across the v-axis into a solid gray reflection (not onto pieces); h/diagonal mirrors
# are hidden; a piece on the h-row makes its whole v-mirror hidden. Hidden cells erase walls and mask targets
# (masked target shows only its centre). Re-extract 4-conn comps; targets merge by bbox gap<=3; names offset by 0-dots.
N = 21
_last = {}


def _pix(state, typ):
    G = {}
    for o in state:
        if o['type'] != typ or not o.get('pixels'):
            continue
        p = o['pixels']
        for i in range(o['w']):
            for j in range(o['h']):
                if p[i][j] != -1:
                    G[(o['x'] + i, o['y'] + j)] = p[i][j]
    return G


def _comps(S):
    S = set(S)
    out = []
    while S:
        s = S.pop()
        st, c = [s], {s}
        while st:
            a, b = st.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if n in S:
                    S.remove(n)
                    c.add(n)
                    st.append(n)
        out.append(c)
    return out


def _blocks(G):
    return {(x // 3, y // 3) for (x, y) in G}


def parse(state):
    W, P, T = _pix(state, 'wall'), _pix(state, 'player'), _pix(state, 'target')
    rc, cc = {}, {}
    for a, b in _blocks(W):
        rc[a] = rc.get(a, 0) + 1
        cc[b] = cc.get(b, 0) + 1
    ax = max(rc, key=lambda k: rc[k]) if rc else 0
    ay = max(cc, key=lambda k: cc[k]) if cc else 0
    pieces = sorted(_comps(_blocks(P)), key=lambda c: min((3 * a, 3 * b) for a, b in c))
    zeros = {(x // 3, y // 3) for G in (W, P) for (x, y), v in G.items() if v == 0}
    sel = None
    for i, c in enumerate(pieces):
        if any(k in zeros and k[0] != ax and k[1] != ay for k in c):
            sel = i
    if sel is None:
        if any(k[0] == ax and k[1] != ay for k in zeros):
            sel = 'h'
        elif any(k[1] == ay and k[0] != ax for k in zeros):
            sel = 'v'
        else:
            sel = 'h'
    counter = [o for o in state if o['type'] == 'counter']
    return {'ax': ax, 'ay': ay, 'sel': sel, 'pieces': pieces, 'targets': _blocks(T), 'counter': counter}


def step(m, action):
    m = dict(m)
    aid = action if isinstance(action, int) else action.get('action_id')
    sel = m['sel']
    if aid == 5:
        order = ['h', 'v'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(sel) + 1) % len(order)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if d is None:
        return m
    if sel == 'h' and d[0]:
        m['ax'] = min(N - 1, max(0, m['ax'] + d[0]))
    elif sel == 'v' and d[1]:
        m['ay'] = min(N - 1, max(0, m['ay'] + d[1]))
    elif isinstance(sel, int):
        moved = {(a + d[0], b + d[1]) for a, b in m['pieces'][sel]}
        if all(0 <= a < N and 0 <= b < N for a, b in moved):
            m['pieces'] = [moved if i == sel else c for i, c in enumerate(m['pieces'])]
    return m


def mirrors(m):
    ax, ay = m['ax'], m['ay']
    allp = set().union(*m['pieces']) if m['pieces'] else set()
    on = lambda k: 0 <= k[0] < N and 0 <= k[1] < N and k not in allp
    vis, hid = set(), set()
    for c in m['pieces']:
        v = {(a, 2 * ay - b) for a, b in c if b != ay}
        h = {(2 * ax - a, b) for a, b in c if a != ax}
        dg = {(2 * ax - a, 2 * ay - b) for a, b in c if a != ax and b != ay}
        if any(a == ax for a, b in c):
            hid |= v
        else:
            vis |= v
        hid |= h | dg
    vis = {k for k in vis if on(k)}
    hid = {k for k in hid if on(k)} - vis
    return vis, hid, allp


def render(m):
    ax, ay, sel = m['ax'], m['ay'], m['sel']
    vis, hid, allp = mirrors(m)
    mask = vis | hid | allp
    selp = m['pieces'][sel] if isinstance(sel, int) else set()
    F = {}
    for a in range(N):
        for b in range(N):
            k = (a, b)
            cells = {}
            if (a == ax or b == ay) and k not in hid:
                dot = (sel == 'h' and a == ax) or (sel == 'v' and b == ay)
                for i in range(3):
                    for j in range(3):
                        if (i, j) != (1, 1):
                            cells[(i, j)] = ('wall', 10)
                        elif dot:
                            cells[(i, j)] = ('wall', 0)
            if k in vis:
                for i in range(3):
                    for j in range(3):
                        cells[(i, j)] = ('reflection', 4)
            if k in allp:
                for i in range(3):
                    for j in range(3):
                        if (i, j) != (1, 1):
                            cells[(i, j)] = ('player', 5)
                        elif k in selp:
                            cells[(i, j)] = ('player', 0)
                        elif (i, j) in cells and cells[(i, j)][1] == 0:
                            cells[(i, j)] = ('player', 0)
            if k in m['targets']:
                for i in range(3):
                    for j in range(3):
                        if (i, j) == (1, 1) or k not in mask:
                            cells[(i, j)] = ('target', 11)
            for (i, j), v in cells.items():
                F[(3 * a + i, 3 * b + j)] = v
    return F


TAGS = {'wall': (['axis'], 1), 'player': (['movable', 'black'], 4),
        'reflection': (['mirror', 'gray'], 3), 'target': (['goal', 'yellow'], 2)}


def _gap(p, q):
    return max(0, q[0] - p[2] - 1, p[0] - q[2] - 1), max(0, q[1] - p[3] - 1, p[1] - q[3] - 1)


def extract(F, m):
    groups = {t: [] for t in TAGS}
    for t in TAGS:
        S = [k for k, v in F.items() if v[0] == t]
        groups[t] = _comps(S)
    bb = lambda c: [min(x for x, y in c), min(y for x, y in c), max(x for x, y in c), max(y for x, y in c)]
    tg = [(bb(c), set(c)) for c in groups['target']]
    merged = True
    while merged:
        merged = False
        for i in range(len(tg)):
            for j in range(i + 1, len(tg)):
                gx, gy = _gap(tg[i][0], tg[j][0])
                if gx <= 3 and gy <= 3:
                    c = tg[i][1] | tg[j][1]
                    tg[i] = (bb(c), c)
                    del tg[j]
                    merged = True
                    break
            if merged:
                break
    groups['target'] = [c for _, c in tg]
    zeros = [k for k, v in F.items() if v[1] == 0]
    hidden = [(3 * a, 3 * b) for a, b in (min(c) for c in _comps(mirrors(m)[1]))]
    out = []
    for t, cs in groups.items():
        cs = sorted(cs, key=lambda c: tuple(bb(c)[:2]))
        others = sorted(z for z in zeros if F[z][0] != t)
        for r, c in enumerate(cs):
            x0, y0, x1, y1 = bb(c)
            idx = r
            if t in ('wall', 'player'):
                idx += sum(1 for z in others if z < (x0, y0))
            if t == 'reflection':
                idx += sum(1 for z in hidden if z < (x0, y0))
            w, h = x1 - x0 + 1, y1 - y0 + 1
            px = [[-1] * h for _ in range(w)]
            for (x, y) in c:
                px[x - x0][y - y0] = F[(x, y)][1]
            tags, layer = TAGS[t]
            if t == 'wall':
                vv = w >= h
                name = 'wall_%s_%d' % ('v' if vv else 'h', idx)
                tags = tags + ['vertical' if vv else 'horizontal']
            else:
                name = '%s_%d' % ({'player': 'piece'}.get(t, t), idx)
            out.append({'name': name, 'type': t, 'tags': tags, 'x': x0, 'y': y0, 'w': w, 'h': h,
                        'layer': layer, 'pixels': px})
    return out + [dict(c) for c in m['counter']]


def _key(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def transition_function(state, action):
    m = parse(state)
    if _last.get('key') == _key(state):
        m['pieces'], m['sel'] = _last['pieces'], _last['sel']
    m2 = step(m, action)
    out = extract(render(m2), m2)
    _last.update(key=_key(out), pieces=m2['pieces'], sel=m2['sel'])
    return out
