# Mechanics: 21x21 lattice of 3x3 cells; row axis A (ACTION1/2) and column axis B (ACTION3/4) are walls; ACTION5 cycles A -> B -> pieces -> A.
# Selected piece moves 3 px with ACTION1-4. Each piece cell mirrors across B into a solid gray reflection (skip on-axis sources, piece-covered dests);
# if a piece touches row A, its whole B-mirror is invisible (erases wall rings). Mirrors across A and across both axes are invisible too.
# Targets render on top; any target cell under a piece/reflection/invisible mirror keeps only its centre pixel. Names rank 4-conn comps by bbox,
# counting foreign 0-dots (walls/pieces) and invisible mirror comps (reflections). Unconfirmed: target grouping, invisible-cell priorities.
import json

N = 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
TAGS = {'target': ['goal', 'yellow'], 'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
_memo = {'out': None, 'model': None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def blocks_of(o, want=None):
    res = set()
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1 and (want is None or v == want):
                res.add(((o['x'] + i) // 3, (o['y'] + j) // 3))
    return res


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    rows, cols = [0] * N, [0] * N
    wb = set()
    for o in walls:
        wb |= blocks_of(o, 10)
    for a, b in wb:
        rows[a] += 1
        cols[b] += 1
    A = max(range(N), key=lambda a: rows[a])
    B = max(range(N), key=lambda b: cols[b])
    targets = set()
    for o in state:
        if o['type'] == 'target':
            targets |= blocks_of(o, 11)
    pieces = []
    for o in sorted([o for o in state if o['type'] == 'player'], key=lambda o: (o['x'], o['y'])):
        pieces.append({'cells': blocks_of(o, 5), 'zero': blocks_of(o, 0)})
    sel = None
    wz = set()
    for o in walls:
        wz |= blocks_of(o, 0)
    if any(a == A and b != B for a, b in wz):
        sel = 'A'
    elif any(b == B and a != A for a, b in wz):
        sel = 'B'
    else:
        for k, p in enumerate(pieces):
            if any(a != A and b != B for a, b in p['zero']) or (p['zero'] and not wz):
                sel = k
        if sel is None:
            sel = 'A'
    return {'A': A, 'B': B, 'sel': sel, 'pieces': [p['cells'] for p in pieces], 'targets': targets}


def step(m, action):
    m = {'A': m['A'], 'B': m['B'], 'sel': m['sel'], 'pieces': [set(p) for p in m['pieces']], 'targets': set(m['targets'])}
    sel = m['sel']
    if action == 5:
        order = ['A', 'B'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(sel) + 1) % len(order)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None:
        return m
    if sel == 'A' and d[0]:
        if 0 <= m['A'] + d[0] < N:
            m['A'] += d[0]
    elif sel == 'B' and d[1]:
        if 0 <= m['B'] + d[1] < N:
            m['B'] += d[1]
    elif isinstance(sel, int):
        moved = {(a + d[0], b + d[1]) for a, b in m['pieces'][sel]}
        others = set().union(*[p for k, p in enumerate(m['pieces']) if k != sel]) if len(m['pieces']) > 1 else set()
        if all(0 <= a < N and 0 <= b < N for a, b in moved) and not (moved & others):
            m['pieces'][sel] = moved
    return m


def mirrors(m):
    A, B = m['A'], m['B']
    allp = set().union(*m['pieces']) if m['pieces'] else set()
    vis, inv = set(), set()
    on = lambda a, b: 0 <= a < N and 0 <= b < N
    for p in m['pieces']:
        hidden = any(a == A for a, b in p)
        for a, b in p:
            if b != B:
                c = (a, 2 * B - b)
                if on(*c) and c not in allp:
                    (inv if hidden else vis).add(c)
            if a != A:
                c = (2 * A - a, b)
                if on(*c):
                    inv.add(c)
                if b != B:
                    c = (2 * A - a, 2 * B - b)
                    if on(*c):
                        inv.add(c)
    inv -= vis
    return vis, inv


def ring(a, b):
    return [(3 * a + i, 3 * b + j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]


def render(m, counter):
    A, B, sel = m['A'], m['B'], m['sel']
    val, own = {}, {}
    wall = {(A, b) for b in range(N)} | {(a, B) for a in range(N)}
    for a, b in wall:
        for p in ring(a, b):
            val[p], own[p] = 10, 'wall'
        if (sel == 'A' and a == A) or (sel == 'B' and b == B):
            c = (3 * a + 1, 3 * b + 1)
            val[c], own[c] = 0, 'wall'
    vis, inv = mirrors(m)
    for a, b in vis:
        for i in range(3):
            for j in range(3):
                p = (3 * a + i, 3 * b + j)
                val[p], own[p] = 4, 'reflection'
    for a, b in inv:
        for p in ring(a, b):
            if own.get(p) == 'wall':
                del val[p], own[p]
    allp = set()
    for k, piece in enumerate(m['pieces']):
        allp |= piece
        for a, b in piece:
            for p in ring(a, b):
                val[p], own[p] = 5, 'player'
            c = (3 * a + 1, 3 * b + 1)
            if sel == k:
                val[c], own[c] = 0, 'player'
            elif val.get(c) == 0:
                own[c] = 'player'
    mask = allp | vis | inv
    for a, b in m['targets']:
        cells = [(3 * a + 1, 3 * b + 1)] if (a, b) in mask else [(3 * a + i, 3 * b + j) for i in range(3) for j in range(3)]
        for p in cells:
            val[p], own[p] = 11, 'target'
    out = [counter] if counter else []
    for t in ('wall', 'player'):
        extra = {p for p, v in val.items() if v == 0 and own[p] != t}
        out += extract(val, own, t, extra)
    invpx = {(3 * a + i, 3 * b + j) for a, b in inv for i in range(3) for j in range(3)}
    out += extract(val, own, 'reflection', invpx - {p for p in own if own[p] == 'reflection'})
    out += targets_out(val, own, m['targets'], mask)
    return out


def comps(pts):
    pts, res = set(pts), []
    while pts:
        st = [pts.pop()]
        c = set(st)
        while st:
            x, y = st.pop()
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if q in pts:
                    pts.remove(q)
                    c.add(q)
                    st.append(q)
        res.append(c)
    return res


def obj(name, t, px, val):
    xs, ys = [p[0] for p in px], [p[1] for p in px]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[val[(x0 + i, y0 + j)] if (x0 + i, y0 + j) in px else -1 for j in range(h)] for i in range(w)]
    if t == 'wall':
        v = w >= h
        name = ('wall_v_' if v else 'wall_h_') + name
        tags = ['axis', 'vertical' if v else 'horizontal']
    else:
        name = {'player': 'piece_', 'reflection': 'reflection_', 'target': 'target_'}[t] + name
        tags = TAGS[t]
    return {'name': name, 'type': t, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': LAYER[t], 'tags': list(tags), 'pixels': pix}


def extract(val, own, t, extra):
    mine = {p for p in own if own[p] == t}
    cs = sorted(comps(mine | extra), key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    res = []
    for k, c in enumerate(cs):
        px = c & mine
        if px:
            res.append(obj(str(k), t, px, val))
    return res


def targets_out(val, own, targets, mask):
    tl = sorted(targets)
    parent = {c: c for c in tl}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c
    for i, c in enumerate(tl):
        for d in tl[i + 1:]:
            da, db = abs(c[0] - d[0]), abs(c[1] - d[1])
            if da + db <= 2 or (sorted((da, db)) == [1, 2] and c not in mask and d not in mask):
                parent[find(c)] = find(d)
    groups = {}
    for c in tl:
        groups.setdefault(find(c), set()).add(c)
    pxs = []
    for g in groups.values():
        px = {(3 * a + i, 3 * b + j) for a, b in g for i in range(3) for j in range(3)}
        pxs.append({p for p in px if own.get(p) == 'target'})
    pxs.sort(key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
    return [obj(str(k), 'target', px, val) for k, px in enumerate(pxs)]


def transition_function(state, action):
    aid = action.get('action_id') if isinstance(action, dict) else action
    counter = next((dict(o) for o in state if o['type'] == 'counter'), None)
    if _memo['out'] is not None and canon(state) == _memo['out']:
        m = _memo['model']
    else:
        m = parse(state)
    m2 = step(m, aid)
    out = render(m2, counter)
    _memo['out'], _memo['model'] = canon(out), m2
    return out
