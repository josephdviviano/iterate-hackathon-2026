# Mechanics: two wall axes on a 21x21 grid of 3x3 blocks: h-axis = block row AX (moved by A1/A2), v-axis = block column AY
# (A3/A4); A5 cycles selection h -> v -> pieces (bbox order) -> h; a selected piece moves x-1/x+1/y-1/y+1 for A1..A4.
# Each piece block (a,b), b!=AY, mirrors to (a,2AY-b) as gray unless covered; a piece touching row AX makes its mirror invisible.
# Invisible mirror cells erase what is beneath and mask targets; masked target blocks show only the centre pixel.
# Re-extraction: 4-conn comps ranked by bbox incl. 0-dot comps; targets merge at cell scale. Counter static (no global rule).
N = 21
_last = {'out': None, 'model': None}


def _canon(s):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _pix(state):
    pm = {}
    for o in state:
        for i, row in enumerate(o.get('pixels', [])):
            for j, v in enumerate(row):
                if v >= 0:
                    pm[(o['x'] + i, o['y'] + j)] = (o['type'], v)
    return pm


def _blocks(pm, typ, val=None):
    out = set()
    for (x, y), (t, v) in pm.items():
        if t == typ and (val is None or v == val) and x < 3 * N and y < 3 * N:
            if (x % 3, y % 3) != (1, 1):
                out.add((x // 3, y // 3))
    return out


def _bcomps(cells):
    cells = set(cells)
    out = []
    while cells:
        st = [cells.pop()]
        c = set(st)
        while st:
            a, b = st.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if n in cells:
                    cells.remove(n)
                    c.add(n)
                    st.append(n)
        out.append(c)
    return out


def _parse(state):
    pm = _pix(state)
    wb = _blocks(pm, 'wall')
    ax = max(range(N), key=lambda r: sum(1 for a, b in wb if a == r))
    ay = max(range(N), key=lambda c: sum(1 for a, b in wb if b == c))
    zeros = {(x // 3, y // 3) for (x, y), (t, v) in pm.items() if v == 0 and t == 'wall'}
    pz = {(x // 3, y // 3) for (x, y), (t, v) in pm.items() if v == 0 and t == 'player'}
    sel = None
    if any(a == ax and b != ay for a, b in zeros):
        sel = 'h'
    elif any(b == ay and a != ax for a, b in zeros):
        sel = 'v'
    pieces = sorted((sorted(c) for c in _bcomps(_blocks(pm, 'player'))), key=lambda c: _bb(c))
    if sel is None:
        for i, p in enumerate(pieces):
            if any(c in pz for c in p):
                sel = i
    tg = {(x // 3, y // 3) for (x, y), (t, v) in pm.items() if t == 'target'}
    other = [o for o in state if 'pixels' not in o]
    return {'ax': ax, 'ay': ay, 'sel': sel, 'pieces': pieces, 'targets': sorted(tg), 'other': other}


def _bb(c):
    return (min(a for a, b in c), min(b for a, b in c))


def _step(m, action):
    m = dict(m)
    m['pieces'] = [list(p) for p in m['pieces']]
    aid = action if isinstance(action, int) else action.get('action_id')
    sel = m['sel']
    if aid == 5:
        order = sorted(range(len(m['pieces'])), key=lambda i: _bb(m['pieces'][i]))
        if sel == 'h':
            m['sel'] = 'v'
        elif sel == 'v':
            m['sel'] = order[0] if order else 'h'
        elif isinstance(sel, int):
            k = order.index(sel)
            m['sel'] = order[k + 1] if k + 1 < len(order) else 'h'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if d is None:
        return m
    if sel == 'h' and d[0]:
        if 0 <= m['ax'] + d[0] < N:
            m['ax'] += d[0]
    elif sel == 'v' and d[1]:
        if 0 <= m['ay'] + d[1] < N:
            m['ay'] += d[1]
    elif isinstance(sel, int):
        p = [(a + d[0], b + d[1]) for a, b in m['pieces'][sel]]
        others = {c for i, q in enumerate(m['pieces']) if i != sel for c in q}
        if all(0 <= a < N and 0 <= b < N for a, b in p) and not (set(p) & others):
            m['pieces'][sel] = p
    return m


def _render(m):
    ax, ay, sel = m['ax'], m['ay'], m['sel']
    occ = {c for p in m['pieces'] for c in p}
    vis, inv = set(), set()
    hid = set()
    for p in m['pieces']:
        touch = any(a == ax for a, b in p)
        for a, b in p:
            if b == ay:
                continue
            d = (a, 2 * ay - b)
            if 0 <= d[1] < N and d not in occ:
                (inv if touch else vis).add(d)
            if a != ax and 0 <= 2 * ax - a < N:
                for hb in (b, 2 * ay - b):
                    if 0 <= hb < N:
                        hid.add((2 * ax - a, hb))
    mask = occ | vis | hid | inv
    layers = []
    walls = {}
    for i in range(N):
        for c in ((ax, i), (i, ay)):
            walls[c] = 0 if (sel == 'h' and c[0] == ax) or (sel == 'v' and c[1] == ay) else -1
    layers.append(('wall', {c: (10, z) for c, z in walls.items()}))
    rl = {c: (4, 4) for c in vis}
    rl.update({c: (-2, -1) for c in inv})
    layers.append(('reflection', rl))
    pl = {}
    for i, p in enumerate(m['pieces']):
        for c in p:
            pl[c] = (5, 0 if sel == i else -1)
    layers.append(('player', pl))
    tl = {c: (-1 if c in mask else 11, 11) for c in m['targets']}
    layers.append(('target', tl))
    pm = {}
    for x in range(3 * N):
        for y in range(3 * N):
            c = (x // 3, y // 3)
            centre = (x % 3, y % 3) == (1, 1)
            for typ, L in reversed(layers):
                if c in L:
                    v = L[c][1] if centre else L[c][0]
                    if v == -2:
                        break
                    if v >= 0:
                        pm[(x, y)] = (typ, v)
                        break
    return pm, inv | hid


def _pcomps(cells):
    cells = set(cells)
    out = []
    while cells:
        st = [cells.pop()]
        c = set(st)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    c.add(n)
                    st.append(n)
        out.append(c)
    return out


def _obj(name, typ, cells, pm, layer, tags):
    xs = [p[0] for p in cells]
    ys = [p[1] for p in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    px = [[pm[(x0 + i, y0 + j)][1] if (x0 + i, y0 + j) in cells else -1 for j in range(h)] for i in range(w)]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': layer, 'tags': tags, 'pixels': px}


def _bbkey(c):
    return (min(p[0] for p in c), min(p[1] for p in c))


def _extract(pm, inv):
    own = {}
    for p, (t, v) in pm.items():
        if v != 0:
            own[p] = t
    for p, (t, v) in pm.items():
        if v == 0:
            x, y = p
            ns = [own[n] for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if n in own]
            own[p] = ns[0] if ns else 'dot'
    zero = {p for p, (t, v) in pm.items() if v == 0}
    out = []
    for typ, layer, tags in (('wall', 1, None), ('player', 4, ['movable', 'black'])):
        cs = sorted(_pcomps({p for p, t in own.items() if t == typ} | zero), key=_bbkey)
        for i, c in enumerate(cs):
            mine = {p for p in c if own[p] == typ}
            if not any(pm[p][1] != 0 for p in mine):
                continue
            if typ == 'wall':
                o = _obj('', typ, mine, pm, layer, None)
                v = o['w'] >= o['h']
                o['name'] = 'wall_%s_%d' % ('v' if v else 'h', i)
                o['tags'] = ['axis', 'vertical' if v else 'horizontal']
            else:
                o = _obj('piece_%d' % i, typ, mine, pm, layer, tags)
            out.append(o)
    rc = [(_bbkey(c), c) for c in _pcomps({p for p, t in own.items() if t == 'reflection'})]
    ic = [((3 * min(a for a, b in c), 3 * min(b for a, b in c)), None) for c in _bcomps(inv)]
    for i, (k, c) in enumerate(sorted(rc + ic, key=lambda t: (t[0], t[1] is None))):
        if c is not None:
            out.append(_obj('reflection_%d' % i, 'reflection', c, pm, 3, ['mirror', 'gray']))
    groups = [[c, _bbox4(c)] for c in _pcomps({p for p, t in own.items() if t == 'target'})]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                A, B = groups[i][1], groups[j][1]
                gx = max(A[0], B[0]) - min(A[2], B[2]) - 1
                gy = max(A[1], B[1]) - min(A[3], B[3]) - 1
                if max(gx, gy) <= 3:
                    c = groups[i][0] | groups[j][0]
                    groups[i] = [c, _bbox4(c)]
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    for i, (c, bb) in enumerate(sorted(groups, key=lambda g: (g[1][0], g[1][1]))):
        out.append(_obj('target_%d' % i, 'target', c, pm, 2, ['goal', 'yellow']))
    return out


def _bbox4(c):
    xs = [p[0] for p in c]
    ys = [p[1] for p in c]
    return (min(xs), min(ys), max(xs), max(ys))


def transition_function(state, action):
    m = None
    if _last['out'] is not None and _canon(state) == _last['out']:
        m = _last['model']
    if m is None:
        m = _parse(state)
    m2 = _step(m, action)
    pm, inv = _render(m2)
    out = [dict(o) for o in m2['other']] + _extract(pm, inv)
    _last['out'] = _canon(out)
    _last['model'] = m2
    return out
