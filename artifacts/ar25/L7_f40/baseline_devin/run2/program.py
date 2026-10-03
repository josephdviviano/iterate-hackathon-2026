# Mechanics: two mirror axes (wall row ah, wall column av, 3x3 blocks with holes), two pieces, static targets.
# ACTION5 cycles selection h-axis -> v-axis -> piece0 -> piece1; ACTION1/2 move h-axis/piece in x, ACTION3/4 v-axis/piece in y (+-3, in bounds).
# Pieces are mirrored across the v-axis (visible gray 4) and across the h-axis and both axes (invisible occluders whose hole shows 4).
# Frame is composited by layer (wall1<target2<reflection3<piece4; holes transparent, selected fill 0) and re-extracted:
# 4-conn components, 1-px dots dropped but indexed, 0-dots indexed in the other of wall/player, targets clustered (gap<=3). Hypothesis: these naming rules.
N = 21
SZ = 63
_mem = {}


def _cells(o):
    out = {}
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def _blocks(cs):
    return {(x // 3, y // 3) for (x, y) in cs}


def parse(state):
    walls, targets, pieces, frame = [], set(), [], {}
    for o in state:
        cs = _cells(o)
        frame.update(cs)
        if o['type'] == 'wall':
            walls += [c for c, v in cs.items() if v == 10]
        elif o['type'] == 'target':
            targets |= _blocks(cs)
        elif o['type'] == 'player':
            pieces.append(((o['x'], o['y']), _blocks([c for c, v in cs.items() if v == 5])))
    rows, cols = {}, {}
    for b in _blocks(walls):
        rows.setdefault(b[0], set()).add(b[1])
        cols.setdefault(b[1], set()).add(b[0])
    ah = max(rows, key=lambda k: len(rows[k]))
    av = max(cols, key=lambda k: len(cols[k]))
    pieces = [p for _, p in sorted(pieces, key=lambda t: t[0])]
    sel = None
    for (x, y), v in frame.items():
        if v != 0 or x % 3 != 1 or y % 3 != 1:
            continue
        b = (x // 3, y // 3)
        if b[0] == ah and b[1] != av:
            sel = 'h'
        elif b[1] == av and b[0] != ah:
            sel = 'v'
        else:
            for i, p in enumerate(pieces):
                if b in p:
                    sel = i
        if sel is not None:
            break
    return {'ah': ah, 'av': av, 'sel': sel, 'pieces': pieces, 'targets': targets}


def step(m, action):
    m = dict(m, pieces=[set(p) for p in m['pieces']])
    order = ['h', 'v'] + list(range(len(m['pieces'])))
    if action == 5:
        m['sel'] = order[(order.index(m['sel']) + 1) % len(order)] if m['sel'] in order else 'h'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action if isinstance(action, int) else 0)
    if d is None:
        return m
    s = m['sel']
    if s == 'h' and d[0]:
        if 0 <= m['ah'] + d[0] < N:
            m['ah'] += d[0]
    elif s == 'v' and d[1]:
        if 0 <= m['av'] + d[1] < N:
            m['av'] += d[1]
    elif isinstance(s, int):
        nb = {(bx + d[0], by + d[1]) for bx, by in m['pieces'][s]}
        if all(0 <= bx < N and 0 <= by < N for bx, by in nb):
            m['pieces'][s] = nb
    return m


def reflections(m):
    ah, av = m['ah'], m['av']
    ok = lambda b: 0 <= b[0] < N and 0 <= b[1] < N
    vis, hid = set(), set()
    for p in m['pieces']:
        for bx, by in p:
            if by != av and ok((bx, 2 * av - by)):
                vis.add((bx, 2 * av - by))
            if bx != ah and ok((2 * ah - bx, by)):
                hid.add((2 * ah - bx, by))
            if bx != ah and by != av and ok((2 * ah - bx, 2 * av - by)):
                hid.add((2 * ah - bx, 2 * av - by))
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    return vis - occ, hid - occ - vis


def composite(m):
    # sprites top-down: (type, id, blocks, ring value, hole fill)
    sp = []
    for i, p in enumerate(m['pieces']):
        sp.append(('player', ('p', i), p, 5, 0 if m['sel'] == i else None))
    vis, hid = reflections(m)
    sp.append(('reflection', 'hid', hid, 'BG', 4))
    sp.append(('reflection', 'vis', vis, 4, 4))
    sp.append(('target', 't', m['targets'], 11, 11))
    hb = {(m['ah'], by) for by in range(N)}
    vb = {(bx, m['av']) for bx in range(N)}
    cross = (m['ah'], m['av'])
    sp.append(('wall', 'h', hb - {cross}, 10, 0 if m['sel'] == 'h' else None))
    sp.append(('wall', 'v', vb - {cross}, 10, 0 if m['sel'] == 'v' else None))
    sp.append(('wall', 'x', {cross}, 10, 0 if m['sel'] in ('h', 'v') else None))
    frame = {}
    for x in range(SZ):
        for y in range(SZ):
            b = (x // 3, y // 3)
            centre = x % 3 == 1 and y % 3 == 1
            hole = None
            res = None
            for typ, sid, blocks, ring, fill in sp:
                if b not in blocks:
                    continue
                if not centre or ring == 11:
                    res = (ring, typ, sid)
                    break
                if hole is None and fill is not None:
                    hole = (fill, typ, sid)
            if res is None:
                res = hole
            if res is not None:
                frame[(x, y)] = res
    return frame


def _comps(cells, near):
    cells = set(cells)
    out = []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            x, y = st.pop()
            for dx in range(-near, near + 1):
                for dy in range(-near, near + 1):
                    if not 0 < abs(dx) + abs(dy) <= near:
                        continue
                    c = (x + dx, y + dy)
                    if c in cells:
                        cells.remove(c)
                        comp.add(c)
                        st.append(c)
        out.append(comp)
    return out


TAGS = {'player': ['movable', 'black'], 'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}


def extract(frame):
    entries = {t: [] for t in LAYER}
    comp_of = {}
    comps = []
    hid = [c for c, f in frame.items() if f[2] == 'hid']
    for comp in _comps(hid, 1):
        entries['reflection'].append((min(comp), None))
    for typ in LAYER:
        cs = [c for c, (v, t, s) in frame.items() if t == typ and v != 0 and s != 'hid']
        for comp in _comps(cs, 6 if typ == 'target' else 1):
            k = len(comps)
            comps.append((typ, comp))
            for c in comp:
                comp_of[c] = k
    pix = [dict((c, frame[c][0]) for c in comp) for _, comp in comps]
    for c, (v, t, s) in frame.items():
        if v != 0:
            continue
        nb = {comp_of[n] for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)) if n in comp_of}
        types = {comps[k][0] for k in nb}
        for k in nb:
            pix[k][c] = 0
        other = 'player' if 'wall' in types else 'wall' if 'player' in types else t
        entries[other].append((c, None))
    for k, (typ, comp) in enumerate(comps):
        p = pix[k]
        if len(p) == 1 and typ != 'target':
            entries[typ].append((min(p), None))
            continue
        x0 = min(c[0] for c in p); y0 = min(c[1] for c in p)
        w = max(c[0] for c in p) - x0 + 1; h = max(c[1] for c in p) - y0 + 1
        grid = [[p.get((x0 + i, y0 + j), -1) for j in range(h)] for i in range(w)]
        o = {'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': LAYER[typ], 'pixels': grid}
        entries[typ].append(((x0, y0), o))
    out = []
    for typ, es in entries.items():
        es.sort(key=lambda e: (e[0], e[1] is None))
        for i, (_, o) in enumerate(es):
            if o is None:
                continue
            if typ == 'wall':
                hz = o['h'] > o['w']
                o['name'] = ('wall_h_%d' if hz else 'wall_v_%d') % i
                o['tags'] = ['axis', 'horizontal' if hz else 'vertical']
            else:
                o['name'] = ('piece_%d' if typ == 'player' else typ + '_%d') % i
                o['tags'] = list(TAGS[typ])
            out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    prev = _mem.get('out')
    if prev is not None and _canon(prev) == _canon(state):
        m = _mem['model']
    if m['sel'] is None:
        m['sel'] = 'h'
    m2 = step(m, action)
    out = extract(composite(m2))
    out += [dict(o) for o in state if o['type'] not in LAYER]
    _mem['out'] = out
    _mem['model'] = m2
    return out


def _canon(s):
    return sorted(repr(sorted(o.items())) for o in s)
