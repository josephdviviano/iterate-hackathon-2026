# Mechanics: 21x21 lattice of 3px blocks; axis A = wall row (ACTION1/2 move it), axis B = wall column (ACTION3/4);
# ACTION5 cycles A -> B -> pieces (by min block) -> A; selected piece moves by one block in 4 dirs (stays on board).
# Visible gray reflection = B-mirror (a,2B-b) of piece blocks; a piece touching row A has its B-mirror hidden; A- and AB-
# mirrors are always hidden. Hidden cells erase wall rings; targets under any piece/mirror cell keep only their centre.
# Re-extract: 4-conn comps, names ranked by (x,y) + foreign 0-dots / hidden comps. Unconfirmed: blocking rules for pieces.
N = 21
S = 63
_last = {'out': None, 'sel': None}


def canon(s):
    return sorted(repr(sorted(o.items())) for o in s)


def blk(X, Y):
    return (X // 3, Y // 3)


def parse(state):
    pix = {}
    for o in state:
        p = o.get('pixels')
        if not p:
            continue
        for i, row in enumerate(p):
            for j, v in enumerate(row):
                if v >= 0:
                    pix[(o['x'] + i, o['y'] + j)] = (o['type'], v, id(o))
    wall_r, wall_c = [0] * N, [0] * N
    wb, tb = set(), set()
    pieces = {}
    for (X, Y), (t, v, oid) in pix.items():
        if X >= S or Y >= S:
            continue
        b = blk(X, Y)
        if t == 'wall':
            wb.add(b)
        elif t == 'target':
            tb.add(b)
        elif t == 'player' and v == 5:
            pieces.setdefault(oid, set()).add(b)
    for (a, c) in wb:
        wall_r[a] += 1
        wall_c[c] += 1
    A = max(range(N), key=lambda r: wall_r[r])
    B = max(range(N), key=lambda c: wall_c[c])
    plist = sorted(pieces.values(), key=lambda s: min(s))
    sel = None
    for (X, Y), (t, v, oid) in pix.items():
        if v != 0 or X % 3 != 1 or Y % 3 != 1:
            continue
        a, c = blk(X, Y)
        if t == 'player' and a != A and c != B:
            for k, ps in enumerate(plist):
                if (a, c) in ps:
                    sel = k
            break
    if sel is None:
        for (X, Y), (t, v, oid) in pix.items():
            if v != 0:
                continue
            a, c = blk(X, Y)
            if a == A and c != B:
                sel = 'A'
                break
            if c == B and a != A:
                sel = 'B'
                break
    return {'A': A, 'B': B, 'pieces': plist, 'targets': tb, 'sel': sel}


def step(m, action):
    sel = m['sel']
    if isinstance(action, dict):
        return m
    order = ['A', 'B'] + list(range(len(m['pieces'])))
    if action == 5:
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else 'A'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None:
        return m
    if sel == 'A':
        if d[1] == 0 and 0 <= m['A'] + d[0] < N:
            m['A'] += d[0]
    elif sel == 'B':
        if d[0] == 0 and 0 <= m['B'] + d[1] < N:
            m['B'] += d[1]
    elif isinstance(sel, int):
        moved = {(a + d[0], c + d[1]) for a, c in m['pieces'][sel]}
        others = set().union(*[p for k, p in enumerate(m['pieces']) if k != sel]) if len(m['pieces']) > 1 else set()
        if all(0 <= a < N and 0 <= c < N for a, c in moved) and not (moved & others):
            m['pieces'][sel] = moved
    return m


def mirrors(m):
    A, B = m['A'], m['B']
    allp = set().union(*m['pieces']) if m['pieces'] else set()
    on = lambda a, c: 0 <= a < N and 0 <= c < N and (a, c) not in allp
    vis, hid = set(), set()
    for ps in m['pieces']:
        touch = any(a == A for a, c in ps)
        for a, c in ps:
            if c != B and on(a, 2 * B - c):
                (hid if touch else vis).add((a, 2 * B - c))
            if a != A and on(2 * A - a, c):
                hid.add((2 * A - a, c))
            if a != A and c != B and on(2 * A - a, 2 * B - c):
                hid.add((2 * A - a, 2 * B - c))
    return vis, hid - vis


def render(m):
    A, B, sel = m['A'], m['B'], m['sel']
    vis, hid = mirrors(m)
    owner = {}
    pieceof = {}
    for k, ps in enumerate(m['pieces']):
        for b in ps:
            pieceof[b] = k
    mask = set(pieceof) | vis | hid
    for a in range(N):
        for c in range(N):
            b = (a, c)
            cells = {}
            if (a == A or c == B) and b not in hid:
                dot = (a == A and sel == 'A') or (c == B and sel == 'B')
                for i in range(3):
                    for j in range(3):
                        if (i, j) != (1, 1):
                            cells[(i, j)] = ('wall', 10)
                        elif dot:
                            cells[(i, j)] = ('wall', 0)
            if b in vis:
                for i in range(3):
                    for j in range(3):
                        cells[(i, j)] = ('reflection', 4)
            if b in pieceof:
                for i in range(3):
                    for j in range(3):
                        if (i, j) != (1, 1):
                            cells[(i, j)] = ('player', 5)
                        elif sel == pieceof[b]:
                            cells[(i, j)] = ('player', 0)
                        elif (i, j) in cells and cells[(i, j)][1] == 0:
                            cells[(i, j)] = ('player', 0)
                        else:
                            cells.pop((i, j), None)
            if b in m['targets']:
                for i in range(3):
                    for j in range(3):
                        if (i, j) == (1, 1) or b not in mask:
                            cells[(i, j)] = ('target', 11)
            for (i, j), tv in cells.items():
                owner[(3 * a + i, 3 * c + j)] = tv
    return owner, hid


def comps(cells):
    cells = set(cells)
    out = []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    st.append(n)
        out.append(comp)
    return out


def bbox(comp):
    xs = [p[0] for p in comp]
    ys = [p[1] for p in comp]
    return min(xs), min(ys), max(xs), max(ys)


def mkobj(name, typ, tags, layer, comp, owner):
    x0, y0, x1, y1 = bbox(comp)
    px = [[owner[(x, y)][1] if (x, y) in comp else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'tags': tags, 'layer': layer, 'x': x0, 'y': y0,
            'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'pixels': px}


def merge_targets(cs):
    cs = [set(c) for c in cs]
    changed = True
    while changed:
        changed = False
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = bbox(cs[i]), bbox(cs[j])
                gx = max(b[0] - a[2], a[0] - b[2])
                gy = max(b[1] - a[3], a[1] - b[3])
                if gx <= 4 and gy <= 4:
                    cs[i] |= cs.pop(j)
                    changed = True
                    break
            if changed:
                break
    return cs


def extract(m):
    owner, hid = render(m)
    bytype = {}
    for p, (t, v) in owner.items():
        bytype.setdefault(t, set()).add(p)
    out = []
    zeros = [(p, t) for p, (t, v) in owner.items() if v == 0]
    for typ, tags, layer in (('wall', None, 1), ('player', ['movable', 'black'], 4)):
        cs = sorted(comps(bytype.get(typ, ())), key=lambda c: bbox(c)[:2])
        for r, c in enumerate(cs):
            tl = bbox(c)[:2]
            off = sum(1 for p, t in zeros if t != typ and p < tl)
            if typ == 'wall':
                x0, y0, x1, y1 = bbox(c)
                v = (x1 - x0) >= (y1 - y0)
                out.append(mkobj('wall_%s_%d' % ('v' if v else 'h', r + off), 'wall',
                                 ['axis', 'vertical' if v else 'horizontal'], 1, c, owner))
            else:
                out.append(mkobj('piece_%d' % (r + off), 'player', tags, layer, c, owner))
    keys = [(bbox(c)[:2], c) for c in comps(bytype.get('reflection', ()))]
    hk = [((3 * min(a for a, _ in hc), 3 * min(b for _, b in hc)), None) for hc in comps(hid)]
    allk = sorted(keys + hk, key=lambda k: k[0])
    for r, (k, c) in enumerate(allk):
        if c is not None:
            out.append(mkobj('reflection_%d' % r, 'reflection', ['mirror', 'gray'], 3, c, owner))
    tc = merge_targets(comps(bytype.get('target', ())))
    for r, c in enumerate(sorted(tc, key=lambda c: bbox(c)[:2])):
        out.append(mkobj('target_%d' % r, 'target', ['goal', 'yellow'], 2, c, owner))
    return out


def transition_function(state, action):
    m = parse(state)
    cont = _last['out'] is not None and canon(state) == _last['out']
    if m['sel'] is None:
        m['sel'] = _last['sel'] if cont else 'A'
    m = step(m, action)
    out = extract(m) + [dict(o) for o in state if o['type'] not in ('wall', 'player', 'reflection', 'target')]
    _last['out'] = canon(out)
    _last['sel'] = m['sel']
    return out
