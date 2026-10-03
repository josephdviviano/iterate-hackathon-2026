# Mechanics: mirror axes (horizontal wall band at row R, vertical band at col C, 3x3-block grid) + holed 3x3-block pieces.
# ACTION5 cycles selection H-axis -> V-axis -> pieces (by x,y); 1/2 move H-axis or piece x-/+3, 3/4 move V-axis or piece y-/+3.
# Gray reflections = pieces mirrored across C; pieces+gray mirrored across R are drawn as invisible 'erasers' (layer 3,
# transparent holes) hiding targets, consuming reflection name indices. Holes show first solid pixel below, else wall/piece fill 0 if selected, reflection fill 4 if on top. Targets: 4-conn comps merged if bboxes within 4 cells.
# Unconfirmed: eraser vs gray order, piece blocking, axis bounds, V-axis selection via ACTION5.
N = 63
LAST = {}


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]; c = set(st)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n); c.add(n); st.append(n)
        out.append(c)
    return out


def cluster(cells, d=4):
    groups = [[c, bb(c)] for c in comps(cells)]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                p, q = groups[i][1], groups[j][1]
                if max(q[0] - p[2], p[0] - q[2], q[1] - p[3], p[1] - q[3]) <= d:
                    c = groups[i][0] | groups.pop(j)[0]
                    groups[i] = [c, bb(c)]
                    merged = True
                    break
            if merged:
                break
    return [g[0] for g in groups]


def bb(c):
    xs = [p[0] for p in c]; ys = [p[1] for p in c]
    return min(xs), min(ys), max(xs), max(ys)


def cellmap(state):
    m = {}
    for o in state:
        for i, r in enumerate(o.get('pixels') or []):
            for j, v in enumerate(r):
                if v != -1:
                    m[(o['x'] + i, o['y'] + j)] = (o['type'], v)
    return m


def parse(state):
    m = cellmap(state)
    wall = [p for p, (t, v) in m.items() if t == 'wall']
    rc = {}
    cc = {}
    for x, y in wall:
        rc[x] = rc.get(x, 0) + 1
        cc[y] = cc.get(y, 0) + 1
    R = max(range(1, N, 3), key=lambda r: sum(rc.get(r + d, 0) for d in (-1, 0, 1)))
    C = max(range(1, N, 3), key=lambda c: sum(cc.get(c + d, 0) for d in (-1, 0, 1)))
    pcells = [p for p, (t, v) in m.items() if t == 'player']
    pieces = []
    for comp in comps(pcells):
        pieces.append(frozenset((x // 3, y // 3) for x, y in comp))
    pieces.sort(key=lambda b: min(b))
    targets = set()
    for (x, y), (t, v) in m.items():
        if t == 'target':
            targets.add((x // 3, y // 3))
    wz = [p for p, (t, v) in m.items() if t == 'wall' and v == 0]
    sel = None
    if any(x == R and y != C for x, y in wz):
        sel = 0
    elif any(y == C and x != R for x, y in wz):
        sel = 1
    else:
        for i, b in enumerate(pieces):
            if any(m.get((3 * bx + 1, 3 * by + 1)) == ('player', 0) for bx, by in b):
                sel = 2 + i
                break
    if sel is None:
        sel = 0
    return {'R': R, 'C': C, 'pieces': pieces, 'targets': frozenset(targets), 'sel': sel}


def step(md, action):
    md = dict(md)
    pieces = list(md['pieces'])
    sel = md['sel']
    if action == 5:
        md['sel'] = (sel + 1) % (2 + len(pieces))
        return md
    d = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}.get(action)
    if d is None:
        return md
    if sel == 0 and d[0]:
        R = md['R'] + d[0]
        if 1 <= R <= N - 2:
            md['R'] = R
    elif sel == 1 and d[1]:
        C = md['C'] + d[1]
        if 1 <= C <= N - 2:
            md['C'] = C
    elif sel >= 2:
        i = sel - 2
        nb = frozenset((bx + d[0] // 3, by + d[1] // 3) for bx, by in pieces[i])
        others = set().union(*[p for j, p in enumerate(pieces) if j != i]) if len(pieces) > 1 else set()
        if all(0 <= bx < N // 3 and 0 <= by < N // 3 for bx, by in nb) and not (nb & others):
            pieces[i] = nb
            md['pieces'] = pieces
    return md


def render(md):
    R, C, sel = md['R'], md['C'], md['sel']
    stacks = {}  # cell -> list of (layer, order, owner, kind, val) kind: 's' solid / 'h' hole(fill)

    def put(cell, layer, owner, kind, val, order=0):
        if 0 <= cell[0] < N and 0 <= cell[1] < N:
            stacks.setdefault(cell, []).append((layer, order, owner, kind, val))

    def block(b, layer, owner, col, fill, order=0):
        bx, by = b
        for i in range(3):
            for j in range(3):
                c = (3 * bx + i, 3 * by + j)
                if i == 1 and j == 1:
                    put(c, layer, owner, 'h', fill, order)
                else:
                    put(c, layer, owner, 's', col, order)
    for y in range(N):
        put((R - 1, y), 1, ('wall',), 's', 10); put((R + 1, y), 1, ('wall',), 's', 10)
        put((R, y), 1, ('wall',), 'h' if y % 3 == 1 else 's', (0 if sel == 0 else None) if y % 3 == 1 else 10)
    for x in range(N):
        if R - 1 <= x <= R + 1:
            if x == R and x % 3 == 1:
                st = stacks[(x, C)]
                stacks[(x, C)] = [(1, 0, ('wall',), 'h', 0 if sel in (0, 1) else None)]
            continue
        put((x, C - 1), 1, ('wall',), 's', 10); put((x, C + 1), 1, ('wall',), 's', 10)
        put((x, C), 1, ('wall',), 'h' if x % 3 == 1 else 's', (0 if sel == 1 else None) if x % 3 == 1 else 10)
    for b in md['targets']:
        for i in range(3):
            for j in range(3):
                put((3 * b[0] + i, 3 * b[1] + j), 2, ('target',), 's', 11)
    a, h = C // 3, R // 3
    vis = []
    for k, p in enumerate(md['pieces']):
        g = frozenset((bx, 2 * a - by) for bx, by in p)
        vis.append(g)
        for b in g:
            block(b, 3, ('refl', k), 4, 4, 0)
    for k, p in enumerate(md['pieces']):
        for b in p | vis[k]:
            block((2 * h - b[0], b[1]), 3, ('eraser', k), -2, None, 1)
    for k, p in enumerate(md['pieces']):
        for b in p:
            block(b, 4, ('player', k), 5, 0 if sel == 2 + k else None)
    out = {}
    for cell, st in stacks.items():
        st.sort(key=lambda e: (-e[0], -e[1]))
        top = st[0][2]
        for layer, order, owner, kind, val in st:
            if kind == 's':
                out[cell] = (owner[0], val)
                break
        else:
            for layer, order, owner, kind, val in st:
                if val is not None and (owner[0] != 'refl' or owner == top):
                    out[cell] = (top[0], val)
                    break
    return out


def obj(name, typ, comp, vals, layer, tags):
    xs = [p[0] for p in comp]; ys = [p[1] for p in comp]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[vals[(x0 + i, y0 + j)] if (x0 + i, y0 + j) in comp else -1 for j in range(h)] for i in range(w)]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': layer, 'tags': tags, 'pixels': pix}


def extract(cells, counter):
    vals = {c: v for c, (t, v) in cells.items()}
    by = {}
    for c, (t, v) in cells.items():
        by.setdefault(t, []).append(c)
    zeros = {t: [c for c, (tt, v) in cells.items() if v == 0 and tt != t] for t in ('wall', 'player')}
    res = [counter] if counter else []

    def ranked(cs, extra):
        items = [(min(c), 0, c) for c in cs] + [(z, 1, None) for z in extra]
        items.sort(key=lambda e: (e[0], e[1]))
        return [(i, it[2]) for i, it in enumerate(items) if it[2] is not None]
    wc = [set(c) for c in comps(by.get('wall', []))]
    for i, c in ranked([(min(p[0] for p in c), min(p[1] for p in c)) and c for c in wc], zeros['wall']):
        o = obj('', 'wall', c, vals, 1, [])
        hz = o['h'] > o['w']
        o['name'] = 'wall_%s_%d' % ('h' if hz else 'v', i)
        o['tags'] = ['axis', 'horizontal' if hz else 'vertical']
        res.append(o)
    for i, c in ranked(comps(by.get('player', [])), zeros['player']):
        res.append(obj('piece_%d' % i, 'player', c, vals, 4, ['movable', 'black']))
    rc = [(c, True) for c in comps(by.get('refl', []))] + [(c, False) for c in comps(by.get('eraser', []))]
    rc.sort(key=lambda e: bbmin(e[0]))
    for i, (c, v) in enumerate(rc):
        if v:
            res.append(obj('reflection_%d' % i, 'reflection', c, vals, 3, ['mirror', 'gray']))
    tc = sorted(cluster(by.get('target', [])), key=bbmin)
    for i, c in enumerate(tc):
        res.append(obj('target_%d' % i, 'target', c, vals, 2, ['goal', 'yellow']))
    return res


def bbmin(c):
    return (min(p[0] for p in c), min(p[1] for p in c))


def canon(s):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def transition_function(state, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    counter = next((dict(o) for o in state if o['type'] == 'counter'), None)
    if LAST.get('out') is not None and canon(state) == LAST['out']:
        md = LAST['md']
    else:
        md = parse(state)
    md = step(md, aid)
    res = extract(render(md), counter)
    LAST['out'] = canon(res); LAST['md'] = md
    return res
