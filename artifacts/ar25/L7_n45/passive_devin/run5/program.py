# Mechanics: two-axes mirror game on a 21x21 grid of 3x3 blocks (x=row, pixels x-major). Model = H band row ah, V column av,
# pieces (block sets), target blocks, selection (H / V / piece). ACTION5 cycles H -> V -> pieces (by min block) -> H; A1/A2 move
# H or the selected piece in x by 3, A3/A4 move V or the piece in y; axes bounded only, pieces blocked by bounds / other pieces.
# Pieces reflect across V as visible gray, across H / both as invisible occluders (the 'mixed reflection' outcomes = re-render).
# Stateless (no hidden state). Unconfirmed: A5 order beyond observed, splitting of merged touching pieces, A6/A7 (no-ops).
N = 21
COL = {'wall': 10, 'player': 5, 'reflection': 4, 'target': 11}


def frame(state):
    g = [[-1] * 64 for _ in range(64)]
    for o in state:
        p = o.get('pixels')
        if not p:
            continue
        for i, row in enumerate(p):
            for j, v in enumerate(row):
                if v != -1 and 0 <= o['x'] + i < 64 and 0 <= o['y'] + j < 64:
                    g[o['x'] + i][o['y'] + j] = v
    return g


def obj_blocks(o, colour, corner):
    out = set()
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            x, y = o['x'] + i, o['y'] + j
            if v == colour and (not corner or (x % 3 == 0 and y % 3 == 0)):
                out.add((x // 3, y // 3))
    return out


def parse(state):
    g = frame(state)
    wall = {(x // 3, y // 3) for x in range(63) for y in range(63) if g[x][y] == 10}
    rows = [sum(1 for (r, c) in wall if r == k) for k in range(N)]
    cols = [sum(1 for (r, c) in wall if c == k) for k in range(N)]
    ah, av = rows.index(max(rows)), cols.index(max(cols))
    pieces = [obj_blocks(o, 5, True) for o in state if o['type'] == 'player']
    pieces = [p for p in pieces if p]
    targets = {(x // 3, y // 3) for x in range(63) for y in range(63) if g[x][y] == 11}
    pb = set().union(*pieces) if pieces else set()
    cen = lambda b: g[3 * b[0] + 1][3 * b[1] + 1]
    sel = None
    if any(cen((ah, c)) == 0 for c in range(N) if c != av and (ah, c) not in pb):
        sel = 'H'
    elif any(cen((r, av)) == 0 for r in range(N) if r != ah and (r, av) not in pb):
        sel = 'V'
    else:
        for i, p in enumerate(pieces):
            if any(cen(b) == 0 for b in p):
                sel = i
        if sel is None and cen((ah, av)) == 0:
            sel = 'H'
    return {'ah': ah, 'av': av, 'pieces': pieces, 'targets': targets, 'sel': sel}


def step(m, action):
    if isinstance(action, dict):
        return m
    pieces = sorted(m['pieces'], key=min)
    sel = m['sel']
    if isinstance(sel, int):
        sp = m['pieces'][sel]
        sel = pieces.index(sp)
    m = dict(m, pieces=pieces, sel=sel)
    if action == 5:
        order = ['H', 'V'] + list(range(len(pieces)))
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else 'H'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None:
        return m
    if sel == 'H' and d[0]:
        m['ah'] = min(N - 1, max(0, m['ah'] + d[0]))
    elif sel == 'V' and d[1]:
        m['av'] = min(N - 1, max(0, m['av'] + d[1]))
    elif isinstance(sel, int):
        moved = {(r + d[0], c + d[1]) for r, c in pieces[sel]}
        others = set().union(set(), *[p for i, p in enumerate(pieces) if i != sel])
        if all(0 <= r < N and 0 <= c < N for r, c in moved) and not moved & others:
            m['pieces'] = [moved if i == sel else p for i, p in enumerate(pieces)]
    return m


def render(m):
    ah, av, sel = m['ah'], m['av'], m['sel']
    pb = set().union(set(), *m['pieces'])
    stacks = {}
    def put(b, spr):
        if 0 <= b[0] < N and 0 <= b[1] < N:
            stacks.setdefault(b, []).append(spr)
    # sprite = (ring colour, owner, solid centre colour or None, hole fill)
    for i, p in enumerate(m['pieces']):
        for b in p:
            put(b, (5, 'player', None, 0 if sel == i else -1))
    for kind in ('vis', 'inv'):
        for p in m['pieces']:
            for r, c in p:
                refl = [(r, 2 * av - c)] if kind == 'vis' else [(2 * ah - r, c), (2 * ah - r, 2 * av - c)]
                for b in refl:
                    if b not in pb and not any(s[1] == 'refl' + kind for s in stacks.get(b, [])):
                        put(b, (4, 'reflvis', None, 4) if kind == 'vis' else (-1, 'reflinv', None, -1))
    for b in m['targets']:
        put(b, (11, 'target', 11, 11))
    for k in range(N):
        for b in ((ah, k), (k, av)):
            if not any(s[1] == 'wall' for s in stacks.get(b, [])):
                on = (sel == 'H' and b[0] == ah) or (sel == 'V' and b[1] == av)
                put(b, (10, 'wall', 0 if on else None, -1))
    g = [[-1] * 63 for _ in range(63)]
    own = [[None] * 63 for _ in range(63)]
    for (r, c), st in stacks.items():
        for i in range(3):
            for j in range(3):
                x, y = 3 * r + i, 3 * c + j
                if (i, j) != (1, 1):
                    g[x][y], own[x][y] = st[0][0], st[0][1]
                    continue
                solid = [s for s in st if s[2] is not None]
                if solid:
                    g[x][y], own[x][y] = solid[0][2], solid[0][1]
                else:
                    g[x][y], own[x][y] = st[0][3], st[0][1]
    return g, own


def comps(cells):
    cells, out = set(cells), []
    while cells:
        s = cells.pop()
        comp, todo = {s}, [s]
        while todo:
            x, y = todo.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.discard(n)
                    comp.add(n)
                    todo.append(n)
        out.append(comp)
    return out


def bbox(cells):
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def make(name, typ, tags, layer, cells, g):
    x0, y0, x1, y1 = bbox(cells)
    pix = [[g[x][y] if (x, y) in cells else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'tags': tags, 'layer': layer, 'x': x0, 'y': y0,
            'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'pixels': pix}


def extract(g, own):
    cells = [(x, y) for x in range(63) for y in range(63)]
    zero = {c for c in cells if g[c[0]][c[1]] == 0}
    out = []
    for typ, col, layer in (('wall', 10, 1), ('player', 5, 4)):
        own_c = {c for c in cells if g[c[0]][c[1]] == col}
        cs = sorted(comps(own_c | zero), key=lambda s: bbox(s)[:2])
        for k, s in enumerate(cs):
            if not s & own_c:
                continue
            x0, y0, x1, y1 = bbox(s)
            if typ == 'wall':
                v = x1 - x0 >= y1 - y0
                out.append(make('wall_%s_%d' % ('v' if v else 'h', k), 'wall',
                                ['axis', 'vertical' if v else 'horizontal'], 1, s, g))
            else:
                out.append(make('piece_%d' % k, 'player', ['movable', 'black'], 4, s, g))
    rc = {c for c in cells if own[c[0]][c[1]] in ('reflvis', 'reflinv') and g[c[0]][c[1]] in (4, -1)}
    vis = {c for c in rc if g[c[0]][c[1]] == 4}
    cs = sorted(comps(rc), key=lambda s: bbox(s)[:2])
    for k, s in enumerate(cs):
        if s & vis:
            out.append(make('reflection_%d' % k, 'reflection', ['mirror', 'gray'], 3, s & vis, g))
    groups = comps({c for c in cells if g[c[0]][c[1]] == 11})
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = bbox(groups[i]), bbox(groups[j])
                gap = max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])
                if gap <= 4:
                    groups[i] = groups[i] | groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    for k, s in enumerate(sorted(groups, key=lambda s: bbox(s)[:2])):
        out.append(make('target_%d' % k, 'target', ['goal', 'yellow'], 2, s, g))
    return out


def transition_function(state, action):
    m = parse(state)
    m2 = step(m, action)
    g, own = render(m2)
    out = extract(g, own) + [dict(o) for o in state if o['type'] == 'counter']
    return out
