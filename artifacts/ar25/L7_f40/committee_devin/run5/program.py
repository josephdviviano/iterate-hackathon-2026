# Mechanics: two mirror axes (h = 3-row band at row ah, v = 3-col band at col av, v drawn over h) and holed 3x3-block pieces.
# ACTION5 cycles selection h -> v -> pieces (by top-left) -> h; A1/A2 move h axis / piece rows -+3, A3/A4 v axis / piece cols -+3.
# Each piece block reflects across v (visible gray, layer 3), across h and both (invisible occluders whose hole shows lower pixels);
# reflections on piece blocks are dropped. Selected sprite shows 0 at holes; render layers then re-extract 4-conn components
# (wall/player 0-dot comps count in index; reflections rank incl. invisible cells; targets merge bbox gap<=4). Unconfirmed: piece blocking, V-vs-H overlap.
N = 63
WALL, TGT, REFL, PIECE = 10, 11, 4, 5


def frame(state):
    g = {}
    for o in state:
        for i, row in enumerate(o.get('pixels') or []):
            for j, v in enumerate(row):
                if v >= 0:
                    g[(o['x'] + i, o['y'] + j)] = (v, o['type'])
    return g


def find_axes(g):
    def score(k, horiz):
        s = 0
        for t in range(N):
            for d in (-1, 1):
                c = (k + d, t) if horiz else (t, k + d)
                if g.get(c, (None,))[0] == WALL:
                    s += 1
        return s
    ah = max(range(1, N, 3), key=lambda r: score(r, True))
    av = max(range(1, N, 3), key=lambda c: score(c, False))
    return ah, av


def blocks_of(o):
    bs = set()
    for i, row in enumerate(o['pixels']):
        for j, v in enumerate(row):
            if v == PIECE:
                x, y = o['x'] + i, o['y'] + j
                bs.add((x - x % 3, y - y % 3))
    return bs


def parse(state):
    g = frame(state)
    ah, av = find_axes(g)
    pieces = sorted((blocks_of(o) for o in state if o['type'] == 'player'), key=min)
    covered = set(b for p in pieces for b in p)
    blk = lambda c: (c[0] - c[0] % 3, c[1] - c[1] % 3)
    is0 = lambda c: g.get(c, (None,))[0] == 0 and blk(c) not in covered
    sel = None
    if any(is0((ah, c)) for c in range(1, N, 3) if abs(c - av) > 1):
        sel = 'h'
    elif any(is0((r, av)) for r in range(1, N, 3) if abs(r - ah) > 1):
        sel = 'v'
    else:
        for k, p in enumerate(pieces):
            if any(g.get((b[0] + 1, b[1] + 1), (None,))[0] == 0 for b in p):
                sel = k
    targets = set(blk(c) for c, (v, t) in g.items() if t == 'target')
    return {'ah': ah, 'av': av, 'sel': sel, 'pieces': pieces, 'targets': targets}


def step(m, action):
    m = dict(m, pieces=[set(p) for p in m['pieces']])
    sel = m['sel']
    if action == 5:
        order = ['h', 'v'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(sel) + 1) % len(order)] if sel in order else 'h'
        return m
    d = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}.get(action) if isinstance(action, int) else None
    if d is None or sel is None:
        return m
    if sel == 'h':
        if d[0] and 1 <= m['ah'] + d[0] <= N - 2:
            m['ah'] += d[0]
    elif sel == 'v':
        if d[1] and 1 <= m['av'] + d[1] <= N - 2:
            m['av'] += d[1]
    else:
        new = set((x + d[0], y + d[1]) for x, y in m['pieces'][sel])
        others = set(b for k, p in enumerate(m['pieces']) if k != sel for b in p)
        if all(0 <= x <= N - 3 and 0 <= y <= N - 3 for x, y in new) and not (new & others):
            m['pieces'][sel] = new
    return m


def block_cells(b):
    return [(b[0] + i, b[1] + j) for i in range(3) for j in range(3)]


def render(m):
    ah, av, sel = m['ah'], m['av'], m['sel']
    stacks = {}  # cell -> list of (kind, val, type), bottom first

    def put(c, kind, val, typ):
        if 0 <= c[0] < N and 0 <= c[1] < N:
            stacks.setdefault(c, []).append((kind, val, typ))
    for c in range(N):
        for r in (ah - 1, ah, ah + 1):
            hole = r == ah and c % 3 == 1
            put((r, c), 'solid' if not hole or sel == 'h' else 'hole', (0 if sel == 'h' else None) if hole else WALL, 'wall')
    for r in range(N):
        for c in (av - 1, av, av + 1):
            hole = c == av and r % 3 == 1
            put((r, c), 'solid' if not hole or sel == 'v' else 'hole', (0 if sel == 'v' else None) if hole else WALL, 'wall')
    for b in m['targets']:
        for c in block_cells(b):
            put(c, 'solid', TGT, 'target')
    occ = set(b for p in m['pieces'] for b in p)
    vis, inv = set(), set()
    for p in m['pieces']:
        for (x, y) in p:
            rx, ry = 2 * ah - 2 - x, 2 * av - 2 - y
            vis.add((x, ry))
            inv.add((rx, y))
            inv.add((rx, ry))
    inb = lambda b: 0 <= b[0] <= N - 3 and 0 <= b[1] <= N - 3
    vis = set(b for b in vis if inb(b) and b not in occ)
    inv = set(b for b in inv if inb(b) and b not in occ and b not in vis)
    invcells = set()
    for b in inv:
        for c in block_cells(b):
            invcells.add(c)
            put(c, 'hole' if c == (b[0] + 1, b[1] + 1) else 'solid', None, 'reflection')
    for b in vis:
        for c in block_cells(b):
            put(c, 'hole' if c == (b[0] + 1, b[1] + 1) else 'solid', REFL, 'reflection')
    for k, p in enumerate(m['pieces']):
        for b in p:
            for c in block_cells(b):
                hole = c == (b[0] + 1, b[1] + 1)
                put(c, 'hole' if hole else 'solid', (0 if sel == k else None) if hole else PIECE, 'player')
    g = {}
    for c, st in stacks.items():
        for kind, val, typ in reversed(st):
            if kind == 'solid':
                if val is not None:
                    g[c] = (val, typ)
                break
        else:
            kind, val, typ = st[-1]
            if val is not None:
                g[c] = (val, typ)
    return g, invcells


def components(cells):
    cells, out = set(cells), []
    while cells:
        s = cells.pop()
        comp, todo = {s}, [s]
        while todo:
            x, y = todo.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    todo.append(n)
        out.append(comp)
    return out


def bbox(cells):
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def make(name, typ, layer, tags, cells, g):
    x0, y0, x1, y1 = bbox(cells)
    pix = [[g[(x, y)][0] if (x, y) in cells else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'layer': layer, 'tags': tags, 'x': x0, 'y': y0,
            'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'pixels': pix}


def merge_targets(comps):
    groups = [set(c) for c in comps]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = bbox(groups[i]), bbox(groups[j])
                dx = max(0, b[0] - a[2], a[0] - b[2])
                dy = max(0, b[1] - a[3], a[1] - b[3])
                if max(dx, dy) <= 4:
                    groups[i] |= groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return groups


def extract(m, counter):
    g, invcells = render(m)
    out = list(counter)
    zeros = set(c for c, (v, t) in g.items() if v == 0)
    for typ, col, layer in (('wall', WALL, 1), ('player', PIECE, 4)):
        own = set(c for c, (v, t) in g.items() if v == col and t == typ)
        comps = sorted(components(own | zeros), key=lambda c: bbox(c)[:2])
        for i, comp in enumerate(comps):
            if not comp & own:
                continue
            x0, y0, x1, y1 = bbox(comp)
            if typ == 'wall':
                v = x1 - x0 >= y1 - y0
                name, tags = ('wall_v_%d' if v else 'wall_h_%d') % i, ['axis', 'vertical' if v else 'horizontal']
            else:
                name, tags = 'piece_%d' % i, ['movable', 'black']
            out.append(make(name, typ, layer, tags, comp, g))
    gray = set(c for c, (v, t) in g.items() if v == REFL)
    comps = sorted(components(gray | invcells), key=lambda c: bbox(c)[:2])
    for i, comp in enumerate(comps):
        if comp & gray:
            out.append(make('reflection_%d' % i, 'reflection', 3, ['mirror', 'gray'], comp & gray, g))
    yel = set(c for c, (v, t) in g.items() if v == TGT)
    for i, comp in enumerate(sorted(merge_targets(components(yel)), key=lambda c: bbox(c)[:2])):
        out.append(make('target_%d' % i, 'target', 2, ['goal', 'yellow'], comp, g))
    return out


def transition_function(state, action):
    counter = [o for o in state if o['type'] == 'counter']
    m = parse(state)
    return extract(step(m, action), counter)
