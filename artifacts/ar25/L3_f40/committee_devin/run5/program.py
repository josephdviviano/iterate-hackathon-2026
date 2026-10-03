# Mechanics: one axis wall (selectable, ACTION1/2 move a horizontal axis by 3), 3x3-cell pieces
# (ACTION1-4 move the selected piece by 3, blocked by bounds/other pieces), ACTION5 cycles selection
# axis -> pieces. Every piece is mirrored across the axis as a gray reflection. Frame is rendered
# wall1<target2<reflection3<piece4; hole centres are transparent over lower content, else 0 (selected).
# Objects re-extracted: 4-conn components; wall index offset by 0-holes; unconfirmed: blocking rules, vertical axes.
import json

DIRS = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}
TAGS = {'wall_h': ['axis', 'horizontal'], 'wall_v': ['axis', 'vertical'],
        'player': ['movable', 'black'], 'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
_mem = {'canon': None, 'model': None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def obj_pixels(o):
    out = {}
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def blk(p):
    return (p[0] // 3 * 3, p[1] // 3 * 3)


def neighbours(c):
    x, y = c
    return [(x - 3, y), (x + 3, y), (x, y - 3), (x, y + 3)]


def cell_groups(cells):
    cells, groups = set(cells), []
    while cells:
        stack, g = [cells.pop()], set()
        while stack:
            c = stack.pop()
            g.add(c)
            for n in neighbours(c):
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        groups.append(frozenset(g))
    return groups


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    counter = [o for o in state if o['type'] == 'counter']
    xmax = counter[0]['x'] if counter and counter[0]['x'] > 0 else 64
    ymax = max([o['y'] + o['h'] for o in walls] + [63])
    vertical = any('vertical' in o.get('tags', []) for o in walls)
    if walls:
        pos = min(o['y'] if vertical else o['x'] for o in walls)
    else:
        pos = None
    axis_sel = any(v == 0 for o in walls for v in obj_pixels(o).values())
    targets = []
    for o in state:
        if o['type'] == 'target':
            targets.append(frozenset(blk(p) for p, v in obj_pixels(o).items() if v == 11))
    others = {}
    for o in state:
        if o['type'] != 'player':
            for p in obj_pixels(o):
                others[p] = True
    pix = {}
    for o in state:
        if o['type'] == 'player':
            pix.update(obj_pixels(o))
    cells = [p for p, v in pix.items() if v == 5 and p[0] % 3 == 0 and p[1] % 3 == 0]
    zero = {c for c in cells if pix.get((c[0] + 1, c[1] + 1)) == 0}
    amb = {c for c in cells if (c[0] + 1, c[1] + 1) not in pix and (c[0] + 1, c[1] + 1) in others}
    sel_cells = set()
    stack = list(zero)
    while stack:
        c = stack.pop()
        if c in sel_cells:
            continue
        sel_cells.add(c)
        stack += [n for n in neighbours(c) if (n in zero or n in amb) and n not in sel_cells]
    pieces = []
    if sel_cells:
        pieces.append(frozenset(sel_cells))
    pieces += cell_groups(set(cells) - sel_cells)
    pieces.sort(key=min)
    if axis_sel or not pieces:
        sel = -1
    elif sel_cells:
        sel = pieces.index(frozenset(sel_cells))
    else:
        sel = 0
    return {'vertical': vertical, 'pos': pos, 'pieces': pieces, 'sel': sel,
            'targets': targets, 'xmax': xmax, 'ymax': ymax}


def step(m, action):
    m = dict(m)
    aid = action.get('action_id') if isinstance(action, dict) else action
    if aid == 5:
        n = len(m['pieces'])
        m['sel'] = m['sel'] + 1 if m['sel'] + 1 < n else (-1 if m['pos'] is not None else 0)
    elif aid in DIRS:
        dx, dy = DIRS[aid]
        if m['sel'] == -1 and m['pos'] is not None:
            d = dy if m['vertical'] else dx
            lim = (m['ymax'] if m['vertical'] else m['xmax']) - 3
            if d and 0 <= m['pos'] + d <= lim:
                m['pos'] += d
        elif 0 <= m['sel'] < len(m['pieces']):
            moved = frozenset((x + dx, y + dy) for x, y in m['pieces'][m['sel']])
            rest = set().union(*[p for i, p in enumerate(m['pieces']) if i != m['sel']])
            ok = all(0 <= x <= m['xmax'] - 3 and 0 <= y <= m['ymax'] - 3 for x, y in moved)
            if ok and not (moved & rest):
                pieces = list(m['pieces'])
                pieces[m['sel']] = moved
                m['pieces'] = pieces
    return m


def mirror(m, c):
    x, y = c
    k = 2 * (m['pos'] + 1)
    return (x, k - y - 2) if m['vertical'] else (k - x - 2, y)


def render(m):
    grid = {}
    inb = lambda p: 0 <= p[0] < m['xmax'] and 0 <= p[1] < m['ymax']

    def put(p, kind, idx, v):
        if inb(p):
            grid[p] = (kind, idx, v)

    if m['pos'] is not None:
        span = m['xmax'] if m['vertical'] else m['ymax']
        for a in range(3):
            for b in range(span):
                p = (b, m['pos'] + a) if m['vertical'] else (m['pos'] + a, b)
                if a == 1 and b % 3 == 1:
                    if m['sel'] == -1:
                        put(p, 'wall', 0, 0)
                else:
                    put(p, 'wall', 0, 10)
    for t, blocks in enumerate(m['targets']):
        for bx, by in blocks:
            for i in range(3):
                for j in range(3):
                    put((bx + i, by + j), 'target', t, 11)
    for layer in ('reflection', 'player'):
        for pi, cells in enumerate(m['pieces']):
            for c in cells:
                if layer == 'reflection':
                    if m['pos'] is None:
                        continue
                    c = mirror(m, c)
                for i in range(3):
                    for j in range(3):
                        p = (c[0] + i, c[1] + j)
                        if (i, j) != (1, 1):
                            put(p, layer, 0, 5 if layer == 'player' else 4)
                        elif p not in grid and inb(p):
                            if layer == 'reflection':
                                put(p, layer, 0, 4)
                            elif pi == m['sel']:
                                put(p, layer, 0, 0)
    return grid


def components(pts):
    pts, comps = set(pts), []
    while pts:
        stack, comp = [pts.pop()], set()
        while stack:
            x, y = stack.pop()
            comp.add((x, y))
            for n in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if n in pts:
                    pts.remove(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def make_obj(pts, grid, typ, layer, tags):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pixels = [[grid[(x0 + i, y0 + j)][2] if (x0 + i, y0 + j) in pts else -1
               for j in range(h)] for i in range(w)]
    return {'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': layer,
            'tags': list(tags), 'pixels': pixels}


def extract(m, grid, passthrough):
    out = [dict(o) for o in passthrough]
    zeros = [p for p, (k, _, v) in grid.items() if v == 0 and k != 'wall']
    wall_objs = [make_obj(c, grid, 'wall', 1, TAGS['wall_v' if m['vertical'] else 'wall_h'])
                 for c in components(p for p, g in grid.items() if g[0] == 'wall')]
    order = sorted([((o['x'], o['y']), 0, id(o)) for o in wall_objs] + [(z, 1, 0) for z in zeros])
    rank = {key[2]: i for i, key in enumerate(order) if key[1] == 0}
    prefix = 'wall_v_' if m['vertical'] else 'wall_h_'
    for o in wall_objs:
        o['name'] = prefix + str(rank[id(o)])
        out.append(o)
    named = []
    for t in range(len(m['targets'])):
        pts = {p for p, g in grid.items() if g[0] == 'target' and g[1] == t}
        if pts:
            named.append(('target', make_obj(pts, grid, 'target', 2, TAGS['target'])))
    for kind, typ, layer, nm in (('reflection', 'reflection', 3, 'reflection'), ('player', 'player', 4, 'piece')):
        for c in components(p for p, g in grid.items() if g[0] == kind):
            named.append((nm, make_obj(c, grid, typ, layer, TAGS[typ])))
    for nm in ('target', 'reflection', 'piece'):
        objs = sorted((o for n, o in named if n == nm), key=lambda o: (o['x'], o['y']))
        for i, o in enumerate(objs):
            o['name'] = '%s_%d' % (nm, i)
            out.append(o)
    return out


def transition_function(state, action):
    c = canon(state)
    model = _mem['model'] if _mem['canon'] == c and _mem['model'] else parse(state)
    passthrough = [o for o in state if o['type'] not in ('wall', 'target', 'reflection', 'player')]
    new = step(model, action)
    out = extract(new, render(new), passthrough)
    _mem['canon'], _mem['model'] = canon(out), new
    return out
