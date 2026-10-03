# Mechanics: 21x21 grid of 3x3 blocks. Walls = horizontal axis row R + vertical axis col C; pieces (player) = block sets.
# Selection (C -> R -> pieces by bbox (x,y) -> C via ACTION5) shows as 0 in hole centres; A1/A2 = x-/+3, A3/A4 = y-/+3
# move the selected piece (blocked by bounds/other pieces) or axis (C: A1/A2, R: A3/A4, bounds only). Reflections of piece
# blocks across R are visible gray; across C / both are invisible erasers. Frame = per-block sprite stacks, re-extracted
# into 4-conn components ranked by bbox (x,y). Unconfirmed: piece cycle order after reordering; axis/piece interactions.
GRID, B = 63, 21
TOP = {'wall': ['axis'], 'player': ['movable', 'black'], 'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
COLOR = {'wall': 10, 'player': 5, 'reflection': 4, 'target': 11}


def paint(state):
    g = [[-1] * GRID for _ in range(GRID)]
    for o in state:
        for i, col in enumerate(o.get('pixels') or []):
            for j, v in enumerate(col):
                if v != -1 and 0 <= o['x'] + i < GRID and 0 <= o['y'] + j < GRID:
                    g[o['x'] + i][o['y'] + j] = v
    return g


def parse(state):
    g = paint(state)
    corner = lambda bx, by: g[3 * bx][3 * by]
    centre = lambda bx, by: g[3 * bx + 1][3 * by + 1]
    wall = {(bx, by) for bx in range(B) for by in range(B) if corner(bx, by) == 10}
    R = max(range(B), key=lambda r: sum((c, r) in wall for c in range(B)))
    C = max(range(B), key=lambda c: sum((c, r) in wall for r in range(B)))
    targets = {(bx, by) for bx in range(B) for by in range(B)
               if any(g[3 * bx + i][3 * by + j] == 11 for i in range(3) for j in range(3))}
    pieces = []
    for o in state:
        if o['type'] != 'player':
            continue
        blocks = set()
        for i, col in enumerate(o['pixels']):
            for j, v in enumerate(col):
                if v == 5:
                    blocks.add(((o['x'] + i) // 3, (o['y'] + j) // 3))
        pieces.append(blocks)
    pieces.sort(key=lambda bs: (min(b[0] for b in bs) if bs else 99, min(b[1] for b in bs) if bs else 99))
    sel = None
    if any(centre(c, R) == 0 for c in range(B) if c != C and (c, R) in wall):
        sel = 'R'
    elif any(centre(C, r) == 0 for r in range(B) if r != R and (C, r) in wall):
        sel = 'C'
    else:
        for k, bs in enumerate(pieces):
            if any(centre(*b) == 0 for b in bs if corner(*b) == 5):
                sel = k
                break
    return {'R': R, 'C': C, 'sel': sel, 'pieces': pieces, 'targets': targets}


DELTA = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def step(m, action):
    sel = m['sel']
    if action == 5:
        n = len(m['pieces'])
        if sel == 'C':
            m['sel'] = 'R'
        elif sel == 'R':
            m['sel'] = 0 if n else 'C'
        elif isinstance(sel, int):
            m['sel'] = sel + 1 if sel + 1 < n else 'C'
        return m
    if action not in DELTA:
        return m
    dx, dy = DELTA[action]
    if sel == 'C' and dx:
        if 0 <= m['C'] + dx < B:
            m['C'] += dx
    elif sel == 'R' and dy:
        if 0 <= m['R'] + dy < B:
            m['R'] += dy
    elif isinstance(sel, int):
        moved = {(x + dx, y + dy) for x, y in m['pieces'][sel]}
        others = set().union(*[p for k, p in enumerate(m['pieces']) if k != sel]) if len(m['pieces']) > 1 else set()
        if all(0 <= x < B and 0 <= y < B for x, y in moved) and not (moved & others):
            m['pieces'][sel] = moved
    return m


def render(m):
    R, C, sel = m['R'], m['C'], m['sel']
    stacks = {}
    def push(b, spr):
        if 0 <= b[0] < B and 0 <= b[1] < B:
            stacks.setdefault(b, []).append(spr)
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    for k, bs in enumerate(m['pieces']):
        for b in bs:
            push(b, (4, 'player', 5, None, 0 if sel == k else -1))
    for bs in m['pieces']:
        for c, r in bs:
            for nb, vis in (((c, 2 * R - r), True), ((2 * C - c, r), False), ((2 * C - c, 2 * R - r), False)):
                if nb not in occupied:
                    push(nb, (3, 'reflection', 4, None, 4) if vis else (3, 'invisible', -1, None, -1))
    for b in m['targets']:
        push(b, (2, 'target', 11, 11, None))
    for c in range(B):
        push((c, R), (1, 'wall', 10, 0 if sel == 'R' else None, -1))
    for r in range(B):
        push((C, r), (1, 'wall', 10, 0 if sel == 'C' else None, -1))
    g = [[-1] * GRID for _ in range(GRID)]
    inv = set()
    for (bx, by), st in stacks.items():
        st.sort(key=lambda s: -s[0])
        top = st[0]
        for i in range(3):
            for j in range(3):
                x, y = 3 * bx + i, 3 * by + j
                if (i, j) == (1, 1):
                    solid = next((s[3] for s in st if s[3] is not None), None)
                    v = solid if solid is not None else top[4]
                else:
                    v = top[2]
                g[x][y] = v
                if top[1] == 'invisible' and v == -1:
                    inv.add((x, y))
    return g, inv


def comps(cells):
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
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def make(typ, name, cells, g, tags):
    x0, y0, x1, y1 = bbox(cells)
    px = [[g[x][y] if (x, y) in cells else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': x1 - x0 + 1, 'h': y1 - y0 + 1,
            'layer': LAYER[typ], 'tags': tags, 'pixels': px}


def extract(g, inv):
    allc = [(x, y) for x in range(GRID) for y in range(GRID)]
    zeros = {c for c in allc if g[c[0]][c[1]] == 0}
    out = []
    for typ, prefix in (('wall', 'wall'), ('player', 'piece')):
        own = {c for c in allc if g[c[0]][c[1]] == COLOR[typ]}
        cs = sorted(comps(own | zeros), key=lambda cc: bbox(cc)[:2])
        for k, cc in enumerate(cs):
            if not cc & own:
                continue
            if typ == 'wall':
                x0, y0, x1, y1 = bbox(cc)
                v = x1 - x0 >= y1 - y0
                out.append(make(typ, 'wall_%s_%d' % ('v' if v else 'h', k), cc, g,
                                TOP[typ] + ['vertical' if v else 'horizontal']))
            else:
                out.append(make(typ, '%s_%d' % (prefix, k), cc, g, TOP[typ]))
    vis = {c for c in allc if g[c[0]][c[1]] == 4}
    cs = sorted(comps(vis | inv), key=lambda cc: bbox(cc)[:2])
    for k, cc in enumerate(cs):
        if cc & vis:
            out.append(make('reflection', 'reflection_%d' % k, cc & vis, g, TOP['reflection']))
    ys = [bbox(cc) for cc in comps({c for c in allc if g[c[0]][c[1]] == 11})]
    merged = True
    while merged:
        merged = False
        for a in range(len(ys)):
            for b in range(a + 1, len(ys)):
                p, q = ys[a], ys[b]
                gap = max(q[0] - p[2], p[0] - q[2], q[1] - p[3], p[1] - q[3])
                if gap <= 4:
                    ys[a] = (min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3]))
                    del ys[b]
                    merged = True
                    break
            if merged:
                break
    for k, bb in enumerate(sorted(ys, key=lambda t: t[:2])):
        cells = {(x, y) for x in range(bb[0], bb[2] + 1) for y in range(bb[1], bb[3] + 1) if g[x][y] == 11}
        out.append(make('target', 'target_%d' % k, cells, g, TOP['target']))
    return out


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get('action_id')
    m = step(parse(state), action)
    g, inv = render(m)
    rest = [dict(o) for o in state if o['type'] not in COLOR]
    return rest + extract(g, inv)
