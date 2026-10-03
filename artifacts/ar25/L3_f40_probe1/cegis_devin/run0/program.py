# Mechanics: mirror-axis puzzle on a 21x21 grid of 3x3 blocks (x = row, pixels x-major). One horizontal axis
# (wall) plus pieces (player); ACTION5 cycles selection axis -> pieces by (x,y) -> axis; ACTION1-4 move the selection
# by one block (axis only vertically, and it may pass under pieces); pieces are blocked by bounds/other pieces.
# Reflections = piece blocks mirrored across the axis row (r -> 2a-r), skipped on axis/piece blocks. Render layers
# wall1<target2<refl3<piece4 + re-extract. Unconfirmed: piece->piece/axis cycle order, piece-onto-axis blocking.
N = 21
SOLID, HOLE = 0, 1
_memo = {}


def canon(state):
    import json
    return tuple(sorted(json.dumps(o, sort_keys=True) for o in state))


def cells_of(o):
    out = {}
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def blocks_of(cells, vals):
    return {(r // 3, c // 3) for (r, c), v in cells.items() if v in vals}


def adj(b):
    r, c = b
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def groups(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        g = set(st)
        while st:
            for nb in adj(st.pop()):
                if nb in blocks:
                    blocks.discard(nb); g.add(nb); st.append(nb)
        out.append(g)
    return out


def parse(state):
    frame = {}
    for o in state:
        for k, v in cells_of(o).items():
            frame[k] = v
    walls = [o for o in state if o['type'] == 'wall']
    wcells = {}
    for o in walls:
        wcells.update(cells_of(o))
    rows = [r for (r, c), v in wcells.items() if v == 10]
    axis = min(rows) // 3 if rows else 0
    axis_sel = any(v == 0 for v in wcells.values())
    targets = []
    for o in sorted((o for o in state if o['type'] == 'target'), key=lambda o: (o['x'], o['y'])):
        targets.append(blocks_of(cells_of(o), {11}))
    tblocks = set().union(*targets) if targets else set()
    pieces, sel = [], None
    for o in state:
        if o['type'] != 'player':
            continue
        pc = cells_of(o)
        bl = blocks_of(pc, {5})
        if axis_sel:
            pieces.append(bl)
            continue
        zero = {b for b in bl if pc.get((3 * b[0] + 1, 3 * b[1] + 1)) == 0}
        if not zero:
            pieces.append(bl)
            continue
        chosen = set(zero)
        grow = True
        while grow:
            grow = False
            for b in list(bl - chosen):
                if b in tblocks and pc.get((3 * b[0] + 1, 3 * b[1] + 1)) is None and any(n in chosen for n in adj(b)):
                    chosen.add(b); grow = True
        pieces.append(chosen)
        rest = bl - chosen
        pieces.extend(groups(rest))
        sel = chosen
    pieces.sort(key=lambda p: min(p))
    if axis_sel:
        s = 'axis'
    else:
        s = next((i for i, p in enumerate(pieces) if p is sel), 'axis')
    return {'axis': axis, 'sel': s, 'pieces': pieces, 'targets': targets}


def step(m, action):
    m = {'axis': m['axis'], 'sel': m['sel'], 'pieces': [set(p) for p in m['pieces']],
         'targets': [set(t) for t in m['targets']]}
    if isinstance(action, dict):
        return m
    if action == 5:
        order = ['axis'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(m['sel']) + 1) % len(order)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None:
        return m
    if m['sel'] == 'axis':
        a = m['axis'] + d[0]
        if d[0] and 0 <= a < N:
            m['axis'] = a
        return m
    i = m['sel']
    moved = {(r + d[0], c + d[1]) for r, c in m['pieces'][i]}
    if any(not (0 <= r < N and 0 <= c < N) for r, c in moved):
        return m
    others = set().union(set(), *(p for j, p in enumerate(m['pieces']) if j != i))
    if moved & others:
        return m
    m['pieces'][i] = moved
    return m


def sprite_cells(blocks, solid, centre):
    out = {}
    for r, c in blocks:
        for i in range(3):
            for j in range(3):
                out[(3 * r + i, 3 * c + j)] = centre if (i, j) == (1, 1) else (SOLID, solid)
    return out


def render(m):
    a, sel = m['axis'], m['sel']
    occ = set().union(set(), *m['pieces'])
    layers = []  # (layer, type, sprite id, cells)
    wall_centre = (SOLID, 0) if sel == 'axis' else (HOLE, -1)
    layers.append((1, 'wall', 0, sprite_cells({(a, c) for c in range(N)}, 10, wall_centre)))
    for k, t in enumerate(m['targets']):
        layers.append((2, 'target', k, sprite_cells(t, 11, (SOLID, 11))))
    refl = set()
    for p in m['pieces']:
        for r, c in p:
            rr = 2 * a - r
            if r != a and 0 <= rr < N and (rr, c) not in occ:
                refl.add((rr, c))
    layers.append((3, 'reflection', 0, sprite_cells(refl, 4, (HOLE, 4))))
    for k, p in enumerate(m['pieces']):
        layers.append((4, 'player', k, sprite_cells(p, 5, (HOLE, 0 if sel == k else -1))))
    layers.sort(key=lambda l: -l[0])
    frame = {}
    keys = set()
    for l in layers:
        keys |= set(l[3])
    for cell in keys:
        stack = [l for l in layers if cell in l[3]]
        hit = next((l for l in stack if l[3][cell][0] == SOLID), None)
        if hit is not None:
            frame[cell] = (hit[3][cell][1], hit[1], hit[2])
        elif stack[0][3][cell][1] >= 0:
            top = stack[0]
            frame[cell] = (top[3][cell][1], top[1], top[2])
    return frame


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        g = set(st)
        while st:
            for nb in adj(st.pop()):
                if nb in cells:
                    cells.discard(nb); g.add(nb); st.append(nb)
        out.append(g)
    return out


def make(name, typ, layer, tags, cells, frame):
    x0 = min(r for r, c in cells); y0 = min(c for r, c in cells)
    w = max(r for r, c in cells) - x0 + 1; h = max(c for r, c in cells) - y0 + 1
    px = [[frame[(x0 + i, y0 + j)][0] if (x0 + i, y0 + j) in cells else -1 for j in range(h)] for i in range(w)]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': layer, 'tags': tags, 'pixels': px}


def extract(frame, scenery):
    out = [dict(o) for o in scenery]
    zeros = {k for k, v in frame.items() if v[0] == 0}
    for typ, pre, layer, tags in (('wall', 'wall_h', 1, ['axis', 'horizontal']), ('player', 'piece', 4, ['movable', 'black'])):
        own = {k for k, v in frame.items() if v[1] == typ and v[0] != 0}
        cs = sorted(comps(own | zeros), key=lambda g: (min(r for r, c in g), min(c for r, c in g)))
        for i, g in enumerate(cs):
            if g & own:
                out.append(make('%s_%d' % (pre, i), typ, layer, list(tags), g, frame))
    rc = {k for k, v in frame.items() if v[1] == 'reflection'}
    cs = sorted(comps(rc), key=lambda g: (min(r for r, c in g), min(c for r, c in g)))
    for i, g in enumerate(cs):
        out.append(make('reflection_%d' % i, 'reflection', 3, ['mirror', 'gray'], g, frame))
    ts = {}
    for k, v in frame.items():
        if v[1] == 'target':
            ts.setdefault(v[2], set()).add(k)
    cs = sorted(ts.values(), key=lambda g: (min(r for r, c in g), min(c for r, c in g)))
    for i, g in enumerate(cs):
        out.append(make('target_%d' % i, 'target', 2, ['goal', 'yellow'], g, frame))
    return out


def transition_function(state, action):
    key = canon(state)
    m = _memo.get('model') if _memo.get('key') == key else None
    if m is None:
        m = parse(state)
    m2 = step(m, action)
    scenery = [o for o in state if o['type'] not in ('wall', 'player', 'reflection', 'target')]
    out = extract(render(m2), scenery)
    _memo['key'] = canon(out)
    _memo['model'] = m2
    return out
