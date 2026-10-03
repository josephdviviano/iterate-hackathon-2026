# Mechanics: one horizontal mirror axis (wall row band, 3x3 blocks) + movable 3x3-block pieces + fixed target sprites.
# Selection (0-filled hole centres) is the axis or one piece; ACTION5 cycles axis -> pieces by min block -> axis.
# ACTION1/2 move the selection x-/+3; ACTION3/4 move a selected piece y-/+3 (axis ignores them). Unconfirmed: blocking
# by board/other pieces/axis row, and A5 order after the first piece. Reflections: piece block b -> (2A-b) if on board.
# Render layers wall<target<reflection<piece, holes filled only if empty; re-extract 4-conn comps, names by bbox (x,y).
N = 21
TAGS = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'],
        'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
PREFIX = {'wall': 'wall_h', 'player': 'piece', 'reflection': 'reflection', 'target': 'target'}
_memo = {'out': None, 'model': None}


def cells(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v


def blocks_of(o):
    return {(x // 3, y // 3) for x, y, _ in cells(o)}


def block_comps(bs):
    bs, comps = set(bs), []
    while bs:
        stack, comp = [bs.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (b[0] + d[0], b[1] + d[1])
                if n in bs:
                    bs.remove(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    rows = {}
    for o in walls:
        for x, y, v in cells(o):
            rows[x // 3] = rows.get(x // 3, 0) + 1
    axis = max(rows, key=rows.get) if rows else 0
    sel = None
    if any(v == 0 for o in walls for _, _, v in cells(o)):
        sel = 'axis'
    targets = [blocks_of(o) for o in state if o['type'] == 'target']
    lower = {}
    for o in state:
        if o['type'] in ('target', 'reflection'):
            for x, y, v in cells(o):
                lower[(x, y)] = v
    pieces, selected = [], None
    for o in state:
        if o['type'] != 'player':
            continue
        pix = {(x, y): v for x, y, v in cells(o)}
        bs = blocks_of(o)
        zero = {b for b in bs if pix.get((3 * b[0] + 1, 3 * b[1] + 1)) == 0}
        amb = {b for b in bs - zero if (3 * b[0] + 1, 3 * b[1] + 1) in lower}
        if zero:
            grp, grow = set(zero), True
            while grow:
                grow = False
                for b in list(amb - grp):
                    if any((b[0] + dx, b[1] + dy) in grp for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        grp.add(b)
                        grow = True
            selected = grp
            pieces.append(grp)
            bs = bs - grp
        pieces.extend(block_comps(bs))
    pieces.sort(key=lambda p: min(p))
    if selected is not None and sel is None:
        sel = pieces.index(selected)
    return {'axis': axis, 'sel': sel, 'pieces': pieces, 'targets': targets}


def step(m, action):
    m = {'axis': m['axis'], 'sel': m['sel'], 'pieces': [set(p) for p in m['pieces']], 'targets': m['targets']}
    if action == 5:
        order = ['axis'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(m['sel']) + 1) % len(order)] if m['sel'] in order else 'axis'
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action)
    if d is None or m['sel'] is None:
        return m
    if m['sel'] == 'axis':
        if d[0] == 0:
            return m
        na = m['axis'] + d[0]
        if 0 <= na < N and not any(b[0] == na for p in m['pieces'] for b in p):
            m['axis'] = na
        return m
    k = m['sel']
    moved = {(b[0] + d[0], b[1] + d[1]) for b in m['pieces'][k]}
    others = set().union(*[p for i, p in enumerate(m['pieces']) if i != k])
    if all(0 <= b[0] < N and 0 <= b[1] < N and b[0] != m['axis'] for b in moved) and not moved & others:
        m['pieces'][k] = moved
    return m


def render(m):
    grid = {}  # (x,y) -> (type, sprite_id, value)

    def paint(b, typ, sid, ring, centre, solid=False):
        for i in range(3):
            for j in range(3):
                c = (3 * b[0] + i, 3 * b[1] + j)
                if i == 1 and j == 1:
                    if centre is not None and (solid or c not in grid):
                        grid[c] = (typ, sid, centre)
                else:
                    grid[c] = (typ, sid, ring)
    A = m['axis']
    for by in range(N):
        paint((A, by), 'wall', 0, 10, 0 if m['sel'] == 'axis' else None)
    for t, bs in enumerate(m['targets']):
        for b in bs:
            paint(b, 'target', t, 11, 11, True)
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    for p in m['pieces']:
        for b in p:
            r = (2 * A - b[0], b[1])
            if 0 <= r[0] < N and r not in occupied:
                paint(r, 'reflection', 0, 4, 4)
    for k, p in enumerate(m['pieces']):
        for b in p:
            paint(b, 'player', k, 5, 0 if m['sel'] == k else None)
    return grid


def comps(cellset):
    cellset, out = set(cellset), []
    while cellset:
        stack, comp = [cellset.pop()], set()
        while stack:
            c = stack.pop()
            comp.add(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in cellset:
                    cellset.remove(n)
                    stack.append(n)
        out.append(comp)
    return out


def make(typ, name, cs, grid):
    xs = [c[0] for c in cs]
    ys = [c[1] for c in cs]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for c in cs:
        pix[c[0] - x0][c[1] - y0] = grid[c][2]
    return {'h': h, 'layer': LAYER[typ], 'name': name, 'pixels': pix, 'tags': list(TAGS[typ]),
            'type': typ, 'w': w, 'x': x0, 'y': y0}


def bbox_key(cs):
    return (min(c[0] for c in cs), min(c[1] for c in cs))


def extract(grid):
    objs = []
    zeros = {c for c, v in grid.items() if v[2] == 0}
    for typ in ('wall', 'player'):
        own = {c for c, v in grid.items() if v[0] == typ}
        cl = sorted(comps(own | zeros), key=bbox_key)
        for i, cs in enumerate(cl):
            if any(grid[c][0] == typ and grid[c][2] != 0 for c in cs):
                objs.append(make(typ, '%s_%d' % (PREFIX[typ], i), cs & own, grid))
    refl = sorted(comps({c for c, v in grid.items() if v[0] == 'reflection'}), key=bbox_key)
    for i, cs in enumerate(refl):
        objs.append(make('reflection', 'reflection_%d' % i, cs, grid))
    sprites = {}
    for c, v in grid.items():
        if v[0] == 'target':
            sprites.setdefault(v[1], set()).add(c)
    for i, cs in enumerate(sorted(sprites.values(), key=bbox_key)):
        objs.append(make('target', 'target_%d' % i, cs, grid))
    return objs


def canon(state):
    return sorted(repr(sorted(o.items())) for o in state)


def transition_function(state, action):
    aid = action.get('action_id') if isinstance(action, dict) else action
    model = None
    if _memo["out"] is not None and canon(state) == _memo['out']:
        model = _memo['model']
    if model is None:
        model = parse(state)
    m = step(model, aid)
    out = [dict(o) for o in state if o['type'] not in LAYER] + extract(render(m))
    _memo['out'], _memo['model'] = canon(out), m
    return out
