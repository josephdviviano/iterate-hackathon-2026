# Mechanics: mirror-axis puzzle on a 3x3-block grid (63x63 board, pixels x-major, x=row).
# Model = {axis top row A, selection (axis | piece k), pieces as block sets, target sprites}.
# ACTION1/2/3/4 move the selection by x-3/x+3/y-3/y+3 (axis ignores 3/4); ACTION5 cycles axis->pieces->axis.
# Pieces reflect across the axis (block row r -> 2A-r, clipped); render layers wall<target<reflection<piece, re-extract.
# Unconfirmed: blocking (piece onto axis/other piece, axis onto piece row, off-board) and ACTION5 order past piece 0.
N = 63
TAGS = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'],
        'target': ['goal', 'yellow'], 'reflection': ['mirror', 'gray']}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
PREFIX = {'wall': 'wall_h', 'target': 'target', 'reflection': 'reflection', 'player': 'piece'}
_mem = {'out': None, 'model': None}


def canon(s):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells_of(o):
    for i, row in enumerate(o.get('pixels', [])):
        for j, v in enumerate(row):
            if v != -1:
                yield (o['x'] + i, o['y'] + j), v


def blk(c):
    return (c[0] // 3 * 3, c[1] // 3 * 3)


def neighbours4(b, step):
    x, y = b
    return [(x - step, y), (x + step, y), (x, y - step), (x, y + step)]


def components(items, step):
    items = set(items)
    out = []
    while items:
        seed = items.pop()
        comp, stack = {seed}, [seed]
        while stack:
            for n in neighbours4(stack.pop(), step):
                if n in items:
                    items.remove(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


# ---------- parsing ----------
def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    players = [o for o in state if o['type'] == 'player']
    targets = [o for o in state if o['type'] == 'target']
    refls = [o for o in state if o['type'] == 'reflection']
    A = walls[0]['x'] if walls else None
    axis_sel = any(v == 0 for o in walls for _, v in cells_of(o))
    lower = set()
    for o in targets + refls:
        lower.update(c for c, _ in cells_of(o))
    tsprites = [frozenset(blk(c) for c, _ in cells_of(o)) for o in targets]
    pieces, sel = [], None
    for o in sorted(players, key=lambda o: (o['x'], o['y'])):
        pix = dict(cells_of(o))
        blocks = {blk(c) for c, v in pix.items() if v == 5}
        zero = {b for b in blocks if pix.get((b[0] + 1, b[1] + 1)) == 0}
        amb = {b for b in blocks if (b[0] + 1, b[1] + 1) in lower}
        if not zero:
            pieces.extend(components(blocks, 3))
            continue
        selb = set(zero)
        grow = True
        while grow:
            grow = False
            for b in list(amb - selb):
                if any(n in selb for n in neighbours4(b, 3)):
                    selb.add(b)
                    grow = True
        sel = len(pieces)
        pieces.append(selb)
        pieces.extend(components(blocks - selb, 3))
    if axis_sel or sel is None:
        sel = 'axis'
    return {'A': A, 'sel': sel, 'pieces': [frozenset(p) for p in pieces],
            'targets': tsprites, 'counter': [o for o in state if o['type'] == 'counter']}


# ---------- dynamics ----------
def piece_ok(m, k, blocks):
    others = set().union(*[p for i, p in enumerate(m['pieces']) if i != k]) if len(m['pieces']) > 1 else set()
    return all(0 <= x <= N - 3 and 0 <= y <= N - 3 and x != m['A'] and (x, y) not in others
               for x, y in blocks)


def step(m, action):
    m = dict(m, pieces=list(m['pieces']))
    if action == 5:
        if m['sel'] == 'axis':
            m['sel'] = 0 if m['pieces'] else 'axis'
        else:
            m['sel'] = m['sel'] + 1 if m['sel'] + 1 < len(m['pieces']) else 'axis'
        return m
    d = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}.get(action)
    if d is None:
        return m
    if m['sel'] == 'axis':
        na = m['A'] + d[0]
        if d[0] and 0 <= na <= N - 3 and not any(b[0] == na for p in m['pieces'] for b in p):
            m['A'] = na
    else:
        k = m['sel']
        nb = frozenset((x + d[0], y + d[1]) for x, y in m['pieces'][k])
        if piece_ok(m, k, nb):
            m['pieces'][k] = nb
    return m


# ---------- rendering ----------
def render(m):
    A, sel = m['A'], m['sel']
    top = {}  # cell -> (kind, sprite id, value)

    def put_block(kind, sid, b, ring, fill):
        x, y = b
        for i in range(3):
            for j in range(3):
                c = (x + i, y + j)
                if (i, j) == (1, 1):
                    if c not in top and fill != -1:
                        top[c] = (kind, sid, fill)
                else:
                    top[c] = (kind, sid, ring)
    if A is not None:
        for y in range(0, N, 3):
            put_block('wall', 0, (A, y), 10, 0 if sel == 'axis' else -1)
    for t, ts in enumerate(m['targets']):
        for b in ts:
            for i in range(3):
                for j in range(3):
                    top[(b[0] + i, b[1] + j)] = ('target', t, 11)
    if A is not None:
        for p in m['pieces']:
            for x, y in p:
                rx = 2 * A - x
                if 0 <= rx <= N - 3:
                    put_block('reflection', 0, (rx, y), 4, 4)
    for k, p in enumerate(m['pieces']):
        for b in p:
            put_block('player', 0, b, 5, 0 if sel == k else -1)
    objs = []
    for kind in ('wall', 'reflection', 'player'):
        cells = {c: v for c, (kd, _, v) in top.items() if kd == kind}
        for comp in components(cells, 1):
            objs.append((kind, {c: cells[c] for c in comp}))
    for t in range(len(m['targets'])):
        cells = {c: v for c, (kd, s, v) in top.items() if kd == 'target' and s == t}
        if cells:
            objs.append(('target', cells))
    built = []
    for kind, cells in objs:
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
        pix = [[cells.get((x0 + i, y0 + j), -1) for j in range(h)] for i in range(w)]
        built.append({'h': h, 'layer': LAYER[kind], 'pixels': pix, 'tags': list(TAGS[kind]),
                      'type': kind, 'w': w, 'x': x0, 'y': y0})
    zeros = sum(v == 0 for o in built if o['type'] != 'wall' for r in o['pixels'] for v in r)
    out = [dict(c) for c in m['counter']]
    for kind in ('wall', 'reflection', 'player', 'target'):
        group = sorted((o for o in built if o['type'] == kind), key=lambda o: (o['x'], o['y']))
        off = zeros if kind == 'wall' else 0
        for i, o in enumerate(group):
            o['name'] = '%s_%d' % (PREFIX[kind], i + off)
            out.append(o)
    return out


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get('action_id')
    if _mem['out'] is not None and canon(state) == _mem['out']:
        m = _mem['model']
    else:
        m = parse(state)
    m2 = step(m, action)
    out = render(m2)
    _mem['out'], _mem['model'] = canon(out), m2
    return out
