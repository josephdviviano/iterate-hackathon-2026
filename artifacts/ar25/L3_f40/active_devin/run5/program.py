# Mechanics: mirror puzzle on a 63x63 board of 3x3 cells (x=row, pixels x-major). A horizontal axis (wall, 3 rows,
# dotted centre row) mirrors every piece block row b to 2*a-b as gray reflections (clipped to the board).
# ACTION5 cycles selection axis -> pieces (sorted by bbox x,y) -> axis; ACTION1-4 move the selection by 3 (axis: rows
# only). Moves blocked off-board, onto the axis rows, or onto another piece (so pieces never enter the axis band).
# Frame = layers wall<target<reflection<piece, transparent holes, re-extracted. Unconfirmed: ACTION6/7 are no-ops.
N = 63
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
DEFAULT_TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
                'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
PREFIX = {'wall': 'wall_h_', 'target': 'target_', 'reflection': 'reflection_', 'player': 'piece_'}
_memory = {'out': None, 'model': None}


def canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells(o):
    p = o.get('pixels') or []
    for i, row in enumerate(p):
        for j, v in enumerate(row):
            yield o['x'] + i, o['y'] + j, v


def blk(r, c):
    return (r // 3 * 3, c // 3 * 3)


def neighbours(b):
    r, c = b
    return [(r - 3, c), (r + 3, c), (r, c - 3), (r, c + 3)]


def components(blocks):
    left, out = set(blocks), []
    while left:
        stack, comp = [left.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for n in neighbours(b):
                if n in left:
                    left.remove(n)
                    stack.append(n)
        out.append(comp)
    return out


# ---------------------------------------------------------------- parse
def parse(state):
    tags = dict(DEFAULT_TAGS)
    for o in state:
        if o['type'] in tags and o.get('tags') is not None:
            tags[o['type']] = o['tags']
    walls = [o for o in state if o['type'] == 'wall']
    ax = min(o['x'] for o in walls)
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    targets = []
    for o in sorted((o for o in state if o['type'] == 'target'), key=lambda o: (o['x'], o['y'])):
        targets.append({blk(r, c) for r, c, v in cells(o) if v >= 0})
    below = {(r, c): v for o in state if o['type'] in ('target', 'reflection') for r, c, v in cells(o) if v >= 0}
    pieces, sel = [], None
    for o in (o for o in state if o['type'] == 'player'):
        px = {(r, c): v for r, c, v in cells(o)}
        blocks = {(r, c) for (r, c), v in px.items() if v >= 0 and r % 3 == 0 and c % 3 == 0}
        centre = {b: px.get((b[0] + 1, b[1] + 1), -1) for b in blocks}
        centre = {b: below.get((b[0] + 1, b[1] + 1), v) if v < 0 else v for b, v in centre.items()}
        seeds = {b for b in blocks if centre[b] == 0}
        if seeds:
            selb, stack = set(seeds), list(seeds)
            while stack:
                for n in neighbours(stack.pop()):
                    if n in blocks and n not in selb and centre[n] != -1:
                        selb.add(n)
                        stack.append(n)
            sel = len(pieces)
            pieces.append(selb)
            blocks = blocks - selb
        pieces.extend(components(blocks))
    if axis_sel:
        sel = 'axis'
    elif sel is None:
        sel = 'axis' if not pieces else 0
    counter = [o for o in state if o['type'] not in LAYER]
    return {'ax': ax, 'sel': sel, 'pieces': pieces, 'targets': targets, 'tags': tags, 'other': counter}


# ---------------------------------------------------------------- dynamics
def piece_key(p):
    return (min(r for r, _ in p), min(c for r, c in p if r == min(r for r, _ in p)))


def bbox_key(p):
    return (min(r for r, _ in p), min(c for _, c in p))


def piece_move_ok(m, i, new):
    others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i]) if len(m['pieces']) > 1 else set()
    for r, c in new:
        if r < 0 or c < 0 or r + 3 > N or c + 3 > N:
            return False
        if m['ax'] - 2 <= r <= m['ax'] + 2:
            return False
        if (r, c) in others:
            return False
    return True


def axis_move_ok(m, a):
    if a < 0 or a + 3 > N:
        return False
    return all(not (a - 2 <= r <= a + 2) for p in m['pieces'] for r, _ in p)


def step(m, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    m = dict(m, pieces=[set(p) for p in m['pieces']])
    if aid == 5:
        order = sorted(range(len(m['pieces'])), key=lambda i: bbox_key(m['pieces'][i]))
        if m['sel'] == 'axis':
            m['sel'] = order[0] if order else 'axis'
        else:
            k = order.index(m['sel'])
            m['sel'] = order[k + 1] if k + 1 < len(order) else 'axis'
    elif aid in (1, 2, 3, 4):
        dr, dc = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}[aid]
        if m['sel'] == 'axis':
            if dr and axis_move_ok(m, m['ax'] + dr):
                m['ax'] += dr
        else:
            i = m['sel']
            new = {(r + dr, c + dc) for r, c in m['pieces'][i]}
            if piece_move_ok(m, i, new):
                m['pieces'][i] = new
    return m


# ---------------------------------------------------------------- render + extract
def render(m):
    # canvas: (r,c) -> (value, owner); owner = (type, sprite index)
    canvas, holes = {}, {}

    def put(r, c, v, owner):
        if 0 <= r < N and 0 <= c < N:
            canvas[(r, c)] = (v, owner)

    def hole(r, c, fill, owner):
        if 0 <= r < N and 0 <= c < N:
            holes[(r, c)] = (fill, owner)

    ax = m['ax']
    wall_sel = m['sel'] == 'axis'
    for i in range(3):
        for c in range(N):
            if i == 1 and c % 3 == 1:
                if wall_sel:
                    hole(ax + i, c, 0, ('wall', 0))
            else:
                put(ax + i, c, 10, ('wall', 0))
    for t, tb in enumerate(m['targets']):
        for r, c in tb:
            for i in range(3):
                for j in range(3):
                    put(r + i, c + j, 11, ('target', t))
                    holes.pop((r + i, c + j), None)
    for k, p in enumerate(m['pieces']):
        for r, c in p:
            rr = 2 * ax - r
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        if (rr + 1, c + 1) not in canvas:
                            hole(rr + 1, c + 1, 4, ('reflection', 0))
                    else:
                        put(rr + i, c + j, 4, ('reflection', 0))
                        holes.pop((rr + i, c + j), None)
    for k, p in enumerate(m['pieces']):
        for r, c in p:
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        if (r + 1, c + 1) not in canvas and m['sel'] == k:
                            hole(r + 1, c + 1, 0, ('player', k))
                    else:
                        put(r + i, c + j, 5, ('player', k))
                        holes.pop((r + i, c + j), None)
    for pos, (v, owner) in holes.items():
        if pos not in canvas:
            canvas[pos] = (v, owner)
    return canvas


def extract(m, canvas):
    out = [dict(o) for o in m['other']]
    groups = {}
    for pos, (v, (typ, idx)) in canvas.items():
        key = (typ, idx) if typ == 'target' else (typ,)
        groups.setdefault(key, set()).add(pos)
    comps = {t: [] for t in LAYER}
    for key, pts in groups.items():
        if key[0] == 'target':
            comps['target'].append(pts)
            continue
        left = set(pts)
        while left:
            stack, comp = [left.pop()], set()
            while stack:
                r, c = stack.pop()
                comp.add((r, c))
                for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if n in left:
                        left.remove(n)
                        stack.append(n)
            comps[key[0]].append(comp)
    zeros = sum(1 for v, (typ, _) in canvas.values() if v == 0 and typ != 'wall')
    for typ, lst in comps.items():
        lst.sort(key=lambda s: (min(r for r, _ in s), min(c for _, c in s)))
        for n, comp in enumerate(lst):
            x0, y0 = min(r for r, _ in comp), min(c for _, c in comp)
            x1, y1 = max(r for r, _ in comp), max(c for _, c in comp)
            pix = [[canvas[(r, c)][0] if (r, c) in comp else -1 for c in range(y0, y1 + 1)]
                   for r in range(x0, x1 + 1)]
            idx = n + zeros if typ == 'wall' else n
            out.append({'name': PREFIX[typ] + str(idx), 'type': typ, 'tags': list(m['tags'][typ]),
                        'x': x0, 'y': y0, 'w': x1 - x0 + 1, 'h': y1 - y0 + 1,
                        'layer': LAYER[typ], 'pixels': pix})
    return out


def transition_function(state, action):
    if _memory['out'] is not None and canon(state) == _memory['out']:
        m = _memory['model']
    else:
        m = parse(state)
    m = step(m, action)
    out = extract(m, render(m))
    _memory['out'], _memory['model'] = canon(out), m
    return out
