# Mechanics: mirror-axis puzzle on 21x21 grid of 3x3 blocks (pixels x-major, x=row). Model = {axis row, selection,
# pieces as block sets, targets}. ACTION1/2 move selection x-/+3, ACTION3/4 move a piece y-/+3 (axis: no-op), ACTION5
# cycles axis->pieces by rank->axis. Pieces may not overlap pieces/axis row/leave board; axis may not enter a piece row.
# Reflections = piece blocks mirrored across axis (clipped). Layers wall<target<reflection<piece, re-extract 4-conn.
# Holes: selected 0 / reflection 4 unless a lower pixel shows. Unconfirmed: cycle order, axis/piece blocking.
NB = 21
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
META = {'wall': (1, ['axis', 'horizontal'], 'wall_h'), 'target': (2, ['goal', 'yellow'], 'target'),
        'reflection': (3, ['mirror', 'gray'], 'reflection'), 'player': (4, ['movable', 'black'], 'piece')}
_last = {'out': None, 'model': None}


def canon(s):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells(o):
    for i, r in enumerate(o.get('pixels') or []):
        for j, v in enumerate(r):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def adj_groups(blocks):
    blocks, groups = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        g = set(st)
        while st:
            bx, by = st.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (bx + dx, by + dy)
                if n in blocks:
                    blocks.remove(n)
                    g.add(n)
                    st.append(n)
        groups.append(g)
    return groups


def rank_key(blocks):
    return (min(b[0] for b in blocks), min(b[1] for b in blocks))


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    axis = min(o['x'] for o in walls) // 3 if walls else None
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    targets = []
    for o in sorted((o for o in state if o['type'] == 'target'), key=lambda o: (o['x'], o['y'])):
        targets.append({(x // 3, y // 3) for x, y, _ in cells(o)})
    other = {}
    for o in state:
        if o['type'] != 'player':
            for x, y, v in cells(o):
                other[(x, y)] = v
    pieces, sel_piece = [], None
    for o in (o for o in state if o['type'] == 'player'):
        pix = {(x, y): v for x, y, v in cells(o)}
        blocks = {(x // 3, y // 3) for x, y in pix}
        cls = {}
        for b in blocks:
            c = (3 * b[0] + 1, 3 * b[1] + 1)
            cls[b] = 'sel' if pix.get(c) == 0 else ('lower' if c in other else 'empty')
        sel = {b for b in blocks if cls[b] == 'sel'}
        if sel:
            grow = True
            while grow:
                grow = False
                for b in blocks - sel:
                    if cls[b] == 'lower' and any((b[0] + dx, b[1] + dy) in sel
                                                 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        sel.add(b)
                        grow = True
            sel_piece = frozenset(sel)
            pieces.append(sel_piece)
            rest = blocks - sel
            pieces.extend(frozenset(g) for g in adj_groups(rest)) if rest else None
        else:
            pieces.append(frozenset(blocks))
    pieces.sort(key=rank_key)
    if axis_sel or sel_piece is None:
        sel = -1 if axis_sel or not pieces else 0
    else:
        sel = pieces.index(sel_piece)
    return {'axis': axis, 'sel': sel, 'pieces': pieces, 'targets': targets}


def inb(b):
    return 0 <= b[0] < NB and 0 <= b[1] < NB


def step(m, action):
    m = dict(m, pieces=list(m['pieces']))
    if action == 5:
        m['sel'] = m['sel'] + 1 if m['sel'] + 1 < len(m['pieces']) else -1
        return m
    if action not in DIRS:
        return m
    dx, dy = DIRS[action]
    if m['sel'] < 0:
        if dy or m['axis'] is None:
            return m
        na = m['axis'] + dx
        if 0 <= na < NB and not any(b[0] == na for p in m['pieces'] for b in p):
            m['axis'] = na
        return m
    i = m['sel']
    moved = frozenset((bx + dx, by + dy) for bx, by in m['pieces'][i])
    occupied = {b for j, p in enumerate(m['pieces']) if j != i for b in p}
    if all(inb(b) and b not in occupied and b[0] != m['axis'] for b in moved):
        m['pieces'][i] = moved
    return m


def render(m):
    val, own = {}, {}

    def put(x, y, v, who):
        val[(x, y)] = v
        own[(x, y)] = who

    def block(b, color, who, centre):
        for i in range(3):
            for j in range(3):
                x, y = 3 * b[0] + i, 3 * b[1] + j
                if i == 1 and j == 1:
                    if centre is not None:
                        put(x, y, centre, who)
                else:
                    put(x, y, color, who)
    a = m['axis']
    if a is not None:
        for by in range(NB):
            block((a, by), 10, ('wall',), 0 if m['sel'] < 0 else None)
    for t, tb in enumerate(m['targets']):
        for b in tb:
            for i in range(3):
                for j in range(3):
                    put(3 * b[0] + i, 3 * b[1] + j, 11, ('target', t))
    if a is not None:
        for p in m['pieces']:
            for bx, by in p:
                rb = (2 * a - bx, by)
                if inb(rb):
                    c = (3 * rb[0] + 1, 3 * rb[1] + 1)
                    block(rb, 4, ('reflection',), None if c in val else 4)
    for k, p in enumerate(m['pieces']):
        for b in p:
            c = (3 * b[0] + 1, 3 * b[1] + 1)
            block(b, 5, ('player',), 0 if k == m['sel'] and c not in val else None)
    return val, own


def components(pts):
    pts, comps = set(pts), []
    while pts:
        st = [pts.pop()]
        g = set(st)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in pts:
                    pts.remove(n)
                    g.add(n)
                    st.append(n)
        comps.append(g)
    return comps


def make_obj(typ, pts, val):
    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
    w, h = max(p[0] for p in pts) - x0 + 1, max(p[1] for p in pts) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for x, y in pts:
        pix[x - x0][y - y0] = val[(x, y)]
    layer, tags, _ = META[typ]
    return {'type': typ, 'layer': layer, 'tags': list(tags), 'x': x0, 'y': y0, 'w': w, 'h': h, 'pixels': pix}


def extract(m, counter):
    val, own = render(m)
    groups = {}
    for p, who in own.items():
        groups.setdefault(who, set()).add(p)
    objs = []
    for who, pts in groups.items():
        if who[0] == 'target':
            objs.append(make_obj('target', pts, val))
        else:
            objs.extend(make_obj(who[0], c, val) for c in components(pts))
    zeros = sum(1 for p, v in val.items() if v == 0 and own[p][0] != 'wall')
    out = list(counter)
    for typ in ('wall', 'target', 'reflection', 'player'):
        lst = sorted((o for o in objs if o['type'] == typ), key=lambda o: (o['x'], o['y']))
        off = zeros if typ == 'wall' else 0
        for i, o in enumerate(lst):
            o['name'] = '%s_%d' % (META[typ][2], i + off)
            out.append(o)
    return out


def transition_function(state, action):
    act = action if isinstance(action, int) else action.get('action_id')
    if _last['out'] is not None and canon(state) == _last['out']:
        model = _last['model']
    else:
        model = parse(state)
    new = step(model, act)
    out = extract(new, [o for o in state if o['type'] not in META])
    _last['out'], _last['model'] = canon(out), new
    return out
