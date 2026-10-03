# Mechanics: mirror-axis puzzle on a 3x3-cell grid (x=row, pixels x-major, 63x63 board). Model = axis row,
# selection (axis or one piece), piece block sets, target block sets. ACTION5 cycles axis->pieces(by x,y)->axis;
# ACTION1/2 move the selection row -3/+3, ACTION3/4 move a piece col -3/+3 (axis ignores 3/4). Blocked if out of
# board, piece onto axis row / another piece, or axis onto a piece row. Reflection block = (2*axis_block - bx, by).
# Render walls1<targets2<reflections3<pieces4 (holes transparent; empty hole: selected 0, reflection 4) + re-extract.
N = 63
B = N // 3
STYLE = {'wall': ('wall_h_', 'wall', ['axis', 'horizontal'], 1),
         'target': ('target_', 'target', ['goal', 'yellow'], 2),
         'reflection': ('reflection_', 'reflection', ['mirror', 'gray'], 3),
         'player': ('piece_', 'player', ['movable', 'black'], 4)}
_last = {'out': None, 'model': None}


def canon(state):
    return sorted(repr(sorted(o.items())) for o in state)


def cells(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v


def components(cellset):
    seen, comps = set(), []
    for c in sorted(cellset):
        if c in seen:
            continue
        stack, comp = [c], set()
        seen.add(c)
        while stack:
            x, y = stack.pop()
            comp.add((x, y))
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cellset and n not in seen:
                    seen.add(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def block_components(blocks):
    return components(set(blocks))


def parse(state):
    owner = {}
    walls, targets, pieces = [], [], []
    axis_sel = False
    for o in state:
        for x, y, v in cells(o):
            owner[(x, y)] = (o['type'], v)
        if o['type'] == 'wall':
            walls.append(o)
            axis_sel = axis_sel or any(v == 0 for _, _, v in cells(o))
        elif o['type'] == 'target':
            targets.append(frozenset({(x // 3, y // 3) for x, y, _ in cells(o)}))
        elif o['type'] == 'player':
            pieces.append(o)
    axis = walls[0]['x'] // 3 if walls else None
    groups, sel = [], None
    for o in pieces:
        blocks = {(x // 3, y // 3) for x, y, v in cells(o) if v == 5}
        cls = {}
        for b in blocks:
            c = owner.get((b[0] * 3 + 1, b[1] * 3 + 1))
            cls[b] = 'sel' if c == ('player', 0) else ('empty' if c is None else 'amb')
        S = {b for b in blocks if cls[b] == 'sel'}
        if S:
            grow = True
            while grow:
                grow = False
                for b in blocks:
                    if b not in S and cls[b] == 'amb' and any(
                            (b[0] + dx, b[1] + dy) in S for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        S.add(b)
                        grow = True
            groups.append(frozenset(S))
            sel = frozenset(S)
            for comp in block_components(blocks - S):
                groups.append(frozenset(comp))
        else:
            groups.append(frozenset(blocks))
    if axis_sel:
        sel = 'axis'
    elif sel is None:
        amb = [g for g in groups if all(owner.get((b[0] * 3 + 1, b[1] * 3 + 1)) is not None for b in g)]
        sel = amb[0] if amb else 'axis'
    if sel == 'axis':
        sel_i = -1
    else:
        sel_i = groups.index(sel)
    return {'axis': axis, 'sel': sel_i, 'pieces': groups, 'targets': targets}


def order(model):
    return sorted(range(len(model['pieces'])), key=lambda i: min(model['pieces'][i]))


def step(model, action):
    m = dict(model)
    m['pieces'] = list(model['pieces'])
    a = action['action_id'] if isinstance(action, dict) else action
    if a == 5:
        seq = [-1] + order(model)
        m['sel'] = seq[(seq.index(model['sel']) + 1) % len(seq)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(a)
    if d is None:
        return m
    if m['sel'] == -1:
        if d[1] != 0:
            return m
        na = m['axis'] + d[0]
        if 0 <= na < B and not any(b[0] == na for g in m['pieces'] for b in g):
            m['axis'] = na
        return m
    i = m['sel']
    moved = frozenset((bx + d[0], by + d[1]) for bx, by in m['pieces'][i])
    others = set().union(*[g for j, g in enumerate(m['pieces']) if j != i]) if len(m['pieces']) > 1 else set()
    ok = all(0 <= bx < B and 0 <= by < B and bx != m['axis'] and (bx, by) not in others for bx, by in moved)
    if ok:
        m['pieces'][i] = moved
    return m


def render(model):
    top = {}  # cell -> (type, value, sprite)
    ax = model['axis']
    if ax is not None:
        for y in range(N):
            for dx in range(3):
                hole = dx == 1 and y % 3 == 1
                if hole:
                    if model['sel'] == -1:
                        top[(ax * 3 + 1, y)] = ('wall', 0, 0)
                else:
                    top[(ax * 3 + dx, y)] = ('wall', 10, 0)

    def draw(blocks, typ, val, hole_fill, sprite):
        for bx, by in blocks:
            if not (0 <= bx < B and 0 <= by < B):
                continue
            for dx in range(3):
                for dy in range(3):
                    c = (bx * 3 + dx, by * 3 + dy)
                    if dx == 1 and dy == 1 and typ != 'target':
                        if c not in top and hole_fill is not None:
                            top[c] = (typ, hole_fill, sprite)
                    else:
                        top[c] = (typ, val, sprite)
    for k, t in enumerate(model['targets']):
        draw(t, 'target', 11, None, k)
    if ax is not None:
        refl = set()
        for g in model['pieces']:
            refl |= {(2 * ax - bx, by) for bx, by in g}
        draw(refl, 'reflection', 4, 4, 0)
    for k, g in enumerate(model['pieces']):
        draw(g, 'player', 5, 0 if model['sel'] == k else None, k)
    return top


def extract(top, counter):
    out = [counter] if counter else []
    by_type = {}
    for c, (typ, v, s) in top.items():
        by_type.setdefault(typ, {})[c] = (v, s)
    zeros = sum(1 for (typ, v, s) in top.values() if v == 0 and typ != 'wall')
    for typ in ('wall', 'target', 'reflection', 'player'):
        cm = by_type.get(typ, {})
        if typ == 'target':
            comps = [{c for c, (v, s) in cm.items() if s == k} for k in sorted({s for v, s in cm.values()})]
        else:
            comps = components(set(cm))
        objs = []
        for comp in comps:
            x0 = min(c[0] for c in comp); y0 = min(c[1] for c in comp)
            x1 = max(c[0] for c in comp); y1 = max(c[1] for c in comp)
            pix = [[cm[(x, y)][0] if (x, y) in comp else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
            objs.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1, pix))
        objs.sort(key=lambda t: (t[0], t[1]))
        pre, otype, tags, layer = STYLE[typ]
        off = zeros if typ == 'wall' else 0
        for k, (x, y, w, h, pix) in enumerate(objs):
            out.append({'name': pre + str(k + off), 'type': otype, 'tags': list(tags), 'layer': layer,
                        'x': x, 'y': y, 'w': w, 'h': h, 'pixels': pix})
    return out


def transition_function(state, action):
    counter = next((dict(o) for o in state if o['type'] == 'counter'), None)
    if _last['out'] is not None and canon(state) == _last['out']:
        model = _last['model']
    else:
        model = parse(state)
    model = step(model, action)
    out = extract(render(model), counter)
    _last['out'], _last['model'] = canon(out), model
    return out
