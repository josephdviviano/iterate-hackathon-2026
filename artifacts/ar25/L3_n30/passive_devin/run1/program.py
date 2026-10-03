# Mechanics: one horizontal mirror axis (wall, 3-row band) + pieces (players, 3x3-block sprites with holes) on a 21x21 block grid.
# ACTION5 cycles selection axis -> pieces (sorted by min block) -> axis; ACTION1/2 move the selection up/down a block, 3/4 move a
# selected piece left/right (axis ignores 3/4). Moves are blocked by bounds, the axis row and other pieces. Reflections = piece blocks
# mirrored across the axis (2A-r); targets are static solid blocks. Frame = layered composite (wall<target<reflection<piece; holes show
# the first lower solid, else topmost fill: selected 0, reflection 4) re-extracted per type. Unconfirmed: piece cycle order, ACTION6/7.
import json

N = 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
        'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
COLOR = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
_memo = {'out': None, 'model': None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells_of(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        c = set(st)
        while st:
            r, q = st.pop()
            for n in ((r + 1, q), (r - 1, q), (r, q + 1), (r, q - 1)):
                if n in blocks:
                    blocks.discard(n); c.add(n); st.append(n)
        out.append(c)
    return out


def parse(state):
    grid = {}
    for o in state:
        for x, y, v in cells_of(o):
            grid[(x, y)] = (o['type'], v)
    walls = [o for o in state if o['type'] == 'wall']
    axis = min(o['x'] for o in walls) // 3
    sel = 'axis' if any(v == 0 for o in walls for _, _, v in cells_of(o)) else None
    pblocks = {(x // 3, y // 3) for o in state if o['type'] == 'player' for x, y, _ in cells_of(o)}
    chosen, amb, empty = set(), set(), set()
    for b in pblocks:
        t = grid.get((3 * b[0] + 1, 3 * b[1] + 1))
        if t == ('player', 0):
            chosen.add(b)
        elif t is not None and t[0] != 'player':
            amb.add(b)
        else:
            empty.add(b)
    pieces = []
    if chosen and sel is None:
        grow = True
        while grow:
            grow = False
            for b in list(amb):
                if any((b[0] + d[0], b[1] + d[1]) in chosen for d in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    chosen.add(b); amb.discard(b); grow = True
        pieces.append(frozenset(chosen))
        sel = 0
    pieces += [frozenset(c) for c in comps(empty | amb)]
    if sel == 0:
        selp = pieces[0]
    pieces.sort(key=min)
    if sel == 0:
        sel = pieces.index(selp)
    targets = []
    for o in state:
        if o['type'] == 'target':
            targets.append(frozenset((x // 3, y // 3) for x, y, _ in cells_of(o)))
    other = [o for o in state if o['type'] not in LAYER]
    return {'axis': axis, 'sel': sel, 'pieces': pieces, 'targets': targets, 'other': other}


def step(m, action):
    m = dict(m, pieces=list(m['pieces']))
    a = action['action_id'] if isinstance(action, dict) else action
    pieces, sel = m['pieces'], m['sel']
    if a == 5:
        order = sorted(range(len(pieces)), key=lambda i: min(pieces[i]))
        if sel == 'axis':
            m['sel'] = order[0] if order else 'axis'
        else:
            k = order.index(sel)
            m['sel'] = order[k + 1] if k + 1 < len(order) else 'axis'
    elif a in (1, 2, 3, 4):
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[a]
        if sel == 'axis':
            na = m['axis'] + dr
            if dc == 0 and 0 <= na < N and not any(r == na for p in pieces for r, _ in p):
                m['axis'] = na
        else:
            moved = frozenset((r + dr, c + dc) for r, c in pieces[sel])
            others = set().union(*[p for i, p in enumerate(pieces) if i != sel])
            if all(0 <= r < N and 0 <= c < N and r != m['axis'] and (r, c) not in others for r, c in moved):
                pieces[sel] = moved
    return m


def block_sprite(blocks, kind, fill):
    sp = {}
    for r, c in blocks:
        for i in range(3):
            for j in range(3):
                if kind == 'solid' or (i, j) != (1, 1):
                    sp[(3 * r + i, 3 * c + j)] = ('S', None)
                else:
                    sp[(3 * r + i, 3 * c + j)] = ('H', fill)
    return sp


def render(m):
    A, sel = m['axis'], m['sel']
    sprites = []  # (layer, type, id, sprite)
    sprites.append((1, 'wall', 0, block_sprite({(A, c) for c in range(N)}, 'holed', 0 if sel == 'axis' else -1)))
    for k, t in enumerate(m['targets']):
        sprites.append((2, 'target', k, block_sprite(t, 'solid', -1)))
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    refl = {(2 * A - r, c) for p in m['pieces'] for r, c in p} - occ
    refl = {(r, c) for r, c in refl if 0 <= r < N}
    sprites.append((3, 'reflection', 0, block_sprite(refl, 'holed', 4)))
    for i, p in enumerate(m['pieces']):
        sprites.append((4, 'player', i, block_sprite(p, 'holed', 0 if sel == i else -1)))
    sprites.sort(key=lambda s: -s[0])
    cells = set().union(*[s[3].keys() for s in sprites])
    own = {}
    for cell in cells:
        stack = [s for s in sprites if cell in s[3]]
        hit = None
        for s in stack:
            if s[3][cell][0] == 'S':
                hit = (s[1], s[2], COLOR[s[1]]); break
        if hit is None:
            for s in stack:
                if s[3][cell][1] >= 0:
                    hit = (s[1], s[2], s[3][cell][1]); break
        if hit:
            own[cell] = hit
    return own


def make_obj(name, typ, cells, own):
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    px = [[-1] * h for _ in range(w)]
    for c in cells:
        px[c[0] - x0][c[1] - y0] = own[c][2]
    return {'name': name, 'type': typ, 'layer': LAYER[typ], 'tags': list(TAGS[typ]),
            'x': x0, 'y': y0, 'w': w, 'h': h, 'pixels': px}


def extract(m, own):
    groups = {'wall': [], 'player': [], 'reflection': [], 'target': []}
    for typ in ('wall', 'player', 'reflection'):
        groups[typ] = comps([c for c, v in own.items() if v[0] == typ])
    for k in range(len(m['targets'])):
        cs = {c for c, v in own.items() if v[0] == 'target' and v[1] == k}
        if cs:
            groups['target'].append(cs)
    out = [dict(o) for o in m['other']]
    zeros = [c for c, v in own.items() if v[0] != 'wall' and v[2] == 0]
    items = [(min(g), 'w', g) for g in groups['wall']] + [(z, 'z', None) for z in zeros]
    items.sort(key=lambda t: t[0])
    for idx, (_, kind, g) in enumerate(items):
        if kind == 'w':
            out.append(make_obj('wall_h_%d' % idx, 'wall', g, own))
    pre = {'player': 'piece', 'reflection': 'reflection', 'target': 'target'}
    for typ in ('player', 'reflection', 'target'):
        objs = [make_obj('', typ, g, own) for g in groups[typ]]
        objs.sort(key=lambda o: (o['x'], o['y']))
        for i, o in enumerate(objs):
            o['name'] = '%s_%d' % (pre[typ], i)
            out.append(o)
    return out


def transition_function(state, action):
    if _memo['out'] is not None and canon(state) == _memo['out']:
        m = _memo['model']
    else:
        m = parse(state)
    m = step(m, action)
    out = extract(m, render(m))
    _memo['out'], _memo['model'] = canon(out), m
    return out
