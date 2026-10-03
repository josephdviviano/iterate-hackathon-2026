# Mechanics: mirror-axis puzzle on a 21x21 grid of 3x3 blocks (x=row, pixels x-major).
# Model = {axis block row, pieces (block sets, ring sprites w/ centre hole), targets (block sets), selection}.
# ACTION5 cycles selection axis -> piece_0 -> piece_1 -> axis; ACTION1-4 move selection by one block (axis rows only);
# moves blocked off-board, onto the axis row / a piece row, or into another piece. Reflections = pieces mirrored
# in the axis. Render layers wall1<target2<reflection3<piece4 (holes transparent, else 0 when selected / 4 for
# reflections) then re-extract: 4-connected comps, targets per sprite, names by bbox (x,y); wall index += #non-wall 0 px.
import json

N = 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
DEF_TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
            'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
DEF_COL = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
PREFIX = {'wall': 'wall_h', 'target': 'target', 'reflection': 'reflection', 'player': 'piece'}
_memo = {'key': None, 'model': None}


def canon(s):
    return json.dumps(sorted(json.dumps(o, sort_keys=True) for o in s))


def cells(o):
    for i, row in enumerate(o.get('pixels', [])):
        for j, v in enumerate(row):
            yield o['x'] + i, o['y'] + j, v


def comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        c = set(st)
        while st:
            a, b = st.pop()
            for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if n in blocks:
                    blocks.discard(n); c.add(n); st.append(n)
        out.append(c)
    return out


def parse(state):
    m = {'tags': dict(DEF_TAGS), 'col': dict(DEF_COL), 'other': []}
    owned = {}
    for o in state:
        for x, y, v in cells(o):
            if v >= 0:
                owned[(x, y)] = o['type']
    walls = [o for o in state if o['type'] == 'wall']
    players = [o for o in state if o['type'] == 'player']
    for o in state:
        t = o['type']
        if t in LAYER:
            m['tags'][t] = list(o['tags'])
            vs = [v for _, _, v in cells(o) if v > 0]
            if vs:
                m['col'][t] = max(set(vs), key=vs.count)
        elif t != 'reflection':
            m['other'].append(o)
    m['axis'] = min(o['x'] for o in walls) // 3 if walls else 0
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    m['targets'] = [{((x // 3), (y // 3)) for x, y, v in cells(o) if v >= 0}
                    for o in sorted((o for o in state if o['type'] == 'target'), key=lambda o: (o['x'], o['y']))]
    pieces, sel = [], None
    for o in sorted(players, key=lambda o: (o['x'], o['y'])):
        pix = {(x, y): v for x, y, v in cells(o)}
        blocks = {(x // 3, y // 3) for (x, y), v in pix.items() if v >= 0}
        kind = {}
        for b in blocks:
            c = (3 * b[0] + 1, 3 * b[1] + 1)
            v = pix.get(c, -1)
            kind[b] = 'sel' if v == 0 else ('amb' if owned.get(c) not in (None, 'player') else 'uns')
        selb = {b for b in blocks if kind[b] == 'sel'}
        if selb:
            grow = set(selb)
            changed = True
            while changed:
                changed = False
                for b in blocks:
                    if b not in grow and kind[b] == 'amb' and any(
                            (b[0] + d0, b[1] + d1) in grow for d0, d1 in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        grow.add(b); changed = True
            for c in comps(grow):
                sel = len(pieces)
                pieces.append({'blocks': c, 'sel': kind})
            rest = blocks - grow
        else:
            rest = blocks
        for c in comps(rest):
            pieces.append({'blocks': c, 'sel': kind})
    pieces.sort(key=lambda p: (min(b[0] for b in p['blocks']), min(b[1] for b in p['blocks'])))
    if sel is not None:
        selected = [p for p in pieces if any(p['sel'][b] == 'sel' for b in p['blocks'])][0]
    m['pieces'] = [p['blocks'] for p in pieces]
    if axis_sel or not pieces:
        m['sel'] = 'axis'
    elif sel is not None:
        m['sel'] = pieces.index(selected)
    else:
        cand = [i for i, p in enumerate(pieces) if all(p['sel'][b] != 'uns' for b in p['blocks'])]
        m['sel'] = cand[0] if cand else 0
    return m


def step(m, action):
    a = action['action_id'] if isinstance(action, dict) else action
    if a == 5:
        order = ['axis'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(m['sel']) + 1) % len(order)]
        return
    if a not in (1, 2, 3, 4):
        return
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[a]
    if m['sel'] == 'axis':
        if d[1]:
            return
        r = m['axis'] + d[0]
        if 0 <= r < N and not any(b[0] == r for p in m['pieces'] for b in p):
            m['axis'] = r
        return
    k = m['sel']
    nb = {(b[0] + d[0], b[1] + d[1]) for b in m['pieces'][k]}
    others = set().union(*[p for i, p in enumerate(m['pieces']) if i != k])
    if all(0 <= x < N and 0 <= y < N and x != m['axis'] for x, y in nb) and not (nb & others):
        m['pieces'][k] = nb


def render(m):
    layers = {}  # cell -> list of (layer, solid, colour, sprite id, type)

    def put(cell, layer, solid, col, sid, t):
        layers.setdefault(cell, []).append((layer, solid, col, sid, t))

    ax = m['axis']
    for y in range(3 * N):
        for i in range(3):
            hole = i == 1 and y % 3 == 1
            put((3 * ax + i, y), 1, not hole, (0 if m['sel'] == 'axis' else -1) if hole else m['col']['wall'], 'w', 'wall')
    for ti, tb in enumerate(m['targets']):
        for bx, by in tb:
            for i in range(3):
                for j in range(3):
                    put((3 * bx + i, 3 * by + j), 2, True, m['col']['target'], ('t', ti), 'target')

    def ring(blocks, layer, col, fill, sid, t):
        for bx, by in blocks:
            for i in range(3):
                for j in range(3):
                    hole = i == 1 and j == 1
                    put((3 * bx + i, 3 * by + j), layer, not hole, fill if hole else col, sid, t)

    for k, p in enumerate(m['pieces']):
        refl = {(2 * ax - bx, by) for bx, by in p if 0 <= 2 * ax - bx < N}
        ring(refl, 3, m['col']['reflection'], m['col']['reflection'], 'r', 'reflection')
        ring(p, 4, m['col']['player'], 0 if m['sel'] == k else -1, 'p', 'player')
    frame = {}
    for cell, lst in layers.items():
        lst.sort(key=lambda e: -e[0])
        top = next((e for e in lst if e[1]), None)
        if top is None:
            top = next((e for e in lst if e[2] >= 0), None)
        if top is not None and top[2] >= 0:
            frame[cell] = top
    return frame


def extract(m, frame):
    groups = {}
    for t in ('wall', 'player', 'reflection'):
        cs = {c for c, e in frame.items() if e[4] == t}
        groups[t] = comps(cs)
    groups['target'] = [{c for c, e in frame.items() if e[3] == ('t', ti)} for ti in range(len(m['targets']))]
    zeros = sum(1 for e in frame.values() if e[2] == 0 and e[4] != 'wall')
    out = list(m['other'])
    for t, gs in groups.items():
        objs = []
        for g in gs:
            if not g:
                continue
            x0, x1 = min(c[0] for c in g), max(c[0] for c in g)
            y0, y1 = min(c[1] for c in g), max(c[1] for c in g)
            pix = [[frame[(x, y)][2] if (x, y) in g else -1 for y in range(y0, y1 + 1)] for x in range(x0, x1 + 1)]
            objs.append({'h': y1 - y0 + 1, 'layer': LAYER[t], 'pixels': pix, 'tags': list(m['tags'][t]),
                         'type': t, 'w': x1 - x0 + 1, 'x': x0, 'y': y0})
        objs.sort(key=lambda o: (o['x'], o['y']))
        off = zeros if t == 'wall' else 0
        for i, o in enumerate(objs):
            o['name'] = '%s_%d' % (PREFIX[t], i + off)
        out.extend(objs)
    return out


def copy_model(m):
    c = dict(m)
    c['pieces'] = [set(p) for p in m['pieces']]
    return c


def transition_function(state, action):
    key = canon(state)
    if _memo['key'] == key:
        m = copy_model(_memo['model'])
    else:
        m = parse(state)
    step(m, action)
    out = extract(m, render(m))
    _memo['key'], _memo['model'] = canon(out), copy_model(m)
    return out
