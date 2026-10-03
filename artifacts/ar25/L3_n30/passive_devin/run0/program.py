# Mechanics: one horizontal mirror axis (wall, 3-row band on a 3x3 block grid) + holed 3x3-block pieces
# (player) + solid yellow target sprites; gray reflections = piece blocks above the axis mirrored to
# block 2A-b (clipped to the 63x63 board). ACTION5 cycles selection axis -> pieces (by min block) -> axis;
# A1/A2 move the selection +-1 block in x (rows), A3/A4 in y (pieces only). Frame = re-render (piece4 >
# refl3 > target2 > wall1, holes show lower solids else topmost fill) + re-extract. Unconfirmed: blocking.
import json

N = 21
COL = {'wall': 10, 'player': 5, 'target': 11, 'reflection': 4}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
PREFIX = {'wall': 'wall_h_', 'player': 'piece_', 'target': 'target_', 'reflection': 'reflection_'}
DEFTAGS = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'],
           'target': ['goal', 'yellow'], 'reflection': ['mirror', 'gray']}
_memo = {'out': None, 'model': None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells_of(o):
    out = {}
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def block_groups(blocks):
    blocks, groups = set(blocks), []
    while blocks:
        stack, g = [blocks.pop()], set()
        while stack:
            b = stack.pop()
            g.add(b)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (b[0] + d[0], b[1] + d[1])
                if n in blocks:
                    blocks.discard(n)
                    stack.append(n)
        groups.append(g)
    return groups


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    players = [o for o in state if o['type'] == 'player']
    targets = [o for o in state if o['type'] == 'target']
    if not walls:
        return None
    axis = max(walls, key=lambda o: o['h'])['x'] // 3
    tcells = {}
    tsprites = []
    for t in targets:
        c = cells_of(t)
        tcells.update(c)
        tsprites.append({(x // 3, y // 3) for (x, y) in c})
    axis_sel = any(v == 0 for w in walls for v in cells_of(w).values())
    pieces, sel = [], 'axis' if axis_sel else None
    for p in players:
        c = cells_of(p)
        blocks = {(x // 3, y // 3) for (x, y), v in c.items() if v == 5 and x % 3 == 0 and y % 3 == 0}
        zero = {b for b in blocks if c.get((b[0] * 3 + 1, b[1] * 3 + 1)) == 0}
        amb = {b for b in blocks - zero if (b[0] * 3 + 1, b[1] * 3 + 1) in tcells}
        if zero and blocks - zero:
            selb, frontier = set(zero), list(zero)
            while frontier:
                b = frontier.pop()
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (b[0] + d[0], b[1] + d[1])
                    if n in amb and n not in selb:
                        selb.add(n)
                        frontier.append(n)
            groups = [selb] + block_groups(blocks - selb)
        else:
            groups = [blocks]
        for g in groups:
            pieces.append(g)
            if zero and g & zero and not axis_sel:
                sel = g
    pieces.sort(key=min)
    if sel is None:
        sel = 'axis'
    elif sel != 'axis':
        sel = pieces.index(sel)
    tsprites.sort(key=min)
    tags = {o['type']: o.get('tags') for o in state if 'tags' in o}
    return {'axis': axis, 'sel': sel, 'pieces': pieces, 'targets': tsprites, 'tags': tags}


def inside(blocks):
    return all(0 <= bx < N and 0 <= by < N for bx, by in blocks)


def step(m, action):
    m = dict(m, pieces=[set(p) for p in m['pieces']])
    aid = action.get('action_id') if isinstance(action, dict) else action
    if aid == 5:
        order = ['axis'] + list(range(len(m['pieces'])))
        m['sel'] = order[(order.index(m['sel']) + 1) % len(order)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if d is None:
        return m
    if m['sel'] == 'axis':
        if d[1] == 0 and 0 <= m['axis'] + d[0] < N:
            m['axis'] += d[0]
        return m
    i = m['sel']
    moved = {(bx + d[0], by + d[1]) for bx, by in m['pieces'][i]}
    others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i]) if len(m['pieces']) > 1 else set()
    if inside(moved) and not (moved & others):
        m['pieces'][i] = moved
    return m


def render(m):
    """cell -> (owner type, colour) over the 63x63 board."""
    A = m['axis']
    stacks = {}  # cell -> list of (layer, type, solid, fill)

    def put(blocks, typ, hole, fill):
        for bx, by in blocks:
            for i in range(3):
                for j in range(3):
                    x, y = bx * 3 + i, by * 3 + j
                    if not (0 <= x < 3 * N and 0 <= y < 3 * N):
                        continue
                    centre = hole and i == 1 and j == 1
                    stacks.setdefault((x, y), []).append((LAYER[typ], typ, not centre, fill))
    put({(A, by) for by in range(N)}, 'wall', True, 0 if m['sel'] == 'axis' else -1)
    for t in m['targets']:
        put(t, 'target', False, -1)
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    refl = set()
    for p in m['pieces']:
        for bx, by in p:
            if bx < A and 2 * A - bx < N:
                refl.add((2 * A - bx, by))
    put(refl - occupied, 'reflection', True, 4)
    for k, p in enumerate(m['pieces']):
        put(p, 'player', True, 0 if m['sel'] == k else -1)
    frame = {}
    for c, st in stacks.items():
        st.sort(key=lambda e: -e[0])
        top = next((e for e in st if e[2]), None)
        if top is not None:
            frame[c] = (top[1], COL[top[1]])
        elif st[0][3] != -1:
            frame[c] = (st[0][1], st[0][3])
    return frame


def components(cells):
    cells, comps = set(cells), []
    while cells:
        stack, g = [cells.pop()], set()
        while stack:
            c = stack.pop()
            g.add(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in cells:
                    cells.discard(n)
                    stack.append(n)
        comps.append(g)
    return comps


def make_obj(typ, idx, cells, frame, tags):
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    pix = [[-1] * (max(ys) - y0 + 1) for _ in range(max(xs) - x0 + 1)]
    for (x, y) in cells:
        pix[x - x0][y - y0] = frame[(x, y)][1]
    return {'name': PREFIX[typ] + str(idx), 'type': typ, 'tags': list(tags.get(typ) or DEFTAGS[typ]),
            'layer': LAYER[typ], 'x': x0, 'y': y0, 'w': len(pix), 'h': len(pix[0]), 'pixels': pix}


def extract(m, frame, keep):
    out = list(keep)
    tags = m.get('tags', {})
    zeros = {c for c, (t, v) in frame.items() if v == 0}
    for typ in ('wall', 'player'):
        own = {c for c, (t, v) in frame.items() if t == typ}
        comps = sorted(components(own | zeros), key=lambda g: (min(c[0] for c in g), min(c[1] for c in g)))
        for k, g in enumerate(comps):
            if g & own:
                out.append(make_obj(typ, k, g, frame, tags))
    own = {c for c, (t, v) in frame.items() if t == 'reflection'}
    comps = sorted(components(own), key=lambda g: (min(c[0] for c in g), min(c[1] for c in g)))
    out += [make_obj('reflection', k, g, frame, tags) for k, g in enumerate(comps)]
    tgs = []
    for t in m['targets']:
        cells = {(bx * 3 + i, by * 3 + j) for bx, by in t for i in range(3) for j in range(3)}
        vis = {c for c in cells if frame.get(c, (None,))[0] == 'target'}
        if vis:
            tgs.append(vis)
    tgs.sort(key=lambda g: (min(c[0] for c in g), min(c[1] for c in g)))
    out += [make_obj('target', k, g, frame, tags) for k, g in enumerate(tgs)]
    return out


def transition_function(state, action):
    model = None
    if _memo['out'] is not None and canon(state) == _memo['out']:
        model = _memo['model']
    if model is None:
        model = parse(state)
    if model is None:
        return state
    model = step(model, action)
    keep = [o for o in state if o['type'] not in COL]
    out = extract(model, render(model), keep)
    _memo['out'], _memo['model'] = canon(out), model
    return out
