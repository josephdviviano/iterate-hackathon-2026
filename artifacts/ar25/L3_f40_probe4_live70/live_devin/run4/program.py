# Mechanics: single mirror axis (wall row) + 3x3-block pieces; ACTION1-4 move the selection (axis: rows only) by one block,
# ACTION5 cycles axis -> pieces (largest first, tie min block) -> axis, ACTION7 undoes the last position change (free).
# Every piece block reflects to block row 2*axis-row (clipped to the 21x21 board); layers wall<target<reflection<piece are
# composited per block (holes show first solid sprite below, else top fill) and re-extracted as 4-connected components.
# Budget: first 64 actions free, then counter shrinks 1/action; once spent the HUD adds one phantom player comp ranked first.
import json

N = 21
RING = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
        'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
_mem = {'canon': None, 'model': None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells(o):
    for i, row in enumerate(o.get('pixels', [])):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v


def comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        c = set(st)
        while st:
            r, k = st.pop()
            for nb in ((r + 1, k), (r - 1, k), (r, k + 1), (r, k - 1)):
                if nb in blocks:
                    blocks.discard(nb)
                    c.add(nb)
                    st.append(nb)
        out.append(c)
    return out


def parse(state):
    grid = {}
    for o in state:
        for x, y, v in cells(o):
            grid[(x, y)] = v
    walls = [o for o in state if o['type'] == 'wall']
    counter = next(o for o in state if o['type'] == 'counter')
    axis = walls[0]['x'] // 3 if walls else N // 2
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    targets = []
    for o in state:
        if o['type'] == 'target':
            targets.append({(x // 3, y // 3) for x, y, v in cells(o) if v == 11})
    centre = lambda b: grid.get((3 * b[0] + 1, 3 * b[1] + 1), -1)
    pieces, sel = [], 'axis' if axis_sel else None
    for o in state:
        if o['type'] != 'player':
            continue
        blocks = {(x // 3, y // 3) for x, y, v in cells(o) if v == 5}
        chosen = set()
        if not axis_sel:
            chosen = {b for b in blocks if centre(b) == 0}
            grow = True
            while grow and chosen:
                grow = False
                for b in blocks - chosen:
                    if centre(b) == 11 and any((b[0] + d[0], b[1] + d[1]) in chosen
                                               for d in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        chosen.add(b)
                        grow = True
        if chosen:
            sel = len(pieces)
            pieces.append(chosen)
        pieces.extend(comps(blocks - chosen))
    if sel is None:
        covered = [i for i, p in enumerate(pieces) if all(centre(b) == 11 for b in p)]
        sel = covered[0] if covered else 'axis'
    n = 128 - counter['h'] if counter['h'] < 64 else 0
    return {'axis': axis, 'sel': sel, 'pieces': pieces, 'targets': targets,
            'n': n, 'hist': [], 'counter': dict(counter)}


def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def step(m, action):
    a = action['action_id'] if isinstance(action, dict) else action
    m = dict(m, pieces=[set(p) for p in m['pieces']], hist=list(m['hist']))
    if a != 7:
        m['n'] += 1
    snap = (m['axis'], [set(p) for p in m['pieces']])
    if a in (1, 2, 3, 4):
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[a]
        if m['sel'] == 'axis':
            if dc == 0 and 0 <= m['axis'] + dr < N:
                m['axis'] += dr
                m['hist'].append(snap)
        else:
            i = m['sel']
            moved = {(r + dr, c + dc) for r, c in m['pieces'][i]}
            others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i])
            if all(0 <= r < N and 0 <= c < N for r, c in moved) and not moved & others:
                m['pieces'][i] = moved
                m['hist'].append(snap)
    elif a == 5:
        cyc = ['axis'] + order(m['pieces'])
        m['sel'] = cyc[(cyc.index(m['sel']) + 1) % len(cyc)]
    elif a == 7 and m['hist']:
        m['axis'], m['pieces'] = m['hist'].pop()
    return m


def render(m):
    # sprites: (layer, type, id, blocks, centre fill, solid centre)
    sp = []
    ax_sel = m['sel'] == 'axis'
    sp.append((1, 'wall', 0, {(m['axis'], c) for c in range(N)}, 0 if ax_sel else -1, ax_sel))
    for t, tb in enumerate(m['targets']):
        sp.append((2, 'target', t, tb, 11, True))
    for i, p in enumerate(m['pieces']):
        rb = {(2 * m['axis'] - r, c) for r, c in p if 0 <= 2 * m['axis'] - r < N}
        sp.append((3, 'reflection', i, rb, 4, False))
    for i, p in enumerate(m['pieces']):
        sp.append((4, 'player', i, p, 0 if m['sel'] == i else -1, False))
    grid = {}
    for r in range(N):
        for c in range(N):
            stack = sorted([s for s in sp if (r, c) in s[3]], key=lambda s: -s[0])
            if not stack:
                continue
            top = stack[0]
            for i in range(3):
                for j in range(3):
                    grid[(3 * r + i, 3 * c + j)] = (RING[top[1]], top[1], top[2])
            cen = next((s for s in stack if s[5]), top)
            v = RING[cen[1]] if cen[1] == 'target' else cen[4]
            if v == -1:
                del grid[(3 * r + 1, 3 * c + 1)]
            else:
                grid[(3 * r + 1, 3 * c + 1)] = (v, cen[1], cen[2])
    return grid


def make(typ, name, cs):
    xs = [p[0] for p in cs]
    ys = [p[1] for p in cs]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x + 1, max(ys) - y + 1
    pix = [[-1] * h for _ in range(w)]
    for (px, py), v in cs.items():
        pix[px - x][py - y] = v
    return {'name': name, 'type': typ, 'x': x, 'y': y, 'w': w, 'h': h,
            'layer': LAYER[typ], 'tags': list(TAGS[typ]), 'pixels': pix}


def cell_comps(keys):
    keys, out = set(keys), []
    while keys:
        st = [keys.pop()]
        c = set(st)
        while st:
            x, y = st.pop()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in keys:
                    keys.discard(nb)
                    c.add(nb)
                    st.append(nb)
        out.append(c)
    return out


def extract(m, grid):
    out = []
    h = max(0, min(64, 128 - m['n']))
    out.append(dict(m['counter'], y=64 - h, h=h))
    zeros = {k for k, v in grid.items() if v[0] == 0}
    bbox = lambda c: (min(p[0] for p in c), min(p[1] for p in c))
    for typ, prefix, offset in (('wall', 'wall_h_', 0), ('player', 'piece_', 1 if h < 64 else 0)):
        own = {k for k, v in grid.items() if v[0] == RING[typ] and v[1] == typ}
        cs = sorted(cell_comps(own | zeros), key=bbox)
        for rank, c in enumerate(cs):
            if c & own:
                out.append(make(typ, prefix + str(rank + offset), {k: grid[k][0] for k in c}))
    own = {k for k, v in grid.items() if v[1] == 'reflection' and v[0] == 4}
    for rank, c in enumerate(sorted(cell_comps(own), key=bbox)):
        out.append(make('reflection', 'reflection_' + str(rank), {k: 4 for k in c}))
    tg = []
    for t in range(len(m['targets'])):
        c = {k: 11 for k, v in grid.items() if v[1] == 'target' and v[2] == t and v[0] == 11}
        if c:
            tg.append(c)
    for rank, c in enumerate(sorted(tg, key=bbox)):
        out.append(make('target', 'target_' + str(rank), c))
    return out


def transition_function(state, action):
    if _mem['canon'] is not None and canon(state) == _mem['canon']:
        m = _mem['model']
    else:
        m = parse(state)
    m = step(m, action)
    out = extract(m, render(m))
    _mem['canon'], _mem['model'] = canon(out), m
    return out
