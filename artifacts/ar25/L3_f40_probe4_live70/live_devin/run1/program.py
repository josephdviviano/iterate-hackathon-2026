# Mechanics: wall axis row band + holed 3x3-block pieces + solid targets + gray reflections (x=row), rendered by layer
# (wall<target<reflection<piece) and re-extracted (walls/players = 4-conn comps of own colour + all 0 cells, by bbox).
# A5 cycles axis -> pieces (size desc, min block) -> axis; A1/A2 move selection +/-1 block row, A3/A4 move a piece +/-1 col;
# A7 undoes the last move (selection kept); reflections 2A-b skip piece blocks. Budget: 64 hidden reserve steps, then
# the bar (row 63) shrinks 1/step. Hypothesis (unconfirmed): the depleted bar ranks first as a phantom player comp (names +1).
N = 21
_H = {'last': None, 'model': None, 'reserve': 64, 'hist': []}


def canon(s):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells_of(o):
    for i, row in enumerate(o.get('pixels', [])):
        for j, v in enumerate(row):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def block_comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        st = [blocks.pop()]; comp = set(st)
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in blocks:
                    blocks.discard(nb); comp.add(nb); st.append(nb)
        out.append(comp)
    return out


def order_key(p):
    return (-len(p), min(p))


def parse(state):
    grid = {}
    m = {'counter': None, 'axis': None, 'sel': None, 'pieces': [], 'targets': []}
    axis_sel = False
    for o in state:
        if o['type'] == 'counter':
            m['counter'] = dict(o)
        for r, c, v in cells_of(o):
            grid[(r, c)] = (v, o['type'], o['name'])
        if o['type'] == 'wall':
            m['axis'] = o['x'] // 3
            if any(v == 0 for _, _, v in cells_of(o)):
                axis_sel = True
    pblocks = {(r // 3, c // 3) for (r, c), (v, t, _) in grid.items()
               if t == 'player' and v == 5 and r % 3 == 0 and c % 3 == 0}
    centre = {b: grid.get((3 * b[0] + 1, 3 * b[1] + 1)) for b in pblocks}
    selb = set() if axis_sel else {b for b in pblocks if centre[b] and centre[b][0] == 0 and centre[b][1] == 'player'}
    grow = list(selb)
    while grow:
        r, c = grow.pop()
        for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if nb in pblocks and nb not in selb and centre[nb] and centre[nb][1] != 'player':
                selb.add(nb); grow.append(nb)
    pieces = block_comps(pblocks - selb)
    sel = None
    if selb:
        sp = set()
        for comp in block_comps(selb):
            sp |= comp
        pieces.append(sp); sel = sp
    elif axis_sel:
        sel = 'axis'
    else:
        cov = [p for p in pieces if all(centre[b] and centre[b][1] != 'player' for b in p)]
        if cov:
            sel = cov[0]
    pieces.sort(key=order_key)
    m['pieces'] = [frozenset(p) for p in pieces]
    m['sel'] = 'axis' if sel == 'axis' else (m['pieces'].index(frozenset(sel)) if sel else None)
    for o in state:
        if o['type'] == 'target':
            m['targets'].append(frozenset((r // 3, c // 3) for r, c, v in cells_of(o) if v == 11))
    return m


def step(m, action):
    m = dict(m, pieces=list(m['pieces']))
    if action == 5:
        order = sorted(range(len(m['pieces'])), key=lambda i: order_key(m['pieces'][i]))
        if m['sel'] is None or not order:
            m['sel'] = 'axis'
        elif m['sel'] == 'axis':
            m['sel'] = order[0]
        else:
            k = order.index(m['sel'])
            m['sel'] = order[k + 1] if k + 1 < len(order) else 'axis'
        return m, False
    if action not in (1, 2, 3, 4) or m['sel'] is None:
        return m, False
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    if m['sel'] == 'axis':
        if dc or not 0 <= m['axis'] + dr < N:
            return m, False
        m['axis'] += dr
        return m, True
    p = m['pieces'][m['sel']]
    np_ = frozenset((r + dr, c + dc) for r, c in p)
    others = set().union(*[q for i, q in enumerate(m['pieces']) if i != m['sel']])
    if any(not (0 <= r < N and 0 <= c < N) for r, c in np_) or np_ & others:
        return m, False
    m['pieces'][m['sel']] = np_
    return m, True


def render(m):
    A = m['axis']
    allp = set().union(*m['pieces']) if m['pieces'] else set()
    refl = set()
    for r, c in allp:
        rr = 2 * A - r
        if r != A and 0 <= rr < N and (rr, c) not in allp:
            refl.add((rr, c))
    # sprites: (layer, kind, name-id, blocks, ring colour, centre fill, centre solid)
    sp = []
    for i, p in enumerate(m['pieces']):
        s = m['sel'] == i
        sp.append((4, 'player', i, p, 5, 0 if s else -1, False))
    sp.append((3, 'reflection', 0, refl, 4, 4, False))
    for i, t in enumerate(m['targets']):
        sp.append((2, 'target', i, t, 11, 11, True))
    s = m['sel'] == 'axis'
    sp.append((1, 'wall', 0, {(A, c) for c in range(N)}, 10, 0 if s else -1, s))
    sp.sort(key=lambda t: -t[0])
    grid = {}
    for br in range(N):
        for bc in range(N):
            stack = [t for t in sp if (br, bc) in t[3]]
            if not stack:
                continue
            top = stack[0]
            for i in range(3):
                for j in range(3):
                    if (i, j) != (1, 1):
                        grid[(3 * br + i, 3 * bc + j)] = (top[4], top[1], top[2])
            sol = [t for t in stack if t[6]]
            t = sol[0] if sol else top
            grid[(3 * br + 1, 3 * bc + 1)] = (t[5], t[1], t[2])
    return grid


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]; comp = {st[0]}
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells:
                    cells.discard(nb); comp.add(nb); st.append(nb)
        out.append(comp)
    return out


def bbox(cs):
    return min(r for r, _ in cs), min(c for _, c in cs)


def mkobj(name, typ, tags, layer, cs, grid):
    x0, y0 = bbox(cs)
    x1, y1 = max(r for r, _ in cs), max(c for _, c in cs)
    pix = [[grid[(r, c)][0] if (r, c) in cs else -1 for c in range(y0, y1 + 1)] for r in range(x0, x1 + 1)]
    return {'name': name, 'type': typ, 'tags': tags, 'layer': layer, 'x': x0, 'y': y0,
            'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'pixels': pix}


def extract(m, grid):
    out = [m['counter']] if m['counter'] else []
    zeros = {k for k, v in grid.items() if v[0] == 0}
    phantom = 1 if m['counter'] and m['counter']['y'] > 0 else 0
    for typ, col, prefix, tags, layer, extra in (
            ('wall', 10, 'wall_h_', ['axis', 'horizontal'], 1, 0),
            ('player', 5, 'piece_', ['movable', 'black'], 4, phantom)):
        own = {k for k, v in grid.items() if v[0] == col}
        cl = sorted(comps(own | zeros), key=bbox)
        for i, cs in enumerate(cl):
            if cs & own:
                out.append(mkobj(prefix + str(i + extra), typ, tags, layer, cs, grid))
    rc = sorted(comps({k for k, v in grid.items() if v[0] == 4 and v[1] == 'reflection'}), key=bbox)
    for i, cs in enumerate(rc):
        out.append(mkobj('reflection_%d' % i, 'reflection', ['mirror', 'gray'], 3, cs, grid))
    tc = []
    for i in range(len(m['targets'])):
        cs = {k for k, v in grid.items() if v[1] == 'target' and v[2] == i}
        if cs:
            tc.append(cs)
    for i, cs in enumerate(sorted(tc, key=bbox)):
        out.append(mkobj('target_%d' % i, 'target', ['goal', 'yellow'], 2, cs, grid))
    return out


def transition_function(state, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    cont = _H['last'] is not None and canon(state) == _H['last']
    m = parse(state)
    if cont:
        pm = _H['model']
        m['pieces'], m['targets'], m['sel'] = list(pm['pieces']), pm['targets'], pm['sel']
    else:
        _H['reserve'], _H['hist'] = 64, []
    if aid == 7:
        if _H['hist']:
            prev = _H['hist'].pop()
            m = dict(m, axis=prev['axis'], pieces=list(prev['pieces']))
        moved = False
    else:
        old = m
        m, moved = step(m, aid)
        if moved:
            _H['hist'].append(old)
    c = m['counter']
    if c and aid != 7:
        if c['y'] > 0 or _H['reserve'] <= 0:
            m['counter'] = dict(c, y=c['y'] + 1, h=c['h'] - 1)
        else:
            _H['reserve'] -= 1
    out = extract(m, render(m))
    _H['last'], _H['model'] = canon(out), m
    return out
