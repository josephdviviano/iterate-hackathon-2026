# Mechanics: single mirror axis (wall band, 3x3-block grid) + holed pieces; A5 cycles axis -> pieces (largest first).
# A1/A2 move the selected axis or piece one block in x, A3/A4 move a piece in y; pieces blocked by bounds and
# each other, never by the axis. Each piece block off the axis row reflects to row 2a-b (either side), skipping
# off-board dests and dests covered by any piece. Frame = layered composite, then re-extraction by colour comps.
# Unconfirmed: piece cycle order beyond the first A5; splitting merged pieces when the axis is selected.
N = 63
NB = 21
WALL, TGT, REF, PIECE = 1, 2, 3, 4
COLOR = {10: 'wall', 5: 'player', 4: 'reflection', 11: 'target'}
_last = {'out': None, 'model': None}


def canon(state):
    return sorted(repr(sorted((k, repr(v)) for k, v in o.items())) for o in state)


def cells_of(o):
    res = {}
    p = o.get('pixels') or []
    for i, row in enumerate(p):
        for j, v in enumerate(row):
            if v != -1:
                res[(o['x'] + i, o['y'] + j)] = v
    return res


def blocks_where(cells, pred):
    return {(x // 3, y // 3) for (x, y), v in cells.items() if pred(x, y, v)}


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    players = [o for o in state if o['type'] == 'player']
    targets = [o for o in state if o['type'] == 'target']
    a = min(o['x'] for o in walls) // 3 if walls else 10
    tcells = {}
    for o in targets:
        tcells.update(cells_of(o))
    tsprites = [blocks_where(cells_of(o), lambda x, y, v: v == 11) for o in targets]
    axis_sel = any(v == 0 for o in walls for v in cells_of(o).values())
    pieces, sel_piece = [], None
    for o in players:
        c = cells_of(o)
        blocks = blocks_where(c, lambda x, y, v: x % 3 == 0 and y % 3 == 0 and v == 5)
        if axis_sel:
            pieces.append((blocks, False))
            continue
        zero = {b for b in blocks if c.get((3 * b[0] + 1, 3 * b[1] + 1)) == 0}
        amb = {b for b in blocks if (3 * b[0] + 1, 3 * b[1] + 1) not in c
               and (3 * b[0] + 1, 3 * b[1] + 1) in tcells}
        if not zero:
            pieces.append((blocks, False))
            continue
        sel = set(zero)
        grow = True
        while grow:
            grow = False
            for b in list(amb - sel):
                if any((b[0] + dx, b[1] + dy) in sel for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    sel.add(b)
                    grow = True
        pieces.append((sel, True))
        for comp in components(blocks - sel):
            pieces.append((comp, False))
    pl = [p for p, _ in pieces]
    if not axis_sel:
        for i, (_, s) in enumerate(pieces):
            if s:
                sel_piece = i
    sel = 'axis' if axis_sel or sel_piece is None else sel_piece
    return {'a': a, 'pieces': pl, 'sel': sel, 'targets': tsprites}


def components(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            x, y = st.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    comp.add(n)
                    st.append(n)
        out.append(comp)
    return out


def step(m, action):
    m = {'a': m['a'], 'pieces': [set(p) for p in m['pieces']], 'sel': m['sel'], 'targets': m['targets']}
    aid = action['action_id'] if isinstance(action, dict) else action
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    if aid == 5:
        order = ['axis'] + sorted(range(len(m['pieces'])), key=lambda i: -len(m['pieces'][i]))
        m['sel'] = order[(order.index(m['sel']) + 1) % len(order)]
    elif aid in d:
        dx, dy = d[aid]
        if m['sel'] == 'axis':
            if dy == 0 and 0 <= m['a'] + dx < NB:
                m['a'] += dx
        else:
            i = m['sel']
            moved = {(x + dx, y + dy) for x, y in m['pieces'][i]}
            others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i])
            if all(0 <= x < NB and 0 <= y < NB for x, y in moved) and not (moved & others):
                m['pieces'][i] = moved
    return m


def reflections(m):
    a = m['a']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    for x, y in occ:
        if x == a:
            continue
        dx = 2 * a - x
        if 0 <= dx < NB and (dx, y) not in occ:
            out.add((dx, y))
    return out


def render(m):
    # sprites: (layer, id, kind, {cell: ('s', colour) | ('h', fill)})
    sprites = []

    def blockspr(blocks, ring, fill, solid_centre=None):
        d = {}
        for bx, by in blocks:
            for i in range(3):
                for j in range(3):
                    c = (3 * bx + i, 3 * by + j)
                    if i == 1 and j == 1:
                        d[c] = ('s', solid_centre) if solid_centre is not None else ('h', fill)
                    else:
                        d[c] = ('s', ring)
        return d
    axis_sel = m['sel'] == 'axis'
    sprites.append((WALL, 'wall', blockspr({(m['a'], y) for y in range(NB)}, 10, None,
                                           0 if axis_sel else None)))
    for k, t in enumerate(m['targets']):
        sprites.append((TGT, ('target', k), blockspr(t, 11, None, 11)))
    sprites.append((REF, 'reflection', blockspr(reflections(m), 4, 4)))
    for k, p in enumerate(m['pieces']):
        sprites.append((PIECE, ('player', k), blockspr(p, 5, 0 if m['sel'] == k else None)))
    sprites.sort(key=lambda s: -s[0])
    frame = {}
    allc = set()
    for s in sprites:
        allc |= set(s[2])
    for c in allc:
        cov = [s for s in sprites if c in s[2]]
        col = own = None
        for s in cov:
            if s[2][c][0] == 's':
                col, own = s[2][c][1], s[1]
                break
        if col is None:
            for s in cov:
                if s[2][c][1] is not None:
                    col, own = s[2][c][1], s[1]
                    break
        if col is None:
            continue
        if col == 0:
            own = cov[0][1]
        frame[c] = (col, own)
    return frame


def kind_of(own):
    return own if isinstance(own, str) else own[0]


def mkobj(name, typ, cells, tags, layer):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    px = [[cells.get((x0 + i, y0 + j), -1) for j in range(h)] for i in range(w)]
    return {'name': name, 'type': typ, 'tags': tags, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': layer, 'pixels': px}


def extract(frame, scenery):
    out = list(scenery)
    zeros = {c for c, (v, _) in frame.items() if v == 0}
    spec = [('wall', 10, 'wall_h_', ['axis', 'horizontal'], WALL, True),
            ('player', 5, 'piece_', ['movable', 'black'], PIECE, True),
            ('reflection', 4, 'reflection_', ['mirror', 'gray'], REF, False)]
    for typ, colr, prefix, tags, layer, with_zero in spec:
        base = {c for c, (v, _) in frame.items() if v == colr}
        cs = base | zeros if with_zero else base | {c for c in zeros if kind_of(frame[c][1]) == typ}
        comps = sorted(components(cs), key=lambda cc: (min(c[0] for c in cc), min(c[1] for c in cc)))
        for idx, comp in enumerate(comps):
            if not comp & base:
                continue
            cells = {c: frame[c][0] for c in comp
                     if frame[c][0] == colr or kind_of(frame[c][1]) == typ}
            out.append(mkobj(prefix + str(idx), typ, cells, list(tags), layer))
    tobjs = []
    tids = sorted({o for _, o in frame.values() if not isinstance(o, str) and o[0] == 'target'})
    for tid in tids:
        cells = {c: v for c, (v, o) in frame.items() if o == tid and v == 11}
        if cells:
            tobjs.append(mkobj('', 'target', cells, ['goal', 'yellow'], TGT))
    tobjs.sort(key=lambda o: (o['x'], o['y']))
    for i, o in enumerate(tobjs):
        o['name'] = 'target_' + str(i)
    return out + tobjs


def transition_function(state, action):
    scenery = [dict(o) for o in state if o['type'] not in ('wall', 'player', 'reflection', 'target')]
    if _last['out'] is not None and canon(state) == _last['out']:
        model = _last['model']
    else:
        model = parse(state)
    nm = step(model, action)
    out = extract(render(nm), scenery)
    _last['out'] = canon(out)
    _last['model'] = nm
    return out
