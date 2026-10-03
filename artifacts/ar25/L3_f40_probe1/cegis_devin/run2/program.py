# Mechanics: a horizontal mirror axis (wall row block a) and 3x3-block pieces live on a 21x21 block grid
# (x = row). ACTION1/2 move the selection one block up/down, ACTION3/4 left/right (pieces only);
# ACTION5 cycles axis -> pieces (by bbox) -> axis. The axis moves freely, also under pieces (step 71).
# Pieces reflect to row block 2a-r (not onto piece blocks); layers wall<target<reflection<piece, re-extracted.
# Unconfirmed: whether pieces may enter the axis row, ACTION6/7 (treated as no-ops), piece cycle order.
import copy

N = 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
        'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
PREFIX = {'wall': 'wall_h', 'target': 'target', 'reflection': 'reflection', 'player': 'piece'}
_memo = {'out': None, 'model': None}


def canon(state):
    return sorted(repr(sorted(o.items())) for o in state)


def cells_of(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells:
                    cells.remove(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    return out


def parse(state):
    shown = {}
    for o in state:
        for r, c, v in cells_of(o):
            shown[(r, c)] = (o['type'], v, o['name'])
    walls = [o for o in state if o['type'] == 'wall']
    a = min(o['x'] for o in walls) // 3
    targets = []
    for o in state:
        if o['type'] == 'target':
            targets.append(frozenset((r // 3, c // 3) for r, c, v in cells_of(o)))
    pblocks = {}
    for o in state:
        if o['type'] == 'player':
            pblocks[o['name']] = {(r // 3, c // 3) for r, c, v in cells_of(o) if v == 5}
    occupied = set().union(*pblocks.values()) if pblocks else set()

    def centre(b):
        return shown.get((3 * b[0] + 1, 3 * b[1] + 1))
    axis_sel = any(v == 0 and r == 3 * a + 1 and (r // 3, c // 3) not in occupied
                   for (r, c), (t, v, n) in shown.items())
    pieces, sel = [], 'axis' if axis_sel else None
    for name, blocks in pblocks.items():
        zero = set() if axis_sel else {b for b in blocks if centre(b) and centre(b)[1] == 0}
        if not zero or zero == blocks:
            pieces.append(frozenset(blocks))
            if zero and sel is None:
                sel = frozenset(blocks)
            continue
        grown, changed = set(zero), True
        while changed:
            changed = False
            for b in blocks - grown:
                cb = centre(b)
                if cb and cb[0] != 'player' and any(
                        (b[0] + d, b[1] + e) in grown for d, e in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    grown.add(b)
                    changed = True
        pieces.append(frozenset(grown))
        sel = frozenset(grown)
        for comp in comps(blocks - grown):
            pieces.append(frozenset(comp))
    if sel is None:
        sel = 'axis'
    return {'a': a, 'sel': sel, 'pieces': pieces, 'targets': targets}


def order(pieces):
    return sorted(pieces, key=lambda p: (min(r for r, c in p), min(c for r, c in p)))


def step(m, action):
    m = dict(m)
    aid = action['action_id'] if isinstance(action, dict) else action
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    pieces = order(m['pieces'])
    if aid == 5:
        if m['sel'] == 'axis':
            m['sel'] = pieces[0] if pieces else 'axis'
        else:
            i = pieces.index(m['sel'])
            m['sel'] = pieces[i + 1] if i + 1 < len(pieces) else 'axis'
    elif d and m['sel'] == 'axis':
        na = m['a'] + d[0]
        if d[0] and 0 <= na < N:
            m['a'] = na
    elif d:
        moved = frozenset((r + d[0], c + d[1]) for r, c in m['sel'])
        others = set().union(*[p for p in pieces if p != m['sel']]) if len(pieces) > 1 else set()
        if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
            m['pieces'] = [moved if p == m['sel'] else p for p in pieces]
            m['sel'] = moved
    return m


def sprite(blocks, colour, fill):
    cells = {}
    for br, bc in blocks:
        for i in range(3):
            for j in range(3):
                hole = fill is not None and i == 1 and j == 1
                cells[(3 * br + i, 3 * bc + j)] = ('hole', fill) if hole else ('solid', colour)
    return cells


def render(m):
    a, sel = m['a'], m['sel']
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    layers = []
    for p in m['pieces']:
        layers.append((4, 'player', None, sprite(p, 5, 0 if p == sel else -1)))
    refl = set()
    for p in m['pieces']:
        for r, c in p:
            rr = 2 * a - r
            if 0 <= rr < N and (rr, c) not in occupied:
                refl.add((rr, c))
    layers.append((3, 'reflection', None, sprite(refl, 4, 4)))
    for k, t in enumerate(m['targets']):
        layers.append((2, 'target', k, sprite(t, 11, None)))
    layers.append((1, 'wall', None, sprite({(a, c) for c in range(N)}, 10, 0 if sel == 'axis' else -1)))
    layers.sort(key=lambda l: -l[0])
    shown = {}
    allcells = set()
    for l in layers:
        allcells |= set(l[3])
    for cell in allcells:
        hole = None
        for _, typ, k, cells in layers:
            if cell not in cells:
                continue
            kind, v = cells[cell]
            if kind == 'solid':
                shown[cell] = (typ, k, v)
                break
            if hole is None and v >= 0:
                hole = (typ, k, v)
        else:
            if hole is not None:
                shown[cell] = hole
    return shown


def make_obj(typ, idx, cells, shown):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    x, y = min(rs), min(cs)
    w, h = max(rs) - x + 1, max(cs) - y + 1
    pix = [[-1] * h for _ in range(w)]
    for r, c in cells:
        pix[r - x][c - y] = shown[(r, c)][2]
    return {'name': '%s_%d' % (PREFIX[typ], idx), 'type': typ, 'tags': list(TAGS[typ]),
            'x': x, 'y': y, 'w': w, 'h': h, 'layer': LAYER[typ], 'pixels': pix}


def extract(shown):
    out = []
    zeros = {cell for cell, (t, k, v) in shown.items() if v == 0}
    key = lambda cs: (min(r for r, c in cs), min(c for r, c in cs))
    for typ in ('wall', 'player'):
        own = {cell for cell, (t, k, v) in shown.items() if t == typ and v != 0}
        for i, comp in enumerate(sorted(comps(own | zeros), key=key)):
            if comp & own:
                out.append(make_obj(typ, i, comp, shown))
    refl = {cell for cell, (t, k, v) in shown.items() if t == 'reflection'}
    for i, comp in enumerate(sorted(comps(refl), key=key)):
        out.append(make_obj('reflection', i, comp, shown))
    tg = {}
    for cell, (t, k, v) in shown.items():
        if t == 'target':
            tg.setdefault(k, set()).add(cell)
    for i, cs in enumerate(sorted(tg.values(), key=key)):
        out.append(make_obj('target', i, cs, shown))
    return out


def transition_function(state, action):
    if _memo['out'] is not None and canon(state) == _memo['out']:
        m = _memo['model']
    else:
        m = parse(state)
    m = step(m, action)
    out = [copy.deepcopy(o) for o in state if o['type'] not in LAYER]
    out += extract(render(m))
    _memo['out'], _memo['model'] = canon(out), m
    return out
