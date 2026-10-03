# Mechanics: mirror-axis game on a 21x21 grid of 3x3 blocks (x=row, y=col, pixels x-major).
# One horizontal axis (wall row A) + holed pieces; ACTION5 cycles selection axis -> pieces
# (largest first, tie by min block) -> axis; A1/A2 move the selection up/down, A3/A4 move a
# piece left/right (axis ignores them). Every piece block b!=A reflects to row 2A-b (gray).
# Layers wall<target<reflection<piece; holes transparent; re-extracted, names ranked w/ 0 dots.
# Hypothesis: piece order by size is a proxy for the game's sprite order (unconfirmed beyond 2 pieces).
N = 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
        'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
PREFIX = {'wall': 'wall_h_', 'target': 'target_', 'reflection': 'reflection_', 'player': 'piece_'}
_memo = {}


def canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells_of(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v


def comps(blocks):
    blocks, out = set(blocks), []
    while blocks:
        stack, comp = [blocks.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (b[0] + d[0], b[1] + d[1])
                if n in blocks:
                    blocks.discard(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(state):
    owner = {}
    for o in state:
        for r, c, v in cells_of(o):
            owner[(r, c)] = (o['name'], o['type'], v)
    axis, axis_sel = None, False
    for o in state:
        if o['type'] == 'wall':
            for r, c, v in cells_of(o):
                if v == 10:
                    axis = r // 3
                if v == 0:
                    axis_sel = True
    targets = {}
    for o in state:
        if o['type'] == 'target':
            targets[o['name']] = {(r // 3, c // 3) for r, c, v in cells_of(o)}

    def centre(b):
        return owner.get((3 * b[0] + 1, 3 * b[1] + 1))

    pieces, sel = [], None
    for o in state:
        if o['type'] != 'player':
            continue
        blocks = {(r // 3, c // 3) for r, c, v in cells_of(o) if v == 5 and r % 3 == 0 and c % 3 == 0}
        zero = {b for b in blocks if centre(b) and centre(b)[0] == o['name'] and centre(b)[2] == 0}
        if axis_sel:
            zero = set()
        amb = {b for b in blocks if centre(b) and centre(b)[0] != o['name']}
        selb = set()
        for comp in comps(zero | amb):
            if comp & zero:
                selb |= comp
        if selb:
            pieces.append(frozenset(selb))
            sel = len(pieces) - 1
        for comp in comps(blocks - selb):
            pieces.append(frozenset(comp))
    if axis_sel:
        sel = 'axis'
    elif sel is None:
        cand = [i for i, p in enumerate(pieces) if all(centre(b) and centre(b)[1] != 'player' for b in p)]
        sel = cand[0] if cand else 'axis'
    return {'axis': axis, 'sel': sel, 'pieces': pieces, 'targets': targets}


def order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def inb(b):
    return 0 <= b[0] < N and 0 <= b[1] < N


def step(m, action):
    m = dict(m, pieces=list(m['pieces']))
    a = action.get('action_id') if isinstance(action, dict) else action
    if a == 5:
        seq = ['axis'] + order(m['pieces'])
        m['sel'] = seq[(seq.index(m['sel']) + 1) % len(seq)]
        return m
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(a)
    if d is None:
        return m
    if m['sel'] == 'axis':
        if d[1] == 0 and 0 <= m['axis'] + d[0] < N:
            m['axis'] += d[0]
        return m
    i = m['sel']
    moved = frozenset((r + d[0], c + d[1]) for r, c in m['pieces'][i])
    others = set().union(*[p for j, p in enumerate(m['pieces']) if j != i])
    if all(inb(b) for b in moved) and not (moved & others):
        m['pieces'][i] = moved
    return m


def render(m):
    A = m['axis']
    sprites = []  # (layer, owner type, sprite id, blocks, ring colour, solid centre?, hole fill)
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    for i, p in enumerate(m['pieces']):
        sprites.append((4, 'player', 'p', p, 5, False, 0 if m['sel'] == i else -1))
    refl = {(2 * A - r, c) for r, c in occ if r != A}
    refl = {b for b in refl if inb(b) and b not in occ}
    sprites.append((3, 'reflection', 'r', refl, 4, False, 4))
    for name, blocks in m['targets'].items():
        sprites.append((2, 'target', name, blocks, 11, True, 11))
    sprites.append((1, 'wall', 'w', {(A, c) for c in range(N)}, 10, False, 0 if m['sel'] == 'axis' else -1))
    sprites.sort(key=lambda s: -s[0])
    grid = {}
    for br in range(N):
        for bc in range(N):
            stack = [s for s in sprites if (br, bc) in s[3]]
            if not stack:
                continue
            top = stack[0]
            for i in range(3):
                for j in range(3):
                    cell = (3 * br + i, 3 * bc + j)
                    if (i, j) != (1, 1):
                        grid[cell] = (top[1], top[2], top[4])
                        continue
                    solid = [s for s in stack if s[5]]
                    if solid:
                        grid[cell] = (solid[0][1], solid[0][2], solid[0][6])
                    else:
                        fill = [s for s in stack if s[6] >= 0]
                        if fill:
                            grid[cell] = (top[1], top[2], fill[0][6])
    return grid


def cell_comps(cells):
    cells, out = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (b[0] + d[0], b[1] + d[1])
                if n in cells:
                    cells.discard(n)
                    stack.append(n)
        out.append(comp)
    return out


def make_obj(typ, idx, cells, grid):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    r1 = max(r for r, c in cells)
    c1 = max(c for r, c in cells)
    pix = [[grid[(r, c)][2] if (r, c) in cells else -1 for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
    return {'name': PREFIX[typ] + str(idx), 'type': typ, 'tags': list(TAGS[typ]), 'x': r0, 'y': c0,
            'w': r1 - r0 + 1, 'h': c1 - c0 + 1, 'layer': LAYER[typ], 'pixels': pix}


def bkey(cells):
    return (min(r for r, c in cells), min(c for r, c in cells))


def extract(grid):
    out = []
    zeros = {k for k, v in grid.items() if v[2] == 0}
    for typ in ('wall', 'player'):
        own = {k for k, v in grid.items() if v[0] == typ}
        cs = sorted(cell_comps(own | zeros), key=bkey)
        for i, comp in enumerate(cs):
            if comp & own:
                out.append(make_obj(typ, i, comp, grid))
    own = {k for k, v in grid.items() if v[0] == 'reflection'}
    for i, comp in enumerate(sorted(cell_comps(own), key=bkey)):
        out.append(make_obj('reflection', i, comp, grid))
    groups = {}
    for k, v in grid.items():
        if v[0] == 'target':
            groups.setdefault(v[1], set()).add(k)
    for i, comp in enumerate(sorted(groups.values(), key=bkey)):
        out.append(make_obj('target', i, comp, grid))
    return out


def transition_function(state, action):
    key = canon(state)
    m = _memo['model'] if _memo.get('key') == key else parse(state)
    m2 = step(m, action)
    out = [dict(o) for o in state if o['type'] not in LAYER] + extract(render(m2))
    _memo['key'] = canon(out)
    _memo['model'] = m2
    return out
