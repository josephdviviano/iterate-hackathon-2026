# Mechanics: single mirror axis (3-row wall band at block row A) + 3x3-block pieces, layered render, re-extract.
# ACTION5 cycles selection axis -> pieces (sorted by min block) -> axis; selection shows as 0 hole fills.
# Selected axis: A1/A2 move A by -/+1 block; selected piece: A1-4 move it 1 block, blocked by bounds/other pieces.
# Piece blocks above the axis reflect to block row 2A-r (clipped, not on piece blocks); targets/counter static.
# Unconfirmed: selection order after the first piece, reflections of blocks below the axis, pieces vs axis row.
N = 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
        'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
PREFIX = {'wall': 'wall_h_', 'target': 'target_', 'reflection': 'reflection_', 'player': 'piece_'}
COLOR = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_memo = {'out': None, 'model': None}


def cells(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def block_comps(blocks):
    blocks, comps = set(blocks), []
    while blocks:
        st = [blocks.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + d[0], c + d[1])
                if n in blocks:
                    blocks.discard(n)
                    comp.add(n)
                    st.append(n)
        comps.append(comp)
    return comps


def parse(state):
    owner = {}
    for o in state:
        for r, c, v in cells(o):
            owner[(r, c)] = (o['type'], v, id(o))
    axis, axis_sel = None, False
    targets, piece_blocks = [], set()
    for o in state:
        if o['type'] == 'wall':
            for r, c, v in cells(o):
                if v == 10 and axis is None:
                    axis = r // 3
                if v == 0:
                    axis_sel = True
        elif o['type'] == 'target':
            targets.append({(r // 3, c // 3) for r, c, v in cells(o)})
        elif o['type'] == 'player':
            for r, c, v in cells(o):
                if v == 5 and r % 3 == 0 and c % 3 == 0:
                    piece_blocks.add((r // 3, c // 3))
    pieces, sel = [], 'axis'
    if axis_sel:
        pieces = block_comps(piece_blocks)
    else:
        zero = {b for b in piece_blocks if owner.get((3 * b[0] + 1, 3 * b[1] + 1), (None, -1))[1] == 0
                and owner[(3 * b[0] + 1, 3 * b[1] + 1)][0] == 'player'}
        amb = {b for b in piece_blocks
               if owner.get((3 * b[0] + 1, 3 * b[1] + 1), ('player',))[0] != 'player'}
        selected = set()
        for comp in block_comps(zero | amb):
            if comp & zero:
                selected |= comp
        pieces = block_comps(piece_blocks - selected)
        if selected:
            pieces.append(selected)
            sel = selected
    pieces = sorted(pieces, key=lambda p: min(p))
    sel_i = None if sel == 'axis' else pieces.index(sel)
    return {'axis': axis, 'sel': sel_i, 'pieces': pieces, 'targets': targets}


def step(m, action):
    m = {'axis': m['axis'], 'sel': m['sel'], 'pieces': [set(p) for p in m['pieces']],
         'targets': m['targets']}
    if action == 5:
        if m['sel'] is None:
            m['sel'] = 0 if m['pieces'] else None
        else:
            m['sel'] = m['sel'] + 1 if m['sel'] + 1 < len(m['pieces']) else None
        return m
    if action not in MOVES:
        return m
    dr, dc = MOVES[action]
    if m['sel'] is None:
        if dc == 0 and 0 <= m['axis'] + dr < N:
            m['axis'] += dr
        return m
    i = m['sel']
    others = set().union(*[p for k, p in enumerate(m['pieces']) if k != i])
    moved = {(r + dr, c + dc) for r, c in m['pieces'][i]}
    if all(0 <= r < N and 0 <= c < N for r, c in moved) and not (moved & others):
        m['pieces'][i] = moved
    return m


def render(m):
    """Return cell -> (type, value, sprite key) after layered compositing."""
    A, sel = m['axis'], m['sel']
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    sprites = []  # (layer, type, key, blocks, solid_centre, fill)
    sprites.append((1, 'wall', 'w', {(A, c) for c in range(N)}, False, 0 if sel is None else -1))
    for k, t in enumerate(m['targets']):
        sprites.append((2, 'target', ('t', k), t, True, 11))
    refl = set()
    for p in m['pieces']:
        for r, c in p:
            if r < A and 2 * A - r < N and (2 * A - r, c) not in occupied:
                refl.add((2 * A - r, c))
    sprites.append((3, 'reflection', 'r', refl, False, 4))
    for k, p in enumerate(m['pieces']):
        sprites.append((4, 'player', ('p', k), p, False, 0 if sel == k else -1))
    sprites.sort(key=lambda s: -s[0])
    stacks = {}
    for s in sprites:
        for b in s[3]:
            stacks.setdefault(b, []).append(s)
    out = {}
    for (br, bc), stack in stacks.items():
        for i in range(3):
            for j in range(3):
                centre = i == 1 and j == 1
                hit = None
                for s in stack:
                    if not centre or s[4]:
                        hit = (s[1], COLOR[s[1]], s[2])
                        break
                if hit is None:
                    fill = next((s[5] for s in stack if s[5] >= 0), -1)
                    if fill >= 0:
                        hit = (stack[0][1], fill, stack[0][2])
                if hit:
                    out[(3 * br + i, 3 * bc + j)] = hit
    return out


def make_obj(typ, idx, cs, vals):
    x0, y0 = min(r for r, c in cs), min(c for r, c in cs)
    x1, y1 = max(r for r, c in cs), max(c for r, c in cs)
    px = [[-1] * (y1 - y0 + 1) for _ in range(x1 - x0 + 1)]
    for r, c in cs:
        px[r - x0][c - y0] = vals[(r, c)]
    return {'name': PREFIX[typ] + str(idx), 'type': typ, 'tags': list(TAGS[typ]), 'layer': LAYER[typ],
            'x': x0, 'y': y0, 'w': x1 - x0 + 1, 'h': y1 - y0 + 1, 'pixels': px}


def cell_comps(cs):
    cs, comps = set(cs), []
    while cs:
        st = [cs.pop()]
        comp = set(st)
        while st:
            r, c = st.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cs:
                    cs.discard(n)
                    comp.add(n)
                    st.append(n)
        comps.append(comp)
    return comps


def extract(img):
    vals = {k: v[1] for k, v in img.items()}
    zeros = {k for k, v in img.items() if v[1] == 0}
    objs = []
    for typ in ('wall', 'player', 'reflection'):
        own = {k for k, v in img.items() if v[0] == typ and v[1] != 0}
        pool = own | zeros if typ != 'reflection' else own
        comps = sorted(cell_comps(pool), key=lambda cs: (min(r for r, c in cs), min(c for r, c in cs)))
        for i, cs in enumerate(comps):
            if cs & own:
                objs.append(make_obj(typ, i, cs, vals))
    tsets = {}
    for k, v in img.items():
        if v[0] == 'target':
            tsets.setdefault(v[2], set()).add(k)
    tl = sorted(tsets.values(), key=lambda cs: (min(r for r, c in cs), min(c for r, c in cs)))
    objs += [make_obj('target', i, cs, vals) for i, cs in enumerate(tl)]
    return objs


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get('action_id')
    model = None
    if _memo['out'] is not None and canon(state) == _memo['out']:
        model = _memo['model']
    if model is None:
        model = parse(state)
    new = step(model, action)
    out = [dict(o) for o in state if o['type'] not in LAYER] + extract(render(new))
    _memo['out'], _memo['model'] = canon(out), new
    return out
