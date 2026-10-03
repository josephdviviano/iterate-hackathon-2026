# Mechanics: one horizontal mirror axis (wall row of 3x3 blocks, 0-dot centres when selected) and 3x3-block pieces.
# ACTION5 cycles selection axis -> pieces (bbox order) -> axis; ACTION1/2 move selection one block up/down, ACTION3/4 move a
# selected piece left/right (axis ignores 3/4). Axis may pass under pieces and pieces may sit on the axis row; pieces stay in
# bounds and may not overlap. Reflection block = (2A-R, C), none on the axis row. Render layers wall<target<reflection<piece,
# holes show lower solids (selected wall dot = solid 0) else own fill, then re-extract. Unconfirmed: A5 order beyond observed.
DEFAULT_TAGS = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'],
                'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
PREFIX = {'wall': 'wall_h_', 'player': 'piece_', 'reflection': 'reflection_', 'target': 'target_'}
NB = 21
_memo = {'out': None, 'model': None}


def cells_of(o):
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
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in blocks:
                    blocks.discard(nb)
                    comp.add(nb)
                    st.append(nb)
        comps.append(comp)
    return comps


def piece_key(p):
    return (min(3 * r for r, c in p), min(3 * c for r, c in p))


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    rows = {}
    for o in walls:
        for x, y, v in cells_of(o):
            rows[x // 3] = rows.get(x // 3, 0) + 1
    axis = max(rows, key=rows.get) if rows else 0
    axis_sel = any(v == 0 for o in walls for _, _, v in cells_of(o))
    own = {}
    for o in state:
        for x, y, v in cells_of(o):
            own[(x, y)] = (o['type'], v)
    targets = []
    for o in state:
        if o['type'] == 'target':
            targets.append({(x // 3, y // 3) for x, y, v in cells_of(o)})
    pieces, sel = [], 'axis' if axis_sel else None
    for o in state:
        if o['type'] != 'player':
            continue
        blocks = {(x // 3, y // 3) for x, y, v in cells_of(o) if v > 0}
        zero = {(x // 3, y // 3) for x, y, v in cells_of(o) if v == 0}
        if axis_sel or not zero:
            pieces.extend(block_comps(blocks))
            continue
        selb = set(zero)
        amb = {b for b in blocks - zero if own.get((3 * b[0] + 1, 3 * b[1] + 1), ('player', -1))[0] != 'player'}
        grow = True
        while grow:
            grow = False
            for b in list(amb):
                if any((b[0] + dr, b[1] + dc) in selb for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    selb.add(b)
                    amb.discard(b)
                    grow = True
        sp = block_comps(selb)
        main = max(sp, key=len)
        pieces.append(main)
        sel = main
        rest = [c for c in sp if c is not main]
        pieces.extend(rest)
        pieces.extend(block_comps(blocks - selb))
    pieces.sort(key=piece_key)
    if sel is None:
        sel = 'axis'
    sel_i = 'axis' if sel == 'axis' else pieces.index(sel)
    return {'axis': axis, 'sel': sel_i, 'pieces': pieces, 'targets': targets}


def step(m, action):
    m = {'axis': m['axis'], 'sel': m['sel'], 'pieces': [set(p) for p in m['pieces']], 'targets': m['targets']}
    if isinstance(action, dict) or action not in (1, 2, 3, 4, 5):
        return m
    if action == 5:
        if m['sel'] == 'axis':
            m['sel'] = 0 if m['pieces'] else 'axis'
        else:
            m['sel'] = m['sel'] + 1 if m['sel'] + 1 < len(m['pieces']) else 'axis'
        return m
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    if m['sel'] == 'axis':
        if dr and 0 <= m['axis'] + dr < NB:
            m['axis'] += dr
        return m
    i = m['sel']
    moved = {(r + dr, c + dc) for r, c in m['pieces'][i]}
    if not all(0 <= r < NB and 0 <= c < NB for r, c in moved):
        return m
    if any(moved & p for j, p in enumerate(m['pieces']) if j != i):
        return m
    m['pieces'][i] = moved
    return m


def render(m):
    # stacks[cell] = list of (layer, solid, colour, type, id); solid False -> hole with fill colour (or None)
    stacks = {}

    def put(r0, c0, typ, ring, centre, solid_centre, ident):
        for i in range(3):
            for j in range(3):
                cell = (r0 + i, c0 + j)
                if cell[0] > 62 or cell[1] > 62:
                    continue
                if i == 1 and j == 1:
                    ent = (LAYER[typ], solid_centre, centre, typ, ident)
                else:
                    ent = (LAYER[typ], True, ring, typ, ident)
                stacks.setdefault(cell, []).append(ent)
    A, selp = m['axis'], m['sel']
    for C in range(NB):
        put(3 * A, 3 * C, 'wall', 10, 0 if selp == 'axis' else None, selp == 'axis', 0)
    for k, t in enumerate(m['targets']):
        for r, c in t:
            put(3 * r, 3 * c, 'target', 11, 11, True, k)
    for k, p in enumerate(m['pieces']):
        for r, c in p:
            put(3 * r, 3 * c, 'player', 5, 0 if selp == k else None, False, k)
            if r != A and 0 <= 2 * A - r < NB:
                put(3 * (2 * A - r), 3 * c, 'reflection', 4, 4, False, k)
    frame = {}
    for cell, st in stacks.items():
        st.sort(key=lambda e: -e[0])
        hit = next((e for e in st if e[1]), None)
        if hit is None and st[0][2] is not None:
            hit = st[0]
        if hit is not None and hit[2] is not None:
            frame[cell] = (hit[2], hit[3], hit[4])
    return frame


def comps(cells):
    cells, out = set(cells), []
    while cells:
        st = [cells.pop()]
        comp = set(st)
        while st:
            x, y = st.pop()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in cells:
                    cells.discard(nb)
                    comp.add(nb)
                    st.append(nb)
        out.append(comp)
    return out


def make_obj(typ, name, cells, frame, tags):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for x, y in cells:
        pix[x - x0][y - y0] = frame[(x, y)][0]
    return {'name': name, 'type': typ, 'tags': list(tags), 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': LAYER[typ], 'pixels': pix}


def extract(frame, templates):
    out = []
    zeros = {c for c, v in frame.items() if v[0] == 0}
    for typ in ('wall', 'player'):
        own = {c for c, v in frame.items() if v[1] == typ and v[0] != 0}
        cs = sorted(comps(own | zeros), key=lambda s: (min(c[0] for c in s), min(c[1] for c in s)))
        for k, s in enumerate(cs):
            if s & own:
                out.append(make_obj(typ, PREFIX[typ] + str(k), s, frame, templates.get(typ, DEFAULT_TAGS[typ])))
    refl = {c for c, v in frame.items() if v[1] == 'reflection' and v[0] != 0}
    for k, s in enumerate(sorted(comps(refl), key=lambda s: (min(c[0] for c in s), min(c[1] for c in s)))):
        out.append(make_obj('reflection', PREFIX['reflection'] + str(k), s, frame,
                            templates.get('reflection', DEFAULT_TAGS['reflection'])))
    groups = {}
    for c, v in frame.items():
        if v[1] == 'target':
            groups.setdefault(v[2], set()).add(c)
    tg = sorted(groups.values(), key=lambda s: (min(c[0] for c in s), min(c[1] for c in s)))
    for k, s in enumerate(tg):
        out.append(make_obj('target', PREFIX['target'] + str(k), s, frame,
                            templates.get('target', DEFAULT_TAGS['target'])))
    return out


def transition_function(state, action):
    templates = {o['type']: o.get('tags', []) for o in state}
    model = None
    if _memo['out'] is not None and canon(state) == _memo['out']:
        model = _memo['model']
    if model is None:
        model = parse(state)
    new = step(model, action)
    out = [dict(o) for o in state if o['type'] not in LAYER]
    out += extract(render(new), templates)
    _memo['out'] = canon(out)
    _memo['model'] = new
    return out
