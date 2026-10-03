# Mechanics: horizontal mirror axis (wall, 3 rows, dotted centre row) + movable 3x3-block pieces (holed) + targets.
# ACTION5 cycles selection axis -> pieces (sorted by position) -> axis; ACTION1/2 move selection up/down 3,
# ACTION3/4 move a selected piece left/right 3 (axis ignores them). Reflections = pieces mirrored about the axis.
# Render: wall1<target2<reflection3<piece4<counter5; holes transparent over lower pixels, else 0 (selected) / 4 (reflection).
# Re-extract: walls/pieces/reflections = 4-connected comps, targets per sprite; unconfirmed: move guards (bounds, axis overlap).
B = 3
GRID = 64
DIRS = {1: (-B, 0), 2: (B, 0), 3: (0, -B), 4: (0, B)}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
DEFAULT_TAGS = {'reflection': ['mirror', 'gray'], 'wall': ['axis', 'horizontal'],
                'player': ['movable', 'black'], 'target': ['goal', 'yellow']}
_memo = {'out': None, 'model': None}


def owned(o):
    for i, r in enumerate(o.get('pixels', [])):
        for j, v in enumerate(r):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def blk(x, y, al):
    return (x - al[0]) // B * B + al[0], (y - al[1]) // B * B + al[1]


def block_groups(blocks):
    blocks, groups = set(blocks), []
    while blocks:
        stack, g = [blocks.pop()], set()
        while stack:
            b = stack.pop(); g.add(b)
            for dx, dy in DIRS.values():
                n = (b[0] + dx, b[1] + dy)
                if n in blocks:
                    blocks.remove(n); stack.append(n)
        groups.append(g)
    return groups


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    ax = walls[0]['x'] if walls else 0
    al = (ax % B, 0)
    m = {'axis': ax, 'span': (min(o['y'] for o in walls), max(o['y'] + o['h'] for o in walls)) if walls else (0, 0),
         'tags': {}, 'pieces': [], 'targets': [], 'sel': None,
         'occ': [o for o in state if o['type'] not in LAYER]}
    for o in state:
        m['tags'].setdefault(o['type'], o.get('tags', []))
    if any(v == 0 for o in walls for _, _, v in owned(o)):
        m['sel'] = 'axis'
    seen = {(x, y) for o in state if o['type'] != 'player' for x, y, _ in owned(o)}
    for o in state:
        if o['type'] == 'target':
            m['targets'].append((set(blk(x, y, al) for x, y, _ in owned(o)), o.get('tags')))
        if o['type'] == 'player':
            px = {(x, y): v for x, y, v in owned(o)}
            bl = set(blk(x, y, al) for x, y in px)
            selb = {b for b in bl if px.get((b[0] + 1, b[1] + 1)) == 0}
            clear = {b for b in bl if (b[0] + 1, b[1] + 1) in seen}
            for g in block_groups(selb | clear):
                if g & selb:
                    selb |= g
            if selb:
                m['sel'] = len(m['pieces'])
                m['pieces'].append(selb)
            m['pieces'].extend(block_groups(bl - selb))
    return m


def bounds_ok(m, cells):
    occ = set()
    for o in m['occ']:
        occ |= {(o['x'] + i, o['y'] + j) for i in range(o['w']) for j in range(o['h'])}
    return all(0 <= x < GRID and 0 <= y < GRID and (x, y) not in occ for x, y in cells)


def block_cells(bs):
    return {(bx + i, by + j) for bx, by in bs for i in range(B) for j in range(B)}


def order(m):
    return sorted(range(len(m['pieces'])), key=lambda k: min(m['pieces'][k]))


def step(m, action):
    sel = m['sel']
    if action == 5:
        o = order(m)
        if sel == 'axis' or sel is None:
            m['sel'] = o[0] if o else 'axis'
        else:
            i = o.index(sel)
            m['sel'] = o[i + 1] if i + 1 < len(o) else 'axis'
    elif action in DIRS:
        dx, dy = DIRS[action]
        axis_rows = set(range(m['axis'], m['axis'] + B))
        if sel == 'axis' and dx:
            nr = m['axis'] + dx
            rows = set(range(nr, nr + B))
            piece_rows = {x for p in m['pieces'] for x, _ in block_cells(p)}
            if bounds_ok(m, [(r, m['span'][0]) for r in rows]) and not rows & piece_rows:
                m['axis'] = nr
        elif isinstance(sel, int):
            nb = {(x + dx, y + dy) for x, y in m['pieces'][sel]}
            cells = block_cells(nb)
            if bounds_ok(m, cells) and not {x for x, _ in cells} & axis_rows:
                m['pieces'][sel] = nb
    return m


def render(m):
    """Return {(x,y): (value, sprite_key)} composited frame."""
    frame = {}
    c2 = 2 * (m['axis'] + 1)

    def paint(key, cells, color, fill):
        lower = dict(frame)
        for (x, y), hole in cells.items():
            if not (0 <= x < GRID and 0 <= y < GRID):
                continue
            if not hole:
                frame[(x, y)] = (color, key)
            elif (x, y) in lower:
                pass
            elif fill is not None:
                frame[(x, y)] = (fill, key)

    def holed(bs, mirror=False):
        out = {}
        for bx, by in bs:
            for i in range(B):
                for j in range(B):
                    x = bx + i
                    out[(c2 - x if mirror else x, by + j)] = (i == 1 and j == 1)
        return out
    wall = {}
    for y in range(*m['span']):
        for i in range(B):
            wall[(m['axis'] + i, y)] = (i == 1 and (y - m['span'][0]) % B == 1)
    paint(('wall', 0), wall, 10, 0 if m['sel'] == 'axis' else None)
    for k, (bs, _) in enumerate(m['targets']):
        paint(('target', k), {c: False for c in block_cells(bs)}, 11, None)
    for k, bs in enumerate(m['pieces']):
        paint(('reflection', k), holed(bs, True), 4, 4)
    for k, bs in enumerate(m['pieces']):
        paint(('player', k), holed(bs), 5, 0 if m['sel'] == k else None)
    for o in m['occ']:
        for i in range(o['w']):
            for j in range(o['h']):
                frame.pop((o['x'] + i, o['y'] + j), None)
    return frame


def components(cells):
    cells, comps = set(cells), []
    while cells:
        stack, g = [cells.pop()], set()
        while stack:
            c = stack.pop(); g.add(c)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + dx, c[1] + dy)
                if n in cells:
                    cells.remove(n); stack.append(n)
        comps.append(g)
    return comps


def make_obj(typ, cells, vals, tags):
    x0 = min(x for x, _ in cells); y0 = min(y for _, y in cells)
    w = max(x for x, _ in cells) - x0 + 1; h = max(y for _, y in cells) - y0 + 1
    pix = [[vals[(x0 + i, y0 + j)] if (x0 + i, y0 + j) in cells else -1 for j in range(h)] for i in range(w)]
    return {'name': None, 'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': LAYER[typ],
            'tags': list(tags), 'pixels': pix}


def extract(m):
    frame = render(m)
    vals = {c: v for c, (v, _) in frame.items()}
    by = {}
    for c, (_, key) in frame.items():
        by.setdefault(key, set()).add(c)
    objs = {'wall': [], 'target': [], 'reflection': [], 'player': []}
    tags = lambda t: m['tags'].get(t) or DEFAULT_TAGS[t]
    for k, (_, tg) in enumerate(m['targets']):
        if ('target', k) in by:
            objs['target'].append(make_obj('target', by[('target', k)], vals, tg or tags('target')))
    for typ in ('wall', 'reflection', 'player'):
        cells = set().union(*[s for (t, _), s in by.items() if t == typ] or [set()])
        objs[typ] = [make_obj(typ, g, vals, tags(typ)) for g in components(cells)]
    black = sum(1 for c, (v, key) in frame.items() if v == 0 and key[0] != 'wall')
    out = [dict(o) for o in m['occ']]
    prefix = {'wall': 'wall_h_', 'target': 'target_', 'reflection': 'reflection_', 'player': 'piece_'}
    for typ, lst in objs.items():
        off = black if typ == 'wall' else 0
        for i, o in enumerate(sorted(lst, key=lambda o: (o['x'], o['y']))):
            o['name'] = prefix[typ] + str(i + off)
            out.append(o)
    return out


def canon(s):
    return sorted(repr(sorted(o.items())) for o in s)


def transition_function(state, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    if _memo['out'] is not None and canon(state) == _memo['out']:
        m = _memo['model']
    else:
        m = parse(state)
    m = step(m, aid)
    out = extract(m)
    _memo['out'], _memo['model'] = canon(out), m
    return out
