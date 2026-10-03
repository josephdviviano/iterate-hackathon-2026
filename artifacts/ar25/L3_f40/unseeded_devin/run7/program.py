# Mechanics: 3x3-block mirror puzzle. A horizontal axis wall (rows ax..ax+2) and pieces (player, 3x3 blocks with a
# centre hole) are selectable; ACTION5 cycles selection axis -> pieces (sorted by x,y) -> axis. Axis: A1/A2 = row -/+3.
# Selected piece: A1/A2/A3/A4 = x-3/x+3/y-3/y+3 (x=row); moves blocked by bounds, other pieces, axis band. Reflections:
# block at row b mirrored to 2*ax-b. Holes are transparent over lower layers, else filled (selected 0, reflection 4).
# Render layers wall1<target2<reflection3<piece4, re-extract (4-conn comps, targets per sprite). Hyp: wall_h index = rank among walls + 0 pixels.
N = 63
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
DEFTAGS = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'],
           'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
COLOR = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
_mem = {'out': None, 'model': None}


def canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells(o):
    p = o.get('pixels') or []
    for i, row in enumerate(p):
        for j, v in enumerate(row):
            if v != -1:
                yield o['x'] + i, o['y'] + j, v


def blk(x, y):
    return (x // 3 * 3, y // 3 * 3)


def adj_blocks(b):
    x, y = b
    return [(x - 3, y), (x + 3, y), (x, y - 3), (x, y + 3)]


def split_cluster(blocks):
    blocks, out = set(blocks), []
    while blocks:
        s = blocks.pop(); comp = {s}; st = [s]
        while st:
            for n in adj_blocks(st.pop()):
                if n in blocks:
                    blocks.discard(n); comp.add(n); st.append(n)
        out.append(comp)
    return out


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    ax = min(o['x'] for o in walls) if walls else 0
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    lower = {}
    targets = []
    for o in state:
        if o['type'] == 'target':
            targets.append({blk(x, y) for x, y, _ in cells(o)})
        if o['type'] in ('target', 'reflection', 'wall'):
            for x, y, v in cells(o):
                lower[(x, y)] = v
    pieces, sel = [], None
    for o in state:
        if o['type'] != 'player':
            continue
        own = {(x, y): v for x, y, v in cells(o)}
        bl = {blk(x, y) for (x, y), v in own.items() if v == 5}
        zero = {b for b in bl if own.get((b[0] + 1, b[1] + 1)) == 0}
        amb = {b for b in bl - zero if (b[0] + 1, b[1] + 1) in lower}
        if zero:
            grown, frontier = set(zero), list(zero)
            while frontier:
                for n in adj_blocks(frontier.pop()):
                    if n in amb and n not in grown:
                        grown.add(n); frontier.append(n)
            sel = len(pieces)
            pieces.append(grown)
            pieces.extend(split_cluster(bl - grown))
        else:
            pieces.append(bl)
    return {'ax': ax, 'sel': 'axis' if axis_sel else sel, 'pieces': pieces, 'targets': targets}


def piece_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (min(b[0] for b in pieces[i]), min(b[1] for b in pieces[i])))


def in_band(bx, ax):
    return ax <= bx + 2 and bx <= ax + 2


def step(m, action):
    a = action if isinstance(action, int) else action.get('action_id')
    pieces = [set(p) for p in m['pieces']]
    ax, sel = m['ax'], m['sel']
    if a == 5:
        order = piece_order(pieces)
        if sel == 'axis' or sel is None:
            sel = order[0] if order else 'axis'
        else:
            k = order.index(sel)
            sel = order[k + 1] if k + 1 < len(order) else 'axis'
    elif sel == 'axis' and a in (1, 2):
        nax = ax + (-3 if a == 1 else 3)
        if 0 <= nax <= N - 3 and not any(in_band(b[0], nax) for p in pieces for b in p):
            ax = nax
    elif isinstance(sel, int) and a in (1, 2, 3, 4):
        dx, dy = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}[a]
        moved = {(x + dx, y + dy) for x, y in pieces[sel]}
        others = set().union(*[p for i, p in enumerate(pieces) if i != sel]) if len(pieces) > 1 else set()
        ok = all(0 <= x <= N - 3 and 0 <= y <= N - 3 and not in_band(x, ax) for x, y in moved)
        if ok and not (moved & others):
            pieces[sel] = moved
    return {'ax': ax, 'sel': sel, 'pieces': pieces, 'targets': m['targets']}


def render(m):
    """Return list of sprites: (type, id, {(x,y): ('s', color) | ('h', fill)})."""
    sprites = []
    ax, sel = m['ax'], m['sel']
    w = {}
    for y in range(N):
        for dx in range(3):
            if dx == 1 and y % 3 == 1:
                w[(ax + 1, y)] = ('h', 0 if sel == 'axis' else -1)
            else:
                w[(ax + dx, y)] = ('s', 10)
    sprites.append(('wall', 0, w))
    for k, t in enumerate(m['targets']):
        sprites.append(('target', k, {(bx + i, by + j): ('s', 11) for bx, by in t for i in range(3) for j in range(3)}))

    def blockspr(blocks, col, fill):
        d = {}
        for bx, by in blocks:
            for i in range(3):
                for j in range(3):
                    d[(bx + i, by + j)] = ('h', fill) if (i, j) == (1, 1) else ('s', col)
        return d
    refl = {(2 * ax - bx, by) for p in m['pieces'] for bx, by in p}
    sprites.append(('reflection', 0, blockspr(refl, 4, 4)))
    for k, p in enumerate(m['pieces']):
        sprites.append(('player', k, blockspr(p, 5, 0 if sel == k else -1)))
    return sprites


def composite(sprites):
    by_layer = sorted(sprites, key=lambda s: -LAYER[s[0]])
    pix = {}
    keys = set()
    for s in sprites:
        keys |= set(s[2])
    for c in keys:
        if not (0 <= c[0] < N and 0 <= c[1] < N):
            continue
        hole = None
        for s in by_layer:
            e = s[2].get(c)
            if e is None:
                continue
            if e[0] == 's':
                pix[c] = (s[0], s[1], e[1]); break
            if hole is None:
                hole = (s[0], s[1], e[1])
        else:
            if hole is not None and hole[2] != -1:
                pix[c] = hole
    return pix


def components(cs):
    cs, out = set(cs), []
    while cs:
        s = cs.pop(); comp = {s}; st = [s]
        while st:
            x, y = st.pop()
            for n in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if n in cs:
                    cs.discard(n); comp.add(n); st.append(n)
        out.append(comp)
    return out


def make_obj(typ, cellmap, tags):
    xs = [c[0] for c in cellmap]; ys = [c[1] for c in cellmap]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    px = [[cellmap.get((x0 + i, y0 + j), -1) for j in range(h)] for i in range(w)]
    return {'type': typ, 'layer': LAYER[typ], 'tags': list(tags), 'x': x0, 'y': y0, 'w': w, 'h': h, 'pixels': px}


def extract(m, tags, extra):
    pix = composite(render(m))
    objs = {'wall': [], 'player': [], 'reflection': [], 'target': []}
    for typ in ('wall', 'player', 'reflection'):
        cs = {c: v[2] for c, v in pix.items() if v[0] == typ}
        for comp in components(cs):
            objs[typ].append(make_obj(typ, {c: cs[c] for c in comp}, tags[typ]))
    for k in range(len(m['targets'])):
        cs = {c: v[2] for c, v in pix.items() if v[0] == 'target' and v[1] == k}
        if cs:
            objs['target'].append(make_obj('target', cs, tags['target']))
    zeros = [(c, None) for c, v in pix.items() if v[2] == 0 and v[0] != 'wall']
    out = list(extra)
    prefix = {'wall': 'wall_h', 'player': 'piece', 'reflection': 'reflection', 'target': 'target'}
    for typ, lst in objs.items():
        if typ == 'wall':
            keys = [((o['x'], o['y']), o) for o in lst] + zeros
            keys.sort(key=lambda t: t[0])
            for r, (_, o) in enumerate(keys):
                if o is not None:
                    o['name'] = 'wall_h_%d' % r
        else:
            lst.sort(key=lambda o: (o['x'], o['y']))
            for r, o in enumerate(lst):
                o['name'] = '%s_%d' % (prefix[typ], r)
        out.extend(lst)
    return out


def transition_function(state, action):
    tags = dict(DEFTAGS)
    for o in state:
        if o['type'] in tags:
            tags[o['type']] = o.get('tags', tags[o['type']])
    extra = [dict(o) for o in state if o['type'] not in LAYER]
    if _mem['out'] is not None and canon(state) == _mem['out']:
        m = _mem['model']
    else:
        m = parse(state)
    m2 = step(m, action)
    out = extract(m2, tags, extra)
    _mem['out'] = canon(out)
    _mem['model'] = m2
    return out
