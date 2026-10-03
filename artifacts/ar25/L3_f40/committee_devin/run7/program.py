# Mechanics: mirror-axis puzzle. Hidden sprites on a 3x3-block lattice: one horizontal axis wall, targets (static),
# pieces (holed 3x3 blocks); every piece casts a gray reflection mirrored across the axis centre row. ACTION5 cycles the
# selection axis -> piece_0 -> piece_1 -> ... -> axis; ACTION1-4 move the selection by 3 (axis only vertically), blocked by
# board edges / axis / other pieces. Render layers wall1<target2<reflection3<piece4, holes transparent over lower layers,
# else filled (selected 0, reflection 4); re-extract components. Hidden state (continuity-gated): sprite split/order. Unconfirmed: blocking rules.
import json

N = 63
DIRS = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
DEFAULT_TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
                'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
PREFIX = {'wall': 'wall_h', 'target': 'target', 'reflection': 'reflection', 'player': 'piece'}
_last = {'out': None, 'model': None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells(o):
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def blocks_of(o):
    return {(x // 3, y // 3) for x, y, _ in cells(o)}


def components(bs):
    bs, out = set(bs), []
    while bs:
        stack, comp = [bs.pop()], set()
        while stack:
            b = stack.pop()
            comp.add(b)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (b[0] + d[0], b[1] + d[1])
                if n in bs:
                    bs.remove(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(state):
    """Stateless recovery of the hidden model from an extracted state."""
    walls = [o for o in state if o['type'] == 'wall']
    tvis = set()
    targets = []
    for o in state:
        if o['type'] == 'target':
            targets.append(blocks_of(o))
            tvis |= {(x, y) for x, y, _ in cells(o)}
    axis_sel = any(v == 0 for o in walls for _, _, v in cells(o))
    sprites, sel = [], 'axis'
    for o in sorted((o for o in state if o['type'] == 'player'), key=lambda o: (o['x'], o['y'])):
        pix = {(x, y): v for x, y, v in cells(o)}
        bs = blocks_of(o)
        zero = {b for b in bs if pix.get((3 * b[0] + 1, 3 * b[1] + 1)) == 0}
        if not zero:
            sprites.append(bs)
            continue
        amb = {b for b in bs - zero if (3 * b[0] + 1, 3 * b[1] + 1) in tvis}
        sel = len(sprites)
        sprites.append(zero | amb)
        sprites.extend(components(bs - zero - amb))
    if axis_sel:
        sel = 'axis'
    tmpl = {}
    for o in state:
        tmpl.setdefault(o['type'], o)
    return {'ax': min(o['x'] for o in walls), 'sel': sel, 'sprites': sprites, 'targets': targets,
            'tmpl': tmpl, 'other': [o for o in state if o['type'] not in LAYER]}


def axis_rows(ax):
    return range(ax, ax + 3)


def step(m, action):
    m = dict(m, sprites=[set(s) for s in m['sprites']])
    aid = action.get('action_id') if isinstance(action, dict) else action
    if aid == 5:
        if m['sel'] == 'axis':
            m['sel'] = 0 if m['sprites'] else 'axis'
        else:
            m['sel'] = m['sel'] + 1 if m['sel'] + 1 < len(m['sprites']) else 'axis'
    elif aid in DIRS:
        dx, dy = DIRS[aid]
        if m['sel'] == 'axis':
            if dx and can_axis(m, m['ax'] + dx):
                m['ax'] += dx
        else:
            k = m['sel']
            moved = {(bx + dx // 3, by + dy // 3) for bx, by in m['sprites'][k]}
            if can_piece(m, k, moved):
                m['sprites'][k] = moved
    return m


def can_axis(m, ax):
    if ax < 0 or ax + 3 > N:
        return False
    rows = set(axis_rows(ax))
    return not any(3 * bx + i in rows for s in m['sprites'] for bx, _ in s for i in range(3))


def can_piece(m, k, moved):
    rows = set(axis_rows(m['ax']))
    others = set().union(*[s for i, s in enumerate(m['sprites']) if i != k]) if len(m['sprites']) > 1 else set()
    for bx, by in moved:
        if bx < 0 or by < 0 or 3 * bx + 3 > N or 3 * by + 3 > N:
            return False
        if any(3 * bx + i in rows for i in range(3)) or (bx, by) in others:
            return False
    return True


def render(m):
    frame = {}  # (x,y) -> (color, kind, owner)

    def put(x, y, v, kind, owner):
        if 0 <= x < N and 0 <= y < N:
            frame[(x, y)] = (v, kind, owner)

    ax = m['ax']
    for y in range(N):
        for i in range(3):
            if i == 1 and y % 3 == 1:
                if m['sel'] == 'axis':
                    put(ax + i, y, 0, 'wall', 0)
            else:
                put(ax + i, y, 10, 'wall', 0)
    for t, bs in enumerate(m['targets']):
        for bx, by in bs:
            for i in range(3):
                for j in range(3):
                    put(3 * bx + i, 3 * by + j, 11, 'target', t)
    mb = (2 * ax) // 3  # block row mirror: rows r -> 2*(ax+1)-r maps block bx to mb-bx
    layers = [('reflection', 4, lambda s, k: {(mb - bx, by) for bx, by in s}, lambda k: 4),
              ('player', 5, lambda s, k: s, lambda k: 0 if m['sel'] == k else None)]
    for kind, col, place, fill in layers:
        for k, s in enumerate(m['sprites']):
            for bx, by in place(s, k):
                for i in range(3):
                    for j in range(3):
                        x, y = 3 * bx + i, 3 * by + j
                        if i == 1 and j == 1:
                            if (x, y) not in frame and fill(k) is not None:
                                put(x, y, fill(k), kind, k)
                        else:
                            put(x, y, col, kind, k)
    return frame


def obj(m, kind, name, pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for (x, y), v in pts.items():
        pix[x - x0][y - y0] = v
    t = m['tmpl'].get(kind, {})
    return {'name': name, 'type': kind, 'x': x0, 'y': y0, 'w': w, 'h': h, 'layer': t.get('layer', LAYER[kind]),
            'tags': list(t.get('tags', DEFAULT_TAGS[kind])), 'pixels': pix}


def pixel_components(pts):
    rest, out = dict(pts), []
    while rest:
        p0 = next(iter(rest))
        stack, comp = [p0], {p0: rest.pop(p0)}
        while stack:
            x, y = stack.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in rest:
                    comp[n] = rest.pop(n)
                    stack.append(n)
        out.append(comp)
    return out


def extract(m, frame):
    groups = []
    for kind in ('wall', 'player', 'reflection'):
        pts = {p: v for p, (v, k, _) in frame.items() if k == kind}
        groups.append((kind, pixel_components(pts)))
    tg = {}
    for p, (v, k, o) in frame.items():
        if k == 'target':
            tg.setdefault(o, {})[p] = v
    groups.append(('target', list(tg.values())))
    zeros = sum(1 for v, k, _ in frame.values() if v == 0 and k != 'wall')
    out = [dict(o) for o in m['other']]
    for kind, comps in groups:
        comps.sort(key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
        off = zeros if kind == 'wall' else 0
        for i, c in enumerate(comps):
            out.append(obj(m, kind, '%s_%d' % (PREFIX[kind], i + off), c))
    return out


def transition_function(state, action):
    model = None
    if _last['out'] is not None and canon(state) == _last['out']:
        model = _last['model']
    if model is None:
        model = parse(state)
    model = step(model, action)
    out = extract(model, render(model))
    _last['out'], _last['model'] = canon(out), model
    return out
