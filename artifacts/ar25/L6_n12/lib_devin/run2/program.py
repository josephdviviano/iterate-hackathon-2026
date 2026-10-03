# Mechanics: two mirror axes (h-axis block row, v-axis block col) + holed 3x3-block pieces + static targets on a 21x21 block board (x=row).
# ACTION5 cycles selection h-axis -> v-axis -> pieces -> h-axis; ACTION1-4 move the selection one block (axes only along their normal;
# blocked by board edge / pieces / axes). Render layers wall_h<wall_v<target<reflection<piece, holes transparent (selected 0, reflection 4),
# then re-extract 4-conn components; wall prefix = majority axis sprite; piece index ranks among wall 0-dots, wall index among other 0-dots.
# Unconfirmed: piece cycle order (largest piece first fits the one observation), h-axis reflections (always off-board here), ACTION6/7 no-ops.
import json

CELL, NB = 3, 21
COLOR = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
NEI = ((1, 0), (-1, 0), (0, 1), (0, -1))
_memo = {}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def obj_pixels(o):
    out = {}
    for i, row in enumerate(o.get('pixels') or []):
        for j, v in enumerate(row):
            if v >= 0:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def block_of(p):
    return (p[0] // CELL, p[1] // CELL)


def centre(b):
    return (CELL * b[0] + 1, CELL * b[1] + 1)


def components(points):
    points, comps = set(points), []
    while points:
        stack, comp = [points.pop()], set()
        while stack:
            p = stack.pop()
            comp.add(p)
            for dx, dy in NEI:
                q = (p[0] + dx, p[1] + dy)
                if q in points:
                    points.remove(q)
                    stack.append(q)
        comps.append(comp)
    return comps


# ---------------------------------------------------------------- model

def axis_blocks(m, kind):
    if kind == 'h':
        return set() if m['h'] is None else {(m['h'], k) for k in range(NB)}
    return set() if m['v'] is None else {(k, m['v']) for k in range(NB)}


def selection_order(m):
    order = [a for a in ('h', 'v') if m[a] is not None]
    return order + list(range(len(m['pieces'])))


def order_pieces(pieces):
    return sorted(pieces, key=lambda p: (-len(p), min(p)))


def parse(state):
    walls = {}
    for o in state:
        if o['type'] == 'wall':
            walls.update(obj_pixels(o))
    wblocks = {block_of(p) for p, v in walls.items() if v == COLOR['wall']}
    rows = {}
    cols = {}
    for bx, by in wblocks:
        rows[bx] = rows.get(bx, 0) + 1
        cols[by] = cols.get(by, 0) + 1
    h = max(rows, key=lambda r: rows[r]) if rows and max(rows.values()) > 2 else None
    v = max(cols, key=lambda c: cols[c]) if cols and max(cols.values()) > 2 else None
    zeros = {block_of(p) for p, val in walls.items() if val == 0}
    sel = None
    if h is not None and any(b[0] == h and b[1] != v for b in zeros):
        sel = 'h'
    elif v is not None and any(b[1] == v and b[0] != h for b in zeros):
        sel = 'v'
    targets = [frozenset(block_of(p) for p in obj_pixels(o)) for o in state if o['type'] == 'target']
    lower = set()
    for o in state:
        if o['type'] in ('wall', 'target', 'reflection'):
            lower |= set(obj_pixels(o))
    pix = {}
    for o in state:
        if o['type'] == 'player':
            pix.update(obj_pixels(o))
    blocks = {block_of(p) for p, val in pix.items() if val == COLOR['player']}
    zero = {b for b in blocks if pix.get(centre(b)) == 0}
    amb = {b for b in blocks if centre(b) not in pix and centre(b) in lower}
    selb, stack = set(), list(zero)
    while stack:
        b = stack.pop()
        if b in selb:
            continue
        selb.add(b)
        stack += [(b[0] + dx, b[1] + dy) for dx, dy in NEI if (b[0] + dx, b[1] + dy) in amb]
    pieces = [frozenset(c) for c in components(blocks - selb)]
    if selb:
        pieces.append(frozenset(selb))
    pieces = order_pieces(pieces)
    if sel is None:
        sel = pieces.index(frozenset(selb)) if selb else 0
    others = [dict(o) for o in state if o['type'] not in LAYER]
    tmpl = {}
    for o in state:
        if o['type'] in ('target', 'reflection', 'player'):
            tmpl.setdefault(o['type'], {'tags': o.get('tags'), 'layer': o.get('layer')})
    return {'h': h, 'v': v, 'sel': sel, 'pieces': pieces, 'targets': targets, 'others': others, 'tmpl': tmpl}


def on_board(blocks):
    return all(0 <= a < NB and 0 <= b < NB for a, b in blocks)


def step(m, action):
    a = action.get('action_id') if isinstance(action, dict) else action
    m = dict(m)
    if a == 5:
        order = selection_order(m)
        m['sel'] = order[(order.index(m['sel']) + 1) % len(order)] if m['sel'] in order else order[0]
        return m
    d = DIRS.get(a)
    if d is None:
        return m
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    if m['sel'] in ('h', 'v'):
        k = m['sel']
        delta = d[0] if k == 'h' else d[1]
        if delta == 0:
            return m
        na = m[k] + delta
        new = dict(m, **{k: na})
        return new if 0 <= na < NB and not axis_blocks(new, k) & occupied else m
    i = m['sel']
    moved = frozenset((bx + d[0], by + d[1]) for bx, by in m['pieces'][i])
    rest = set().union(*[p for j, p in enumerate(m['pieces']) if j != i]) if len(m['pieces']) > 1 else set()
    if on_board(moved) and not moved & rest and not moved & (axis_blocks(m, 'h') | axis_blocks(m, 'v')):
        pieces = list(m['pieces'])
        pieces[i] = moved
        m['pieces'] = pieces
    return m


# ---------------------------------------------------------------- render

def sprites(m):
    """Low -> high: (kind, sid, blocks, hole_fill); hole_fill 'solid' | value | None."""
    sp = [('wall', 'h', axis_blocks(m, 'h'), 0 if m['sel'] == 'h' else None),
          ('wall', 'v', axis_blocks(m, 'v'), 0 if m['sel'] == 'v' else None)]
    for i, t in enumerate(m['targets']):
        sp.append(('target', i, set(t), 'solid'))
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    refl = set()
    if m['v'] is not None:
        for p in m['pieces']:
            refl |= {(bx, 2 * m['v'] - by) for bx, by in p}
    sp.append(('reflection', 0, refl - occupied, COLOR['reflection']))
    for i, p in enumerate(m['pieces']):
        sp.append(('player', i, set(p), 0 if m['sel'] == i else None))
    return sp


def composite(sp):
    stacks = {}
    for s in reversed(sp):
        for b in s[2]:
            if 0 <= b[0] < NB and 0 <= b[1] < NB:
                stacks.setdefault(b, []).append(s)
    frame = {}
    for b, stack in stacks.items():
        for i in range(CELL):
            for j in range(CELL):
                p = (CELL * b[0] + i, CELL * b[1] + j)
                is_c = (i, j) == (1, 1)
                hole = None
                for kind, sid, _, fill in stack:
                    if not is_c or fill == 'solid':
                        frame[p] = (kind, sid, COLOR[kind])
                        break
                    if fill is not None and hole is None:
                        hole = (kind, sid, fill)
                else:
                    if hole is not None:
                        frame[p] = hole
    return frame


def make_obj(kind, pts, tags, layer):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for (x, y), v in pts.items():
        pix[x - x0][y - y0] = v
    return {'name': None, 'type': kind, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': layer, 'tags': list(tags), 'pixels': pix}


def rank_names(objs, prefix, dots):
    """Name objects prefix_k where k = rank by (x, y) among the objects plus the given 0-dot pixels."""
    keys = sorted([((o['x'], o['y']), 1, idx) for idx, o in enumerate(objs)] + [(d, 0, -1) for d in dots])
    for k, (_, is_obj, idx) in enumerate(keys):
        if is_obj:
            objs[idx]['name'] = prefix(objs[idx]) + str(k)


def render(m):
    frame = composite(sprites(m))
    by_kind, by_target = {}, {}
    for p, (kind, sid, v) in frame.items():
        if kind == 'target':
            by_target.setdefault(sid, {})[p] = v
        else:
            by_kind.setdefault(kind, {})[p] = (sid, v)
    tm = m['tmpl']
    out = []
    walls = []
    wp = by_kind.get('wall', {})
    for comp in components(wp):
        nh = sum(1 for p in comp if wp[p][0] == 'h')
        horiz = nh > len(comp) - nh
        o = make_obj('wall', {p: wp[p][1] for p in comp}, ['axis', 'horizontal' if horiz else 'vertical'], LAYER['wall'])
        o['_pre'] = 'wall_h_' if horiz else 'wall_v_'
        walls.append(o)
    wall_dots = [p for p, (sid, v) in wp.items() if v == 0]
    other_dots = [p for p, (kind, sid, v) in frame.items() if v == 0 and kind != 'wall']
    rank_names(walls, lambda o: o.pop('_pre'), other_dots)
    out += walls
    for kind, prefix, dots in (('player', 'piece_', wall_dots), ('reflection', 'reflection_', [])):
        pts = by_kind.get(kind, {})
        t = tm.get(kind, {})
        tags = t.get('tags') or (['movable', 'black'] if kind == 'player' else ['mirror', 'gray'])
        objs = [make_obj(kind, {p: pts[p][1] for p in c}, tags, t.get('layer') or LAYER[kind]) for c in components(pts)]
        rank_names(objs, lambda o, pr=prefix: pr, dots)
        out += objs
    t = tm.get('target', {})
    tobjs = [make_obj('target', pts, t.get('tags') or ['goal', 'yellow'], t.get('layer') or LAYER['target'])
             for _, pts in sorted(by_target.items())]
    rank_names(tobjs, lambda o: 'target_', [])
    out += tobjs
    return out + [dict(o) for o in m['others']]


def transition_function(state, action):
    global _memo
    model = _memo['model'] if _memo.get('out') == canon(state) else parse(state)
    new = step(model, action)
    out = render(new)
    _memo = {'out': canon(out), 'model': new}
    return out
