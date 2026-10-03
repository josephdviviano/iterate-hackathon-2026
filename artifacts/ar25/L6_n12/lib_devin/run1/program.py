# Mechanics: 21x21 board of 3px blocks (x = row); an h-axis row and a v-axis column (layer 1), targets (2),
# gray v-axis reflections of pieces (3, by -> 2v-by, clipped, not under pieces) and holed pieces (4).
# ACTION1-4 move the selection one block (axis only along its normal; blocked by edges/pieces/axes);
# ACTION5 cycles h-axis -> v-axis -> pieces (largest first, then (x,y)) -> h; selected holes show 0.
# Unconfirmed: piece cycle order beyond the first pick, h-axis reflections (all off-board here), clicks/undo.
import json

CELL, N = 3, 21
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
COLOR = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
TAGS = {'wall': ['axis'], 'target': ['goal', 'yellow'], 'reflection': ['mirror', 'gray'],
        'player': ['movable', 'black']}
PREFIX = {'target': 'target_', 'reflection': 'reflection_', 'player': 'piece_'}
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))
MEMO = {}


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
            for dx, dy in NB:
                q = (p[0] + dx, p[1] + dy)
                if q in points:
                    points.remove(q)
                    stack.append(q)
        comps.append(comp)
    return comps


# ---------------------------------------------------------------- parsing (stateless recovery)

def find_axes(wall_blocks):
    rows, cols = {}, {}
    for bx, by in wall_blocks:
        rows[bx] = rows.get(bx, 0) + 1
        cols[by] = cols.get(by, 0) + 1
    h = max(rows, key=lambda r: (rows[r], -r)) if rows and max(rows.values()) >= 2 else None
    v = max(cols, key=lambda c: (cols[c], -c)) if cols and max(cols.values()) >= 2 else None
    if h is None and v is None and wall_blocks:
        h = min(wall_blocks)[0]
    return h, v


def piece_order(pieces):
    return sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), min(pieces[i])))


def parse(state):
    wall_pix, play_pix, lower = {}, {}, set()
    for o in state:
        px = obj_pixels(o)
        if o['type'] == 'wall':
            wall_pix.update(px)
        elif o['type'] == 'player':
            play_pix.update(px)
        if o['type'] in ('wall', 'target', 'reflection'):
            lower |= set(px)
    h, v = find_axes({block_of(p) for p, c in wall_pix.items() if c == COLOR['wall']})
    zb = {block_of(p) for p, c in wall_pix.items() if c == 0}
    sel = None
    if any(b[0] == h and b[1] != v for b in zb):
        sel = 'h'
    elif any(b[1] == v and b[0] != h for b in zb):
        sel = 'v'
    elif zb:
        sel = 'h' if h is not None else 'v'
    targets = [frozenset(block_of(p) for p in obj_pixels(o)) for o in state if o['type'] == 'target']
    blocks = {block_of(p) for p, c in play_pix.items() if c == COLOR['player']}
    zero = {b for b in blocks if play_pix.get(centre(b)) == 0}
    amb = {b for b in blocks if centre(b) not in play_pix and centre(b) in lower}
    selb, stack = set(), list(zero)
    while stack:
        b = stack.pop()
        if b not in selb:
            selb.add(b)
            stack += [(b[0] + dx, b[1] + dy) for dx, dy in NB if (b[0] + dx, b[1] + dy) in amb]
    pieces = [frozenset(c) for c in components(blocks - selb)]
    if selb:
        pieces.append(frozenset(selb))
    pieces.sort(key=min)
    if sel is None:
        if selb:
            sel = pieces.index(frozenset(selb))
        elif pieces:
            sel = piece_order(pieces)[0]
        else:
            sel = 'h' if h is not None else 'v'
    templates = {}
    for o in state:
        templates.setdefault(o['type'], {'layer': o.get('layer'), 'tags': o.get('tags')})
    others = [dict(o) for o in state if o['type'] not in LAYER]
    return {'h': h, 'v': v, 'sel': sel, 'pieces': pieces, 'targets': targets,
            'order': piece_order(pieces), 'others': others, 'templates': templates}


# ---------------------------------------------------------------- dynamics

def axis_line(kind, k):
    return {(k, j) for j in range(N)} if kind == 'h' else {(j, k) for j in range(N)}


def cycle(m):
    seq = [a for a in ('h', 'v') if m[a] is not None] + list(m['order'])
    if not seq:
        return m['sel']
    i = seq.index(m['sel']) if m['sel'] in seq else -1
    return seq[(i + 1) % len(seq)]


def move_axis(m, kind, d):
    step = d[0] if kind == 'h' else d[1]
    if step == 0:
        return m[kind]
    na = m[kind] + step
    occupied = set().union(*m['pieces']) if m['pieces'] else set()
    if not 0 <= na < N or axis_line(kind, na) & occupied:
        return m[kind]
    return na


def move_piece(m, k, d):
    moved = frozenset((bx + d[0], by + d[1]) for bx, by in m['pieces'][k])
    rest = set().union(set(), *[p for i, p in enumerate(m['pieces']) if i != k])
    for a in ('h', 'v'):
        if m[a] is not None:
            rest |= axis_line(a, m[a])
    if any(not (0 <= bx < N and 0 <= by < N) for bx, by in moved) or moved & rest:
        return m['pieces']
    out = list(m['pieces'])
    out[k] = moved
    return out


def step(m, action):
    a = action.get('action_id') if isinstance(action, dict) else action
    if a == 5:
        return dict(m, sel=cycle(m))
    d = DIRS.get(a)
    if d is None:
        return m
    if m['sel'] in ('h', 'v'):
        return dict(m, **{m['sel']: move_axis(m, m['sel'], d)})
    return dict(m, pieces=move_piece(m, m['sel'], d))


# ---------------------------------------------------------------- rendering

def sprites(m):
    """Low -> high: (kind, id, blocks, hole fill: 'solid' | value | None)."""
    sp = []
    for a in ('h', 'v'):
        if m[a] is not None:
            sp.append(('wall', a, axis_line(a, m[a]), 0 if m['sel'] == a else None))
    for i, t in enumerate(m['targets']):
        sp.append(('target', i, set(t), 'solid'))
    occupied = set().union(set(), *m['pieces'])
    if m['v'] is not None:
        refl = {(bx, 2 * m['v'] - by) for p in m['pieces'] for bx, by in p} - occupied
        sp.append(('reflection', 0, refl, COLOR['reflection']))
    for i, p in enumerate(m['pieces']):
        sp.append(('player', i, set(p), 0 if m['sel'] == i else None))
    return sp


def composite(sp):
    by_block = {}
    for s in reversed(sp):
        for b in s[2]:
            if 0 <= b[0] < N and 0 <= b[1] < N:
                by_block.setdefault(b, []).append(s)
    frame = {}
    for b, stack in by_block.items():
        for i in range(CELL):
            for j in range(CELL):
                p = (CELL * b[0] + i, CELL * b[1] + j)
                hole = None
                for kind, sid, _, fill in stack:
                    if (i, j) != (1, 1) or fill == 'solid':
                        frame[p] = (kind, sid, COLOR[kind])
                        break
                    if fill is not None and hole is None:
                        hole = (kind, sid, fill)
                else:
                    if hole is not None:
                        frame[p] = hole
    return frame


def make_object(kind, pts, templates, tags=None):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for (x, y), v in pts.items():
        pix[x - x0][y - y0] = v
    t = templates.get(kind, {})
    return {'name': None, 'type': kind, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': t.get('layer') or LAYER[kind],
            'tags': tags if tags is not None else list(t.get('tags') or TAGS[kind]), 'pixels': pix}


def wall_tag(comp, frame):
    nh = sum(1 for p in comp if frame[p][1] == 'h')
    return ['axis', 'horizontal' if nh * 2 >= len(comp) else 'vertical']


def extract(frame, templates):
    by_kind, by_target = {}, {}
    for p, (kind, sid, v) in frame.items():
        if kind == 'target':
            by_target.setdefault(sid, {})[p] = v
        else:
            by_kind.setdefault(kind, {})[p] = v
    objs = {'target': [make_object('target', c, templates) for c in by_target.values()]}
    for kind in ('wall', 'reflection', 'player'):
        pix = by_kind.get(kind, {})
        objs[kind] = []
        for comp in components(pix):
            tags = wall_tag(comp, frame) if kind == 'wall' else None
            objs[kind].append(make_object(kind, {p: pix[p] for p in comp}, templates, tags))
    # walls and pieces are ranked together with the 0 (selection) pixels of the other type
    zeros = {k: [p for p, (kd, _, v) in frame.items() if kd == k and v == 0] for k in ('wall', 'player')}
    out = []
    for kind, lst in objs.items():
        other = {'wall': 'player', 'player': 'wall'}.get(kind)
        keys = [((o['x'], o['y']), 1, o) for o in lst] + [(p, 0, None) for p in zeros.get(other, [])]
        rank = 0
        for _, _, o in sorted(keys, key=lambda k: (k[0], k[1])):
            if o is not None:
                pre = PREFIX.get(kind) or ('wall_h_' if 'horizontal' in o['tags'] else 'wall_v_')
                o['name'] = pre + str(rank)
            rank += 1
        out += lst
    return out


def render(m):
    return extract(composite(sprites(m)), m['templates']) + [dict(o) for o in m['others']]


def transition_function(state, action):
    global MEMO
    m = MEMO['model'] if MEMO and MEMO['out'] == canon(state) else parse(state)
    m2 = step(m, action)
    out = render(m2)
    MEMO = {'out': canon(out), 'model': m2}
    return out
