# Mechanics (no hidden counter needed; step 71 = axis under pieces): one horizontal mirror axis (wall, block row A) + holed 3x3-block pieces (player) + gray reflections
# + solid yellow targets, rendered on a 21x21 block grid (x = row, y = col, pixels row-major in x). ACTION5 cycles the
# selection axis -> pieces (largest first, tie min block) -> axis; A1/A2/A3/A4 move the selection by one block
# (axis moves only vertically and passes under pieces; pieces are blocked by bounds and other pieces). Every off-axis
# piece block b (above OR below A) reflects to 2A-b unless a piece holds it; layers composited, frame re-extracted.
G = 21
DEFAULT_TAGS = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'],
                'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
PREFIX = {'wall': 'wall_h', 'player': 'piece', 'reflection': 'reflection'}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
COLOR = {10: 'wall', 5: 'player', 4: 'reflection', 11: 'target'}
_memory = {'out': None, 'model': None}


def canon(state):
    import json
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cells(o):
    p = o.get('pixels') or []
    for i, row in enumerate(p):
        for j, v in enumerate(row):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def comps(points):
    points = set(points)
    out = []
    while points:
        s = points.pop()
        comp, stack = {s}, [s]
        while stack:
            r, c = stack.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in points:
                    points.discard(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


def parse(state):
    frame = {}
    tags = dict(DEFAULT_TAGS)
    model = {'axis': None, 'axis_sel': False, 'targets': [], 'pieces': [], 'sel': None, 'others': []}
    piece_blocks = set()
    for o in state:
        t = o.get('type')
        if t in DEFAULT_TAGS:
            tags[t] = o.get('tags', tags[t])
        if t not in LAYER:
            model['others'].append(o)
            continue
        tb = set()
        for r, c, v in cells(o):
            frame[(r, c)] = v
            if t == 'wall' and v == 10:
                model['axis'] = r // 3
            if t == 'wall' and v == 0:
                model['axis_sel'] = True
            if t == 'target' and v == 11:
                tb.add((r // 3, c // 3))
            if t == 'player' and v == 5:
                piece_blocks.add((r // 3, c // 3))
        if t == 'target' and tb:
            model['targets'].append(frozenset(tb))
    model['tags'] = tags
    groups = comps(piece_blocks)
    pieces = []
    sel = None
    if model['axis_sel']:
        pieces = groups
    else:
        def centre(b):
            return frame.get((3 * b[0] + 1, 3 * b[1] + 1), -1)
        for g in groups:
            zero = {b for b in g if centre(b) == 0}
            amb = {b for b in g if centre(b) not in (0, -1)}
            if not zero and amb == g and sel is None:
                zero = set(g)
            if zero and sel is None:
                s = set(zero)
                grow = True
                while grow:
                    grow = False
                    for b in amb - s:
                        if any(n in s for n in ((b[0] + 1, b[1]), (b[0] - 1, b[1]), (b[0], b[1] + 1), (b[0], b[1] - 1))):
                            s.add(b)
                            grow = True
                sel = frozenset(s)
                pieces.append(sel)
                pieces.extend(comps(g - s))
            else:
                pieces.append(g)
    model['pieces'] = [frozenset(p) for p in pieces]
    model['sel'] = sel
    return model


def order(pieces):
    return sorted(pieces, key=lambda p: (-len(p), min(p)))


def step(model, action):
    m = dict(model)
    m['pieces'] = list(model['pieces'])
    aid = action.get('action_id') if isinstance(action, dict) else action
    if aid == 5:
        ordered = order(m['pieces'])
        if m['axis_sel'] or m['sel'] is None:
            if ordered:
                m['axis_sel'], m['sel'] = False, ordered[0]
        else:
            i = ordered.index(m['sel'])
            if i + 1 < len(ordered):
                m['sel'] = ordered[i + 1]
            else:
                m['axis_sel'], m['sel'] = True, None
        return m
    delta = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(aid)
    if delta is None:
        return m
    dr, dc = delta
    if m['axis_sel']:
        if dr and 0 <= m['axis'] + dr < G:
            m['axis'] += dr
        return m
    if m['sel'] is None:
        return m
    moved = frozenset((r + dr, c + dc) for r, c in m['sel'])
    others = set().union(*[p for p in m['pieces'] if p != m['sel']]) if len(m['pieces']) > 1 else set()
    if all(0 <= r < G and 0 <= c < G for r, c in moved) and not (moved & others):
        m['pieces'] = [moved if p == m['sel'] else p for p in m['pieces']]
        m['sel'] = moved
    return m


def reflections(m):
    A = m['axis']
    occ = set().union(*m['pieces']) if m['pieces'] else set()
    out = set()
    for r, c in occ:
        if r == A:
            continue
        d = (2 * A - r, c)
        if 0 <= d[0] < G and d not in occ:
            out.add(d)
    return out


def render(m):
    # per block: stack of sprites top-down: (type, sprite id, ring colour, centre solid?, centre value/fill)
    stacks = {}

    def push(b, entry):
        stacks.setdefault(b, []).append(entry)
    for p in m['pieces']:
        fill = 0 if p == m['sel'] else -1
        for b in p:
            push(b, (4, 'player', None, 5, False, fill))
    for b in reflections(m):
        push(b, (3, 'reflection', None, 4, False, 4))
    for k, t in enumerate(m['targets']):
        for b in t:
            push(b, (2, 'target', k, 11, True, 11))
    if m['axis'] is not None:
        for c in range(G):
            push((m['axis'], c), (1, 'wall', None, 10, m['axis_sel'], 0 if m['axis_sel'] else -1))
    frame = {}
    for (br, bc), st in stacks.items():
        st.sort(key=lambda e: -e[0])
        top = st[0]
        for i in range(3):
            for j in range(3):
                if (i, j) != (1, 1):
                    frame[(3 * br + i, 3 * bc + j)] = (top[3], top[1], top[2])
        val = None
        for e in st:
            if e[4]:
                val = (e[5], e[1], e[2])
                break
        if val is None:
            for e in st:
                if e[5] >= 0:
                    val = (e[5], e[1], e[2])
                    break
        if val is not None:
            frame[(3 * br + 1, 3 * bc + 1)] = val
    return frame


def make_obj(name, typ, pts, frame, tags):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for r, c in pts:
        pix[r - x0][c - y0] = frame[(r, c)][0]
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': LAYER[typ], 'tags': list(tags[typ]), 'pixels': pix}


def extract(m, frame):
    out = [dict(o) for o in m['others']]
    zeros = {p for p, v in frame.items() if v[0] == 0}
    bbox = lambda s: (min(p[0] for p in s), min(p[1] for p in s))
    for typ in ('wall', 'player', 'reflection'):
        own = {p for p, v in frame.items() if v[0] >= 0 and COLOR.get(v[0]) == typ}
        pts = own | zeros if typ != 'reflection' else own
        cs = sorted(comps(pts), key=bbox)
        for i, cmp in enumerate(cs):
            if cmp & own:
                out.append(make_obj('%s_%d' % (PREFIX[typ], i), typ, cmp, frame, m['tags']))
    tg = {}
    for p, v in frame.items():
        if v[0] == 11:
            tg.setdefault(v[2], set()).add(p)
    for i, cmp in enumerate(sorted(tg.values(), key=bbox)):
        out.append(make_obj('target_%d' % i, 'target', cmp, frame, m['tags']))
    return out


def transition_function(state, action):
    model = None
    if _memory['out'] is not None and canon(state) == _memory['out']:
        model = _memory['model']
    if model is None:
        model = parse(state)
    new = step(model, action)
    result = extract(new, render(new))
    _memory['out'] = canon(result)
    _memory['model'] = new
    return result
