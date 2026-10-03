# Mechanics: mirror-axis puzzle on a 3x3-cell grid. Sprites: wall axis (full-width row of holed cells), holed pieces,
# solid targets, gray reflections of every piece across the axis (row R -> 2a-R, clipped to the 63-row board).
# ACTION5 cycles selection axis -> pieces (sorted by top-left) -> axis; ACTION1/2 move the selection up/down 3, ACTION3/4
# left/right 3 (axis only vertical). Holes are transparent over lower layers, else filled (selected 0, reflection 4).
# Render layers wall1<target2<reflection3<piece4, re-extract (same-type 4-connected merge; wall_h index offset by #0 dots
# in pieces). Unconfirmed: blocking rules (bounds/overlap/axis crossing), cycle order past the first piece, ACTION6/7 no-op.
import json

TAGS = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'],
        'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
COLOR = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
PREFIX = {'wall': 'wall_h_', 'target': 'target_', 'reflection': 'reflection_', 'player': 'piece_'}
_mem = {'canon': None, 'model': None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cell_of(px, py):
    return (px // 3 * 3, py // 3 * 3)


def obj_pixels(o):
    for i, row in enumerate(o.get('pixels', [])):
        for j, v in enumerate(row):
            if v >= 0:
                yield o['x'] + i, o['y'] + j, v


def cell_components(cells):
    cells, comps = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            c = stack.pop()
            comp.add(c)
            for d in ((3, 0), (-3, 0), (0, 3), (0, -3)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def parse(state):
    others = [o for o in state if o['type'] not in LAYER]
    size = next((o['x'] for o in others if o['type'] == 'counter'), 63)
    walls = [o for o in state if o['type'] == 'wall']
    axis = min(o['x'] for o in walls)
    axis_sel = any(v == 0 for o in walls for _, _, v in obj_pixels(o))
    targets = []
    for o in state:
        if o['type'] == 'target':
            targets.append({cell_of(px, py) for px, py, v in obj_pixels(o)})
    tcells = set().union(*targets) if targets else set()
    cells, centre = set(), {}
    for o in state:
        if o['type'] != 'player':
            continue
        for px, py, v in obj_pixels(o):
            c = cell_of(px, py)
            if (px - c[0], py - c[1]) == (1, 1):
                centre[c] = v
            else:
                cells.add(c)
    sel = {c for c in cells if centre.get(c) == 0}
    amb = {c for c in cells if c not in sel and c in tcells}
    if sel:
        grown = True
        while grown:
            grown = False
            for c in list(amb):
                if any((c[0] + dx, c[1] + dy) in sel for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3))):
                    sel.add(c)
                    amb.discard(c)
                    grown = True
    elif not axis_sel:
        for comp in cell_components(cells):
            if comp <= tcells:
                sel = comp
                break
    pieces = cell_components(cells - sel) + ([sel] if sel else [])
    pieces.sort(key=lambda p: min(p))
    selected = -1 if axis_sel or not sel else pieces.index(sel)
    return {'size': size, 'axis': axis, 'pieces': pieces, 'sel': selected,
            'targets': targets, 'others': others}


def step(m, action):
    m = dict(m, pieces=[set(p) for p in m['pieces']])
    size, pieces = m['size'], m['pieces']
    if action == 5:
        m['sel'] = m['sel'] + 1 if m['sel'] + 1 < len(pieces) else -1
        return m
    d = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}.get(action)
    if d is None:
        return m
    if m['sel'] < 0:
        if d[1] != 0:
            return m
        a = m['axis'] + d[0]
        blocked = a < 0 or a + 2 >= size or any(c[0] == a for p in pieces for c in p)
        if not blocked:
            m['axis'] = a
        return m
    k = m['sel']
    moved = {(c[0] + d[0], c[1] + d[1]) for c in pieces[k]}
    rest = set().union(*(p for i, p in enumerate(pieces) if i != k))
    ok = all(0 <= r and r + 2 < size and 0 <= c and c + 2 < size and r != m['axis'] for r, c in moved)
    if ok and not (moved & rest):
        pieces[k] = moved
    return m


def sprites(m):
    a, out = m['axis'], []
    out.append({'type': 'wall', 'cells': {(a, c) for c in range(0, m['size'] - 2, 3)},
                'fill': 0 if m['sel'] < 0 else -1, 'solid_centre': False, 'id': 0})
    for i, t in enumerate(m['targets']):
        out.append({'type': 'target', 'cells': t, 'fill': -1, 'solid_centre': True, 'id': i})
    for i, p in enumerate(m['pieces']):
        out.append({'type': 'reflection', 'cells': {(2 * a - r, c) for r, c in p},
                    'fill': 4, 'solid_centre': False, 'id': i})
        out.append({'type': 'player', 'cells': p, 'fill': 0 if m['sel'] == i else -1,
                    'solid_centre': False, 'id': i})
    out.sort(key=lambda s: -LAYER[s['type']])
    return out


def render(m):
    size, owner = m['size'], {}
    sps = sprites(m)
    for px in range(size):
        for py in range(size):
            c = cell_of(px, py)
            is_centre = (px - c[0], py - c[1]) == (1, 1)
            hole = None
            for s in sps:
                if c not in s['cells']:
                    continue
                if s['solid_centre'] or not is_centre:
                    owner[(px, py)] = (s, COLOR[s['type']])
                    hole = None
                    break
                if hole is None:
                    hole = s
            if hole is not None and hole['fill'] >= 0:
                owner[(px, py)] = (hole, hole['fill'])
    return owner


def make_obj(typ, name, pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    pix = [[-1] * h for _ in range(w)]
    for (px, py), v in pts.items():
        pix[px - x0][py - y0] = v
    return {'name': name, 'type': typ, 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': LAYER[typ], 'tags': list(TAGS[typ]), 'pixels': pix}


def pixel_components(pts):
    left, comps = set(pts), []
    while left:
        stack, comp = [left.pop()], {}
        while stack:
            p = stack.pop()
            comp[p] = pts[p]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (p[0] + dx, p[1] + dy)
                if n in left:
                    left.remove(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def extract(m):
    owner = render(m)
    bytype = {t: {} for t in LAYER}
    target_pts = {}
    for p, (s, v) in owner.items():
        bytype[s['type']][p] = v
        if s['type'] == 'target':
            target_pts.setdefault(s['id'], {})[p] = v
    dots = sum(1 for p, v in bytype['player'].items() if v == 0)
    result = [dict(o) for o in m['others']]
    for typ in ('wall', 'player', 'reflection', 'target'):
        groups = list(target_pts.values()) if typ == 'target' else pixel_components(bytype[typ])
        objs = [make_obj(typ, '', g) for g in groups]
        objs.sort(key=lambda o: (o['x'], o['y']))
        base = dots if typ == 'wall' else 0
        for i, o in enumerate(objs):
            o['name'] = PREFIX[typ] + str(base + i)
        result.extend(objs)
    return result


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get('action_id')
    model = _mem['model'] if _mem['canon'] == canon(state) else None
    if model is None:
        model = parse(state)
    model = step(model, action)
    out = extract(model)
    _mem['canon'], _mem['model'] = canon(out), model
    return out
