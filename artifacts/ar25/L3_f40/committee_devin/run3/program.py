# Mechanics: a mirror axis (wall, 3px thick, full width) and 3x3-cell pieces; action 5 cycles selection
# axis -> piece_0 -> piece_1 -> ...; actions 1/2/3/4 move the selection one cell up/down/left/right
# (axis only along its normal). Every piece cell is mirrored across the axis as a gray reflection cell.
# Render layers wall<target<reflection<piece, cell centres show what lies beneath (else 0/4), re-extract
# 4-connected components; wall names offset by #black piece pixels. Unconfirmed: blocking rules, cycle past last piece.
import json

N = 63
C = 3
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
TAGS = {'wall': {'horizontal': ['axis', 'horizontal'], 'vertical': ['axis', 'vertical']},
        'player': ['movable', 'black'], 'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
_memo = {'out': None, 'world': None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


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
            r, c = stack.pop()
            comp.add((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def parse(state):
    walls = [o for o in state if o['type'] == 'wall']
    orient = 'vertical' if walls and 'vertical' in walls[0].get('tags', []) else 'horizontal'
    if walls:
        axis = min(o['x'] for o in walls) if orient == 'horizontal' else min(o['y'] for o in walls)
    else:
        axis = None
    axis_sel = any(v == 0 for o in walls for _, _, v in obj_pixels(o))
    targets = []
    for o in sorted((o for o in state if o['type'] == 'target'), key=lambda o: (o['x'], o['y'])):
        targets.append({(r // C, c // C) for r, c, v in obj_pixels(o) if v == 11})
    pcells, centre = set(), {}
    for o in state:
        if o['type'] != 'player':
            continue
        for r, c, v in obj_pixels(o):
            cell = (r // C, c // C)
            if v == 5:
                pcells.add(cell)
            if r % C == 1 and c % C == 1:
                centre[cell] = v
    tcells = set().union(*targets) if targets else set()
    sprites, sel = [], -1
    for comp in cell_components(pcells):
        black = {k for k in comp if centre.get(k) == 0}
        clear = {k for k in comp if k not in centre and k not in tcells}
        if black and clear and not axis_sel:
            sprites.append(comp - clear)
            sprites.append(clear)
        else:
            sprites.append(comp)
    sprites.sort(key=lambda s: min(s))
    if not axis_sel:
        for i, s in enumerate(sprites):
            if any(centre.get(k) == 0 for k in s):
                sel = i
                break
        else:
            for i, s in enumerate(sprites):
                if all(k not in centre and k in tcells for k in s):
                    sel = i
                    break
    keep = [o for o in state if o['type'] not in ('wall', 'target', 'player', 'reflection')]
    return {'orient': orient, 'axis': axis, 'sel': sel, 'targets': targets,
            'sprites': [set(s) for s in sprites], 'keep': keep}


def axis_cells(w):
    if w['axis'] is None:
        return set()
    a = w['axis'] // C
    n = N // C
    return {(a, k) for k in range(n)} if w['orient'] == 'horizontal' else {(k, a) for k in range(n)}


def inside(cells):
    return all(0 <= r < N // C and 0 <= c < N // C for r, c in cells)


def step(w, action):
    if action == 5:
        w['sel'] = w['sel'] + 1 if w['sel'] + 1 < len(w['sprites']) else -1
        return
    if action not in DIRS:
        return
    dr, dc = DIRS[action]
    occupied = set().union(*w['sprites']) if w['sprites'] else set()
    if w['sel'] < 0:
        if w['axis'] is None:
            return
        d = dr if w['orient'] == 'horizontal' else dc
        if d == 0:
            return
        nw = dict(w, axis=w['axis'] + d * C)
        cells = axis_cells(nw)
        if inside(cells) and not (cells & occupied):
            w['axis'] = nw['axis']
        return
    me = w['sprites'][w['sel']]
    moved = {(r + dr, c + dc) for r, c in me}
    if inside(moved) and not (moved & (occupied - me)) and not (moved & axis_cells(w)):
        w['sprites'][w['sel']] = moved


def reflect(w, cell):
    a = w['axis'] // C
    r, c = cell
    return (2 * a - r, c) if w['orient'] == 'horizontal' else (r, 2 * a - c)


def render(w):
    grid = {}

    def put(r, c, owner, v):
        if 0 <= r < N and 0 <= c < N:
            grid[(r, c)] = (owner, v)

    if w['axis'] is not None:
        for k in range(N):
            for t in range(C):
                r, c = (w['axis'] + t, k) if w['orient'] == 'horizontal' else (k, w['axis'] + t)
                dot = t == 1 and k % C == 1
                if dot and w['sel'] >= 0:
                    continue
                put(r, c, ('wall',), 0 if dot else 10)
    for i, cells in enumerate(w['targets']):
        for R, Cc in cells:
            for i2 in range(C):
                for j2 in range(C):
                    put(R * C + i2, Cc * C + j2, ('target', i), 11)
    layers = []
    if w['axis'] is not None:
        layers.append(('reflection', [reflect(w, k) for s in w['sprites'] for k in s], 4, 4))
    for i, s in enumerate(w['sprites']):
        layers.append(('player', list(s), 5, 0 if i == w['sel'] else None))
    for kind, cells, col, centre in layers:
        for R, Cc in cells:
            for i2 in range(C):
                for j2 in range(C):
                    r, c = R * C + i2, Cc * C + j2
                    if i2 == 1 and j2 == 1:
                        if centre is not None and (r, c) not in grid:
                            put(r, c, (kind,), centre)
                    else:
                        put(r, c, (kind,), col)
    return grid


def make_obj(name, typ, tags, pix):
    xs = [p[0] for p in pix]
    ys = [p[1] for p in pix]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    px = [[-1] * h for _ in range(w)]
    for (r, c), v in pix.items():
        px[r - x0][c - y0] = v
    return {'name': name, 'type': typ, 'tags': list(tags), 'x': x0, 'y': y0, 'w': w, 'h': h,
            'layer': LAYER[typ], 'pixels': px}


def pixel_components(pix):
    left, comps = dict(pix), []
    while left:
        start = next(iter(left))
        stack, comp = [start], {start: left.pop(start)}
        while stack:
            r, c = stack.pop()
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in left:
                    comp[n] = left.pop(n)
                    stack.append(n)
        comps.append(comp)
    return sorted(comps, key=lambda p: (min(k[0] for k in p), min(k[1] for k in p)))


def extract(w, grid):
    out = [dict(o) for o in w['keep']]
    by = {}
    for pos, (owner, v) in grid.items():
        key = owner if owner[0] == 'target' else owner[0]
        by.setdefault(key, {})[pos] = v
    tgt = [make_obj(None, 'target', TAGS['target'], by[('target', i)])
           for i in range(len(w['targets'])) if ('target', i) in by]
    tgt.sort(key=lambda o: (o['x'], o['y']))
    for i, o in enumerate(tgt):
        o['name'] = 'target_%d' % i
    out += tgt
    for typ, prefix in (('player', 'piece'), ('reflection', 'reflection')):
        for i, comp in enumerate(pixel_components(by.get(typ, {}))):
            out.append(make_obj('%s_%d' % (prefix, i), typ, TAGS[typ], comp))
    black = sum(1 for v in by.get('player', {}).values() if v == 0)
    tag = 'h' if w['orient'] == 'horizontal' else 'v'
    for i, comp in enumerate(pixel_components(by.get('wall', {}))):
        out.append(make_obj('wall_%s_%d' % (tag, i + black), 'wall', TAGS['wall'][w['orient']], comp))
    return out


def copy_world(w):
    return {'orient': w['orient'], 'axis': w['axis'], 'sel': w['sel'],
            'targets': [set(t) for t in w['targets']], 'sprites': [set(s) for s in w['sprites']],
            'keep': [dict(o) for o in w['keep']]}


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get('action_id')
    if _memo['out'] is not None and canon(state) == _memo['out']:
        world = copy_world(_memo['world'])
        world['keep'] = [o for o in state if o['type'] not in ('wall', 'target', 'player', 'reflection')]
    else:
        world = parse(state)
    step(world, action)
    out = extract(world, render(world))
    _memo['out'] = canon(out)
    _memo['world'] = copy_world(world)
    return out
