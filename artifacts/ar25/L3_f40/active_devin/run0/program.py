# Mechanics: mirror puzzle on a 21x21 grid of 3x3 cells (63x63 px). One vertical axis (type wall, 3px wide,
# full height, dotted centre column) and player pieces (cells with centre holes); reflection = pieces mirrored
# across the axis (holes filled grey unless a target lies beneath). ACTION5 cycles selection axis->pieces;
# ACTION1-4 move the selected object one cell (axis: left/right only). Frame is re-rendered by layer and
# re-extracted; wall names offset by #black(0) pixels elsewhere. Unconfirmed: blocking rules, ACTION6/7 (no-op).
import json

CELL, N = 3, 21
SIZE = CELL * N
LAYER = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
PREFIX = {'wall': 'wall_h', 'target': 'target', 'reflection': 'reflection', 'player': 'piece'}
DEFAULT_TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
                'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
COLOR = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
_memory = {'canon': None, 'world': None}


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def owner_grid(state):
    g = {}
    for o in state:
        for i, col in enumerate(o.get('pixels', [])):
            for j, v in enumerate(col):
                if v >= 0:
                    g[(o['x'] + i, o['y'] + j)] = (o['type'], v, o['name'])
    return g


def cell_pixels(cx, cy):
    return [(cx * CELL + i, cy * CELL + j) for i in range(CELL) for j in range(CELL)]


def is_center(X, Y):
    return X % CELL == 1 and Y % CELL == 1


def cell_groups(cells):
    cells, groups = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            c = stack.pop()
            comp.add(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        groups.append(comp)
    return groups


def decode(state):
    """Recover hidden sprites from the extracted objects of a frame."""
    g = owner_grid(state)
    walls = [o for o in state if o['type'] == 'wall']
    axis = walls[0]['x'] // CELL if walls else N // 2
    axis_sel = any(v == 0 for o in walls for c in o['pixels'] for v in c)
    piece_cells, sel_cells, known_unsel = set(), set(), set()
    target_cells = {}
    for cx in range(N):
        for cy in range(N):
            pts = cell_pixels(cx, cy)
            ring = [p for p in pts if not is_center(*p)]
            centre = (cx * CELL + 1, cy * CELL + 1)
            cv = g.get(centre)
            if any(g.get(p, (None,))[0] == 'player' for p in ring):
                piece_cells.add((cx, cy))
                if cv and cv[0] == 'player' and cv[1] == 0:
                    sel_cells.add((cx, cy))
                elif cv is None:
                    known_unsel.add((cx, cy))
            if cv and cv[0] == 'target':
                target_cells.setdefault(cv[2], set()).add((cx, cy))
    pieces = []
    if sel_cells:
        ambiguous = piece_cells - sel_cells - known_unsel
        for grp in cell_groups(sel_cells | ambiguous):
            if grp & sel_cells:
                pieces.append({'cells': grp, 'sel': True})
                piece_cells -= grp
    for grp in cell_groups(piece_cells):
        pieces.append({'cells': grp, 'sel': False})
    tags = {o['type']: o['tags'] for o in state if 'tags' in o}
    others = [o for o in state if o['type'] not in LAYER]
    return {'axis': axis, 'axis_sel': axis_sel, 'pieces': pieces,
            'targets': list(target_cells.values()), 'tags': tags, 'others': others}


def piece_order(world):
    return sorted(world['pieces'], key=lambda p: min(p['cells']))


def blocked_piece(world, piece, cells):
    if any(not (0 <= x < N and 0 <= y < N) or x == world['axis'] for x, y in cells):
        return True
    return any(cells & q['cells'] for q in world['pieces'] if q is not piece)


def step(world, action):
    aid = action.get('action_id') if isinstance(action, dict) else action
    if aid == 5:
        order = piece_order(world)
        cur = -1 if world['axis_sel'] else next((i for i, p in enumerate(order) if p['sel']), -1)
        for p in order:
            p['sel'] = False
        world['axis_sel'] = False
        nxt = cur + 1
        if nxt >= len(order):
            world['axis_sel'] = True
        else:
            order[nxt]['sel'] = True
    elif aid in MOVES:
        dx, dy = MOVES[aid]
        if world['axis_sel']:
            if dy == 0:
                na = world['axis'] + dx
                if 0 <= na < N and not any(x == na for p in world['pieces'] for x, _ in p['cells']):
                    world['axis'] = na
        else:
            for p in world['pieces']:
                if p['sel']:
                    cells = {(x + dx, y + dy) for x, y in p['cells']}
                    if not blocked_piece(world, p, cells):
                        p['cells'] = cells
    return world


def sprites(world):
    """Yield (type, sprite_id, solid_cells, hole_fill) in render order (top first)."""
    out = []
    for k, p in enumerate(piece_order(world)):
        out.append(('player', 'p%d' % k, p['cells'], 0 if p['sel'] else None))
    refl = set()
    for p in world['pieces']:
        for x, y in p['cells']:
            mx = 2 * world['axis'] - x
            if 0 <= mx < N:
                refl.add((mx, y))
    out.append(('reflection', 'r', refl, 4))
    for k, t in enumerate(world['targets']):
        out.append(('target', 't%d' % k, t, 'solid'))
    out.append(('wall', 'w', {(world['axis'], y) for y in range(N)}, 0 if world['axis_sel'] else None))
    return out


def render(world):
    layers = sprites(world)
    lookup = [(t, sid, cells, fill) for t, sid, cells, fill in layers]
    g = {}
    for X in range(SIZE):
        for Y in range(SIZE):
            c = (X // CELL, Y // CELL)
            hole_fill = None
            for t, sid, cells, fill in lookup:
                if c not in cells:
                    continue
                if not is_center(X, Y) or fill == 'solid':
                    g[(X, Y)] = (t, COLOR[t], sid)
                    break
                if fill is not None and hole_fill is None:
                    hole_fill = (t, fill, sid)
            else:
                if hole_fill is not None:
                    g[(X, Y)] = hole_fill
    return g


def components(points):
    points, comps = set(points), []
    while points:
        stack, comp = [points.pop()], []
        while stack:
            p = stack.pop()
            comp.append(p)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (p[0] + d[0], p[1] + d[1])
                if n in points:
                    points.remove(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def make_obj(t, pts, g, tags):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0, w, h = min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    pix = [[-1] * h for _ in range(w)]
    for X, Y in pts:
        pix[X - x0][Y - y0] = g[(X, Y)][1]
    return {'type': t, 'tags': list(tags.get(t, DEFAULT_TAGS[t])), 'layer': LAYER[t],
            'x': x0, 'y': y0, 'w': w, 'h': h, 'pixels': pix}


def extract(world, g):
    by_type = {}
    for p, (t, v, sid) in g.items():
        key = sid if t == 'target' else t
        by_type.setdefault((t, key), []).append(p)
    objs = {t: [] for t in LAYER}
    for (t, key), pts in by_type.items():
        groups = [pts] if t == 'target' else components(pts)
        for comp in groups:
            objs[t].append(make_obj(t, comp, g, world['tags']))
    black = sum(1 for (t, v, s) in g.values() if v == 0 and t != 'wall')
    result = []
    for t, lst in objs.items():
        lst.sort(key=lambda o: (o['x'], o['y']))
        base = black if t == 'wall' else 0
        for k, o in enumerate(lst):
            o['name'] = '%s_%d' % (PREFIX[t], base + k)
            result.append(o)
    return result + [dict(o) for o in world['others']]


def copy_world(w):
    return {'axis': w['axis'], 'axis_sel': w['axis_sel'],
            'pieces': [{'cells': set(p['cells']), 'sel': p['sel']} for p in w['pieces']],
            'targets': [set(t) for t in w['targets']], 'tags': w['tags'], 'others': w['others']}


def transition_function(state, action):
    c = canon(state)
    if _memory['canon'] == c and _memory['world'] is not None:
        world = copy_world(_memory['world'])
    else:
        world = decode(state)
    world['others'] = [o for o in state if o['type'] not in LAYER]
    world = step(world, action)
    out = extract(world, render(world))
    _memory['canon'] = canon(out)
    _memory['world'] = copy_world(world)
    return out
