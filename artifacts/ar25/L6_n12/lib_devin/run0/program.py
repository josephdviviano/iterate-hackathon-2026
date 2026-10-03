# Actions 1-4 move the selected piece or the selected axis normal by one 3px block.
# Action 5 cycles horizontal axis, vertical axis, then pieces; empty centres show selection.
# Reflections mirror pieces across axes; wall/target/reflection/player layers are composited.
# Pixel components are re-extracted; black-dot components contribute to wall/player indices.
# Unconfirmed: later selection order (largest first), collisions, on-board double mirrors, 6/7.
import copy
import json

CELL = 3
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
NEIGHBOURS = tuple(DIRS.values())
COLORS = {'wall': 10, 'target': 11, 'reflection': 4, 'player': 5}
LAYERS = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
TAGS = {'wall': ['axis', 'horizontal'], 'target': ['goal', 'yellow'],
        'reflection': ['mirror', 'gray'], 'player': ['movable', 'black']}
_memo = None


def canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def pixels(obj):
    return {(obj['x'] + i, obj['y'] + j): v
            for i, row in enumerate(obj.get('pixels', []))
            for j, v in enumerate(row) if v >= 0}


def block(p):
    return p[0] // CELL, p[1] // CELL


def centre(b):
    return CELL * b[0] + 1, CELL * b[1] + 1


def components(points):
    remaining, result = set(points), []
    while remaining:
        stack, group = [remaining.pop()], set()
        while stack:
            p = stack.pop()
            group.add(p)
            for dx, dy in NEIGHBOURS:
                q = p[0] + dx, p[1] + dy
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(frozenset(group))
    return result


def axis_blocks(orient, pos, n):
    return {(pos, k) if orient == 'horizontal' else (k, pos) for k in range(n)}


def parse_model(state):
    n = next((o['x'] // CELL for o in state if o['type'] == 'counter'), 21)
    wall = {}
    piece_pix = {}
    lower = set()
    templates = {}
    for o in state:
        templates.setdefault(o['type'], copy.deepcopy(o))
        p = pixels(o)
        if o['type'] == 'wall':
            wall.update(p)
        if o['type'] == 'player':
            piece_pix.update(p)
        elif o['type'] in COLORS:
            lower.update(p)
    wb = {block(p) for p, v in wall.items() if v == 10}
    row_counts = [sum(b[0] == k for b in wb) for k in range(n)]
    col_counts = [sum(b[1] == k for b in wb) for k in range(n)]
    axes = []
    if max(row_counts, default=0) > 1:
        axes.append(('horizontal', row_counts.index(max(row_counts))))
    if max(col_counts, default=0) > 1:
        axes.append(('vertical', col_counts.index(max(col_counts))))
    zeros = {block(p) for p, v in wall.items() if v == 0}
    sel = None
    for i, (orient, pos) in enumerate(axes):
        coordinate = 0 if orient == 'horizontal' else 1
        if zeros and all(b[coordinate] == pos for b in zeros):
            sel = i
    blocks = {block(p) for p, v in piece_pix.items() if v == 5}
    chosen = {b for b in blocks if piece_pix.get(centre(b)) == 0}
    ambiguous = {b for b in blocks if centre(b) not in piece_pix and centre(b) in lower}
    stack = list(chosen)
    while stack:
        b = stack.pop()
        for dx, dy in NEIGHBOURS:
            q = b[0] + dx, b[1] + dy
            if q in ambiguous and q not in chosen:
                chosen.add(q)
                stack.append(q)
    pieces = components(blocks - chosen)
    if chosen:
        pieces.append(frozenset(chosen))
    pieces.sort(key=lambda p: (-len(p), min(p)))
    if chosen:
        sel = len(axes) + pieces.index(frozenset(chosen))
    if sel is None:
        sel = len(axes) if pieces else 0
    targets = [frozenset(block(p) for p in pixels(o)) for o in state if o['type'] == 'target']
    return {'n': n, 'axes': axes, 'pieces': pieces, 'sel': sel, 'targets': targets,
            'templates': templates,
            'others': [copy.deepcopy(o) for o in state if o['type'] not in COLORS]}


def inside_board(blocks, n):
    return all(0 <= x < n and 0 <= y < n for x, y in blocks)


def hits_axis(blocks, axes, n):
    return any(blocks & axis_blocks(orient, pos, n) for orient, pos in axes)


def hits_piece(blocks, pieces, exclude=None):
    return any(blocks & p for i, p in enumerate(pieces) if i != exclude)


def update_wall(model, index, direction):
    orient, pos = model['axes'][index]
    pos += direction[0 if orient == 'horizontal' else 1]
    if 0 <= pos < model['n'] and not hits_piece(axis_blocks(orient, pos, model['n']), model['pieces']):
        model['axes'][index] = orient, pos


def update_player(model, index, direction):
    dx, dy = direction
    moved = frozenset((x + dx, y + dy) for x, y in model['pieces'][index])
    if (inside_board(moved, model['n']) and not hits_piece(moved, model['pieces'], index)
            and not hits_axis(moved, model['axes'], model['n'])):
        model['pieces'][index] = moved


def step(model, action):
    model = copy.deepcopy(model)
    aid = action.get('action_id') if isinstance(action, dict) else action
    na = len(model['axes'])
    if aid == 5:
        model['sel'] = (model['sel'] + 1) % (na + len(model['pieces']))
    elif aid in DIRS:
        if model['sel'] < na:
            update_wall(model, model['sel'], DIRS[aid])
        else:
            update_player(model, model['sel'] - na, DIRS[aid])
    return model


def reflected(blocks, orient, pos):
    if orient == 'horizontal':
        return {(2 * pos - x, y) for x, y in blocks}
    return {(x, 2 * pos - y) for x, y in blocks}


def update_reflection(model):
    original = set().union(*model['pieces'])
    reflections = set()
    for orient, pos in model['axes']:
        reflections |= reflected(original | reflections, orient, pos)
    return reflections - original


def scene_sprites(model):
    sprites = []
    for i, (orient, pos) in enumerate(model['axes']):
        sprites.append(('wall', i, axis_blocks(orient, pos, model['n']), 0 if model['sel'] == i else None))
    for i, t in enumerate(model['targets']):
        sprites.append(('target', i, t, 'solid'))
    sprites.append(('reflection', 0, update_reflection(model), 4))
    for i, p in enumerate(model['pieces']):
        sprites.append(('player', i, p, 0 if model['sel'] == len(model['axes']) + i else None))
    return sprites


def composite(model):
    cells = {}
    for kind, sid, blocks, fill in reversed(scene_sprites(model)):
        for b in blocks:
            if inside_board([b], model['n']):
                cells.setdefault(b, []).append((kind, sid, fill))
    frame = {}
    for b, stack in cells.items():
        for i in range(CELL):
            for j in range(CELL):
                p = CELL * b[0] + i, CELL * b[1] + j
                fallback = None
                for kind, sid, fill in stack:
                    if (i, j) != (1, 1) or fill == 'solid':
                        frame[p] = kind, sid, COLORS[kind]
                        break
                    if fill is not None and fallback is None:
                        fallback = kind, sid, fill
                else:
                    if fallback is not None:
                        frame[p] = fallback
    return frame


def make_object(kind, points, template):
    x, y = min(p[0] for p in points), min(p[1] for p in points)
    w, h = max(p[0] for p in points) - x + 1, max(p[1] for p in points) - y + 1
    pix = [[-1] * h for _ in range(w)]
    for (px, py), value in points.items():
        pix[px - x][py - y] = value
    obj = copy.deepcopy(template) if template else {'type': kind, 'tags': TAGS[kind], 'layer': LAYERS[kind]}
    obj.update(x=x, y=y, w=w, h=h, pixels=pix)
    return obj


def component_key(comp):
    return min(p[0] for p in comp), min(p[1] for p in comp)


def extract_objects(frame, templates):
    result = []
    dots = {p for p, (_, _, v) in frame.items() if v == 0}
    for kind in ('wall', 'player', 'reflection'):
        owned = {p: v for p, (k, _, v) in frame.items() if k == kind}
        points = set(owned)
        if kind in ('wall', 'player'):
            points |= dots
        comps = sorted(components(points), key=component_key)
        for index, comp in enumerate(comps):
            visible = {p: owned[p] for p in comp if p in owned}
            if not visible:
                continue
            obj = make_object(kind, visible, templates.get(kind))
            if kind == 'wall':
                orient = 'vertical' if obj['w'] >= obj['h'] else 'horizontal'
                obj['tags'] = ['axis', orient]
                obj['name'] = ('wall_v_' if orient == 'vertical' else 'wall_h_') + str(index)
            else:
                obj['name'] = ('piece_' if kind == 'player' else 'reflection_') + str(index)
            result.append(obj)
    targets = {}
    for p, (kind, sid, v) in frame.items():
        if kind == 'target':
            targets.setdefault(sid, {})[p] = v
    for i, points in enumerate(sorted(targets.values(), key=component_key)):
        obj = make_object('target', points, templates.get('target'))
        obj['name'] = 'target_' + str(i)
        result.append(obj)
    return result


def render_state(model):
    return extract_objects(composite(model), model['templates']) + copy.deepcopy(model['others'])


def transition_function(state, action):
    global _memo
    key = canon(state)
    model = _memo['model'] if _memo and _memo['out'] == key else parse_model(state)
    model = step(model, action)
    out = render_state(model)
    _memo = {'model': model, 'out': canon(out)}
    return out
