# A 21x21 board uses 3x3 cells; x is the row and y is the column.
# Actions 1-4 move the selection; 5 cycles axis then logical pieces.
# Pieces mirror across a horizontal axis; targets and HUD stay fixed.
# Layered rendering, transparent holes and components determine all fields.
# Hypothesis: pieces cannot enter the axis; clicks/reset are unobserved no-ops.
from copy import deepcopy
import json


_last_state = None
_last_model = None
DELTAS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def canonical(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def visible_pixels(obj):
    return {(obj['x'] + i, obj['y'] + j): v
            for i, row in enumerate(obj.get('pixels', []))
            for j, v in enumerate(row) if v >= 0}


def blocks(obj):
    return {(x // 3, y // 3) for x, y in visible_pixels(obj)}


def components(points):
    remaining = set(points)
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        part, stack = {root}, [root]
        while stack:
            x, y = stack.pop()
            for p in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def infer_model(state):
    walls = [o for o in state if o['type'] == 'wall']
    axis = min(o['x'] for o in walls) // 3
    frame = {}
    for obj in sorted(state, key=lambda o: o.get('layer', 0)):
        frame.update(visible_pixels(obj))
    pieces, selected = [], -1
    for obj in sorted((o for o in state if o['type'] == 'player'),
                      key=lambda o: (o['x'], o['y'])):
        cells = blocks(obj)
        pixels = visible_pixels(obj)
        marked = {c for c in cells
                  if pixels.get((3*c[0]+1, 3*c[1]+1)) == 0}
        plain = {c for c in cells
                 if frame.get((3*c[0]+1, 3*c[1]+1), -1) == -1}
        if marked and plain:
            assigned = {c: True for c in marked}
            assigned.update({c: False for c in plain})
            pending = cells - assigned.keys()
            while pending:
                growth = {}
                for x, y in pending:
                    neighbours = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
                    known = [assigned[c] for c in neighbours if c in assigned]
                    if known:
                        growth[(x,y)] = any(known)
                if not growth:
                    growth = {c: True for c in pending}
                assigned.update(growth)
                pending -= growth.keys()
            chosen = {c for c in cells if assigned[c]}
            selected = len(pieces)
            pieces.append(chosen)
            pieces.extend(components(cells - chosen))
        else:
            if marked:
                selected = len(pieces)
            pieces.append(cells)
    targets = [blocks(o) for o in state if o['type'] == 'target']
    return {'axis': axis, 'pieces': pieces, 'selected': selected,
            'targets': targets}


def on_board(cells):
    return all(0 <= x < 21 and 0 <= y < 21 for x, y in cells)


def touches_axis(cells, axis):
    return any(x == axis for x, y in cells)


def overlaps_piece(cells, others):
    return any(cells & other for other in others)


def update_wall(model, action):
    if model['selected'] != -1 or action not in (1, 2):
        return
    axis = model['axis'] + DELTAS[action][0]
    if 0 <= axis < 21 and not any(touches_axis(p, axis)
                                 for p in model['pieces']):
        model['axis'] = axis


def update_player(model, action):
    selected = model['selected']
    if selected < 0 or action not in DELTAS:
        return
    dx, dy = DELTAS[action]
    cells = {(x+dx, y+dy) for x, y in model['pieces'][selected]}
    others = [p for i, p in enumerate(model['pieces']) if i != selected]
    if (on_board(cells) and not touches_axis(cells, model['axis'])
            and not overlaps_piece(cells, others)):
        model['pieces'][selected] = cells


def update_target(model, action):
    return model['targets']


def update_reflection(model):
    return [{(2*model['axis']-x, y) for x, y in piece}
            for piece in model['pieces']]


def update_counter(state, action):
    return [deepcopy(o) for o in state if o['type'] == 'counter']


def object_from_pixels(points, template, name):
    x = min(p[0] for p in points)
    y = min(p[1] for p in points)
    w = max(p[0] for p in points) - x + 1
    h = max(p[1] for p in points) - y + 1
    obj = deepcopy(template)
    obj.update(name=name, x=x, y=y, w=w, h=h,
               pixels=[[points.get((x+i, y+j), -1) for j in range(h)]
                       for i in range(w)])
    return obj


def render(model, state, action):
    templates = {o['type']: o for o in state}
    if 'reflection' not in templates:
        templates['reflection'] = {'type': 'reflection', 'layer': 3,
                                   'tags': ['mirror', 'gray']}
    frame, owners = {}, {}

    def paint(x, y, color, owner):
        if 0 <= x < 63 and 0 <= y < 63:
            frame[(x,y)] = color
            owners[(x,y)] = owner

    axis = 3*model['axis']
    for x in range(axis, axis+3):
        for y in range(63):
            if x % 3 == 1 and y % 3 == 1:
                if model['selected'] != -1:
                    continue
                color = 0
            else:
                color = 10
            paint(x, y, color, ('wall', 0))
    for i, cells in enumerate(update_target(model, action)):
        for bx, by in cells:
            for dx in range(3):
                for dy in range(3):
                    paint(3*bx+dx, 3*by+dy, 11, ('target', i))
    for i, cells in enumerate(update_reflection(model)):
        for bx, by in cells:
            for dx in range(3):
                for dy in range(3):
                    x, y = 3*bx+dx, 3*by+dy
                    if dx == dy == 1 and frame.get((x,y)) == 11:
                        continue
                    paint(x, y, 4, ('reflection', i))
    for i, cells in enumerate(model['pieces']):
        for bx, by in cells:
            for dx in range(3):
                for dy in range(3):
                    x, y = 3*bx+dx, 3*by+dy
                    if dx == dy == 1:
                        if (x,y) in frame or i != model['selected']:
                            continue
                        color = 0
                    else:
                        color = 5
                    paint(x, y, color, ('player', i))
    output = update_counter(state, action)
    output.extend(deepcopy(o) for o in state
                  if o['type'] not in ('counter','wall','player',
                                       'reflection','target'))
    offset = sum(v == 0 and owners[p][0] != 'wall'
                 for p, v in frame.items())
    for typ, prefix in [('wall','wall_h'), ('player','piece'),
                        ('reflection','reflection'), ('target','target')]:
        points = {p: frame[p] for p, owner in owners.items() if owner[0] == typ}
        if typ == 'target':
            groups = [{p for p in points if owners[p] == ('target', i)}
                      for i in range(len(model['targets']))]
        else:
            groups = components(points)
        objects = [object_from_pixels({p: points[p] for p in group},
                                      templates[typ], '')
                   for group in groups if group]
        objects.sort(key=lambda o: (o['x'], o['y']))
        for i, obj in enumerate(objects):
            obj['name'] = prefix + '_' + str(i + (offset if typ == 'wall' else 0))
        output.extend(objects)
    return output


def transition_function(state, action):
    global _last_state, _last_model
    action_id = action.get('action_id') if isinstance(action, dict) else action
    if not any(o['type'] == 'wall' for o in state):
        _last_state, _last_model = None, None
        return deepcopy(state)
    if _last_model is not None and canonical(state) == _last_state:
        model = deepcopy(_last_model)
    else:
        model = infer_model(state)
    if action_id == 5:
        model['selected'] += 1
        if model['selected'] >= len(model['pieces']):
            model['selected'] = -1
    else:
        update_wall(model, action_id)
        update_player(model, action_id)
    output = render(model, state, action_id)
    _last_state, _last_model = canonical(output), deepcopy(model)
    return output
