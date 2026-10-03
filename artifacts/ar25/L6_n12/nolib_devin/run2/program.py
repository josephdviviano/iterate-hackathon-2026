# Axes and holed pieces occupy 3x3 cells; arrows move the selected item.
# Black hole centres select horizontal axis, vertical axis, or a piece.
# Targets stay fixed; mirrored pieces occlude lower layers and clip at edges.
# Rendered components and black dots determine names, bounds, and pixels.
# Unobserved: later selection order, clicks/reset, and collision outcomes.
from copy import deepcopy


def cells_of(obj, color):
    return {(obj['x'] + i, obj['y'] + j)
            for i, row in enumerate(obj.get('pixels', []))
            for j, value in enumerate(row) if value == color}


def blocks_of(obj, color):
    return {(x // 3, y // 3) for x, y in cells_of(obj, color)}


def neighbours(point):
    x, y = point
    return ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))


def components(points):
    points = set(points)
    groups = []
    while points:
        p = min(points)
        points.remove(p)
        group, pending = {p}, [p]
        while pending:
            for q in neighbours(pending.pop()):
                if q in points:
                    points.remove(q)
                    group.add(q)
                    pending.append(q)
        groups.append(group)
    return groups


def bbox(points):
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def decode(state):
    limit = next((o['x'] for o in state if o['type'] == 'counter'), 63)
    size = limit // 3
    walls = set().union(*(blocks_of(o, 10) for o in state
                          if o['type'] == 'wall'))
    rows = {x: sum(a == x for a, b in walls) for x in range(size)}
    cols = {y: sum(b == y for a, b in walls) for y in range(size)}
    ax = max(rows, key=rows.get) if walls and max(rows.values()) > size // 2 else None
    ay = max(cols, key=cols.get) if walls and max(cols.values()) > size // 2 else None
    players = sorted((o for o in state if o['type'] == 'player'),
                     key=lambda o: (o['x'], o['y']))
    pieces = [blocks_of(o, 5) for o in players]
    wall_dots = set().union(*(cells_of(o, 0) for o in state
                              if o['type'] == 'wall'))
    selection = ('v' if wall_dots and len({y for x, y in wall_dots}) == 1
                 else 'h')
    for i, o in enumerate(players):
        if cells_of(o, 0):
            selection = i
    targets = [(blocks_of(o, 11), deepcopy(o)) for o in state
               if o['type'] == 'target']
    return {'size': size, 'ax': ax, 'ay': ay, 'pieces': pieces,
            'selection': selection, 'targets': targets,
            'static': [deepcopy(o) for o in state
                       if o['type'] not in ('wall', 'player', 'reflection', 'target')]}


def within_board(blocks, size):
    return all(0 <= x < size and 0 <= y < size for x, y in blocks)


def hits_axis(blocks, ax, ay):
    return any(x == ax or y == ay for x, y in blocks)


def hits_piece(blocks, pieces, index):
    return any(blocks & piece for i, piece in enumerate(pieces) if i != index)


def update_wall(model, dx, dy):
    selected, size = model['selection'], model['size']
    if selected == 'h' and model['ax'] is not None:
        new = model['ax'] + dx
        if 0 <= new < size:
            model['ax'] = new
    elif selected == 'v' and model['ay'] is not None:
        new = model['ay'] + dy
        if 0 <= new < size:
            model['ay'] = new


def update_player(model, dx, dy):
    index = model['selection']
    moved = {(x + dx, y + dy) for x, y in model['pieces'][index]}
    if (within_board(moved, model['size'])
            and not hits_axis(moved, model['ax'], model['ay'])
            and not hits_piece(moved, model['pieces'], index)):
        model['pieces'][index] = moved


def step(model, action):
    aid = action.get('action_id') if isinstance(action, dict) else action
    selected = model['selection']
    if aid == 5:
        order = ([a for a, key in [('h', 'ax'), ('v', 'ay')]
                  if model[key] is not None]
                 + list(reversed(range(len(model['pieces'])))))
        if order:
            model['selection'] = order[(order.index(selected) + 1) % len(order)]
    elif aid in (1, 2, 3, 4):
        dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[aid]
        if isinstance(selected, int):
            update_player(model, dx, dy)
        else:
            update_wall(model, dx, dy)


def make_object(points, values, template, name, kind, tags, layer):
    x, y, right, bottom = bbox(points)
    obj = deepcopy(template)
    obj.update(name=name, type=kind, tags=tags, layer=layer,
               x=x, y=y, w=right-x, h=bottom-y,
               pixels=[[values.get((i, j), -1) if (i, j) in points else -1
                        for j in range(y, bottom)] for i in range(x, right)])
    return obj


def render(model, state):
    size, selected = model['size'], model['selection']
    sprites = []

    def sprite(blocks, owner, color, hole, fill=-1):
        pixels = {}
        for x, y in blocks:
            if 0 <= x < size and 0 <= y < size:
                for i in range(3):
                    for j in range(3):
                        pixels[3*x+i, 3*y+j] = (None if hole and i == j == 1 else color)
        sprites.append((owner, pixels, fill))

    ax, ay = model['ax'], model['ay']
    if ax is not None:
        sprite({(ax, y) for y in range(size)}, ('wall', 0), 10, True,
               0 if selected == 'h' else -1)
    if ay is not None:
        sprite({(x, ay) for x in range(size)}, ('wall', 0), 10, True,
               0 if selected == 'v' else -1)
    for i, (blocks, template) in enumerate(model['targets']):
        sprite(blocks, ('target', i), 11, False)
    occupied = set().union(*model['pieces']) if model['pieces'] else set()
    for piece in model['pieces']:
        if ax is not None:
            mirrored = {(2*ax-x, y) for x, y in piece} - occupied
            sprite(mirrored, ('hidden', 0), -1, True, 4)
            if ay is not None:
                mirrored = {(2*ax-x, 2*ay-y) for x, y in piece} - occupied
                sprite(mirrored, ('hidden', 0), -1, True, 4)
        if ay is not None:
            mirrored = {(x, 2*ay-y) for x, y in piece} - occupied
            sprite(mirrored, ('reflection', 0), 4, True, 4)
    for i, piece in enumerate(model['pieces']):
        sprite(piece, ('player', 0), 5, True, 0 if selected == i else -1)
    stacks = {}
    for owner, pixels, fill in sprites:
        for p, value in pixels.items():
            stacks.setdefault(p, []).append((owner, value, fill))
    groups, values = {}, {}
    for p, stack in stacks.items():
        chosen = next((s for s in reversed(stack) if s[1] is not None), None)
        if chosen is not None:
            owner, value, fill = chosen
        else:
            chosen = next((s for s in reversed(stack) if s[2] >= 0), None)
            if chosen is None:
                continue
            owner, unused, value = chosen
        values[p] = value
        groups.setdefault(owner, set()).add(p)
    result = deepcopy(model['static'])
    zeros = {p for p, value in values.items() if value == 0}
    for kind in ('wall', 'player', 'reflection'):
        points = groups.get((kind, 0), set())
        comps = sorted(components(points), key=lambda c: bbox(c)[:2])
        outside_dots = zeros - points if kind in ('wall', 'player') else set()
        extras = components(groups.get(('hidden', 0), set())) if kind == 'reflection' else []
        all_boxes = [bbox(c)[:2] for c in comps + extras]
        all_boxes += list(outside_dots)
        template = next((o for o in state if o['type'] == kind), {})
        for comp in comps:
            x, y, right, bottom = bbox(comp)
            index = sum(pos < (x, y) for pos in all_boxes)
            if kind == 'wall':
                orient = 'h' if right-x < bottom-y else 'v'
                name, tags, layer = ('wall_' + orient + '_' + str(index),
                                     ['axis', 'horizontal' if orient == 'h' else 'vertical'], 1)
            elif kind == 'player':
                name, tags, layer = 'piece_' + str(index), ['movable', 'black'], 4
            else:
                name, tags, layer = 'reflection_' + str(index), ['mirror', 'gray'], 3
            result.append(make_object(comp, values, template, name, kind, tags, layer))
    target_groups = [(points, model['targets'][i][1])
                     for (kind, i), points in groups.items() if kind == 'target']
    for i, (points, template) in enumerate(sorted(target_groups, key=lambda item: bbox(item[0])[:2])):
        result.append(make_object(points, values, template, 'target_' + str(i),
                                  'target', template['tags'], template['layer']))
    return result


def transition_function(state, action):
    model = decode(state)
    step(model, action)
    return render(model, state)
