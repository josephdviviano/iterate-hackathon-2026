# A5 cycles column axis, row axis, then pieces; arrows move the selection.
# Pieces move one 3x3 cell, bounded by the board and other real pieces.
# Row mirrors are gray; column/double mirrors are transparent erasers.
# A piece on the column axis also has an invisible whole-piece row mirror.
# Recompose layers and re-extract names; click/reset behavior is unobserved.
from copy import deepcopy
from collections import Counter


def components(points):
    remaining = set(points)
    groups = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group, todo = {seed}, [seed]
        while todo:
            x, y = todo.pop()
            for q in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return sorted(groups, key=bbox)


def bbox(points):
    return (min(x for x, y in points), min(y for x, y in points),
            max(x for x, y in points), max(y for x, y in points))


def pixel_cells(obj):
    return {(obj['x']+i, obj['y']+j): value
            for i, col in enumerate(obj.get('pixels', []))
            for j, value in enumerate(col) if value >= 0}


def parse_state(state):
    pixels = {}
    for obj in state:
        pixels.update(pixel_cells(obj))
    walls = {(x//3, y//3) for (x, y), v in pixels.items() if v == 10}
    c = Counter(x for x, y in walls).most_common(1)[0][0]
    r = Counter(y for x, y in walls).most_common(1)[0][0]
    occupied = {(x//3, y//3) for (x, y), v in pixels.items() if v == 5}
    pieces = components(occupied)
    zeros = {(x//3, y//3) for (x, y), v in pixels.items() if v == 0}
    if any(x == c and y != r and (x, y) not in occupied for x, y in zeros):
        selected = 0
    elif any(y == r and x != c and (x, y) not in occupied for x, y in zeros):
        selected = 1
    else:
        selected = next((i+2 for i, p in enumerate(pieces) if p & zeros), 0)
    targets = {(x//3, y//3) for (x, y), v in pixels.items() if v == 11}
    return {'c': c, 'r': r, 'pieces': pieces, 'selected': selected,
            'targets': targets}


def inside_board(points):
    return all(0 <= x < 21 and 0 <= y < 21 for x, y in points)


def player_clear(candidate, pieces, index):
    return inside_board(candidate) and not any(
        candidate & p for i, p in enumerate(pieces) if i != index)


def update_wall(model, dx, dy):
    if model['selected'] == 0:
        model['c'] = min(20, max(0, model['c']+dx))
    elif model['selected'] == 1:
        model['r'] = min(20, max(0, model['r']+dy))


def update_player(model, dx, dy):
    index = model['selected']-2
    if index < 0:
        return
    candidate = {(x+dx, y+dy) for x, y in model['pieces'][index]}
    if player_clear(candidate, model['pieces'], index):
        model['pieces'][index] = candidate


def piece_on_column_axis(piece, column):
    return any(x == column for x, y in piece)


def update_reflection(model):
    visible, invisible = set(), set()
    c, r = model['c'], model['r']
    for piece in model['pieces']:
        row_mirror = {(x, 2*r-y) for x, y in piece}
        if piece_on_column_axis(piece, c):
            invisible.update(row_mirror)
        else:
            visible.update(row_mirror)
        invisible.update((2*c-x, y) for x, y in piece)
        invisible.update((2*c-x, 2*r-y) for x, y in piece)
    return visible, invisible


def update_target(model):
    return model['targets']


def render(model):
    stacks = {}

    def add(blocks, ring, fill, solid=False):
        for q in blocks:
            if inside_board([q]):
                stacks.setdefault(q, []).append((ring, fill, solid))

    for i, piece in enumerate(model['pieces']):
        add(piece, 5, 0 if model['selected'] == i+2 else -1)
    visible, invisible = update_reflection(model)
    add(visible, 4, 4)
    add(invisible, -2, -1)
    add(update_target(model), 11, 11, True)
    add({(model['c'], y) for y in range(21)}, 10,
        0 if model['selected'] == 0 else -1, model['selected'] == 0)
    add({(x, model['r']) for x in range(21)}, 10,
        0 if model['selected'] == 1 else -1, model['selected'] == 1)
    pixels = {}
    for (x, y), stack in stacks.items():
        ring = stack[0][0]
        center = next((fill for _, fill, solid in stack if solid), stack[0][1])
        for i in range(3):
            for j in range(3):
                value = center if i == j == 1 else ring
                if value != -1:
                    pixels[(3*x+i, 3*y+j)] = value
    return pixels


def target_components_near(a, b):
    ax, ay, az, aw = bbox(a)
    bx, by, bz, bw = bbox(b)
    return max(bx-az, ax-bz, by-aw, ay-bw) <= 4


def merge_target_components(groups):
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                if target_components_near(groups[i], groups[j]):
                    groups[i] |= groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return sorted(groups, key=bbox)


def make_object(points, pixels, kind, index):
    x, y, right, bottom = bbox(points)
    w, h = right-x+1, bottom-y+1
    layer, prefix, tags = {
        'player': (4, 'piece', ['movable', 'black']),
        'reflection': (3, 'reflection', ['mirror', 'gray']),
        'target': (2, 'target', ['goal', 'yellow']),
        'wall': (1, 'wall', ['axis'])}[kind]
    if kind == 'wall':
        vertical = w >= h
        prefix = 'wall_v' if vertical else 'wall_h'
        tags.append('vertical' if vertical else 'horizontal')
    return {'name': prefix+'_'+str(index), 'type': kind, 'x': x, 'y': y,
            'w': w, 'h': h, 'layer': layer, 'tags': tags,
            'pixels': [[pixels.get((x+i, y+j), -1)
                        if (x+i, y+j) in points else -1
                        for j in range(h)] for i in range(w)]}


def extract(pixels):
    result = []
    zeros = {q for q, v in pixels.items() if v == 0}
    for kind, value in [('wall', 10), ('player', 5),
                        ('reflection', 4), ('target', 11)]:
        own = {q for q, v in pixels.items() if v == value}
        search = own
        if kind in ('wall', 'player'):
            search = own | zeros
        elif kind == 'reflection':
            search = own | {q for q, v in pixels.items() if v == -2}
        groups = components(search)
        if kind == 'target':
            groups = merge_target_components(groups)
        for index, group in enumerate(groups):
            if not (group & own):
                continue
            if kind == 'reflection':
                group = group & own
            result.append(make_object(group, pixels, kind, index))
    return result


def update_counter(state):
    return [deepcopy(o) for o in state
            if o['type'] not in ('wall', 'player', 'reflection', 'target')]


def transition_function(state, action):
    action_id = action.get('action_id') if isinstance(action, dict) else action
    if action_id not in (1, 2, 3, 4, 5):
        return deepcopy(state)
    model = parse_state(state)
    if action_id == 5:
        model['selected'] = (model['selected']+1) % (len(model['pieces'])+2)
    else:
        dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action_id]
        update_wall(model, dx, dy)
        update_player(model, dx, dy)
    return update_counter(state) + extract(render(model))
