# Arrows move the selected axis or piece on a three-pixel grid; ACTION5 cycles selection.
# Black hole pixels identify selection; the HUD counter is static in every observation.
# Pieces produce vertical gray and horizontal/double invisible mirror images.
# Layers occlude lower rings, then connected components determine names and pixels.
# Unobserved clicks/reset are no-ops; fully hidden selection defaults to the horizontal axis.
from copy import deepcopy
from collections import Counter


def object_pixels(obj):
    return {(obj['x'] + i, obj['y'] + j): value
            for i, row in enumerate(obj.get('pixels', []))
            for j, value in enumerate(row) if value >= 0}


def components(points):
    remaining = set(points)
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group = {seed}
        queue = [seed]
        while queue:
            x, y = queue.pop()
            for p in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    queue.append(p)
        result.append(group)
    return sorted(result, key=lambda g: (min(x for x, y in g),
                                         min(y for x, y in g)))


def bounds(group):
    return (min(x for x, y in group), min(y for x, y in group),
            max(x for x, y in group), max(y for x, y in group))


def touching_bboxes(a, b):
    ax, ay, ar, ab = bounds(a)
    bx, by, br, bb = bounds(b)
    return max(ax-br, bx-ar) <= 4 and max(ay-bb, by-ab) <= 4


def target_components(points):
    groups = components(points)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                if touching_bboxes(groups[i], groups[j]):
                    groups[i] |= groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return sorted(groups, key=lambda g: bounds(g)[:2])


def decode_state(state):
    frame = {}
    for obj in sorted(state, key=lambda o: o.get('layer', 0)):
        frame.update(object_pixels(obj))
    wall_blocks = {(x//3, y//3) for (x, y), c in frame.items() if c == 10}
    horizontal = Counter(x for x, y in wall_blocks).most_common(1)[0][0]
    vertical = Counter(y for x, y in wall_blocks).most_common(1)[0][0]
    pieces = []
    for obj in sorted(state, key=lambda o: (o['x'], o['y'])):
        if obj['type'] == 'player':
            blocks = {(x//3, y//3) for (x, y), c in object_pixels(obj).items() if c == 5}
            if blocks:
                pieces.append(blocks)
    selection = 0
    zeros = {(x//3, y//3) for (x, y), c in frame.items() if c == 0}
    if any(y == vertical and x != horizontal for x, y in zeros):
        selection = 1
    elif any(x == horizontal and y != vertical for x, y in zeros):
        selection = 0
    else:
        for i, blocks in enumerate(pieces):
            if blocks & zeros:
                selection = 2+i
                break
    targets = {(x//3, y//3) for (x, y), c in frame.items() if c == 11}
    counter = next((o for o in state if o['type'] == 'counter'), None)
    size = counter['x']//3 if counter else 21
    return {'h': horizontal, 'v': vertical, 'pieces': pieces,
            'selection': selection, 'targets': targets, 'size': size}


def inside_board(blocks, size):
    return all(0 <= x < size and 0 <= y < size for x, y in blocks)


def piece_collision(blocks, pieces, moving):
    return any(blocks & other for i, other in enumerate(pieces) if i != moving)


def update_wall(model, dx, dy):
    axis = 'h' if model['selection'] == 0 else 'v'
    delta = dx if axis == 'h' else dy
    position = model[axis] + delta
    if 0 <= position < model['size']:
        model[axis] = position


def update_player(model, dx, dy):
    i = model['selection'] - 2
    blocks = {(x+dx, y+dy) for x, y in model['pieces'][i]}
    if inside_board(blocks, model['size']) and not piece_collision(blocks, model['pieces'], i):
        model['pieces'][i] = blocks


def update_reflection(model):
    source = set().union(*model['pieces']) if model['pieces'] else set()
    h, v = model['h'], model['v']
    gray = {(x, 2*v-y) for x, y in source} - source
    invisible = ({(2*h-x, y) for x, y in source} |
                 {(2*h-x, 2*v-y) for x, y in source}) - source
    size = model['size']
    gray = {(x, y) for x, y in gray if 0 <= x < size and 0 <= y < size}
    invisible = {(x, y) for x, y in invisible if 0 <= x < size and 0 <= y < size}
    return gray, invisible


def update_target(model):
    return model['targets']


def update_counter(state):
    return [deepcopy(o) for o in state if o['type'] not in
            ('wall', 'player', 'target', 'reflection')]


def draw_blocks(frame, blocks, color, hole=None):
    for bx, by in sorted(blocks):
        for i in range(3):
            for j in range(3):
                p = (3*bx+i, 3*by+j)
                if i == j == 1 and hole is not None:
                    if frame.get(p, -1) == -1:
                        frame[p] = hole
                else:
                    frame[p] = color


def make_object(group, frame, name, kind, layer, tags):
    x, y, right, bottom = bounds(group)
    return {'name': name, 'type': kind, 'x': x, 'y': y,
            'w': right-x+1, 'h': bottom-y+1, 'layer': layer, 'tags': tags,
            'pixels': [[frame.get((a, b), -1) if (a, b) in group else -1
                        for b in range(y, bottom+1)] for a in range(x, right+1)]}


def extract_state(frame, state):
    result = update_counter(state)
    for kind, colors, primary, layer, prefix, tags in (
        ('wall', {10, 0}, 10, 1, 'wall', ['axis']),
        ('player', {5, 0}, 5, 4, 'piece', ['movable', 'black']),
        ('reflection', {4, -2}, 4, 3, 'reflection', ['mirror', 'gray']),
        ('target', {11}, 11, 2, 'target', ['goal', 'yellow'])):
        points = {p for p, c in frame.items() if c in colors}
        groups = target_components(points) if kind == 'target' else components(points)
        for i, group in enumerate(groups):
            if not any(frame[p] == primary for p in group):
                continue
            name = prefix + '_' + str(i)
            otags = tags[:]
            if kind == 'wall':
                x, y, right, bottom = bounds(group)
                direction = 'v' if right-x >= bottom-y else 'h'
                name = 'wall_' + direction + '_' + str(i)
                otags += ['vertical' if direction == 'v' else 'horizontal']
            result.append(make_object(group, frame, name, kind, layer, otags))
    return result


def render_state(model, state):
    frame = {}
    size, h, v = model['size'], model['h'], model['v']
    selection = model['selection']
    draw_blocks(frame, {(h, y) for y in range(size)}, 10, 0 if selection == 0 else -1)
    draw_blocks(frame, {(x, v) for x in range(size)}, 10, 0 if selection == 1 else -1)
    if selection in (0, 1):
        frame[3*h+1, 3*v+1] = 0
    draw_blocks(frame, update_target(model), 11)
    gray, invisible = update_reflection(model)
    draw_blocks(frame, invisible, -2, -2)
    draw_blocks(frame, gray, 4, 4)
    for i, blocks in enumerate(model['pieces']):
        for bx, by in blocks:
            p = (3*bx+1, 3*by+1)
            if frame.get(p, -1) == -1 and selection == 2+i:
                frame[p] = 0
        draw_blocks(frame, blocks, 5, -1)
    return extract_state(frame, state)


def transition_function(state, action):
    model = decode_state(state)
    action_id = action.get('action_id') if isinstance(action, dict) else action
    if action_id == 5:
        model['selection'] = (model['selection']+1) % (len(model['pieces'])+2)
    elif action_id in (1, 2, 3, 4):
        dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action_id]
        if model['selection'] < 2:
            update_wall(model, dx, dy)
        else:
            update_player(model, dx, dy)
    else:
        return deepcopy(state)
    return render_state(model, state)
