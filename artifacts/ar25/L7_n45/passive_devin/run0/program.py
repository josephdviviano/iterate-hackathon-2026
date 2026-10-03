# Black centre pixels select an axis or piece; ACTION5 cycles selection.
# Direction actions translate the selected axis or piece by one 3x3 cell.
# Pieces create vertical gray mirrors and invisible horizontal/double mirrors.
# Layered holes, clipping, and connected components determine pixels and names.
# Click/reset semantics and unobserved collision cases remain unconfirmed.
from copy import deepcopy
from collections import Counter


def object_pixels(obj):
    return {(obj['x'] + i, obj['y'] + j): value
            for i, row in enumerate(obj.get('pixels', []))
            for j, value in enumerate(row) if value != -1}


def block_cells(pixels):
    return {(x // 3, y // 3) for x, y in pixels}


def connected_components(cells):
    remaining = set(cells)
    result = []
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
        result.append(group)
    return sorted(result, key=lambda g: (min(x for x, y in g),
                                        min(y for x, y in g)))


def bounds(cells):
    return (min(x for x, y in cells), min(y for x, y in cells),
            max(x for x, y in cells), max(y for x, y in cells))


def selected_axis(wall_pixels, row, col):
    zeros = {p for p, v in wall_pixels.items() if v == 0}
    if any(x // 3 == row and y // 3 != col for x, y in zeros):
        return 0
    if any(y // 3 == col and x // 3 != row for x, y in zeros):
        return 1
    return None


def parse_model(state):
    walls = {}
    pieces, targets = [], set()
    selected_piece = None
    for obj in state:
        pixels = object_pixels(obj)
        if obj['type'] == 'wall':
            walls.update(pixels)
        elif obj['type'] == 'player':
            pieces.append((block_cells(pixels), any(v == 0 for v in pixels.values())))
        elif obj['type'] == 'target':
            targets.update(block_cells(pixels))
    wb = block_cells({p: v for p, v in walls.items() if v == 10})
    row = Counter(x for x, y in wb).most_common(1)[0][0]
    col = Counter(y for x, y in wb).most_common(1)[0][0]
    pieces.sort(key=lambda item: min(item[0]))
    selection = selected_axis(walls, row, col)
    if selection is None:
        for i, (blocks, selected) in enumerate(pieces):
            if selected:
                selected_piece = i + 2
                break
        selection = selected_piece if selected_piece is not None else 0
    return {'row': row, 'col': col, 'pieces': [p[0] for p in pieces],
            'targets': targets, 'selection': selection}


def inside_board(blocks):
    return all(0 <= x < 21 and 0 <= y < 21 for x, y in blocks)


def overlaps_other_piece(blocks, pieces, selected):
    return any(blocks & other for i, other in enumerate(pieces) if i != selected)


def update_wall(model, action):
    selection = model['selection']
    if selection == 0 and action in (1, 2):
        row = model['row'] + (-1 if action == 1 else 1)
        if 0 <= row < 21:
            model['row'] = row
    elif selection == 1 and action in (3, 4):
        col = model['col'] + (-1 if action == 3 else 1)
        if 0 <= col < 21:
            model['col'] = col


def update_player(model, action):
    if model['selection'] < 2 or action not in (1, 2, 3, 4):
        return
    index = model['selection'] - 2
    dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    moved = {(x + dx, y + dy) for x, y in model['pieces'][index]}
    if inside_board(moved) and not overlaps_other_piece(moved, model['pieces'], index):
        model['pieces'][index] = moved


def update_reflection(model):
    row, col = model['row'], model['col']
    pieces = set().union(*model['pieces'])
    visible = {(x, 2 * col - y) for x, y in pieces} - pieces
    invisible = ({(2 * row - x, y) for x, y in pieces} |
                 {(2 * row - x, 2 * col - y) for x, y in pieces}) - pieces
    return visible, invisible


def update_target(model):
    return model['targets']


def update_counter(obj):
    return deepcopy(obj)


def render_pixels(model):
    stacks = {}

    def add(blocks, color, fill=-1, solid_center=False):
        for bx, by in sorted(blocks):
            if not (0 <= bx < 21 and 0 <= by < 21):
                continue
            for i in range(3):
                for j in range(3):
                    center = i == j == 1
                    value = color if not center or solid_center else fill
                    solid = not center or solid_center or value == 0 and color == 10
                    stacks.setdefault((3*bx+i, 3*by+j), []).append((solid, value))

    add({(model['row'], y) for y in range(21)}, 10,
        0 if model['selection'] == 0 else -1)
    add({(x, model['col']) for x in range(21)}, 10,
        0 if model['selection'] == 1 else -1)
    add(update_target(model), 11, solid_center=True)
    visible, invisible = update_reflection(model)
    add(invisible, -2)
    add(visible, 4, 4)
    for i, blocks in enumerate(model['pieces']):
        add(blocks, 5, 0 if model['selection'] == i+2 else -1)
    pixels = {}
    for p, stack in stacks.items():
        value = stack[-1][1]
        for solid, candidate in reversed(stack):
            if solid:
                value = candidate
                break
        if value != -1:
            pixels[p] = value
    return pixels


def make_object(kind, name, cells, pixels):
    x, y, xmax, ymax = bounds(cells)
    tags = {'player': ['movable', 'black'], 'reflection': ['mirror', 'gray'],
            'target': ['goal', 'yellow']}
    if kind == 'wall':
        orientation = 'horizontal' if ymax-y > xmax-x else 'vertical'
        obj_tags = ['axis', orientation]
    else:
        obj_tags = tags[kind]
    return {'name': name, 'type': kind, 'x': x, 'y': y,
            'w': xmax-x+1, 'h': ymax-y+1,
            'layer': {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}[kind],
            'tags': obj_tags,
            'pixels': [[pixels.get((a, b), -1) if (a, b) in cells else -1
                        for b in range(y, ymax+1)] for a in range(x, xmax+1)]}


def target_components(cells):
    groups = connected_components(cells)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            a, b, c, d = bounds(groups[i])
            for j in range(i+1, len(groups)):
                e, f, g, h = bounds(groups[j])
                gap = max(0, a-g, e-c, b-h, f-d)
                if gap <= 4:
                    groups[i] |= groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return sorted(groups, key=lambda g: bounds(g)[:2])


def extract_objects(pixels, state):
    result = [update_counter(o) for o in state if o['type'] not in
              ('wall', 'player', 'reflection', 'target')]
    zeros = {p for p, v in pixels.items() if v == 0}
    for kind, color in (('wall', 10), ('player', 5)):
        primary = {p for p, v in pixels.items() if v == color}
        for index, group in enumerate(connected_components(primary | zeros)):
            if not group & primary:
                continue
            if kind == 'wall':
                x, y, xmax, ymax = bounds(group)
                prefix = 'wall_h_' if ymax-y > xmax-x else 'wall_v_'
            else:
                prefix = 'piece_'
            result.append(make_object(kind, prefix+str(index), group, pixels))
    reflection_cells = {p for p, v in pixels.items() if v in (4, -2)}
    for index, group in enumerate(connected_components(reflection_cells)):
        visible = {p for p in group if pixels[p] == 4}
        if visible:
            result.append(make_object('reflection', 'reflection_'+str(index), visible, pixels))
    target_cells = {p for p, v in pixels.items() if v == 11}
    for index, group in enumerate(target_components(target_cells)):
        result.append(make_object('target', 'target_'+str(index), group, pixels))
    return result


def transition_function(state, action):
    action_id = action.get('action_id') if isinstance(action, dict) else action
    if action_id not in (1, 2, 3, 4, 5):
        return deepcopy(state)
    model = parse_model(state)
    if action_id == 5:
        model['selection'] = (model['selection'] + 1) % (len(model['pieces']) + 2)
    else:
        update_wall(model, action_id)
        update_player(model, action_id)
    return extract_objects(render_pixels(model), state)
