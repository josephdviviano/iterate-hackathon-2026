# Axes and selected pieces move on a three-pixel lattice; ACTION5 cycles selection.
# Vertical mirrors are gray; horizontal and double mirrors are invisible occluders.
# Sprite holes reveal lower solid pixels, otherwise selection dots or mirror gray.
# Re-extraction merges touching pixels and counts black dots in component names.
# Click/reset behavior and collisions outside the observed layouts are unconfirmed.
from copy import deepcopy
from collections import Counter


def pixel_map(obj):
    return {(obj['x'] + i, obj['y'] + j): value
            for i, row in enumerate(obj.get('pixels', []))
            for j, value in enumerate(row) if value >= 0}


def components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        todo = [seed]
        group = {seed}
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


def template(state, kind):
    for obj in state:
        if obj['type'] == kind:
            return {k: deepcopy(v) for k, v in obj.items()
                    if k not in ('name', 'x', 'y', 'w', 'h', 'pixels')}
    return {'type': kind, 'layer': 3, 'tags': ['mirror', 'gray']}


def parse_model(state):
    wall = {}
    players = []
    targets = set()
    for obj in state:
        pixels = pixel_map(obj)
        if obj['type'] == 'wall':
            wall.update(pixels)
        elif obj['type'] == 'player':
            blocks = {(x//3, y//3) for (x, y), v in pixels.items() if v == 5}
            players.append((blocks, pixels))
        elif obj['type'] == 'target':
            targets.update((x//3, y//3) for (x, y), v in pixels.items() if v == 11)
    row_count = Counter(x//3 for (x, y), v in wall.items() if v == 10)
    col_count = Counter(y//3 for (x, y), v in wall.items() if v == 10)
    hr = max(row_count, key=row_count.get)
    vc = max(col_count, key=col_count.get)
    players.sort(key=lambda p: bounds(p[0]))
    zeros = [q for q, v in wall.items() if v == 0]
    if any(x//3 == hr and y//3 != vc for x, y in zeros):
        selection = 0
    elif any(y//3 == vc and x//3 != hr for x, y in zeros):
        selection = 1
    else:
        selection = next((i+2 for i, (_, p) in enumerate(players)
                          if 0 in p.values()), 0)
    return {'hr': hr, 'vc': vc, 'pieces': [p[0] for p in players],
            'targets': targets, 'selection': selection,
            'size': next((o['x']//3 for o in state if o['type'] == 'counter'), 21)}


def within_board(blocks, size):
    return all(0 <= r < size and 0 <= c < size for r, c in blocks)


def piece_collision(blocks, others):
    return any(blocks & other for other in others)


def update_wall(model, action):
    selection = model['selection']
    if selection == 0 and action in (1, 2):
        model['hr'] = max(0, min(model['size']-1,
                                model['hr'] + (-1 if action == 1 else 1)))
    elif selection == 1 and action in (3, 4):
        model['vc'] = max(0, min(model['size']-1,
                                model['vc'] + (-1 if action == 3 else 1)))


def update_player(model, action):
    index = model['selection']-2
    if index < 0 or action not in (1, 2, 3, 4):
        return
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    moved = {(r+dr, c+dc) for r, c in model['pieces'][index]}
    others = [p for i, p in enumerate(model['pieces']) if i != index]
    if within_board(moved, model['size']) and not piece_collision(moved, others):
        model['pieces'][index] = moved


def update_reflection(model):
    pieces = set().union(*model['pieces'])
    vertical, hidden = set(), set()
    for r, c in pieces:
        vertical.add((r, 2*model['vc']-c))
        hidden.add((2*model['hr']-r, c))
        hidden.add((2*model['hr']-r, 2*model['vc']-c))
    return vertical-pieces, hidden-pieces-vertical


def update_target(model):
    return model['targets']


def update_counter(state):
    return [deepcopy(o) for o in state if o['type'] not in
            ('wall', 'player', 'reflection', 'target')]


def sprite(blocks, color, fill, size):
    result = {}
    for r, c in blocks:
        if not (0 <= r < size and 0 <= c < size):
            continue
        for i in range(3):
            for j in range(3):
                result[(3*r+i, 3*c+j)] = (None, fill) if i == j == 1 else (color, -1)
    return result


def target_groups(cells):
    groups = components(cells)
    changed = True
    while changed:
        changed = False
        for i, group in enumerate(groups):
            a, b, c, d = bounds(group)
            for j in range(i+1, len(groups)):
                e, f, g, h = bounds(groups[j])
                dx = max(0, a-g, e-c)
                dy = max(0, b-h, f-d)
                if max(dx, dy) <= 4:
                    groups[i] |= groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return sorted(groups, key=lambda g: bounds(g)[:2])


def extracted_object(base, name, group, pixels):
    x, y, right, bottom = bounds(group)
    obj = deepcopy(base)
    obj.update(name=name, x=x, y=y, w=right-x+1, h=bottom-y+1)
    obj['pixels'] = [[pixels.get((i, j), -1) if (i, j) in group else -1
                      for j in range(y, bottom+1)] for i in range(x, right+1)]
    return obj


def render_state(state, model):
    size, hr, vc, sel = (model[k] for k in ('size', 'hr', 'vc', 'selection'))
    layers = []
    layers.append(sprite({(hr, c) for c in range(size)}, 10, 0 if sel == 0 else -1, size))
    layers.append(sprite({(r, vc) for r in range(size)}, 10, 0 if sel == 1 else -1, size))
    targets = {(3*r+i, 3*c+j): (11, -1) for r, c in update_target(model)
               for i in range(3) for j in range(3)}
    layers.append(targets)
    vertical, hidden = update_reflection(model)
    layers.append(sprite(hidden, -2, 4, size))
    layers.append(sprite(vertical, 4, 4, size))
    for i, blocks in enumerate(model['pieces']):
        layers.append(sprite(blocks, 5, 0 if sel == i+2 else -1, size))
    canvas = {}
    coords = set().union(*(set(s) for s in layers))
    for q in coords:
        entries = [s[q] for s in reversed(layers) if q in s]
        solid = next((v for v, fill in entries if v is not None), None)
        if solid is not None:
            canvas[q] = solid
        else:
            fill = next((fill for v, fill in entries if fill >= 0), -1)
            if fill >= 0:
                canvas[q] = fill
    output = update_counter(state)
    for kind, color, prefix in [('wall', 10, 'wall'), ('player', 5, 'piece'),
                                 ('reflection', 4, 'reflection')]:
        base = template(state, kind)
        allowed = {q for q, v in canvas.items() if v == color or
                   (v == 0 and kind != 'reflection')}
        if kind == 'reflection':
            allowed.update(q for q, v in canvas.items() if v == -2)
        for index, group in enumerate(components(allowed)):
            if not any(canvas[q] == color for q in group):
                continue
            if kind == 'reflection' and not any(
                    (x//3, y//3) in vertical and canvas[(x, y)] == 4
                    for x, y in group):
                continue
            name = prefix
            if kind == 'wall':
                x, y, right, bottom = bounds(group)
                orientation = 'vertical' if right-x >= bottom-y else 'horizontal'
                base['tags'] = ['axis', orientation]
                name += '_v' if orientation == 'vertical' else '_h'
            pixels = {q: canvas[q] for q in group if canvas[q] >= 0}
            output.append(extracted_object(base, name+'_'+str(index), group, pixels))
    yellow = {q for q, v in canvas.items() if v == 11}
    for i, group in enumerate(target_groups(yellow)):
        output.append(extracted_object(template(state, 'target'), 'target_'+str(i),
                                       group, {q: 11 for q in group}))
    return output


def transition_function(state, action):
    model = parse_model(state)
    action_id = action.get('action_id') if isinstance(action, dict) else action
    if action_id == 5:
        model['selection'] = (model['selection']+1) % (2+len(model['pieces']))
    else:
        update_wall(model, action_id)
        update_player(model, action_id)
    return render_state(state, model)
