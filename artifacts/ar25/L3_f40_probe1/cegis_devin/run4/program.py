# Selection is read from black centres: wall dots take precedence over player dots.
# Arrows translate the selected axis or piece by one 3x3 cell; axes pass under pieces.
# Reflections mirror piece blocks, excluding blocks occupied by any original piece.
# Layered holes expose lower solids; rendered components determine names and pixels.
# Hidden history is unnecessary here; unobserved clicks and full cycle order are hypotheses.
from collections import Counter
from copy import deepcopy


def pixels_of(obj):
    return {(obj['x'] + i, obj['y'] + j): v
            for i, row in enumerate(obj.get('pixels', []))
            for j, v in enumerate(row) if v != -1}


def blocks_of(obj, colour):
    return {(x // 3, y // 3) for (x, y), v in pixels_of(obj).items()
            if v == colour}


def components(cells):
    remaining = set(cells)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            x, y = stack.pop()
            for p in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return sorted(groups, key=lambda g: (min(x for x, y in g),
                                         min(y for x, y in g)))


def axis_selected(state):
    return any(0 in pixels_of(o).values() for o in state
               if o['type'] == 'wall')


def decode(state):
    frame = {}
    for obj in state:
        frame.update(pixels_of(obj))
    wall_blocks = set().union(*(blocks_of(o, 10) for o in state
                               if o['type'] == 'wall'))
    axis = Counter(x for x, y in wall_blocks).most_common(1)[0][0]
    selected_axis = axis_selected(state)
    pieces = []
    for obj in state:
        if obj['type'] != 'player':
            continue
        blocks = blocks_of(obj, 5)
        marked = {p for p in blocks if frame.get((3*p[0]+1, 3*p[1]+1)) == 0}
        if marked and not selected_axis:
            # A selected sprite's centre cannot be empty, even under a lower solid.
            selected = {p for p in blocks
                        if (3*p[0]+1, 3*p[1]+1) in frame}
            pieces.append((selected, True))
            pieces.extend((g, False) for g in components(blocks - selected))
        else:
            pieces.append((blocks, False))
    pieces.sort(key=lambda p: min(p[0]))
    selection = -1
    if not selected_axis:
        selection = next((i for i, (_, active) in enumerate(pieces) if active), -1)
    targets = [blocks_of(o, 11) for o in state if o['type'] == 'target']
    return axis, [p[0] for p in pieces], selection, targets


def inside_board(blocks):
    return all(0 <= x < 21 and 0 <= y < 21 for x, y in blocks)


def overlaps_player(blocks, pieces, selected):
    return any(blocks & other for i, other in enumerate(pieces) if i != selected)


def update_wall(axis, selection, action):
    if selection == -1 and action in (1, 2):
        axis = max(0, min(20, axis + (-1 if action == 1 else 1)))
    return axis


def update_player(pieces, selection, action):
    if selection < 0 or action not in (1, 2, 3, 4):
        return pieces
    dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
    moved = {(x + dx, y + dy) for x, y in pieces[selection]}
    if inside_board(moved) and not overlaps_player(moved, pieces, selection):
        pieces[selection] = moved
    return pieces


def update_reflection(pieces, axis):
    occupied = set().union(*pieces)
    return {(2*axis-x, y) for x, y in occupied
            if 0 <= 2*axis-x < 21 and (2*axis-x, y) not in occupied}


def update_target(targets):
    return targets


def update_counter(state):
    return [deepcopy(o) for o in state if o['type'] not in
            ('wall', 'player', 'reflection', 'target')]


def render(axis, pieces, selection, targets):
    stack = {}

    def draw(blocks, colour, hole=False, fill=-1, solid_centre=False):
        for bx, by in blocks:
            for i in range(3):
                for j in range(3):
                    centre = i == 1 and j == 1
                    solid = colour if not (hole and centre) else None
                    if centre and solid_centre:
                        solid = fill
                    p = (3*bx+i, 3*by+j)
                    stack.setdefault(p, []).append((solid, fill))

    draw({(axis, y) for y in range(21)}, 10, True,
         0 if selection == -1 else -1, selection == -1)
    for target in update_target(targets):
        draw(target, 11)
    draw(update_reflection(pieces, axis), 4, True, 4)
    for i, piece in enumerate(pieces):
        draw(piece, 5, True, 0 if i == selection else -1)
    frame = {}
    for p, layers in stack.items():
        value = next((solid for solid, fill in reversed(layers)
                      if solid is not None), layers[-1][1])
        if value != -1:
            frame[p] = value
    return frame


def make_object(cells, frame, kind, index):
    x, y = min(p[0] for p in cells), min(p[1] for p in cells)
    w = max(p[0] for p in cells) - x + 1
    h = max(p[1] for p in cells) - y + 1
    prefixes = {'wall': 'wall_h', 'player': 'piece',
                'reflection': 'reflection', 'target': 'target'}
    tags = {'wall': ['axis', 'horizontal'], 'player': ['movable', 'black'],
            'reflection': ['mirror', 'gray'], 'target': ['goal', 'yellow']}
    layers = {'wall': 1, 'target': 2, 'reflection': 3, 'player': 4}
    pixels = [[-1 for j in range(h)] for i in range(w)]
    for px, py in cells:
        pixels[px-x][py-y] = frame[px, py]
    return dict(name=prefixes[kind]+'_'+str(index), type=kind, x=x, y=y,
                w=w, h=h, pixels=pixels, layer=layers[kind], tags=tags[kind])


def extract(frame, targets):
    output = []
    zeros = {p for p, colour in frame.items() if colour == 0}
    for kind, colour in (('wall', 10), ('player', 5), ('reflection', 4)):
        primary = {p for p, v in frame.items() if v == colour}
        cells = primary | zeros if kind in ('wall', 'player') else primary
        for i, group in enumerate(components(cells)):
            if group & primary:
                output.append(make_object(group, frame, kind, i))
    visible_targets = []
    for blocks in targets:
        cells = {p for p, v in frame.items()
                 if v == 11 and (p[0]//3, p[1]//3) in blocks}
        if cells:
            visible_targets.append(cells)
    visible_targets.sort(key=lambda g: (min(x for x, y in g),
                                         min(y for x, y in g)))
    for i, group in enumerate(visible_targets):
        output.append(make_object(group, frame, 'target', i))
    return output


def transition_function(state, action):
    action_id = action.get('action_id') if isinstance(action, dict) else action
    if action_id == 7:
        return deepcopy(state)
    axis, pieces, selection, targets = decode(state)
    if action_id == 5:
        selection = selection + 1 if selection + 1 < len(pieces) else -1
    elif action_id == 6:
        # Click coordinates are conventional (column, row), unlike extractor x/y.
        block = (action['y']//3, action['x']//3)
        selection = next((i for i, p in enumerate(pieces) if block in p), selection)
        if block[0] == axis and not any(block in p for p in pieces):
            selection = -1
    else:
        axis = update_wall(axis, selection, action_id)
        pieces = update_player(pieces, selection, action_id)
    frame = render(axis, pieces, selection, targets)
    return update_counter(state) + extract(frame, targets)
