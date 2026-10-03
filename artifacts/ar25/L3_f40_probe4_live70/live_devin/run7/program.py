# Arrows move the selected 3x3-block piece or horizontal mirror axis.
# Action 5 cycles axis then pieces by size; reflections and goals are layered.
# Budget starts at 128, displays at most 64, and spent HUD offsets piece names.
# Continuity-gated history supports free action-7 movement undo, keeping selection.
# Clicks, equal-size cycle ties, and exhausted-budget behavior are unconfirmed.
from copy import deepcopy
import json

_last = None
_turns = 0
_history = []


def signature(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def components(cells):
    remaining = set(cells)
    groups = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group, todo = {seed}, [seed]
        while todo:
            for p in neighbors(todo.pop()):
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    todo.append(p)
        groups.append(group)
    return sorted(groups, key=lambda g: (min(r for r, c in g),
                                         min(c for r, c in g)))


def read_pixels(state):
    frame = {}
    for o in state:
        for r, row in enumerate(o.get('pixels', [])):
            for c, value in enumerate(row):
                if value >= 0:
                    frame[o['x'] + r, o['y'] + c] = value
    return frame


def axis_selected(state):
    return any(0 in row for o in state if o['type'] == 'wall'
               for row in o.get('pixels', []))


def parse_scene(state):
    frame = read_pixels(state)
    axis = next(r // 3 for (r, c), v in frame.items() if v == 10)
    blocks = {(r // 3, c // 3) for (r, c), v in frame.items() if v == 5}
    selected = set()
    wall_selected = axis_selected(state)
    if not wall_selected:
        selected = {p for p in blocks if frame.get((3*p[0]+1, 3*p[1]+1)) == 0}
        ambiguous = {p for p in blocks
                     if frame.get((3*p[0]+1, 3*p[1]+1), -1) not in (-1, 0, 5)}
        for group in components(selected | ambiguous):
            if group & selected:
                selected |= group
    pieces = []
    for group in components(blocks):
        if group & selected and group - selected:
            pieces.extend(components(group & selected))
            pieces.extend(components(group - selected))
        else:
            pieces.append(group)
    pieces.sort(key=lambda p: (-len(p), min(p)))
    selection = -1 if wall_selected else None
    for i, p in enumerate(pieces):
        if p & selected:
            selection = i
    if selection is None:
        for i, p in enumerate(pieces):
            if all(frame.get((3*r+1, 3*c+1), -1) not in (-1, 0, 5)
                   for r, c in p):
                selection = i
                break
    goals = components({(r // 3, c // 3)
                        for (r, c), v in frame.items() if v == 11})
    return axis, pieces, goals, selection


def within_board(blocks):
    return all(0 <= r < 21 and 0 <= c < 21 for r, c in blocks)


def piece_collision(candidate, pieces, index):
    return any(candidate & p for i, p in enumerate(pieces) if i != index)


def update_wall(axis, selection, action):
    if selection == -1 and action in (1, 2):
        return max(0, min(20, axis + (-1 if action == 1 else 1)))
    return axis


def update_players(pieces, selection, action):
    if selection is not None and selection >= 0 and action in (1, 2, 3, 4):
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
        candidate = {(r+dr, c+dc) for r, c in pieces[selection]}
        if within_board(candidate) and not piece_collision(candidate, pieces, selection):
            pieces[selection] = candidate
    return pieces


def update_selection(selection, pieces, action):
    if action != 5:
        return selection
    if selection is None or selection == len(pieces)-1:
        return -1
    return selection + 1


def update_reflections(pieces, axis):
    return {(2*axis-r, c) for p in pieces for r, c in p
            if r != axis and 0 <= 2*axis-r < 21}


def update_counter(counter, action, turns):
    if action in (1, 2, 3, 4, 5):
        turns += 1
    out = deepcopy(counter)
    out['h'] = max(0, min(64, 128-turns))
    out['y'] = 64-out['h']
    return out, turns


def render_scene(axis, pieces, goals, selection):
    stacks = {}
    def sprite(blocks, layer, ring, solid, fill):
        for p in blocks:
            stacks.setdefault(p, []).append((layer, ring, solid, fill))
    sprite({(axis, c) for c in range(21)}, 1, 10,
           0 if selection == -1 else None, 0 if selection == -1 else -1)
    for goal in goals:
        sprite(goal, 2, 11, 11, 11)
    sprite(update_reflections(pieces, axis), 3, 4, None, 4)
    for i, piece in enumerate(pieces):
        sprite(piece, 4, 5, None, 0 if selection == i else -1)
    frame = {}
    for (r, c), stack in stacks.items():
        stack.sort(key=lambda item: item[0], reverse=True)
        center = next((s[2] for s in stack if s[2] is not None), stack[0][3])
        for dr in range(3):
            for dc in range(3):
                value = center if (dr, dc) == (1, 1) else stack[0][1]
                if value >= 0:
                    frame[3*r+dr, 3*c+dc] = value
    return frame


def make_object(cells, frame, kind, name, layer, tags):
    x, y = min(r for r, c in cells), min(c for r, c in cells)
    w, h = max(r for r, c in cells)-x+1, max(c for r, c in cells)-y+1
    pixels = [[-1]*h for _ in range(w)]
    for r, c in cells:
        pixels[r-x][c-y] = frame[r, c]
    return dict(name=name, type=kind, tags=tags, layer=layer,
                x=x, y=y, w=w, h=h, pixels=pixels)


def extract_scene(frame, goals, counter, state):
    output = [counter]
    zeros = {p for p, v in frame.items() if v == 0}
    specifications = [('wall', 10, 'wall_h', 1, ['axis', 'horizontal']),
                      ('player', 5, 'piece', 4, ['movable', 'black']),
                      ('reflection', 4, 'reflection', 3, ['mirror', 'gray'])]
    for kind, value, prefix, layer, tags in specifications:
        template = next((o for o in state if o['type'] == kind), None)
        if template:
            tags = template.get('tags', tags)
        own = {p for p, v in frame.items() if v == value}
        groups = components(own | zeros if kind in ('player', 'wall') else own)
        offset = int(kind == 'player' and counter['h'] < 64)
        for i, cells in enumerate(groups):
            if cells & own:
                output.append(make_object(cells, frame, kind, f'{prefix}_{i+offset}',
                                          layer, tags))
    target_objects = []
    for goal in goals:
        cells = {p for p, v in frame.items()
                 if v == 11 and (p[0]//3, p[1]//3) in goal}
        if cells:
            target_objects.append(make_object(cells, frame, 'target', '', 2,
                                               ['goal', 'yellow']))
    target_objects.sort(key=lambda o: (o['x'], o['y']))
    for i, obj in enumerate(target_objects):
        obj['name'] = f'target_{i}'
    output.extend(target_objects)
    output.extend(deepcopy(o) for o in state
                  if o['type'] not in ('wall', 'player', 'target', 'reflection', 'counter'))
    return output


def transition_function(state, action):
    global _last, _turns, _history
    action_id = action.get('action_id') if isinstance(action, dict) else action
    counter = next(o for o in state if o['type'] == 'counter')
    if signature(state) != _last:
        _turns = 0 if counter['h'] == 64 else 128-counter['h']
        _history = []
    axis, pieces, goals, selection = parse_scene(state)
    before = (axis, deepcopy(pieces))
    if action_id == 7:
        if _history:
            axis, pieces = _history.pop()
    else:
        axis = update_wall(axis, selection, action_id)
        pieces = update_players(pieces, selection, action_id)
        selection = update_selection(selection, pieces, action_id)
        if before != (axis, pieces):
            _history.append(before)
    counter, _turns = update_counter(counter, action_id, _turns)
    result = extract_scene(render_scene(axis, pieces, goals, selection),
                           goals, counter, state)
    _last = signature(result)
    return result
