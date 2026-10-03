# Arrow keys move cyan blocks by four cells, with mirrored horizontal motion.
# Clicking a marker arms both players and selects the marker for arrow motion.
# Clicking a player restores block mode; markers and players obstruct motion.
# Timer bars grow at three pixels per seven actions; naming follows scan order.
# Three occluded maze cells are inferred; other walls and reset phases are unknown.
from copy import deepcopy
import json

_last_result = None
_elapsed = 0
_hidden_walls = frozenset(((14, 38), (6, 18), (38, 14)))


def canonical(state):
    return tuple(sorted(json.dumps(o, sort_keys=True) for o in state))


def is_timer(obj):
    return obj['type'] == 'wall' and 'color_0' in obj['tags']


def is_active(obj):
    return obj['type'] == 'marker' and 'active' in obj['tags']


def contains(obj, x, y):
    return obj['x'] <= x < obj['x'] + obj['w'] and obj['y'] <= y < obj['y'] + obj['h']


def overlaps(a, b):
    return (a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
            and a[1] < b[1] + b[3] and b[1] < a[1] + a[3])


def tile(obj):
    if obj['type'] == 'marker':
        return obj['x'] - 1, obj['y'] - 1, 4, 4
    return obj['x'], obj['y'], obj['w'], obj['h']


def blocked_destination(obj, dx, dy, actors):
    x, y, w, h = tile(obj)
    dest = x + dx, y + dy, w, h
    if not (2 <= dest[0] <= 58 and 2 <= dest[1] <= 58):
        return True
    if dest[:2] in _hidden_walls:
        return True
    return any(other is not obj and overlaps(dest, tile(other)) for other in actors)


def update_player(obj, context):
    result = deepcopy(obj)
    if context['mode'] == 'marker':
        result.pop('pixels', None)
        result['tags'] = ['color_1', 'armed', 'player']
    else:
        result['tags'] = ['cyan', 'block', 'player']
        result['pixels'] = [[10] * result['w'] for _ in range(result['h'])]
    if context['action'] in (1, 2, 3, 4) and context['mode'] == 'block':
        dx, dy = context['delta']
        if obj['x'] >= 32:
            dx = -dx
        if not blocked_destination(obj, dx, dy, context['actors']):
            result['x'] += dx
            result['y'] += dy
    return result


def update_marker(obj, context):
    result = deepcopy(obj)
    active = obj is context['selected'] and context['mode'] == 'marker'
    result['tags'] = ['color_11', 'active', 'marker'] if active else ['color_9', 'inactive', 'marker']
    if active and context['action'] in (1, 2, 3, 4):
        dx, dy = context['delta']
        if not blocked_destination(obj, dx, dy, context['actors']):
            result['x'] += dx
            result['y'] += dy
    return result


def update_wall(obj, context):
    return None if is_timer(obj) else deepcopy(obj)


def update_hazard(obj, context):
    return deepcopy(obj)


def timer_bars(width):
    if not width:
        return []
    return [dict(name='', type='wall', tags=['color_0', 'wall'],
                 x=x, y=y, w=width, h=1, layer=0, visible=True)
            for x, y in ((64 - width, 0), (0, 63))]


def assign_names(state, mode):
    blocks = [o for o in state if o['type'] == 'player'] if mode == 'block' else []
    for obj in blocks:
        obj['name'] = 'block_left' if obj['x'] < 32 else 'block_right'
    others = [o for o in state if o not in blocks]
    for index, obj in enumerate(sorted(others, key=lambda o: (o['y'], o['x'])), len(blocks)):
        color = next(tag.split('_')[1] for tag in obj['tags'] if tag.startswith('color_'))
        obj['name'] = obj['type'] + '_' + color + '_' + str(index)
    return state


def transition_function(state, action):
    global _last_result, _elapsed
    if canonical(state) != _last_result:
        width = max((o['w'] for o in state if is_timer(o)), default=0)
        _elapsed = max(0, (7 * width + 2) // 3 - 1)
    action_id = action.get('action_id') if isinstance(action, dict) else action
    actors = [o for o in state if o['type'] in ('player', 'marker')]
    selected = next((o for o in actors if is_active(o)), None)
    mode = 'marker' if selected else 'block'
    if action_id == 6 and isinstance(action, dict):
        hit = next((o for o in actors if contains(o, action['x'], action['y'])), None)
        if hit is not None and hit['type'] == 'marker':
            selected, mode = hit, 'marker'
        elif hit is not None and hit['type'] == 'player':
            selected, mode = None, 'block'
    context = dict(action=action_id, selected=selected, mode=mode, actors=actors,
                   delta={1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(action_id, (0, 0)))
    updates = {'player': update_player, 'marker': update_marker,
               'wall': update_wall, 'hazard': update_hazard}
    result = []
    for obj in state:
        changed = updates[obj['type']](obj, context) if obj['type'] in updates else deepcopy(obj)
        if changed is not None:
            result.append(changed)
    _elapsed += 1
    result.extend(timer_bars(min(64, 3 * (_elapsed + 1) // 7)))
    assign_names(result, mode)
    _last_result = canonical(result)
    return result
