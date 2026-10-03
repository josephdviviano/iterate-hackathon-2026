# Blocks move four cells vertically or horizontally in mirrored directions.
# Clicking inactive markers arms players; active clicks do nothing; player clicks disarm.
# Armed movement steers only the active marker; occupied cells and maze walls block it.
# Timer bars advance every action; exact-state continuity preserves their hidden phase.
# Three level maze cells are inferred; unseen walls and ACTION7 effects are unconfirmed.
from copy import deepcopy
import json


_last_state = None
_age = 0
_maze_cells = frozenset({(14, 38), (6, 18), (38, 14)})
_directions = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}


def _canonical(state):
    return tuple(sorted(json.dumps(o, sort_keys=True) for o in state))


def _is_active(marker):
    return 'active' in marker['tags']


def _is_armed(player):
    return 'armed' in player['tags']


def _is_timer(obj):
    return obj['type'] == 'wall' and 'color_0' in obj['tags'] and obj['h'] == 1


def _click_hits(obj, action):
    return (isinstance(action, dict) and action.get('action_id') == 6
            and obj.get('visible', True)
            and obj['x'] <= action['x'] < obj['x'] + obj['w']
            and obj['y'] <= action['y'] < obj['y'] + obj['h'])


def _selection(players, markers, action):
    armed = any(_is_armed(p) for p in players)
    active = next((i for i, m in enumerate(markers) if _is_active(m)), None)
    clicked = next((i for i, m in enumerate(markers) if _click_hits(m, action)), None)
    if clicked is not None:
        if not _is_active(markers[clicked]):
            armed, active = True, clicked
    elif armed and any(_click_hits(p, action) for p in players):
        armed, active = False, None
    return armed, active


def _marker_cell(marker):
    return marker['x'] - 1, marker['y'] - 1


def _cell_blocked(cell, occupied, xmin=2, xmax=58):
    x, y = cell
    return (not (xmin <= x <= xmax and 2 <= y <= 58)
            or cell in _maze_cells or cell in occupied)


def _update_player(player, action, was_armed, armed, markers):
    obj = deepcopy(player)
    if not was_armed and action in _directions:
        dx, dy = _directions[action]
        left = obj['x'] < 32
        if not left:
            dx = -dx
        cell = obj['x'] + dx, obj['y'] + dy
        occupied = {_marker_cell(m) for m in markers}
        if not _cell_blocked(cell, occupied, 2 if left else 34, 26 if left else 58):
            obj['x'], obj['y'] = cell
    if armed:
        obj['tags'] = ['color_1', 'armed', 'player']
        obj.pop('pixels', None)
    else:
        obj['tags'] = ['cyan', 'block', 'player']
        obj['pixels'] = [[10] * obj['w'] for _ in range(obj['h'])]
        obj['name'] = 'block_left' if obj['x'] < 32 else 'block_right'
    return obj


def _update_marker(marker, index, action, was_armed, armed, active, players, markers):
    obj = deepcopy(marker)
    if was_armed and _is_active(marker) and action in _directions:
        dx, dy = _directions[action]
        cell = obj['x'] - 1 + dx, obj['y'] - 1 + dy
        occupied = {_marker_cell(m) for i, m in enumerate(markers) if i != index}
        occupied.update((p['x'], p['y']) for p in players)
        if not _cell_blocked(cell, occupied):
            obj['x'], obj['y'] = cell[0] + 1, cell[1] + 1
    selected = armed and index == active
    obj['tags'] = ['color_11' if selected else 'color_9',
                   'active' if selected else 'inactive', 'marker']
    return obj


def _update_wall(wall):
    return deepcopy(wall)


def _update_hazard(hazard):
    return deepcopy(hazard)


def _timer_walls(width):
    if width == 0:
        return []
    return [dict(name='', type='wall', tags=['color_0', 'wall'],
                 x=x, y=y, w=width, h=1, layer=0, visible=True)
            for x, y in ((64 - width, 0), (0, 63))]


def _rename(objects, armed):
    fixed = [o for o in objects if o['type'] == 'player' and not armed]
    scanned = sorted((o for o in objects if o not in fixed),
                     key=lambda o: (o['y'], o['x']))
    for rank, obj in enumerate(scanned, len(fixed)):
        color = next(t[6:] for t in obj['tags'] if t.startswith('color_'))
        obj['name'] = '{}_{}_{}'.format(obj['type'], color, rank)
    return fixed + scanned


def transition_function(state, action):
    global _last_state, _age
    fingerprint = _canonical(state)
    if fingerprint != _last_state:
        width = max((o['w'] for o in state if _is_timer(o)), default=0)
        _age = (7 * width - 1) // 3 if width else 0
    players = [o for o in state if o['type'] == 'player']
    markers = [o for o in state if o['type'] == 'marker']
    was_armed = any(_is_armed(p) for p in players)
    armed, active = _selection(players, markers, action)
    movement = action if isinstance(action, int) else 6
    result = [_update_player(p, movement, was_armed, armed, markers) for p in players]
    result.extend(_update_marker(m, i, movement, was_armed, armed, active, players, markers)
                  for i, m in enumerate(markers))
    for obj in state:
        if obj['type'] == 'wall' and not _is_timer(obj):
            result.append(_update_wall(obj))
        elif obj['type'] == 'hazard':
            result.append(_update_hazard(obj))
        elif obj['type'] not in ('wall', 'hazard', 'player', 'marker'):
            result.append(deepcopy(obj))
    _age += 1
    result.extend(_timer_walls(min(64, 3 * (_age + 1) // 7)))
    result = _rename(result, armed)
    _last_state = _canonical(result)
    return result
