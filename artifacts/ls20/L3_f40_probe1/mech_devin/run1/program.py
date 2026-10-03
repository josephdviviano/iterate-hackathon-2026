# Arrows move the player by one 5-cell tile; inferred walls block and a portal warps.
# A pending chamber target covered by the proposed move rejects it without cost.
# Otherwise budget burns 2; covered refuel rings are consumed and refill it to 42.
# Covered lower-layer targets hide, then return; memory requires exact continuity.
# Unseen maze cells, chamber unlocking, and actions 5/6/7 are unconfirmed.
from copy import deepcopy
import json


_last_state = None
_hidden_targets = []
_maze_walls = {(39, 5), (34, 10)}
_portals = {(9, 5): (34, 5)}


def canonical(state):
    return tuple(sorted(json.dumps(o, sort_keys=True) for o in state))


def covered_by(obj, player):
    return (player['x'] <= obj['x'] and player['y'] <= obj['y']
            and obj['x'] + obj['w'] <= player['x'] + player['w']
            and obj['y'] + obj['h'] <= player['y'] + player['h'])


def chamber_entry_locked(candidate, objects):
    pending = any(o['type'] == 'counter' and 'pending' in o.get('tags', [])
                  for o in objects)
    return pending and any(o['type'] == 'target'
                           and 'chamber' in o.get('tags', [])
                           and covered_by(o, candidate) for o in objects)


def wall_blocks(candidate, objects):
    if (candidate['x'], candidate['y']) in _maze_walls:
        return True
    return any(o['type'] == 'wall'
               and candidate['x'] < o['x'] + o['w']
               and o['x'] < candidate['x'] + candidate['w']
               and candidate['y'] < o['y'] + o['h']
               and o['y'] < candidate['y'] + candidate['h'] for o in objects)


def update_player(player, action_id, objects):
    candidate = deepcopy(player)
    dx, dy = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}.get(
        action_id, (0, 0))
    candidate['x'] += dx
    candidate['y'] += dy
    if chamber_entry_locked(candidate, objects):
        return deepcopy(player), True
    if wall_blocks(candidate, objects):
        return deepcopy(player), False
    if (dx or dy) and (candidate['x'], candidate['y']) in _portals:
        candidate['x'], candidate['y'] = _portals[candidate['x'], candidate['y']]
    return candidate, False


def update_target(target, player):
    occluded = (target.get('layer', 0) < player.get('layer', 1)
                and covered_by(target, player))
    return None if occluded else target


def update_refuel(ring, player):
    return None if covered_by(ring, player) else ring


def update_counter(counter, refueled):
    if 'budget' in counter.get('tags', []):
        right = counter['x'] + counter['w']
        counter['w'] = 42 if refueled else max(0, counter['w'] - 2)
        counter['x'] = right - counter['w']
        counter['tags'] = ['budget', str(counter['w'])]
    return counter


def update_button(button, player):
    return button


def transition_function(state, action):
    global _last_state, _hidden_targets
    objects = deepcopy(state)
    if canonical(state) != _last_state:
        _hidden_targets = []
    names = {o['name'] for o in objects}
    objects.extend(deepcopy(o) for o in _hidden_targets if o['name'] not in names)
    original_player = next((o for o in objects if o['type'] == 'player'), None)
    if original_player is None:
        _hidden_targets = []
        _last_state = canonical(state)
        return deepcopy(state)
    action_id = action.get('action_id') if isinstance(action, dict) else action
    player, rejected = update_player(original_player, action_id, objects)
    if rejected:
        _last_state = canonical(state)
        return deepcopy(state)
    refueled = any(o['type'] == 'refuel' and covered_by(o, player) for o in objects)
    result, hidden = [], []
    for obj in objects:
        kind = obj['type']
        if kind == 'player':
            updated = player if obj['name'] == original_player['name'] else obj
        elif kind == 'target':
            updated = update_target(obj, player)
            if updated is None:
                hidden.append(obj)
        elif kind == 'refuel':
            updated = update_refuel(obj, player)
        elif kind == 'counter':
            updated = update_counter(obj, refueled)
        elif kind == 'button':
            updated = update_button(obj, player)
        else:
            updated = obj
        if updated is not None:
            result.append(updated)
    _hidden_targets = deepcopy(hidden)
    _last_state = canonical(result)
    return result
