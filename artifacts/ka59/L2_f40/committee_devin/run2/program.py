# Arrows move the active token by one three-cell lattice step when unobstructed.
# Contact slides a chain of black tokens to a wall without moving the player.
# Clicking a black token makes it the player and removes the old active token.
# Targets and portal metadata stay fixed; black tokens are named in row order.
# Internal walls are inferred from chamber/socket geometry, not exposed pixels.
from copy import deepcopy
from math import gcd


def overlaps(a, b):
    return (a['x'] < b['x'] + b['w'] and b['x'] < a['x'] + a['w']
            and a['y'] < b['y'] + b['h'] and b['y'] < a['y'] + a['h'])


def translated(obj, dx, dy):
    result = dict(obj)
    result['x'] += dx
    result['y'] += dy
    return result


def lattice_step(state):
    step = 0
    for obj in state:
        if obj['type'] in ('player', 'block'):
            step = gcd(step, gcd(obj['w'], obj['h']))
    return step or 3


def portal_geometry(state, step):
    portal = next((o for o in state if o['type'] == 'portal'), None)
    if portal is None:
        return None, []
    x, y, w, h = (portal[k] for k in ('x', 'y', 'w', 'h'))
    interior = (x + step, y + step, x + w - step, y + h - step)
    mid_x = x + (w // step // 2) * step
    mid_y = y + (h // step // 2) * step
    walls = [{'x': mid_x, 'y': mid_y - step,
              'w': interior[2] - mid_x, 'h': step}]
    for socket in state:
        if socket['type'] != 'target':
            continue
        if socket['x'] >= mid_x or socket['y'] < mid_y:
            continue
        mouth_x = socket['x'] + socket['w'] - 1 + step
        mouth_y = socket['y'] + socket['h'] - 1
        walls.append({'x': interior[0], 'y': mouth_y,
                      'w': max(0, mouth_x - interior[0]),
                      'h': max(0, interior[3] - mouth_y)})
        walls.append({'x': mid_x - step, 'y': mouth_y + step,
                      'w': step,
                      'h': max(0, interior[3] - mouth_y - step)})
    return interior, walls


def blocked_by_portal(obj, geometry):
    interior, walls = geometry
    if interior is None:
        return False
    left, top, right, bottom = interior
    if (obj['x'] < left or obj['y'] < top
            or obj['x'] + obj['w'] > right
            or obj['y'] + obj['h'] > bottom):
        return True
    return any(w['w'] > 0 and w['h'] > 0 and overlaps(obj, w)
               for w in walls)


def push_chain(blocks, roots, player, dx, dy, geometry):
    moving = set(roots)
    while True:
        growing = set(moving)
        pending = list(moving)
        blocked = False
        while pending:
            i = pending.pop()
            dest = translated(blocks[i], dx, dy)
            if blocked_by_portal(dest, geometry) or overlaps(dest, player):
                blocked = True
                break
            for j, other in enumerate(blocks):
                if j not in growing and overlaps(dest, other):
                    growing.add(j)
                    pending.append(j)
        if blocked:
            return
        for i in growing:
            blocks[i]['x'] += dx
            blocks[i]['y'] += dy
        moving = growing


def update_player(player, blocks, action, step, geometry):
    direction = {1: (0, -step), 2: (0, step),
                 3: (-step, 0), 4: (step, 0)}.get(action)
    if direction is None:
        return
    dx, dy = direction
    destination = translated(player, dx, dy)
    if blocked_by_portal(destination, geometry):
        return
    touching = [i for i, block in enumerate(blocks)
                if overlaps(destination, block)]
    if touching:
        push_chain(blocks, touching, player, dx, dy, geometry)
    else:
        player['x'] += dx
        player['y'] += dy


def select_block(player, blocks, click):
    for i, block in enumerate(blocks):
        if (block['x'] <= click['x'] < block['x'] + block['w']
                and block['y'] <= click['y'] < block['y'] + block['h']):
            for key in ('x', 'y', 'w', 'h'):
                player[key] = block[key]
            blocks.pop(i)
            return


def update_blocks(blocks):
    for i, block in enumerate(sorted(blocks, key=lambda b: (b['y'], b['x']))):
        block['name'] = 'token_' + str(i)


def update_target(target):
    return target


def update_portal(portal):
    return portal


def transition_function(state, action):
    result = deepcopy(state)
    player = next((o for o in result if o['type'] == 'player'), None)
    blocks = [o for o in result if o['type'] == 'block']
    if player is None:
        return result
    step = lattice_step(result)
    geometry = portal_geometry(result, step)
    action_id = action.get('action_id') if isinstance(action, dict) else action
    if action_id == 6 and isinstance(action, dict):
        select_block(player, blocks, action)
    else:
        update_player(player, blocks, action_id, step, geometry)
    update_blocks(blocks)
    output = [player] + blocks
    for obj in result:
        if obj['type'] == 'target':
            output.append(update_target(obj))
        elif obj['type'] == 'portal':
            output.append(update_portal(obj))
        elif obj['type'] not in ('player', 'block'):
            output.append(obj)
    return output
