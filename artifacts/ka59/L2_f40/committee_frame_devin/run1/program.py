# Arrows move the player by three pixels, or kick a block without moving.
# Kicked blocks slide and collect contacted blocks into a rigid push chain.
# Frame colour 2 blocks all; colour 15 blocks players but not kicked blocks.
# Non-fitting socket interiors stop blocks; clicking possesses a block.
# Tokens are named in reading order; other actions and socket filling are unconfirmed.
from copy import deepcopy


def overlaps(a, b):
    return (a['x'] < b['x'] + b['w'] and b['x'] < a['x'] + a['w']
            and a['y'] < b['y'] + b['h'] and b['y'] < a['y'] + a['h'])


def translated(obj, dx, dy):
    result = dict(obj)
    result['x'] += dx
    result['y'] += dy
    return result


def contains(rect, x, y):
    return (rect['x'] <= x < rect['x'] + rect['w']
            and rect['y'] <= y < rect['y'] + rect['h'])


def terrain_blocks(rect, frame, is_player):
    solid = (2, 15) if is_player else (2,)
    for y in range(rect['y'], rect['y'] + rect['h']):
        for x in range(rect['x'], rect['x'] + rect['w']):
            if not (0 <= y < len(frame) - 1 and 0 <= x < len(frame[y])):
                return True
            if frame[y][x] in solid:
                return True
    return False


def socket_blocks(rect, targets):
    for target in targets:
        inner = dict(target, x=target['x'] + 1, y=target['y'] + 1,
                     w=target['w'] - 2, h=target['h'] - 2)
        if overlaps(rect, inner) and (rect['w'], rect['h']) != (inner['w'], inner['h']):
            return True
    return False


def directed_contact(player, blocks, dx, dy):
    destination = translated(player, dx, dy)
    return {i for i, block in enumerate(blocks) if overlaps(destination, block)}


def push_chain(blocks, moving, dx, dy):
    moving = set(moving)
    while True:
        added = {j for i in moving for j, block in enumerate(blocks)
                 if j not in moving and overlaps(translated(blocks[i], dx, dy), block)}
        if not added:
            return moving
        moving.update(added)


def block_blocked(rect, player, targets, frame):
    return (terrain_blocks(rect, frame, False) or overlaps(rect, player)
            or socket_blocks(rect, targets))


def update_blocks(blocks, moving, player, targets, frame, dx, dy):
    while moving:
        moving = push_chain(blocks, moving, dx, dy)
        destinations = {i: translated(blocks[i], dx, dy) for i in moving}
        if any(block_blocked(rect, player, targets, frame)
               for rect in destinations.values()):
            break
        for i, rect in destinations.items():
            blocks[i].update(x=rect['x'], y=rect['y'])


def update_player(player, blocks, action, frame):
    action_id = action.get('action_id') if isinstance(action, dict) else action
    if action_id == 6:
        for block in blocks:
            if contains(block, action['x'], action['y']):
                for key in ('x', 'y', 'w', 'h'):
                    player[key] = block[key]
                blocks.remove(block)
                break
        return None
    direction = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}.get(action_id)
    if direction is None:
        return None
    dx, dy = direction
    moving = directed_contact(player, blocks, dx, dy)
    if moving:
        return moving, dx, dy
    destination = translated(player, dx, dy)
    if not terrain_blocks(destination, frame, True):
        player.update(x=destination['x'], y=destination['y'])
    return None


def update_targets(targets):
    return targets


def update_portals(portals):
    return portals


def transition_function(state, action, frame):
    objects = deepcopy(state)
    player = next((o for o in objects if o['type'] == 'player'), None)
    if player is None:
        return objects
    blocks = [o for o in objects if o['type'] == 'block']
    targets = update_targets([o for o in objects if o['type'] == 'target'])
    portals = update_portals([o for o in objects if o['type'] == 'portal'])
    kicked = update_player(player, blocks, action, frame)
    if kicked is not None:
        moving, dx, dy = kicked
        update_blocks(blocks, moving, player, targets, frame, dx, dy)
    for i, block in enumerate(sorted(blocks, key=lambda o: (o['y'], o['x']))):
        block['name'] = 'token_' + str(i)
    other = [o for o in objects if o['type'] not in ('player', 'block', 'target', 'portal')]
    return [player] + blocks + targets + portals + other
