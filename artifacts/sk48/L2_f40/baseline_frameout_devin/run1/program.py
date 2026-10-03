# The ring moves vertically; the beam extends/retracts in ring-sized steps.
# Beam-carried blocks and newly contacted blocks move as collision chains.
# Vertical obstruction cancels motion; horizontal obstruction slides the beam.
# The next block alone at the tip is collected and marked in the HUD.
# HUD ticks every three arrows; initial full-bar reserve and actions 5/7 are unconfirmed.
from copy import deepcopy

_last_frame = None
_last_state = None
_meter_remaining = 4


def rectangle(o):
    return o['x'], o['y'], o['w'], o['h']


def touches(a, b):
    x, y, w, h = a
    u, v, s, t = b
    return x < u+s and u < x+w and y < v+t and v < y+h


def shifted(o, dx, dy):
    return o['x']+dx, o['y']+dy, o['w'], o['h']


def on_beam(block, arm):
    return touches(rectangle(block), rectangle(arm))


def hooked(block, arm):
    return (on_beam(block, arm) and
            block['x']+block['w'] == arm['x']+arm['w']-1 and
            any(t in block['tags'] for t in ('next', 'collected')))


def push_chain(blocks, movers, solids, dx, dy):
    movers = set(movers)
    while True:
        occupied = solids + [shifted(blocks[i], dx, dy) for i in movers]
        added = {i for i, b in enumerate(blocks) if i not in movers
                 and any(touches(rectangle(b), r) for r in occupied)}
        if not added:
            return movers
        movers |= added


def valid_chain(blocks, movers, dx, dy, ring, floor):
    left, top, right, bottom = floor
    for i in movers:
        x, y, w, h = shifted(blocks[i], dx, dy)
        if x < left or x+w > right or y < top or y+h > bottom:
            return False
        if touches((x, y, w, h), rectangle(ring)):
            return False
    return True


def move_blocks(blocks, movers, dx, dy):
    for i in movers:
        blocks[i]['x'] += dx
        blocks[i]['y'] += dy


def update_player(ring, arm, blocks, action, floor):
    if action not in (1, 2):
        return
    dy = ring['h'] * (-1 if action == 1 else 1)
    candidate_ring = dict(ring, y=ring['y']+dy)
    candidate_arm = dict(arm, y=arm['y']+dy)
    if not floor[1] <= candidate_ring['y'] <= floor[3]-ring['h']:
        return
    carried = [i for i, b in enumerate(blocks) if on_beam(b, arm)]
    movers = push_chain(blocks, carried,
                        [rectangle(candidate_ring), rectangle(candidate_arm)], 0, dy)
    if valid_chain(blocks, movers, 0, dy, candidate_ring, floor):
        ring['y'] += dy
        arm['y'] += dy
        move_blocks(blocks, movers, 0, dy)


def update_arm(ring, arm, blocks, action, floor):
    if action not in (3, 4):
        return
    dx = ring['w'] * (-1 if action == 3 else 1)
    new_width = arm['w']+dx
    if new_width < 1 or arm['x']+new_width > floor[2]:
        return
    if any(hooked(b, arm) for b in blocks):
        movers = [i for i, b in enumerate(blocks) if on_beam(b, arm)]
        solids = []
    elif dx > 0:
        movers = []
        solids = [(arm['x']+arm['w'], arm['y'], dx, arm['h'])]
    else:
        movers, solids = [], []
    movers = push_chain(blocks, movers, solids, dx, 0)
    if valid_chain(blocks, movers, dx, 0, ring, floor):
        move_blocks(blocks, movers, dx, 0)
    arm['w'] = new_width
    arm['pixels'] = [[2 if (x+y-1) % 3 == 0 else 1
                      for x in range(new_width)] for y in range(arm['h'])]


def update_collection(blocks, arm):
    was_collected = [b for b in blocks if 'collected' in b['tags']]
    was_next = [b for b in blocks if 'next' in b['tags']]
    carried = [b for b in blocks if on_beam(b, arm)]
    collected = [b for b in was_collected if hooked(b, arm)]
    if len(carried) == 1 and hooked(carried[0], arm):
        if carried[0] not in collected:
            collected.append(carried[0])
    remaining = [b for b in blocks if b not in collected]
    priorities = [b for b in was_collected+was_next if b in remaining]
    next_block = (priorities[0] if priorities else
                  max(remaining, key=lambda b: b['x']) if remaining else None)
    for b in blocks:
        status = 'collected' if b in collected else 'next' if b is next_block else 'todo'
        b['tags'] = [t for t in b['tags'] if t not in ('next', 'todo', 'collected')]+[status]


def meter_row(frame):
    for y, row in enumerate(frame):
        if len(set(row)) <= 2 and 2 in row and all(c in (2, 3) for c in row):
            return y
    return len(frame)


def floor_bounds(frame, hud):
    cells = [(x, y) for y in range(hud) for x, c in enumerate(frame[y]) if c == 4]
    return (min(x for x, y in cells), min(y for x, y in cells),
            max(x for x, y in cells)+1, max(y for x, y in cells)+1)


def restore_terrain(frame, state, ring, floor):
    left, top, right, bottom = floor
    result = [row[:] for row in frame]
    rail_left = ring['x']+ring['w']//2-1
    for o in state:
        if o['type'] not in ('player', 'arm', 'block'):
            continue
        for y in range(o['y'], o['y']+o['h']):
            for x in range(o['x'], o['x']+o['w']):
                if not (0 <= y < len(result) and 0 <= x < len(result[y])):
                    continue
                color = 4 if left <= x < right and top <= y < bottom else 5
                if rail_left <= x < rail_left+2 and top+2 <= y < bottom-2:
                    color = 2 if (y-top-2) % ring['h'] < 2 else 3
                result[y][x] = color
    return result


def draw_object(result, o):
    for dy, row in enumerate(o['pixels']):
        for dx, color in enumerate(row):
            x, y = o['x']+dx, o['y']+dy
            if 0 <= y < len(result) and 0 <= x < len(result[y]) and color >= 0:
                result[y][x] = color


def draw_collection(result, blocks, hud):
    for b in blocks:
        color = next(c for row in b['pixels'] for c in row if c not in (0, -1))
        cells = [(x, y) for y in range(hud+1, len(result))
                 for x, c in enumerate(result[y]) if c == color]
        if not cells:
            continue
        left, top = min(x for x, y in cells), min(y for x, y in cells)
        right, bottom = max(x for x, y in cells), max(y for x, y in cells)
        fill = 0 if 'collected' in b['tags'] else color
        for y in range(top+1, bottom):
            for x in range(left+1, right):
                result[y][x] = fill


def transition_function(state, action, frame):
    global _last_frame, _last_state, _meter_remaining
    aid = action.get('action_id') if isinstance(action, dict) else action
    hud = meter_row(frame)
    continuous = (frame == _last_frame and _last_state is not None and
                  sorted(state, key=lambda o: o['name']) ==
                  sorted(_last_state, key=lambda o: o['name']))
    if not continuous:
        _meter_remaining = 4 if hud < len(frame) and 3 not in frame[hud] else 3
    objects = deepcopy(state)
    ring = next((o for o in objects if o['type'] == 'player'), None)
    arm = next((o for o in objects if o['type'] == 'arm'), None)
    if ring is None or arm is None:
        return [row[:] for row in frame]
    blocks = [o for o in objects if o['type'] == 'block']
    floor = floor_bounds(frame, hud)
    result = restore_terrain(frame, state, ring, floor)
    update_player(ring, arm, blocks, aid, floor)
    update_arm(ring, arm, blocks, aid, floor)
    update_collection(blocks, arm)
    for o in sorted(objects, key=lambda o: o.get('layer', 0)):
        if o['type'] in ('player', 'arm', 'block'):
            draw_object(result, o)
    draw_collection(result, blocks, hud)
    if aid in (1, 2, 3, 4) and hud < len(frame):
        _meter_remaining -= 1
        if _meter_remaining == 0:
            remaining = [x for x, c in enumerate(result[hud]) if c == 2]
            if remaining:
                result[hud][max(remaining)] = 3
            _meter_remaining = 3
    _last_state = deepcopy(objects)
    _last_frame = [row[:] for row in result]
    return result
