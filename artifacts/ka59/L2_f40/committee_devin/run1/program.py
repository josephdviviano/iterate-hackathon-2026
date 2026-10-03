# Mechanics: grid step 3. Player (white token) moves 3 px per A1-A4 (up/down/left/right) inside the floor;
# a move into a block launches it: the block slides 3 px per tick until blocked, shoving any block in front
# (chain); the player stays put. Blocks are stopped by walls and by socket interiors (bbox minus 1-px rim)
# unless they fit the interior exactly (then they drop in). A6 click on a block: player takes that block's
# bbox, block removed. Blocks renamed token_i by (y, x). Hypothesis: floor = room x30-59,y30-59 + corridor y42-50.
import copy

STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
FLOOR = [(30, 30, 59, 59), (0, 42, 29, 50)]  # (x0, y0, x1, y1) inclusive; walls elsewhere


def cells(r):
    x, y, w, h = r
    return [(x + i, y + j) for i in range(w) for j in range(h)]


def on_floor(r):
    return all(any(a <= cx <= c and b <= cy <= d for a, b, c, d in FLOOR) for cx, cy in cells(r))


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o['x'], o['y'], o['w'], o['h'])


def shift(r, dx, dy):
    return (r[0] + dx * STEP, r[1] + dy * STEP, r[2], r[3])


def socket_inner(s):
    return (s['x'] + 1, s['y'] + 1, s['w'] - 2, s['h'] - 2)


def socket_blocks(r, sockets):
    for s in sockets:
        inner = socket_inner(s)
        if overlap(r, inner) and r != inner:
            return True
    return False


def push_chain(i, blocks, sockets, dx, dy):
    """Return the set of block indices that move if block i moves one step, or None if blocked."""
    moving, todo = set(), [i]
    while todo:
        k = todo.pop()
        if k in moving:
            continue
        moving.add(k)
        nr = shift(blocks[k], dx, dy)
        if not on_floor(nr) or socket_blocks(nr, sockets):
            return None
        for j, b in enumerate(blocks):
            if j not in moving and overlap(nr, b):
                todo.append(j)
    return moving


def slide(i, blocks, sockets, dx, dy):
    for _ in range(64):
        moving = push_chain(i, blocks, sockets, dx, dy)
        if moving is None:
            break
        for k in moving:
            blocks[k] = shift(blocks[k], dx, dy)
        if any(blocks[k] == socket_inner(s) for k in moving for s in sockets):
            break
    return blocks


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o['type'] == 'player'), None)
    block_objs = [o for o in state if o['type'] == 'block']
    sockets = [o for o in state if o['type'] == 'target']
    others = [o for o in state if o['type'] not in ('player', 'block')]
    blocks = [rect(o) for o in block_objs]
    if player is None:
        return state
    pr = rect(player)
    if isinstance(action, dict):
        if action.get('action_id') == 6:
            cx, cy = action['x'], action['y']
            hit = [k for k, b in enumerate(blocks) if overlap((cx, cy, 1, 1), b)]
            if hit:
                pr = blocks[hit[0]]
                del blocks[hit[0]]
    elif action in DIRS:
        dx, dy = DIRS[action]
        nr = shift(pr, dx, dy)
        hit = [k for k, b in enumerate(blocks) if overlap(nr, b)]
        if hit:
            blocks = slide(hit[0], blocks, sockets, dx, dy)
        elif on_floor(nr):
            pr = nr
    template = block_objs[0] if block_objs else None
    out = []
    p = dict(player)
    p['x'], p['y'], p['w'], p['h'] = pr
    out.append(p)
    for n, b in enumerate(sorted(blocks, key=lambda r: (r[1], r[0]))):
        o = dict(template) if template else {}
        o['tags'] = list(template['tags'])
        o['name'] = 'token_%d' % n
        o['x'], o['y'], o['w'], o['h'] = b
        out.append(o)
    return out + others
