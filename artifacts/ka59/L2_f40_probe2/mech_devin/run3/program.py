# Mechanics: arrows (1-4) move the 3x3 player 3px within an invisible FLOOR (union of rects).
# If the player's next rect overlaps a block, the block is kicked instead (player stays): the
# kicked chain slides 3px/step, absorbing blocks its next shift overlaps, until any member leaves
# the FLOOR, hits the player, or overlaps a non-fitting socket interior. Click on a block = player
# takes its rect, block consumed. Blocks renamed token_i by (y,x). Unconfirmed: A5/A7, socket fill.
import copy

FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]
DIRS = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def on_floor(r):
    for px in range(r[0], r[0] + r[2]):
        for py in range(r[1], r[1] + r[3]):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_blocks(r, sockets):
    for s in sockets:
        inner = (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, player, sockets):
    return not on_floor(r) or overlaps(r, rect(player)) or socket_blocks(r, sockets)


def push_chain(seed, blocks, d):
    chain = [seed]
    grown = True
    while grown:
        grown = False
        for b in blocks:
            if b in chain:
                continue
            if any(overlaps(shift(rect(c), d), rect(b)) for c in chain):
                chain.append(b)
                grown = True
    return chain


def kick(seed, blocks, player, sockets, d):
    while True:
        chain = push_chain(seed, blocks, d)
        if any(block_blocked(shift(rect(c), d), player, sockets) for c in chain):
            return
        for c in chain:
            c["x"] += d[0]
            c["y"] += d[1]


def rename(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    players = [o for o in state if o["type"] == "player"]
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    if not players:
        return state
    player = players[0]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            cx, cy = action["x"], action["y"]
            for b in blocks:
                if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]:
                    for k in ("x", "y", "w", "h"):
                        player[k] = b[k]
                    state.remove(b)
                    blocks.remove(b)
                    rename(blocks)
                    break
        return state
    d = DIRS.get(action)
    if d is None:
        return state
    nr = shift(rect(player), d)
    hit = [b for b in blocks if overlaps(nr, rect(b))]
    if hit:
        kick(hit[0], blocks, player, sockets, d)
        rename(blocks)
    elif on_floor(nr):
        player["x"], player["y"] = nr[0], nr[1]
    return state
