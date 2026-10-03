# Mechanics: arrows (A1-4 = up/down/left/right) move the player 3px inside an invisible FLOOR
# (room x30-59,y30-59 + corridor x6-29,y42-50, per-pixel coverage). If the player's next rect hits a
# block, the block is kicked instead (player stays) and slides 3px/tick, chaining blocks it hits, until
# any member would leave FLOOR, hit the player, or enter a non-fitting socket interior. A6 click on a
# block: player takes that block's rect, block consumed; blocks re-ranked token_i by (y,x). Stateless.
import copy

STEP = 3
FLOOR = [(30, 30, 59, 59), (6, 42, 29, 50)]  # inclusive (x0, y0, x1, y1); unconfirmed beyond observed paths
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shifted(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    for x in range(r[0], r[0] + r[2]):
        for y in range(r[1], r[1] + r[3]):
            if not any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_rejects(r, sockets):
    """Guard: a block may only overlap a socket interior that exactly fits its size."""
    for s in sockets:
        inner = socket_interior(s)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, player_r, sockets):
    return not on_floor(r) or overlaps(r, player_r) or socket_rejects(r, sockets)


def push_chain(group, blocks, dx, dy):
    """Closure of blocks hit by the shifted rects of the group."""
    group = set(group)
    changed = True
    while changed:
        changed = False
        for i, b in enumerate(blocks):
            if i in group:
                continue
            if any(overlaps(shifted(rect(blocks[j]), dx, dy), rect(b)) for j in group):
                group.add(i)
                changed = True
    return group


def kick(blocks, first, dx, dy, player_r, sockets):
    group = {first}
    moved = False
    while True:
        group = push_chain(group, blocks, dx, dy)
        if any(block_blocked(shifted(rect(blocks[i]), dx, dy), player_r, sockets) for i in group):
            return moved
        for i in group:
            blocks[i]["x"] += dx
            blocks[i]["y"] += dy
        moved = True


def rename_blocks(blocks):
    for k, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % k


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if player is None:
        return state

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        hit = [b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]]
        if not hit:
            return state
        b = hit[0]
        for k in ("x", "y", "w", "h"):
            player[k] = b[k]
        blocks = [o for o in blocks if o is not b]
        rename_blocks(blocks)
        return [player] + blocks + others

    if action not in DIRS:
        return state
    ux, uy = DIRS[action]
    dx, dy = ux * STEP, uy * STEP
    nxt = shifted(rect(player), dx, dy)
    hit = [i for i, b in enumerate(blocks) if overlaps(nxt, rect(b))]
    if hit:
        kick(blocks, hit[0], dx, dy, rect(player), sockets)
        rename_blocks(blocks)
    elif on_floor(nxt):
        player["x"] += dx
        player["y"] += dy
    return [player] + blocks + others
