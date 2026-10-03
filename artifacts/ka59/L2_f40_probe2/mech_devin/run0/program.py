# Mechanics: arrows move the player 3 px within an invisible FLOOR (room + corridors); if the player's
# next rect overlaps a block it kicks it instead (player stays): the kicked group slides 3 px/step,
# absorbing blocks its shifted rects overlap, until any member would leave the floor, hit the player or
# overlap a socket interior (bbox minus 1px border) of a different size. Click on a block: the player
# takes its rect, the block is consumed. Blocks renamed token_i by (y,x). Unconfirmed: fitting-socket entry, A5/A7.
import copy

FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]  # inclusive x0,y0,x1,y1
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 3


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    x, y, w, h = r
    for px in range(x, x + w):
        for py in range(y, y + h):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_blocks(r, sockets):
    for s in sockets:
        inner = (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)
        if overlaps(r, inner) and (inner[2], inner[3]) != (r[2], r[3]):
            return True
    return False


def block_blocked(r, player, sockets):
    return not on_floor(r) or overlaps(r, rect(player)) or socket_blocks(r, sockets)


def push_chain(group, blocks, dx, dy):
    """Closure of blocks overlapped by the shifted group; returns indices."""
    group = set(group)
    changed = True
    while changed:
        changed = False
        shifted = [rect(blocks[i], dx, dy) for i in group]
        for j, b in enumerate(blocks):
            if j not in group and any(overlaps(rect(b), s) for s in shifted):
                group.add(j)
                changed = True
    return group


def kick(start, blocks, player, sockets, dx, dy):
    moved = False
    group = set(start)
    while True:
        group = push_chain(group, blocks, dx, dy)
        if any(block_blocked(rect(blocks[i], dx, dy), player, sockets) for i in group):
            return moved
        for i in group:
            blocks[i]["x"] += dx
            blocks[i]["y"] += dy
        moved = True


def rename(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if player is None:
        return state
    if isinstance(action, dict):
        if action.get("action_id") != 6:
            return state
        cx, cy = action["x"], action["y"]
        hit = [b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]]
        if not hit:
            return state
        b = hit[0]
        player.update(x=b["x"], y=b["y"], w=b["w"], h=b["h"])
        blocks = [o for o in blocks if o is not b]
        rename(blocks)
        return [player] + blocks + others
    if action not in DIRS:
        return state
    dx, dy = DIRS[action]
    dx, dy = dx * STEP, dy * STEP
    nxt = rect(player, dx, dy)
    hit = [i for i, b in enumerate(blocks) if overlaps(rect(b), nxt)]
    if hit:
        kick(hit, blocks, player, sockets, dx, dy)
        rename(blocks)
        return [player] + blocks + others
    if on_floor(nxt):
        player["x"] += dx
        player["y"] += dy
    return [player] + blocks + others
