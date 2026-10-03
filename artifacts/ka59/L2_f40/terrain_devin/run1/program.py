# Mechanics: arrows (1 up,2 down,3 left,4 right) move the player 3px inside a hidden FLOOR (room + west corridor).
# If the player's next rect overlaps a block it kicks it instead (player stays): the block slides 3px per tick,
# pushing every block its shifted rect overlaps (chain closure) until any member would leave the floor, overlap
# the player, or enter a socket interior it does not fit. Click on a block: it becomes the player (takes its rect).
# Blocks re-ranked token_i by (y,x). Terrain/portal/sockets static. Unconfirmed: floor extent beyond observed stops.
import copy

STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
FLOOR = [(30, 30, 30, 30), (6, 42, 24, 9)]  # (x, y, w, h)


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    x, y, w, h = r
    for px in range(x, x + w):
        for py in range(y, y + h):
            if not any(fx <= px < fx + fw and fy <= py < fy + fh for fx, fy, fw, fh in FLOOR):
                return False
    return True


def interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_blocks(r, sockets):
    for s in sockets:
        i = interior(s)
        if overlap(r, i) and (r[2], r[3]) != (i[2], i[3]):
            return True
    return False


def shifted(r, d):
    return (r[0] + d[0] * STEP, r[1] + d[1] * STEP, r[2], r[3])


def push_chain(start, blocks, d):
    group = {start}
    changed = True
    while changed:
        changed = False
        for i in list(group):
            nr = shifted(rect(blocks[i]), d)
            for j, b in enumerate(blocks):
                if j not in group and overlap(nr, rect(b)):
                    group.add(j)
                    changed = True
    return group


def block_blocked(r, player, sockets):
    return not on_floor(r) or overlap(r, rect(player)) or socket_blocks(r, sockets)


def kick(start, blocks, player, sockets, d):
    moved = False
    while True:
        group = push_chain(start, blocks, d)
        if any(block_blocked(shifted(rect(blocks[i]), d), player, sockets) for i in group):
            return moved
        for i in group:
            blocks[i]["x"] += d[0] * STEP
            blocks[i]["y"] += d[1] * STEP
        moved = True


def rename(blocks):
    blocks.sort(key=lambda b: (b["y"], b["x"]))
    for i, b in enumerate(blocks):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    players = [o for o in state if o["type"] == "player"]
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if not players:
        return state
    player = players[0]
    if isinstance(action, dict):
        if action.get("action_id") != 6:
            return state
        cx, cy = action["x"], action["y"]
        hit = [b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]]
        if not hit:
            return state
        b = hit[0]
        for k in ("x", "y", "w", "h"):
            player[k] = b[k]
        blocks.remove(b)
        rename(blocks)
        return [player] + blocks + others
    if action not in DIRS:
        return state
    d = DIRS[action]
    nr = shifted(rect(player), d)
    hit = [i for i, b in enumerate(blocks) if overlap(nr, rect(b))]
    if hit:
        kick(hit[0], blocks, player, sockets, d)
    elif on_floor(nr):
        player["x"], player["y"] = nr[0], nr[1]
    rename(blocks)
    return [player] + blocks + others
