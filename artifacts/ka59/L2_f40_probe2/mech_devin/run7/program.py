# Mechanics: arrows (1 up, 2 down, 3 left, 4 right) move the player 3 cells on a hidden FLOOR.
# If the player's next rect overlaps a block it kicks it: the player stays and the block slides
# 3/step, pushing every block it meets (chain closure), until any member would leave the floor,
# overlap the player, or overlap a non-fitting socket interior. Click on a block: it becomes the player.
# Hypothesis (unconfirmed): FLOOR = 4 undrawn rects; fitting-socket entry and actions 5/7 unobserved.
import copy

STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
# Undrawn bounds of the playable area, inclusive (x0, y0, x1, y1):
FLOOR = [(30, 30, 59, 59),   # main room
         (15, 42, 29, 50),   # lower corridor
         (6, 42, 14, 47),    # neck to socket_44_8
         (18, 30, 29, 35)]   # upper corridor (step-78 stop at x=18)


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shifted(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def on_floor(r):
    for px in range(r[0], r[0] + r[2]):
        for py in range(r[1], r[1] + r[3]):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


# Guard: block vs socket — a block may not overlap a socket interior it does not fit.
def socket_blocks(r, sockets):
    for s in sockets:
        inner = socket_interior(s)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


# Guard: block vs floor/player/socket.
def block_blocked(r, player_r, sockets):
    return (not on_floor(r)) or overlaps(r, player_r) or socket_blocks(r, sockets)


# Interaction: block pushes block — closure of blocks hit by the shifted group.
def push_chain(start, blocks, d):
    group = {start}
    changed = True
    while changed:
        changed = False
        for i, b in enumerate(blocks):
            if i in group:
                continue
            if any(overlaps(shifted(rect(blocks[j]), d), rect(b)) for j in group):
                group.add(i)
                changed = True
    return group


# Interaction: player kicks block — slide the chain until blocked.
def kick(start, blocks, d, player_r, sockets):
    moved = False
    while True:
        group = push_chain(start, blocks, d)
        if any(block_blocked(shifted(rect(blocks[i]), d), player_r, sockets) for i in group):
            return moved
        for i in group:
            blocks[i]["x"] += d[0]
            blocks[i]["y"] += d[1]
        moved = True


def rename_blocks(blocks):
    for k, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % k


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
        player.update(x=b["x"], y=b["y"], w=b["w"], h=b["h"])
        blocks = [o for o in blocks if o is not b]
        rename_blocks(blocks)
        return [player] + blocks + others
    if action not in DIRS:
        return state
    d = DIRS[action]
    pr = rect(player)
    nxt = shifted(pr, d)
    hit = [i for i, b in enumerate(blocks) if overlaps(nxt, rect(b))]
    if hit:
        kick(hit[0], blocks, d, pr, sockets)
        rename_blocks(blocks)
    elif on_floor(nxt):
        player["x"], player["y"] = nxt[0], nxt[1]
    return [player] + blocks + others
