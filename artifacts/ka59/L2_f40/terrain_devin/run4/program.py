# Mechanics: arrows (1 up,2 down,3 left,4 right) move the player 3 px within an invisible FLOOR
# (room x30-59,y30-59 + corridor x6-29,y42-50, per-pixel coverage). If the player's next rect hits a
# block it kicks it instead (player stays): the push-chain group slides 3 px per tick until any member
# would leave the floor, hit the player, or enter a socket interior (bbox inset 1) of a different size.
# Click on a block: player takes its rect, old player vanishes; blocks renamed token_i by (y,x). Unconfirmed: fitting socket entry, A5/A7.
import copy

FLOOR = [(30, 30, 59, 59), (6, 42, 29, 50)]
STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shifted(r, d):
    return (r[0] + d[0] * STEP, r[1] + d[1] * STEP, r[2], r[3])


def on_floor(r):
    for px in range(r[0], r[0] + r[2]):
        for py in range(r[1], r[1] + r[3]):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_blocks(r, sockets):
    for s in sockets:
        inner = socket_interior(s)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def push_chain(start, blocks, d):
    group = {start}
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if i in group:
                continue
            if any(overlaps(shifted(rect(blocks[j]), d), rect(blocks[i])) for j in group):
                group.add(i)
                changed = True
    return group


def block_blocked(r, player, sockets):
    return not on_floor(r) or overlaps(r, rect(player)) or socket_blocks(r, sockets)


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
        blocks.remove(b)
        rename_blocks(blocks)
        return [player] + blocks + others
    d = DIRS.get(action)
    if d is None:
        return state
    nxt = shifted(rect(player), d)
    hit = [i for i, b in enumerate(blocks) if overlaps(nxt, rect(b))]
    if hit:
        moved = False
        for i in hit:
            moved = kick(i, blocks, player, sockets, d) or moved
        if moved:
            rename_blocks(blocks)
        return [player] + blocks + others
    if on_floor(nxt):
        player["x"], player["y"] = nxt[0], nxt[1]
    return [player] + blocks + others
