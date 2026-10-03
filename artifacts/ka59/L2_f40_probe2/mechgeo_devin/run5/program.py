# Mechanics: arrows move the player 3 cells (A1 up, A2 down, A3 left, A4 right) inside FLOOR; a
# player step that would overlap a block kicks it instead: the block slides 3/tick (push chain grows
# when it would overlap another block) until any member leaves FLOOR, hits the player or enters a
# socket interior (bbox minus 1px rim) of a different size; player stays. Click on a block = the
# player takes that block's rect, block removed; blocks re-ranked token_i by (y,x). Unconfirmed: FLOOR rects are inferred (no visible object marks x=18); A5/A7 unseen (no-op).
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


def socket_interior(t):
    return (t["x"] + 1, t["y"] + 1, t["w"] - 2, t["h"] - 2)


def socket_blocks(r, sockets):
    for t in sockets:
        inner = socket_interior(t)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, player, sockets):
    return not on_floor(r) or overlaps(r, rect(player)) or socket_blocks(r, sockets)


def push_chain(seed, blocks, dx, dy):
    chain = [seed]
    grew = True
    while grew:
        grew = False
        for b in blocks:
            if b in chain:
                continue
            if any(overlaps(rect(c, dx, dy), rect(b)) for c in chain):
                chain.append(b)
                grew = True
    return chain


def kick(seed, blocks, player, sockets, dx, dy):
    moved = False
    while True:
        chain = push_chain(seed, blocks, dx, dy)
        if any(block_blocked(rect(c, dx, dy), player, sockets) for c in chain):
            return moved
        for c in chain:
            c["x"] += dx
            c["y"] += dy
        moved = True


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    s = copy.deepcopy(state)
    players = [o for o in s if o["type"] == "player"]
    blocks = [o for o in s if o["type"] == "block"]
    sockets = [o for o in s if o["type"] == "target"]
    if not players:
        return s
    player = players[0]
    if isinstance(action, dict):
        if action.get("action_id") != 6:
            return s
        cx, cy = action["x"], action["y"]
        hit = [b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]]
        if not hit:
            return s
        b = hit[0]
        player.update(x=b["x"], y=b["y"], w=b["w"], h=b["h"])
        s.remove(b)
        blocks.remove(b)
        rename_blocks(blocks)
        return s
    if action not in DIRS:
        return s
    ddx, ddy = DIRS[action]
    dx, dy = ddx * STEP, ddy * STEP
    nr = rect(player, dx, dy)
    hit = [b for b in blocks if overlaps(nr, rect(b))]
    if hit:
        kick(hit[0], blocks, player, sockets, dx, dy)
        rename_blocks(blocks)
        return s
    if on_floor(nr):
        player["x"] += dx
        player["y"] += dy
    return s
