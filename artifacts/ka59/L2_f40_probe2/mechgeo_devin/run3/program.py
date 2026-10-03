# Mechanics: arrows (1 up,2 down,3 left,4 right) step 3 px. Player moves if its new rect is on the floor and hits no block.
# Kick: if the player's next rect overlaps blocks, they slide as a push chain (closure of blocks hit by shifted rects) until
# any member would leave the floor, hit the player or enter a non-fitting socket interior (bbox minus 1px border); player stays.
# Click on a block: player takes the block's rect, block removed; blocks re-ranked token_i by (y,x). Actions 5/7: no-op.
# Unconfirmed: walls are not in the extractor (no pixels); FLOOR rects (incl. upper corridor x18-29,y30-35) are inferred.
import copy

FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]  # inclusive x0,y0,x1,y1
DIRS = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    x, y, w, h = r
    for px in range(x, x + w):
        for py in range(y, y + h):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_blocks(block_rect, sockets):
    for s in sockets:
        inner = socket_interior(s)
        if overlap(block_rect, inner) and (block_rect[2], block_rect[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, player, sockets):
    return (not on_floor(r)) or overlap(r, rect(player)) or socket_blocks(r, sockets)


def push_chain(seed, blocks, dx, dy):
    group = list(seed)
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if any(b is g for g in group):
                continue
            if any(overlap(rect(g, dx, dy), rect(b)) for g in group):
                group.append(b)
                changed = True
    return group


def kick(seed, blocks, player, sockets, dx, dy):
    moved = False
    while True:
        group = push_chain(seed, blocks, dx, dy)
        if any(block_blocked(rect(g, dx, dy), player, sockets) for g in group):
            return moved
        for g in group:
            g["x"] += dx
            g["y"] += dy
        moved = True


def rename_blocks(blocks):
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
        if action.get("action_id") != 6:
            return state
        cx, cy = action["x"], action["y"]
        hit = [b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]]
        if not hit:
            return state
        b = hit[0]
        player.update(x=b["x"], y=b["y"], w=b["w"], h=b["h"])
        state = [o for o in state if o is not b]
        rename_blocks([o for o in state if o["type"] == "block"])
        return state
    if action not in DIRS:
        return state
    dx, dy = DIRS[action]
    nxt = rect(player, dx, dy)
    hit = [b for b in blocks if overlap(nxt, rect(b))]
    if hit:
        kick(hit, blocks, player, sockets, dx, dy)
        rename_blocks(blocks)
        return state
    if on_floor(nxt):
        player["x"] += dx
        player["y"] += dy
    return state
