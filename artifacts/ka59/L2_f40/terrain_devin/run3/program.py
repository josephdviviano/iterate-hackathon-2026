# Mechanics: arrows (1 up,2 down,3 left,4 right) move the 3px player one step of 3 inside the floor.
# If the player's next rect hits a block, the block is kicked instead (player stays): the hit group
# slides 3px/step, absorbing every block it touches, until any member would leave the floor, hit the
# player, or enter a socket interior it does not exactly fit. Click (6) on a block: player takes its
# rect, block consumed. Blocks re-ranked token_i by (y,x). Terrain/portal/targets static.
# Unconfirmed: hidden FLOOR rects (terrain pixels do not match observed collisions); actions 5/7 no-op.
import copy

FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 3


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


def socket_interior(t):
    return (t["x"] + 1, t["y"] + 1, t["w"] - 2, t["h"] - 2)


def socket_blocks(r, sockets):
    for t in sockets:
        inner = socket_interior(t)
        if overlap(r, inner) and r != inner:
            return True
    return False


def player_can_move(r):
    return on_floor(r)


def slide(group, blocks, player, sockets, dx, dy):
    group = list(group)
    while True:
        grew = True
        while grew:
            grew = False
            for b in blocks:
                if b in group:
                    continue
                if any(overlap(rect(g, dx, dy), rect(b)) for g in group):
                    group.append(b)
                    grew = True
        nxt = [rect(g, dx, dy) for g in group]
        if any(not on_floor(r) or overlap(r, rect(player)) or socket_blocks(r, sockets) for r in nxt):
            return
        for g in group:
            g["x"] += dx
            g["y"] += dy
        if any(rect(g) == socket_interior(t) for g in group for t in sockets):
            return


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def click(player, blocks, x, y):
    for b in blocks:
        if b["x"] <= x < b["x"] + b["w"] and b["y"] <= y < b["y"] + b["h"]:
            player["x"], player["y"], player["w"], player["h"] = b["x"], b["y"], b["w"], b["h"]
            blocks.remove(b)
            return True
    return False


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if player is None:
        return state
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(player, blocks, action["x"], action["y"])
    elif action in DIRS:
        dx, dy = DIRS[action][0] * STEP, DIRS[action][1] * STEP
        nr = rect(player, dx, dy)
        hit = [b for b in blocks if overlap(nr, rect(b))]
        if hit:
            slide(hit, blocks, player, sockets, dx, dy)
        elif player_can_move(nr):
            player["x"] += dx
            player["y"] += dy
    rename_blocks(blocks)
    return [player] + blocks + others
