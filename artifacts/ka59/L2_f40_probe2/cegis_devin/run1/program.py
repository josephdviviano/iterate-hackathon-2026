# Mechanics: arrows move the player 3px on an invisible FLOOR; if its next rect overlaps a block, the
# block is kicked instead (player stays) and slides 3/step, gathering blocks it hits into a chain, until
# any member would leave the floor, overlap the player, or overlap a socket interior it does not fit.
# Click on a block: the player takes that block's rect, the block is consumed; blocks re-ranked by (y,x).
# Hypothesis (unconfirmed): FLOOR rects inferred from stops (room, lower corridor, socket mouth, upper corridor x>=18).
import copy

STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
FLOOR = [(30, 30, 30, 30),   # main room
         (15, 42, 15, 9),    # lower corridor
         (6, 42, 9, 6),      # mouth leading into socket_44_8
         (18, 30, 12, 6)]    # upper corridor (token_0 stops at x=18)


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    x, y, w, h = r
    for cy in range(y, y + h):
        for cx in range(x, x + w):
            if not any(f[0] <= cx < f[0] + f[2] and f[1] <= cy < f[1] + f[3] for f in FLOOR):
                return False
    return True


def socket_blocks(r, sockets):
    for s in sockets:
        inner = (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)
        if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_can_move(r, player_r, sockets):
    return on_floor(r) and not overlap(r, player_r) and not socket_blocks(r, sockets)


def slide(blocks, first, dx, dy, player_r, sockets):
    chain = {first}
    moved = False
    while True:
        while True:  # grow chain with blocks hit by the shifted members
            new = {j for j, b in enumerate(blocks) if j not in chain and
                   any(overlap(shift(rect(blocks[i]), dx, dy), rect(b)) for i in chain)}
            if not new:
                break
            chain |= new
        if not all(block_can_move(shift(rect(blocks[i]), dx, dy), player_r, sockets) for i in chain):
            return moved
        for i in chain:
            blocks[i]["x"] += dx
            blocks[i]["y"] += dy
        moved = True


def rename(blocks):
    for k, b in enumerate(sorted(blocks, key=lambda b: (b["y"], b["x"]))):
        b["name"] = "token_%d" % k


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    if player is None:
        return state
    if isinstance(action, dict):
        if action.get("action_id") != 6:
            return state
        cx, cy = action["x"], action["y"]
        hit = next((b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]), None)
        if hit is None:
            return state
        player.update(x=hit["x"], y=hit["y"], w=hit["w"], h=hit["h"])
        state = [o for o in state if o is not hit]
        rename([o for o in state if o["type"] == "block"])
        return state
    if action not in DIRS:
        return state
    dx, dy = DIRS[action][0] * STEP, DIRS[action][1] * STEP
    pr = rect(player)
    nxt = shift(pr, dx, dy)
    kicked = next((j for j, b in enumerate(blocks) if overlap(nxt, rect(b))), None)
    if kicked is not None:
        if slide(blocks, kicked, dx, dy, pr, sockets):
            rename(blocks)
        return state
    if on_floor(nxt):
        player["x"], player["y"] = nxt[0], nxt[1]
    return state
