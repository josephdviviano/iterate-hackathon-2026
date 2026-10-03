# Mechanics: arrows (1-4 = up/down/left/right) move the player 3 px inside an invisible FLOOR
# (room + upper corridor + lower corridor + socket neck). If the player's next rect hits a block,
# the block is kicked instead: it slides 3 px/step, absorbing blocks it runs into, until any member
# would leave the floor, hit the player or enter a non-fitting socket interior; player stays put.
# Click on a block: the player takes its rect, block consumed. Blocks renamed token_i by (y,x). Unconfirmed: fitting-socket entry, actions 5/7.
import copy

STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
# inclusive (x0, y0, x1, y1) floor rects inferred from blocked moves and slide stops
FLOOR = [(30, 30, 59, 59), (18, 30, 29, 35), (15, 42, 29, 50), (6, 42, 14, 47)]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shifted(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    for px in range(r[0], r[0] + r[2]):
        for py in range(r[1], r[1] + r[3]):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_blocks(r, sockets):
    """A block may not enter the interior of a socket whose shape it does not fit."""
    for s in sockets:
        inner = socket_interior(s)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, player_r, sockets):
    return not on_floor(r) or overlaps(r, player_r) or socket_blocks(r, sockets)


def push_chain(blocks, start, dx, dy, player_r, sockets):
    """Slide the chain led by blocks[start]; returns True if anything moved."""
    moved = False
    group = {start}
    while True:
        while True:
            grown = {j for j, b in enumerate(blocks) if j not in group and
                     any(overlaps(shifted(rect(blocks[i]), dx, dy), rect(b)) for i in group)}
            if not grown:
                break
            group |= grown
        if any(block_blocked(shifted(rect(blocks[i]), dx, dy), player_r, sockets) for i in group):
            return moved
        for i in group:
            blocks[i]["x"] += dx
            blocks[i]["y"] += dy
        moved = True


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda b: (b["y"], b["x"]))):
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

    if isinstance(action, dict) and action.get("action_id") == 6:
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
    dx, dy = DIRS[action]
    nxt = shifted(rect(player), dx, dy)
    kicked = [i for i, b in enumerate(blocks) if overlaps(nxt, rect(b))]
    if kicked:
        if push_chain(blocks, kicked[0], dx, dy, rect(player), sockets):
            rename_blocks(blocks)
        return [player] + blocks + others
    if on_floor(nxt):
        player["x"], player["y"] = nxt[0], nxt[1]
    return [player] + blocks + others
