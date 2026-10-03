# Mechanics: arrows (1 up,2 down,3 left,4 right) move the player 3 cells within an invisible FLOOR.
# If the player's next rect overlaps a block, the player stays and the block is kicked: it slides
# 3/step, absorbing blocks its shifted rect hits, until any member would leave the floor, hit the
# player or enter a non-fitting socket interior. Click on a block: player takes its rect, block gone.
# Blocks re-named token_i by (y,x). Hypothesis (unconfirmed): floor shape beyond observed stops; no hidden state.
import copy

STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
# Inferred walkable floor (x, y, w, h): room, upper corridor, lower corridor, neck to the socket.
FLOOR = [(30, 30, 30, 30), (18, 30, 12, 6), (15, 42, 15, 9), (6, 42, 9, 6)]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shifted(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    for x in range(r[0], r[0] + r[2]):
        for y in range(r[1], r[1] + r[3]):
            if not any(f[0] <= x < f[0] + f[2] and f[1] <= y < f[1] + f[3] for f in FLOOR):
                return False
    return True


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_rejects(r, sockets):
    """Guard block/socket: a block may not overlap the interior of a socket it does not fit."""
    for s in sockets:
        inner = socket_interior(s)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def slide_chain(blocks, start, dx, dy, player_r, sockets):
    """Kicked block slides, collecting blocks it runs into, until any member is blocked."""
    group = {start}
    while True:
        while True:
            grow = {j for j in range(len(blocks)) if j not in group and any(
                overlaps(shifted(rect(blocks[i]), dx, dy), rect(blocks[j])) for i in group)}
            if not grow:
                break
            group |= grow
        new = [shifted(rect(blocks[i]), dx, dy) for i in group]
        if any(not on_floor(r) or overlaps(r, player_r) or socket_rejects(r, sockets) for r in new):
            return
        for i in group:
            blocks[i]["x"] += dx
            blocks[i]["y"] += dy


def rename_blocks(blocks):
    blocks.sort(key=lambda b: (b["y"], b["x"]))
    for i, b in enumerate(blocks):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target" and "socket" in o.get("tags", [])]
    others = [o for o in state if o["type"] not in ("player", "block")]
    aid = action.get("action_id") if isinstance(action, dict) else action

    if aid == 6 and player is not None:
        cx, cy = action["x"], action["y"]
        hit = next((b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]), None)
        if hit is not None:
            player.update(x=hit["x"], y=hit["y"], w=hit["w"], h=hit["h"])
            blocks.remove(hit)
    elif aid in DIRS and player is not None:
        dx, dy = DIRS[aid]
        dx, dy = dx * STEP, dy * STEP
        pr = rect(player)
        nxt = shifted(pr, dx, dy)
        kicked = next((i for i, b in enumerate(blocks) if overlaps(nxt, rect(b))), None)
        if kicked is not None:
            slide_chain(blocks, kicked, dx, dy, pr, sockets)
        elif on_floor(nxt):
            player["x"] += dx
            player["y"] += dy

    rename_blocks(blocks)
    return ([player] if player is not None else []) + blocks + others
