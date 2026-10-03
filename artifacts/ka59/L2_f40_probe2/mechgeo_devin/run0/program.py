# Mechanics: A1-A4 move the white player 3 cells (up/down/left/right) inside the portal's walled floor.
# If the player's next rect hits a pushable block it stays put and kicks: the block (plus every block its
# shifted rect touches, recursively) slides in 3-cell steps until any member would leave the floor, overlap the
# player, or enter a socket interior (bbox minus 1px border) of a different size. Click on a block: player takes
# its rect, block removed; blocks renamed token_i by (y,x). Hypothesis: floor = portal wall layout (pixels not extracted).
import copy

STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
# Walkable floor of the portal (wall) object, inclusive rects relative to the portal origin:
# main room, lower corridor, socket neck, upper corridor.
PORTAL_FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shifted(r, d):
    return (r[0] + d[0] * STEP, r[1] + d[1] * STEP, r[2], r[3])


def floor_rects(state):
    portal = next((o for o in state if o["type"] == "portal"), None)
    px, py = (portal["x"], portal["y"]) if portal else (0, 0)
    return [(px + a, py + b, px + c, py + e) for a, b, c, e in PORTAL_FLOOR]


def on_floor(r, floor):
    for x in range(r[0], r[0] + r[2]):
        for y in range(r[1], r[1] + r[3]):
            if not any(a <= x <= c and b <= y <= e for a, b, c, e in floor):
                return False
    return True


def socket_blocks(r, sockets):
    for s in sockets:
        inner = (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, floor, sockets, player):
    return not on_floor(r, floor) or socket_blocks(r, sockets) or overlaps(r, rect(player))


def push_chain(blocks, start, d):
    group = [start]
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if b in group:
                continue
            if any(overlaps(shifted(rect(g), d), rect(b)) for g in group):
                group.append(b)
                changed = True
    return group


def kick(blocks, start, d, floor, sockets, player):
    moved = False
    while True:
        group = push_chain(blocks, start, d)
        if any(block_blocked(shifted(rect(g), d), floor, sockets, player) for g in group):
            return moved
        for g in group:
            g["x"] += d[0] * STEP
            g["y"] += d[1] * STEP
        moved = True


def rename_blocks(state):
    blocks = sorted((o for o in state if o["type"] == "block"), key=lambda o: (o["y"], o["x"]))
    for i, b in enumerate(blocks):
        b["name"] = "token_%d" % i


def move(state, d):
    player = next((o for o in state if o["type"] == "player"), None)
    if player is None:
        return state
    floor = floor_rects(state)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    nxt = shifted(rect(player), d)
    hit = [b for b in blocks if overlaps(nxt, rect(b))]
    if hit:
        kick(blocks, hit[0], d, floor, sockets, player)
    elif on_floor(nxt, floor):
        player["x"], player["y"] = nxt[0], nxt[1]
    return state


def click(state, x, y):
    player = next((o for o in state if o["type"] == "player"), None)
    target = next((o for o in state if o["type"] == "block"
                   and o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]), None)
    if player is None or target is None:
        return state
    player["x"], player["y"], player["w"], player["h"] = target["x"], target["y"], target["w"], target["h"]
    return [o for o in state if o is not target]


def transition_function(state, action):
    state = copy.deepcopy(state)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            state = click(state, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        state = move(state, DIRS[action])
    rename_blocks(state)
    return state
