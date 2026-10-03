# Mechanics: arrows (1 up,2 down,3 left,4 right) move the 3-step player inside FLOOR; if the player's
# next rect hits a block it kicks instead (player stays): the push chain (closure of hit blocks) slides 3/tick
# until any member leaves FLOOR, hits the player, or enters a socket interior of a different size.
# Click on a block: player takes that block's rect, block disappears; blocks re-ranked token_i by (y,x).
# Unconfirmed: FLOOR rects (incl. upper corridor x18-29,y30-35 giving the step-78 stop) are undrawn; fitting-socket entry, A5/A7.
import copy

STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
# Inclusive (x0, y0, x1, y1) walkable rects: main room, lower corridor, socket neck, upper corridor.
FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]


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


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_blocks(r, sockets):
    for s in sockets:
        inner = socket_interior(s)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def push_chain(seed, blocks, dx, dy):
    group = [seed]
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if b in group:
                continue
            if any(overlaps(rect(g, dx, dy), rect(b)) for g in group):
                group.append(b)
                changed = True
    return group


def block_blocked(b, dx, dy, player, sockets):
    r = rect(b, dx, dy)
    return (not on_floor(r)) or overlaps(r, rect(player)) or socket_blocks(r, sockets)


def kick(seed, blocks, player, sockets, dx, dy):
    while True:
        group = push_chain(seed, blocks, dx, dy)
        if any(block_blocked(b, dx, dy, player, sockets) for b in group):
            return
        for b in group:
            b["x"] += dx
            b["y"] += dy


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

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        hit = [b for b in blocks if overlaps(rect(b), (cx, cy, 1, 1))]
        if not hit:
            return state
        b = hit[0]
        player["x"], player["y"], player["w"], player["h"] = b["x"], b["y"], b["w"], b["h"]
        state = [o for o in state if o is not b]
        blocks = [o for o in blocks if o is not b]
        rename_blocks(blocks)
        return state

    if action not in DIRS:
        return state
    dx, dy = DIRS[action]
    nxt = rect(player, dx, dy)
    hit = [b for b in blocks if overlaps(nxt, rect(b))]
    if hit:
        kick(hit[0], blocks, player, sockets, dx, dy)
        rename_blocks(blocks)
        return state
    if on_floor(nxt):
        player["x"] += dx
        player["y"] += dy
    return state
