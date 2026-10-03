# Mechanics: player (3px step) and blocks live on a hidden FLOOR (room + two west corridors + socket neck).
# Player A1-4: moves 3 if its next rect stays on FLOOR and touches no block; if it would overlap a block,
#   it KICKS: that block slides 3px/step, absorbing any block its next rect overlaps (push chain) until any
#   member is blocked (off FLOOR, player overlap, or non-fitting socket interior); player stays put.
# A6 click on a block: the block becomes the player (keeps its bbox); old player vanishes. Blocks re-named token_i by (y,x).
# Unconfirmed: fitting-socket entry (socket fill/tag change), A5/A7 effects, exact floor beyond observed cells.
import copy

STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
FLOOR = [(30, 30, 59, 59),   # main room
         (18, 30, 29, 35),   # upper west corridor
         (15, 42, 29, 50),   # lower west corridor
         (6, 42, 14, 47)]    # neck to socket_44_8


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cell_on_floor(x, y):
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in FLOOR)


def on_floor(r):
    x, y, w, h = r
    return all(cell_on_floor(i, j) for i in range(x, x + w) for j in range(y, y + h))


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


# --- named guards -------------------------------------------------------
def socket_blocks(block_rect, sockets):
    """A socket rejects a block whose rect enters its interior but does not fit it exactly."""
    for s in sockets:
        inner = socket_interior(s)
        if overlaps(block_rect, inner) and block_rect != inner:
            return True
    return False


def block_blocked(r, player, sockets):
    return (not on_floor(r)) or overlaps(r, rect(player)) or socket_blocks(r, sockets)


def push_chain(seed, blocks, dx, dy):
    """Blocks moved together when `seed` shifts by (dx,dy): closure under next-rect overlap."""
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


# --- per-type rules -----------------------------------------------------
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


def player_move(player, blocks, sockets, d):
    dx, dy = d[0] * STEP, d[1] * STEP
    nxt = rect(player, dx, dy)
    hit = [b for b in blocks if overlaps(nxt, rect(b))]
    if hit:
        kick(hit[0], blocks, player, sockets, d[0] * STEP, d[1] * STEP)
        return
    if on_floor(nxt):
        player["x"] += dx
        player["y"] += dy


def click(state, player, blocks, x, y):
    for b in blocks:
        bx, by, bw, bh = rect(b)
        if bx <= x < bx + bw and by <= y < by + bh:
            player["x"], player["y"], player["w"], player["h"] = bx, by, bw, bh
            state.remove(b)
            blocks.remove(b)
            return


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    players = [o for o in state if o["type"] == "player"]
    if not players:
        return state
    player = players[0]
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target" and "socket" in o.get("tags", [])]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(state, player, blocks, action["x"], action["y"])
    elif action in DIRS:
        player_move(player, blocks, sockets, DIRS[action])
    rename_blocks(blocks)
    return state
