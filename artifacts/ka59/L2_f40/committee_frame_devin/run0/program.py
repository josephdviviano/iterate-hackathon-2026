# Mechanics: arrows move the player 3 cells; a frame cell of colour 15 or 2 (or off-board) blocks the player.
# If the player's next rect overlaps a block, the block is kicked instead (player stays): the push chain
# slides 3 cells per tick, absorbing blocks it touches, until any member hits colour 2, leaves the board,
# hits the player, or enters a socket interior (inset 1) it does not exactly fit. Click on a block = the
# player takes its rect, the block vanishes. Blocks re-named token_i by (y,x). Unconfirmed: fitting-socket docking, A5/A7.
STEP = 3
PLAYER_WALLS = (15, 2)
BLOCK_WALLS = (2,)
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def hits_colour(r, frame, colours):
    x, y, w, h = r
    if frame is None:
        return False
    H, W = len(frame), len(frame[0])
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return True
    return any(frame[yy][xx] in colours for yy in range(y, y + h) for xx in range(x, x + w))


def off_board(r, frame):
    W = len(frame[0]) if frame else 64
    H = len(frame) if frame else 64
    return r[0] < 0 or r[1] < 0 or r[0] + r[2] > W or r[1] + r[3] > H


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_refuses(block_rect, sockets):
    for s in sockets:
        inner = socket_interior(s)
        if overlap(block_rect, inner) and (block_rect[2], block_rect[3]) != (inner[2], inner[3]):
            return True
    return False


def block_can_move(r, frame, player, sockets):
    if off_board(r, frame) or hits_colour(r, frame, BLOCK_WALLS):
        return False
    if overlap(r, rect(player)):
        return False
    return not socket_refuses(r, sockets)


def push_chain(seed, blocks, dx, dy):
    chain = list(seed)
    grew = True
    while grew:
        grew = False
        for b in blocks:
            if any(b is c for c in chain):
                continue
            if any(overlap(rect(c, dx, dy), rect(b)) for c in chain):
                chain.append(b)
                grew = True
    return chain


def slide(seed, blocks, dx, dy, frame, player, sockets):
    chain = list(seed)
    while True:
        chain = push_chain(chain, blocks, dx, dy)
        if not all(block_can_move(rect(c, dx, dy), frame, player, sockets) for c in chain):
            return
        for c in chain:
            c["x"] += dx
            c["y"] += dy


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def step_player(player, blocks, sockets, action, frame):
    dx, dy = DIRS[action]
    dx, dy = dx * STEP, dy * STEP
    nxt = rect(player, dx, dy)
    kicked = [b for b in blocks if overlap(nxt, rect(b))]
    if kicked:
        slide(kicked, blocks, dx, dy, frame, player, sockets)
        return
    if hits_colour(nxt, frame, PLAYER_WALLS):
        return
    player["x"] += dx
    player["y"] += dy


def click(state, player, blocks, cx, cy):
    for b in blocks:
        if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]:
            for k in ("x", "y", "w", "h"):
                player[k] = b[k]
            state.remove(b)
            return


def transition_function(state, action, frame=None):
    state = [dict(o) for o in state]
    players = [o for o in state if o["type"] == "player"]
    if not players:
        return state
    player = players[0]
    sockets = [o for o in state if o["type"] == "target"]
    blocks = [o for o in state if o["type"] == "block"]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(state, player, blocks, action["x"], action["y"])
    elif action in DIRS:
        step_player(player, blocks, sockets, action, frame)
    rename_blocks([o for o in state if o["type"] == "block"])
    return state
