# Mechanics: A1-4 move the player 3 cells; a move into a wall cell (colour 15 or 2) or off-board is a no-op.
# If the player's next rect overlaps a block it stays and kicks: the block slides 1 cell/step in that direction,
# blocks it touches join the chain, until any chain block would hit colour 2, leave the board or enter a socket interior.
# Click (A6) on a block: the player takes that block's rect, the block is removed; blocks re-ranked token_i by (y,x).
# Unconfirmed: whether a block whose size equals a socket interior may enter it (assumed yes); A5/A7 assumed no-ops.
import copy

PLAYER_WALLS = {15, 2}
BLOCK_WALLS = {2}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 3


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def hits_colour(r, frame, colours):
    H, W = len(frame), len(frame[0])
    x, y, w, h = r
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return True
    return any(frame[yy][xx] in colours for yy in range(y, y + h) for xx in range(x, x + w))


def socket_interiors(state):
    return [(o["x"] + 1, o["y"] + 1, o["w"] - 2, o["h"] - 2)
            for o in state if o.get("type") == "target" and "socket" in o.get("tags", [])]


def block_blocked(r, frame, sockets):
    if hits_colour(r, frame, BLOCK_WALLS):
        return True
    return any(overlap(r, s) and (r[2], r[3]) != (s[2], s[3]) for s in sockets)


def slide_chain(blocks, start, dx, dy, frame, sockets):
    pos = {i: rect(b) for i, b in enumerate(blocks)}
    chain = {start}
    while True:
        changed = True
        while changed:
            changed = False
            for i in list(chain):
                nr = shift(pos[i], dx, dy)
                for j in pos:
                    if j not in chain and overlap(nr, pos[j]):
                        chain.add(j)
                        changed = True
        nxt = {i: shift(pos[i], dx, dy) for i in chain}
        if any(block_blocked(r, frame, sockets) for r in nxt.values()):
            break
        pos.update(nxt)
    for i, b in enumerate(blocks):
        b["x"], b["y"] = pos[i][0], pos[i][1]


def rename_blocks(blocks):
    for k, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % k


def transition_function(state, action, frame=None):
    state = copy.deepcopy(state)
    player = next((o for o in state if o.get("type") == "player"), None)
    blocks = [o for o in state if o.get("type") == "block"]
    others = [o for o in state if o.get("type") not in ("player", "block")]
    if player is None or frame is None:
        return state
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            cx, cy = action["x"], action["y"]
            hit = next((b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]), None)
            if hit is not None:
                player["x"], player["y"], player["w"], player["h"] = hit["x"], hit["y"], hit["w"], hit["h"]
                blocks.remove(hit)
                rename_blocks(blocks)
        return [player] + blocks + others
    if action not in DIRS:
        return state
    dx, dy = DIRS[action]
    nr = shift(rect(player), dx * STEP, dy * STEP)
    kicked = [i for i, b in enumerate(blocks) if overlap(nr, rect(b))]
    if kicked:
        slide_chain(blocks, kicked[0], dx, dy, frame, socket_interiors(state))
        for i in kicked[1:]:
            slide_chain(blocks, i, dx, dy, frame, socket_interiors(state))
        rename_blocks(blocks)
    elif not hits_colour(nr, frame, PLAYER_WALLS):
        player["x"], player["y"] = nr[0], nr[1]
    return [player] + blocks + others
