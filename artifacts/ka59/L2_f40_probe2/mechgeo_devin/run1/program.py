# Mechanics: arrows (1-4) move the player 3 cells; if the player's next rect overlaps a block it is kicked
# instead (player stays) and slides 3/step as a push chain (blocks its shifted rects overlap join) until any
# member would leave the floor, overlap the player, or enter a socket interior (bbox minus 1px) it doesn't fit.
# Click (6) on a block: player takes the block's rect, the block vanishes; blocks are renamed token_i by (y,x).
# Unconfirmed: the floor (walls are not extracted: 'portal' is one bbox, no pixels); fitting socket entry; A5/A7.
import copy

DIRS = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}
FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]  # inclusive x0,y0,x1,y1


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shifted(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            if not any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in FLOOR):
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


def block_blocked(r, player_r, sockets):
    return not on_floor(r) or overlaps(r, player_r) or socket_blocks(r, sockets)


def push_chain(chain, blocks, d):
    """Grow chain with every block its shifted rects overlap; return the closed chain."""
    chain = list(chain)
    grown = True
    while grown:
        grown = False
        for b in blocks:
            if b in chain:
                continue
            if any(overlaps(shifted(rect(c), d), rect(b)) for c in chain):
                chain.append(b)
                grown = True
    return chain


def kick(hit, blocks, player_r, sockets, d):
    chain = hit
    moved = False
    while True:
        chain = push_chain(chain, blocks, d)
        if any(block_blocked(shifted(rect(c), d), player_r, sockets) for c in chain):
            return moved
        for c in chain:
            c["x"] += d[0]
            c["y"] += d[1]
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

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        for b in blocks:
            if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]:
                player["x"], player["y"], player["w"], player["h"] = b["x"], b["y"], b["w"], b["h"]
                state.remove(b)
                blocks.remove(b)
                rename_blocks(blocks)
                break
        return state

    if action not in DIRS:
        return state
    d = DIRS[action]
    nxt = shifted(rect(player), d)
    hit = [b for b in blocks if overlaps(nxt, rect(b))]
    if hit:
        if kick(hit, blocks, rect(player), sockets, d):
            rename_blocks(blocks)
        return state
    if on_floor(nxt):
        player["x"], player["y"] = nxt[0], nxt[1]
    return state
