# Mechanics: arrows move the 3x3 player 3px within a hidden FLOOR (room + corridors). If the player's
# next rect overlaps a block, the block is KICKED instead (player stays): the push chain (closure of
# blocks hit) slides 3px/step until any member leaves the floor, hits the player, or enters a socket
# interior it does not fit. Click on a block = player takes its rect, block consumed. Blocks are renamed
# token_i by (y,x). Unconfirmed: floor shape beyond observed stops, fitting-socket entry, actions 5/7.
import copy

DIRS = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}
# inclusive floor rects (x0, y0, x1, y1): main room, lower corridor, socket alcove, upper corridor
FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    for x in range(r[0], r[0] + r[2]):
        for y in range(r[1], r[1] + r[3]):
            if not any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_blocks(r, sockets):
    """A socket interior (bbox inset 1) rejects blocks that do not exactly fit it."""
    for s in sockets:
        inner = (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)
        if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, player_r, sockets):
    return not on_floor(r) or overlap(r, player_r) or socket_blocks(r, sockets)


def push_chain(start, blocks, dx, dy):
    chain = set(start)
    grew = True
    while grew:
        grew = False
        for i, b in enumerate(blocks):
            if i in chain:
                continue
            if any(overlap(shift(rect(blocks[j]), dx, dy), rect(b)) for j in chain):
                chain.add(i)
                grew = True
    return chain


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if player is None:
        return state
    pr = rect(player)

    if isinstance(action, dict):
        cx, cy = action.get("x"), action.get("y")
        hit = next((b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]), None)
        if hit is None:
            return state
        player["x"], player["y"], player["w"], player["h"] = hit["x"], hit["y"], hit["w"], hit["h"]
        blocks = [b for b in blocks if b is not hit]
        rename_blocks(blocks)
        return [player] + blocks + others

    if action not in DIRS:
        return state
    dx, dy = DIRS[action]
    nr = shift(pr, dx, dy)
    kicked = [i for i, b in enumerate(blocks) if overlap(nr, rect(b))]
    if kicked:
        moved = False
        while True:
            chain = push_chain(kicked, blocks, dx, dy)
            if any(block_blocked(shift(rect(blocks[i]), dx, dy), pr, sockets) for i in chain):
                break
            for i in chain:
                blocks[i]["x"] += dx
                blocks[i]["y"] += dy
            kicked = sorted(chain)
            moved = True
        if moved:
            rename_blocks(blocks)
        return [player] + blocks + others
    if on_floor(nr):
        player["x"], player["y"] = nr[0], nr[1]
    return [player] + blocks + others
