# Mechanics: kick/slide token game. Arrows move the player 3px inside an invisible FLOOR
# (room + upper/lower corridors + socket neck). If the player's next rect hits a block, the
# block is kicked instead (player stays) and slides 3px/step, chaining every block it hits,
# until any chain member would leave the floor, hit the player, or enter a non-fitting socket
# interior. Click on a block: player takes its rect, block consumed. Blocks renamed token_i by
# (y,x). Unconfirmed: hidden floor beyond observed paths, fitting-socket entry, actions 5/7.
import copy

FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]
DELTA = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def on_floor(r):
    for px in range(r[0], r[0] + r[2]):
        for py in range(r[1], r[1] + r[3]):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_blocks(r, sockets):
    for s in sockets:
        inner = (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, player_r, sockets):
    return not on_floor(r) or overlaps(r, player_r) or socket_blocks(r, sockets)


def push_chain(start, rects, d):
    chain = {start}
    grew = True
    while grew:
        grew = False
        for i in list(chain):
            nr = shift(rects[i], d)
            for j, r in enumerate(rects):
                if j not in chain and overlaps(nr, r):
                    chain.add(j)
                    grew = True
    return chain


def slide(start, rects, d, player_r, sockets):
    rects = list(rects)
    moved = False
    while True:
        chain = push_chain(start, rects, d)
        if any(block_blocked(shift(rects[i], d), player_r, sockets) for i in chain):
            return rects, moved
        for i in chain:
            rects[i] = shift(rects[i], d)
        moved = True


def rename_blocks(blocks):
    blocks.sort(key=lambda o: (o["y"], o["x"]))
    for i, b in enumerate(blocks):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if player is None:
        return state
    if isinstance(action, dict):
        cx, cy = action.get("x"), action.get("y")
        hit = next((b for b in blocks if b["x"] <= cx < b["x"] + b["w"]
                    and b["y"] <= cy < b["y"] + b["h"]), None)
        if hit is None:
            return state
        player.update(x=hit["x"], y=hit["y"], w=hit["w"], h=hit["h"])
        blocks = [b for b in blocks if b is not hit]
        rename_blocks(blocks)
        return [player] + blocks + others
    d = DELTA.get(action)
    if d is None:
        return state
    pr = rect(player)
    nr = shift(pr, d)
    rects = [rect(b) for b in blocks]
    hit = next((i for i, r in enumerate(rects) if overlaps(nr, r)), None)
    if hit is not None:
        rects, moved = slide(hit, rects, d, pr, sockets)
        if not moved:
            return state
        for b, r in zip(blocks, rects):
            b["x"], b["y"] = r[0], r[1]
        rename_blocks(blocks)
        return [player] + blocks + others
    if on_floor(nr):
        player["x"], player["y"] = nr[0], nr[1]
    return state
