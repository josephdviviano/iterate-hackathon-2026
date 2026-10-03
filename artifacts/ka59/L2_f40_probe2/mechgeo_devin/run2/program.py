# Mechanics: A1-A4 move the player 3 cells (up/down/left/right); if its next rect hits a block it kicks
# instead: the push chain (blocks hit by shifted blocks) slides 3 at a time until any member is blocked; the
# player stays. Blocked = leaves the floor (room, lower corridor and the visible socket bboxes, per pixel), hits
# the player, or enters a socket interior (bbox minus border) whose size it does not match. A6 on a block makes
# its rect the player (old player gone); tokens re-rank token_i by (y,x). Unconfirmed: the step-78 stop at x=18.
import copy

STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
ROOM = (30, 30, 30, 30)
CORRIDOR = (13, 42, 17, 9)


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shifted(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def inside(px, py, r):
    return r[0] <= px < r[0] + r[2] and r[1] <= py < r[1] + r[3]


def interior(s):
    return (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)


def on_floor(r, sockets):
    areas = [ROOM, CORRIDOR] + sockets
    return all(any(inside(px, py, a) for a in areas)
               for px in range(r[0], r[0] + r[2]) for py in range(r[1], r[1] + r[3]))


def socket_rejects(r, sockets):
    for s in sockets:
        inn = interior(s)
        if overlaps(r, inn) and (r[2], r[3]) != (inn[2], inn[3]):
            return True
    return False


def block_blocked(r, player_r, sockets):
    return not on_floor(r, sockets) or overlaps(r, player_r) or socket_rejects(r, sockets)


def push_chain(start, blocks, d):
    chain = {start}
    grew = True
    while grew:
        grew = False
        for i in list(chain):
            nr = shifted(blocks[i], d)
            for j, b in enumerate(blocks):
                if j not in chain and overlaps(nr, b):
                    chain.add(j)
                    grew = True
    return chain


def slide(blocks, idxs, d, player_r, sockets):
    moved = False
    while True:
        new = {i: shifted(blocks[i], d) for i in idxs}
        if any(block_blocked(new[i], player_r, sockets) for i in idxs):
            return moved
        for i in idxs:
            blocks[i] = new[i]
        moved = True
        extra = set()
        for i in idxs:
            extra |= push_chain(i, blocks, d)
        idxs = idxs | extra


def rename_tokens(objs):
    toks = sorted([o for o in objs if o["type"] == "block"], key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(toks):
        o["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    block_objs = [o for o in state if o["type"] == "block"]
    sockets = [rect(o) for o in state if o["type"] == "target"]
    if player is None:
        return state
    if isinstance(action, dict):
        if action.get("action_id") != 6:
            return state
        cx, cy = action["x"], action["y"]
        hit = next((o for o in block_objs if inside(cx, cy, rect(o))), None)
        if hit is None:
            return state
        player.update(x=hit["x"], y=hit["y"], w=hit["w"], h=hit["h"])
        state = [o for o in state if o is not hit]
        rename_tokens(state)
        return state
    d = DIRS.get(action)
    if d is None:
        return state
    pr = rect(player)
    npr = shifted(pr, d)
    blocks = [rect(o) for o in block_objs]
    hits = {i for i, b in enumerate(blocks) if overlaps(npr, b)}
    if hits:
        chain = set()
        for i in hits:
            chain |= push_chain(i, blocks, d)
        if slide(blocks, chain, d, pr, sockets):
            for o, b in zip(block_objs, blocks):
                o["x"], o["y"] = b[0], b[1]
            rename_tokens(state)
        return state
    if on_floor(npr, sockets):
        player["x"], player["y"] = npr[0], npr[1]
    return state
