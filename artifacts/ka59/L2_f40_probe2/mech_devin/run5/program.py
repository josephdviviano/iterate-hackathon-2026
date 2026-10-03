# Mechanics: player (white token) moves 3 px per arrow (1 up,2 down,3 left,4 right), confined to a hidden FLOOR.
# Kick: if the player's next rect overlaps a block, the player stays and the block slides 3 px per tick as a push
# chain (blocks hit by shifted rects join) until any member leaves the FLOOR, hits the player, or enters a socket
# interior (bbox minus 1px border) whose size does not fit it. Click on a block: player takes that block's rect,
# old player vanishes; blocks re-named token_i by (y,x). Hypothesis (unconfirmed): FLOOR rects beyond observed; A5/A7 no-op.
import copy

# inclusive (x0, y0, x1, y1): room, lower corridor, socket alcove, upper corridor
FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]
DELTA = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    x, y, w, h = r
    for px in range(x, x + w):
        for py in range(y, y + h):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_blocks(r, sockets):
    # a block may not enter a socket interior unless it exactly fits it
    for s in sockets:
        inner = (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)
        if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def shift(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def push_chain(seed, blocks, d):
    group = {seed}
    changed = True
    while changed:
        changed = False
        for i in list(group):
            nr = shift(blocks[i], d)
            for j, b in enumerate(blocks):
                if j not in group and overlap(nr, b):
                    group.add(j)
                    changed = True
    return group


def kick(seed, blocks, d, player, sockets):
    blocks = list(blocks)
    moved = False
    while True:
        group = push_chain(seed, blocks, d)
        new = {i: shift(blocks[i], d) for i in group}
        if any(not on_floor(r) or overlap(r, player) or socket_blocks(r, sockets) for r in new.values()):
            return blocks, moved
        for i, r in new.items():
            blocks[i] = r
        moved = True


def rename_blocks(objs):
    blocks = sorted((o for o in objs if o["type"] == "block"), key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(blocks):
        o["name"] = "token_%d" % i


def transition_function(state, action):
    objs = copy.deepcopy(state)
    players = [o for o in objs if o["type"] == "player"]
    block_objs = [o for o in objs if o["type"] == "block"]
    sockets = [rect(o) for o in objs if o["type"] == "target"]
    if not players:
        return objs
    p = players[0]
    if isinstance(action, dict):
        if action.get("action_id") != 6:
            return objs
        cx, cy = action["x"], action["y"]
        for b in block_objs:
            if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]:
                p["x"], p["y"], p["w"], p["h"] = b["x"], b["y"], b["w"], b["h"]
                objs.remove(b)
                rename_blocks(objs)
                break
        return objs
    if action not in DELTA:
        return objs
    d = DELTA[action]
    pr = rect(p)
    nr = shift(pr, d)
    blocks = [rect(b) for b in block_objs]
    hit = [i for i, b in enumerate(blocks) if overlap(nr, b)]
    if hit:
        blocks, moved = kick(hit[0], blocks, d, pr, sockets)
        if moved:
            for o, r in zip(block_objs, blocks):
                o["x"], o["y"] = r[0], r[1]
            rename_blocks(objs)
        return objs
    if on_floor(nr):
        p["x"], p["y"] = nr[0], nr[1]
    return objs
