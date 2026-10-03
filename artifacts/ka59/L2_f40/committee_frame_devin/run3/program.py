# Mechanics: arrows (1-4) move the player 3 cells; a move is blocked when its new rect leaves the portal
#   bbox or covers a wall-coloured frame cell (15 or 2). If the new rect overlaps a block, the block is
#   kicked instead (player stays): the push chain slides 1 cell/tick until any member would hit colour 2,
#   leave the portal bbox, overlap the player, or enter a socket interior (bbox inset 1) of another size.
# Click (6) on a block: the player takes that block's rect and the block disappears. Blocks re-rank by (y,x).
# Unconfirmed: fitting-socket entry (tags kept), actions 5/7 (treated as no-ops).
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


def inside(r, bound):
    return r[0] >= bound[0] and r[1] >= bound[1] and r[0] + r[2] <= bound[0] + bound[2] and r[1] + r[3] <= bound[1] + bound[3]


def hits_colour(r, frame, colours, occupied):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            if (x, y) in occupied:
                continue
            if not (0 <= y < len(frame) and 0 <= x < len(frame[0])):
                return True
            if frame[y][x] in colours:
                return True
    return False


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_blocks(r, sockets):
    for s in sockets:
        inner = socket_interior(s)
        if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def cells(rects):
    out = set()
    for r in rects:
        for y in range(r[1], r[1] + r[3]):
            for x in range(r[0], r[0] + r[2]):
                out.add((x, y))
    return out


def push_chain(seeds, blocks, dx, dy):
    group = set(seeds)
    changed = True
    while changed:
        changed = False
        for i in list(group):
            nr = shift(blocks[i], dx, dy)
            for j, b in enumerate(blocks):
                if j not in group and overlap(nr, b):
                    group.add(j)
                    changed = True
    return group


def block_blocked(r, frame, bound, player, sockets, drawn):
    return (not inside(r, bound) or hits_colour(r, frame, BLOCK_WALLS, drawn)
            or overlap(r, player) or socket_blocks(r, sockets))


def kick(seed, blocks, dx, dy, frame, bound, player, sockets, drawn):
    blocks = list(blocks)
    moved = False
    while True:
        group = push_chain(seed, blocks, dx, dy)
        new = {i: shift(blocks[i], dx, dy) for i in group}
        if any(block_blocked(r, frame, bound, player, sockets, drawn) for r in new.values()):
            return blocks, moved
        for i, r in new.items():
            blocks[i] = r
        moved = True


def rename(objs):
    blocks = sorted((o for o in objs if o["type"] == "block"), key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(blocks):
        o["name"] = "token_%d" % i
    return objs


def transition_function(state, action, frame=None):
    objs = copy.deepcopy(state)
    player = next((o for o in objs if o["type"] == "player"), None)
    block_objs = [o for o in objs if o["type"] == "block"]
    sockets = [o for o in objs if o["type"] == "target"]
    portal = next((o for o in objs if o["type"] == "portal"), None)
    bound = rect(portal) if portal else (0, 0, 64, 64)
    if frame is None:
        frame = [[1] * 64 for _ in range(64)]
    if player is None:
        return objs
    drawn = cells([rect(o) for o in block_objs] + [rect(player)])

    if isinstance(action, dict):
        if action.get("action_id") == 6:
            cx, cy = action["x"], action["y"]
            for o in block_objs:
                if o["x"] <= cx < o["x"] + o["w"] and o["y"] <= cy < o["y"] + o["h"]:
                    player["x"], player["y"], player["w"], player["h"] = o["x"], o["y"], o["w"], o["h"]
                    objs.remove(o)
                    return rename(objs)
        return objs

    if action not in DIRS:
        return objs
    ux, uy = DIRS[action]
    pr = rect(player)
    nr = shift(pr, ux * STEP, uy * STEP)
    blocks = [rect(o) for o in block_objs]
    hit = [i for i, b in enumerate(blocks) if overlap(nr, b)]
    if hit:
        new_blocks, moved = kick(hit, blocks, ux, uy, frame, bound, pr, sockets, drawn)
        if moved:
            for o, r in zip(block_objs, new_blocks):
                o["x"], o["y"] = r[0], r[1]
        return rename(objs)
    if inside(nr, bound) and not hits_colour(nr, frame, PLAYER_WALLS, drawn):
        player["x"], player["y"] = nr[0], nr[1]
    return objs
