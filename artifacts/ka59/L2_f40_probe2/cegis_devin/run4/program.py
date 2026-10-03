# Mechanics: arrows (1-4 = up/down/left/right) move the player 3 cells inside an invisible FLOOR
# (room, lower corridor, socket alcove, upper corridor). If the player's next rect overlaps a block,
# the block is kicked instead (player stays) and slides 3/step, absorbing every block its shift hits,
# until any member would leave the floor, overlap the player, or enter a non-fitting socket interior.
# Click on a block: player takes its rect, block consumed; blocks renamed token_i by (y,x). Unconfirmed: fitting socket entry, A5/A7.
import copy

# Inclusive floor rects (x0, y0, x1, y1) inferred from where slides and walks stop.
FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]
DELTA = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}


def cells(r):
    x, y, w, h = r
    return [(i, j) for i in range(x, x + w) for j in range(y, y + h)]


def on_floor(r):
    return all(any(a <= i <= c and b <= j <= d for a, b, c, d in FLOOR) for i, j in cells(r))


def overlap(r, s):
    return r[0] < s[0] + s[2] and s[0] < r[0] + r[2] and r[1] < s[1] + s[3] and s[1] < r[1] + r[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shifted(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def socket_blocks(r, sockets):
    """Guard block<->target: a block may not enter the interior of a socket it does not fit."""
    for s in sockets:
        inner = (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)
        if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def slide(blocks, start, d, player, sockets):
    """Block update: kicked chain slides until any member is blocked."""
    group = {start}
    while True:
        # grow the chain by every block the shifted group would hit
        changed = True
        while changed:
            changed = False
            for i, b in enumerate(blocks):
                if i in group:
                    continue
                if any(overlap(shifted(blocks[g], d), b) for g in group):
                    group.add(i)
                    changed = True
        new = {g: shifted(blocks[g], d) for g in group}
        if any(not on_floor(r) or overlap(r, player) or socket_blocks(r, sockets) for r in new.values()):
            return blocks
        for g, r in new.items():
            blocks[g] = r


def rename(objs):
    blocks = sorted((o for o in objs if o["type"] == "block"), key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(blocks):
        o["name"] = "token_%d" % i
    return objs


def transition_function(state, action):
    objs = copy.deepcopy(state)
    player = next((o for o in objs if o["type"] == "player"), None)
    blk_objs = [o for o in objs if o["type"] == "block"]
    sockets = [o for o in objs if o["type"] == "target"]
    if player is None:
        return objs
    if isinstance(action, dict):
        if action.get("action_id") != 6:
            return objs
        cx, cy = action["x"], action["y"]
        for o in blk_objs:
            if o["x"] <= cx < o["x"] + o["w"] and o["y"] <= cy < o["y"] + o["h"]:
                for k in ("x", "y", "w", "h"):
                    player[k] = o[k]
                objs.remove(o)
                return rename(objs)
        return objs
    if action not in DELTA:
        return objs
    d = DELTA[action]
    prect = rect(player)
    nxt = shifted(prect, d)
    blocks = [rect(o) for o in blk_objs]
    hit = [i for i, b in enumerate(blocks) if overlap(nxt, b)]
    if hit:
        blocks = slide(blocks, hit[0], d, prect, sockets)
        for o, r in zip(blk_objs, blocks):
            o["x"], o["y"] = r[0], r[1]
        return rename(objs)
    if on_floor(nxt):
        player["x"], player["y"] = nxt[0], nxt[1]
    return objs
