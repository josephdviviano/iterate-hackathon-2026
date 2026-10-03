# Mechanics: kick/slide token sokoban. Arrows move the player 3px; if the player's next rect hits a block, the
# block is kicked instead (player stays) and the hit chain slides 3px/step, absorbing blocks it runs into, until any
# member would leave the hidden FLOOR, overlap the player, or overlap a socket interior of a different size.
# Click on a block: the player takes its rect and the block is consumed. Blocks are renamed token_i by (y,x).
# Hypothesis (unconfirmed): FLOOR rects are inferred from stops (walls invisible); fitting socket entry unobserved.
import copy

STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
# inclusive pixel rects (x0, y0, x1, y1) of walkable floor
FLOOR = [(30, 30, 59, 59),   # main room
         (15, 42, 29, 50),   # wide west corridor
         (6, 42, 14, 47),    # narrow alcove into socket_44_8
         (18, 30, 29, 35)]   # upper west corridor (step-78 stop at x=18)


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shifted(r, d):
    return (r[0] + d[0] * STEP, r[1] + d[1] * STEP, r[2], r[3])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    x, y, w, h = r
    for px in range(x, x + w):
        for py in range(y, y + h):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_blocks(r, sockets):
    for s in sockets:
        inner = socket_interior(s)
        if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def chain_step(group, blocks, d, player_r, sockets):
    """Grow the moving group by every block its shifted rects hit; return group or None if blocked."""
    group = set(group)
    while True:
        moved = [shifted(rect(blocks[i]), d) for i in group]
        hit = {j for j, b in enumerate(blocks) if j not in group and any(overlap(m, rect(b)) for m in moved)}
        if not hit:
            break
        group |= hit
    for i in group:
        m = shifted(rect(blocks[i]), d)
        if not on_floor(m) or overlap(m, player_r) or socket_blocks(m, sockets):
            return None
    return group


def kick(blocks, first, d, player_r, sockets):
    group = {first}
    while True:
        g = chain_step(group, blocks, d, player_r, sockets)
        if g is None:
            return
        for i in g:
            blocks[i]["x"] += d[0] * STEP
            blocks[i]["y"] += d[1] * STEP
        group = g


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda b: (b["y"], b["x"]))):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    players = [o for o in state if o["type"] == "player"]
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if not players:
        return state
    player = players[0]

    if isinstance(action, dict):
        if action.get("action_id") == 6:
            cx, cy = action["x"], action["y"]
            for b in blocks:
                if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]:
                    player["x"], player["y"], player["w"], player["h"] = b["x"], b["y"], b["w"], b["h"]
                    blocks.remove(b)
                    break
    elif action in DIRS:
        d = DIRS[action]
        nxt = shifted(rect(player), d)
        hit = [i for i, b in enumerate(blocks) if overlap(nxt, rect(b))]
        if hit:
            kick(blocks, hit[0], d, rect(player), sockets)
        elif on_floor(nxt):
            player["x"], player["y"] = nxt[0], nxt[1]

    rename_blocks(blocks)
    return [player] + blocks + others
