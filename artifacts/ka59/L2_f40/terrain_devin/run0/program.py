# Mechanics: arrows (1-4 = up/down/left/right) move the player 3px if its new rect stays on FLOOR.
# Kick: if the player's next rect overlaps a block, the player stays and that block slides 3px/tick,
#   chaining any block its shifted rect hits, until a member would leave FLOOR, hit the player, or
#   enter a socket interior (bbox minus 1px border) whose size differs. Click on a block: the player
#   takes that block's rect and the block is consumed. Blocks re-named token_i by (y,x). Hypothesis: FLOOR (hidden walls) inferred from moves; terrain pixels do not match it.
import copy

FLOOR = [(30, 30, 59, 59), (6, 42, 29, 50)]  # inclusive (x0, y0, x1, y1)
DELTA = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    x, y, w, h = r
    for px in range(x, x + w):
        for py in range(y, y + h):
            if not any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def shift(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def socket_blocks(r, sockets):
    for s in sockets:
        inner = (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)
        if overlaps(r, inner) and (inner[2], inner[3]) != (r[2], r[3]):
            return True
    return False


def block_ok(r, player_r, sockets):
    return on_floor(r) and not overlaps(r, player_r) and not socket_blocks(r, sockets)


def slide(blocks, start, d, player_r, sockets):
    group = set(start)
    for _ in range(200):
        while True:
            moved = [shift(rect(blocks[i]), d) for i in group]
            extra = {j for j, b in enumerate(blocks) if j not in group
                     and any(overlaps(m, rect(b)) for m in moved)}
            if not extra:
                break
            group |= extra
        if not all(block_ok(m, player_r, sockets) for m in moved):
            return
        for i in group:
            blocks[i]["x"] += d[0]
            blocks[i]["y"] += d[1]


def rename(blocks):
    for k, b in enumerate(sorted(blocks, key=lambda b: (b["y"], b["x"]))):
        b["name"] = "token_%d" % k


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if player is None:
        return state
    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        hit = next((b for b in blocks if b["x"] <= cx < b["x"] + b["w"]
                    and b["y"] <= cy < b["y"] + b["h"]), None)
        if hit is not None:
            player["x"], player["y"], player["w"], player["h"] = rect(hit)
            blocks = [b for b in blocks if b is not hit]
    elif action in DELTA:
        d = DELTA[action]
        nxt = shift(rect(player), d)
        kicked = [i for i, b in enumerate(blocks) if overlaps(nxt, rect(b))]
        if kicked:
            slide(blocks, kicked, d, rect(player), sockets)
        elif on_floor(nxt):
            player["x"], player["y"] = nxt[0], nxt[1]
    rename(blocks)
    return [player] + blocks + others
