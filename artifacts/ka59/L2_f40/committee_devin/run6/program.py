# Mechanics: player (3px token) moves 3 per arrow inside an inferred floor (room x,y 30..59 + corridor y 42..50 west).
# Arrow into an adjacent block kicks it instead (player stays): the block slides 3/step, collecting every block it
# hits into a chain, until any member would leave the floor, overlap the player or a non-fitting socket interior.
# Click (ACTION6) on a block: the block becomes the player (takes its rect), old player vanishes; blocks renamed token_i by (y,x).
# Unconfirmed: floor extents beyond observed moves, fitting-socket behaviour (assumed enterable); no hidden state needed.
import copy

FLOOR = [(30, 30, 59, 59), (6, 42, 29, 50)]  # inclusive x0,y0,x1,y1
DIRS = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}


def cells(r):
    x, y, w, h = r
    return {(i, j) for i in range(x, x + w) for j in range(y, y + h)}


def on_floor(r):
    return all(any(a <= i <= c and b <= j <= d for a, b, c, d in FLOOR) for i, j in cells(r))


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_blocks(r, sockets):
    for s in sockets:
        inn = interior(s)
        if overlap(r, inn) and not (r[2] == inn[2] and r[3] == inn[3]):
            return True
    return False


def shift(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def kick(start, blocks, player, sockets, d):
    chain = {start}
    while True:
        moved = {i: shift(rect(blocks[i]), d) for i in chain}
        hit = {j for j, b in enumerate(blocks) if j not in chain
               and any(overlap(m, rect(b)) for m in moved.values())}
        if hit:
            chain |= hit
            continue
        if any(not on_floor(m) or overlap(m, player) or socket_blocks(m, sockets) for m in moved.values()):
            return
        for i, m in moved.items():
            blocks[i]["x"], blocks[i]["y"] = m[0], m[1]


def transition_function(state, action):
    st = copy.deepcopy(state)
    player = next((o for o in st if o["type"] == "player"), None)
    blocks = [o for o in st if o["type"] == "block"]
    sockets = [o for o in st if o["type"] == "target"]
    others = [o for o in st if o["type"] not in ("player", "block")]
    if isinstance(action, dict):
        if action.get("action_id") == 6 and player is not None:
            cx, cy = action["x"], action["y"]
            for i, b in enumerate(blocks):
                if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]:
                    player.update(x=b["x"], y=b["y"], w=b["w"], h=b["h"])
                    blocks.pop(i)
                    break
    elif action in DIRS and player is not None:
        d = DIRS[action]
        nr = shift(rect(player), d)
        hit = [i for i, b in enumerate(blocks) if overlap(nr, rect(b))]
        if hit:
            kick(hit[0], blocks, rect(player), sockets, d)
        elif on_floor(nr):
            player["x"], player["y"] = nr[0], nr[1]
    blocks.sort(key=lambda b: (b["y"], b["x"]))
    for i, b in enumerate(blocks):
        b["name"] = "token_%d" % i
    return ([player] if player else []) + blocks + others
