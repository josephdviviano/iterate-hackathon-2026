# Mechanics: player moves 3px per arrow (A1 up, A2 down, A3 left, A4 right) on an inferred floor
# (room x30-59,y30-59 + corridor y42-50 running left to x5); walls themselves are not extracted.
# Player-block guard 'kick': a block the move would overlap is kicked instead (player stays) and the chain
# slides 3px/step, collecting every block it would hit, until any member leaves the floor or enters a
# non-fitting socket interior. A6 click on a block: player takes its rect, block consumed; blocks renamed by (y,x).
import copy

STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
FLOOR = [(30, 30, 30, 30), (5, 42, 25, 9)]  # (x, y, w, h); hypothesis inferred from blocked moves


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    x, y, w, h = r
    return all(any(fx <= px < fx + fw and fy <= py < fy + fh for fx, fy, fw, fh in FLOOR)
               for px in range(x, x + w) for py in range(y, y + h))


def interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_blocks(r, sockets):
    # Block-socket guard: a block may not overlap a socket interior unless it fits it exactly.
    for s in sockets:
        i = interior(s)
        if overlap(r, i) and not (r[2:] == i[2:] and r[0] >= s["x"] and r[1] >= s["y"]
                                  and r[0] + r[2] <= s["x"] + s["w"] and r[1] + r[3] <= s["y"] + s["h"]):
            return True
    return False


def shift(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def slide(blocks, chain, d, sockets, player):
    moved = False
    while True:
        nxt = {i: shift(rect(blocks[i]), d) for i in chain}
        hit = [j for j in range(len(blocks)) if j not in chain
               and any(overlap(rect(blocks[j]), r) for r in nxt.values())]
        if hit:
            chain |= set(hit)
            continue
        if any(not on_floor(r) or socket_blocks(r, sockets) or overlap(r, player) for r in nxt.values()):
            return moved
        for i, r in nxt.items():
            blocks[i]["x"], blocks[i]["y"] = r[0], r[1]
        moved = True
        if any(r[2:] == interior(s)[2:] and r[:2] == interior(s)[:2] for r in nxt.values() for s in sockets):
            return moved


def rename(blocks):
    for k, b in enumerate(sorted(blocks, key=lambda b: (b["y"], b["x"]))):
        b["name"] = "token_%d" % k


def transition_function(state, action):
    st = copy.deepcopy(state)
    player = next(o for o in st if o["type"] == "player")
    blocks = [o for o in st if o["type"] == "block"]
    sockets = [o for o in st if o["type"] == "target"]
    others = [o for o in st if o["type"] not in ("player", "block")]
    if isinstance(action, dict):
        cx, cy = action["x"], action["y"]
        for b in blocks:
            if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]:
                player["x"], player["y"], player["w"], player["h"] = b["x"], b["y"], b["w"], b["h"]
                blocks.remove(b)
                break
    elif action in DIRS:
        d = DIRS[action]
        target = shift(rect(player), d)
        hit = {i for i, b in enumerate(blocks) if overlap(rect(b), target)}
        if hit:
            slide(blocks, hit, d, sockets, rect(player))
        elif on_floor(target):
            player["x"], player["y"] = target[0], target[1]
    rename(blocks)
    return [player] + blocks + others
