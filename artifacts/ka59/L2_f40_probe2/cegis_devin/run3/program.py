# Mechanics: arrows (1-4) move the player 3px; if its shifted rect overlaps a block, that block is kicked
# instead (player stays) and slides 3px/step, absorbing any block its next shift would overlap, until any
# member would leave the floor, overlap the player, or enter a socket interior of a different size.
# Click (6) on a block: player takes the block's rect, the block is consumed. Blocks renamed token_i by (y,x).
# Hypothesis: floor = room + inferred corridors (upper one ends at x=18, lower narrows to y42-47 for x<15);
# the unseen continuation to socket_11_10, fitting-socket entry and actions 5/7 are unconfirmed (no-ops).
import copy

STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
FLOOR = [(30, 30, 30, 30),   # main room x30-59, y30-59
         (18, 30, 12, 6),    # upper corridor x18-29, y30-35
         (15, 42, 15, 9),    # lower corridor x15-29, y42-50
         (6, 42, 9, 6)]      # lower corridor neck x6-14, y42-47 (into socket_44_8)


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    x, y, w, h = r
    return all(any(fx <= cx < fx + fw and fy <= cy < fy + fh for fx, fy, fw, fh in FLOOR)
               for cx in range(x, x + w) for cy in range(y, y + h))


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def blocked_by_socket(r, sockets):
    for s in sockets:
        inner = socket_interior(s)
        if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_can_occupy(r, player, sockets):
    return on_floor(r) and not overlap(r, rect(player)) and not blocked_by_socket(r, sockets)


def slide(chain, blocks, player, sockets, dx, dy):
    """Slide a kicked chain until blocked; chain grows with blocks its next shift overlaps."""
    while True:
        changed = True
        while changed:
            changed = False
            for b in blocks:
                if b in chain:
                    continue
                if any(overlap(rect(c, dx, dy), rect(b)) for c in chain):
                    chain.append(b)
                    changed = True
        if not all(block_can_occupy(rect(c, dx, dy), player, sockets) for c in chain):
            return
        for c in chain:
            c["x"] += dx
            c["y"] += dy


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    state = copy.deepcopy(state)
    players = [o for o in state if o["type"] == "player"]
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    if not players:
        return state
    player = players[0]
    if isinstance(action, dict):
        if action.get("action_id") != 6:
            return state
        cx, cy = action["x"], action["y"]
        hit = [b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]]
        if not hit:
            return state
        b = hit[0]
        player.update(x=b["x"], y=b["y"], w=b["w"], h=b["h"])
        state = [o for o in state if o is not b]
        rename_blocks([o for o in state if o["type"] == "block"])
        return state
    if action not in DIRS:
        return state
    dx, dy = DIRS[action]
    nxt = rect(player, dx, dy)
    kicked = [b for b in blocks if overlap(nxt, rect(b))]
    if kicked:
        slide(kicked, blocks, player, sockets, dx, dy)
        rename_blocks(blocks)
    elif on_floor(nxt):
        player["x"] += dx
        player["y"] += dy
    return state
