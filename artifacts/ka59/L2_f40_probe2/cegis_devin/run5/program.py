# Mechanics: arrows move the player 3 cells inside an invisible FLOOR (union of rects); if the player's
# next rect overlaps a block it kicks it instead (player stays): the block slides 3/step, absorbing blocks
# its shifted rect hits, until any chain member would leave the floor, overlap the player, or enter a
# socket interior (bbox minus 1px border) of a different size. Click on a block: player takes its rect,
# block consumed. Blocks renamed token_i by (y,x). Hypothesis: floor shape off the observed paths is a guess.
STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
# inclusive cell rects (x0, y0, x1, y1): room, lower corridor, socket neck, upper west corridor
FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shifted(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def on_floor(r):
    for x in range(r[0], r[0] + r[2]):
        for y in range(r[1], r[1] + r[3]):
            if not any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in FLOOR):
                return False
    return True


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def blocked_by_socket(r, sockets):
    for s in sockets:
        inner = socket_interior(s)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_can_occupy(r, player_r, sockets):
    return on_floor(r) and not overlaps(r, player_r) and not blocked_by_socket(r, sockets)


def slide(blocks, start, d, player_r, sockets):
    chain = {start}
    while True:
        grew = True
        while grew:
            grew = False
            for i in chain.copy():
                nr = shifted(rect(blocks[i]), d)
                for j, b in enumerate(blocks):
                    if j not in chain and overlaps(nr, rect(b)):
                        chain.add(j)
                        grew = True
        if not all(block_can_occupy(shifted(rect(blocks[i]), d), player_r, sockets) for i in chain):
            return
        for i in chain:
            blocks[i]["x"] += d[0]
            blocks[i]["y"] += d[1]


def rename_blocks(blocks):
    blocks.sort(key=lambda b: (b["y"], b["x"]))
    for i, b in enumerate(blocks):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    objs = [dict(o) for o in state]
    players = [o for o in objs if o["type"] == "player"]
    blocks = [o for o in objs if o["type"] == "block"]
    sockets = [o for o in objs if o["type"] == "target"]
    others = [o for o in objs if o["type"] not in ("player", "block")]
    if not players:
        return objs
    player = players[0]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            cx, cy = action["x"], action["y"]
            for b in blocks:
                if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]:
                    player.update(x=b["x"], y=b["y"], w=b["w"], h=b["h"])
                    blocks.remove(b)
                    break
    elif action in DIRS:
        d = DIRS[action]
        nr = shifted(rect(player), d)
        hit = [i for i, b in enumerate(blocks) if overlaps(nr, rect(b))]
        if hit:
            slide(blocks, hit[0], d, rect(player), sockets)
        elif on_floor(nr):
            player["x"], player["y"] = nr[0], nr[1]
    rename_blocks(blocks)
    return [player] + blocks + others
