# Mechanics: arrows move the player 3 px (1 up, 2 down, 3 left, 4 right) inside a hidden FLOOR
# (room + upper/lower corridors + socket neck). If the player's next rect hits blocks, the player
# stays and the hit blocks (plus every block they push, a chain) slide 3 px per tick until any chain
# member would leave the floor, hit the player, or enter a socket interior it does not fit. Click on a
# block: player takes that block's rect, block removed; blocks renamed token_i by (y,x). Hypothesis: the
# fitting-socket fill and actions 5/7 are unobserved (treated as no-ops); no hidden state was needed.
import copy

FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]
DELTA = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def on_floor(r):
    for px in range(r[0], r[0] + r[2]):
        for py in range(r[1], r[1] + r[3]):
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


def block_blocked(r, player_r, sockets):
    return not on_floor(r) or overlap(r, player_r) or socket_blocks(r, sockets)


def push_chain(start, blocks, d):
    group, frontier = set(start), list(start)
    while frontier:
        i = frontier.pop()
        nr = shift(blocks[i], d)
        for j, b in enumerate(blocks):
            if j not in group and overlap(nr, b):
                group.add(j)
                frontier.append(j)
    return group


def slide(start, blocks, d, player_r, sockets):
    blocks = list(blocks)
    while True:
        group = push_chain(start, blocks, d)
        if any(block_blocked(shift(blocks[i], d), player_r, sockets) for i in group):
            return blocks
        for i in group:
            blocks[i] = shift(blocks[i], d)


def rename(blocks, template):
    out = []
    for k, r in enumerate(sorted(blocks, key=lambda r: (r[1], r[0]))):
        o = copy.deepcopy(template)
        o.update(name="token_%d" % k, x=r[0], y=r[1], w=r[2], h=r[3])
        out.append(o)
    return out


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    block_objs = [o for o in state if o["type"] == "block"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    sockets = [o for o in state if o["type"] == "target"]
    if player is None or not block_objs:
        return state
    template = block_objs[0]
    blocks = [rect(o) for o in sorted(block_objs, key=lambda o: (o["y"], o["x"]))]
    pr = rect(player)

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        hit = [i for i, b in enumerate(blocks) if b[0] <= cx < b[0] + b[2] and b[1] <= cy < b[1] + b[3]]
        if not hit:
            return state
        b = blocks.pop(hit[0])
        player.update(x=b[0], y=b[1], w=b[2], h=b[3])
    elif action in DELTA:
        d = DELTA[action]
        nr = shift(pr, d)
        hit = [i for i, b in enumerate(blocks) if overlap(nr, b)]
        if hit:
            blocks = slide(hit, blocks, d, pr, sockets)
        elif on_floor(nr):
            player.update(x=nr[0], y=nr[1])
        else:
            return state
    else:
        return state
    return [player] + rename(blocks, template) + others
