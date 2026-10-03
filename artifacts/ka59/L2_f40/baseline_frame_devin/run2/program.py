# Mechanics: arrows (1 up,2 down,3 left,4 right) move the player 3 cells; blocked by wall colours 15/2 in the frame.
# Kick: if the player's next rect overlaps a block, the player stays and the block slides 1 cell/tick in that
# direction, pushing every block it touches (chain); the whole chain stops on colour-2 cells, the player, or a
# socket interior (target bbox inset 1) whose size differs from the block's. Colour 15 is passable for blocks only.
# Click on a block: player takes its rect, block removed; tokens re-ranked token_i by (y,x). Unconfirmed: fitting socket entry.
import copy

DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 3
WALL_ALL = {2}
WALL_PLAYER = {15, 2}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shifted(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def cells(r):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            yield x, y


def hits_colour(r, frame, bad):
    H, W = len(frame), len(frame[0])
    for x, y in cells(r):
        if not (0 <= x < W and 0 <= y < H) or frame[y][x] in bad:
            return True
    return False


def socket_interiors(state):
    out = []
    for o in state:
        if o["type"] == "target":
            out.append((o["x"] + 1, o["y"] + 1, o["w"] - 2, o["h"] - 2))
    return out


def block_blocked(r, frame, sockets, player_r):
    if hits_colour(r, frame, WALL_ALL):
        return True
    if overlap(r, player_r):
        return True
    for s in sockets:
        if overlap(r, s) and (r[2], r[3]) != (s[2], s[3]):
            return True
    return False


def push_chain(start, blocks, dx, dy):
    chain = {start}
    grew = True
    while grew:
        grew = False
        for i in chain.copy():
            nr = shifted(blocks[i], dx, dy)
            for j, b in enumerate(blocks):
                if j not in chain and overlap(nr, b):
                    chain.add(j)
                    grew = True
    return chain


def slide(start, blocks, dx, dy, frame, sockets, player_r):
    while True:
        chain = push_chain(start, blocks, dx, dy)
        new = {i: shifted(blocks[i], dx, dy) for i in chain}
        if any(block_blocked(r, frame, sockets, player_r) for r in new.values()):
            return blocks
        for i, r in new.items():
            blocks[i] = r


def rename(blocks_objs):
    blocks_objs.sort(key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(blocks_objs):
        o["name"] = "token_%d" % i


def transition_function(state, action, frame=None):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    others = [o for o in state if o["type"] not in ("player", "block")]
    if isinstance(action, dict):
        cx, cy = action.get("x"), action.get("y")
        hit = next((b for b in blocks if overlap(rect(b), (cx, cy, 1, 1))), None)
        if hit is not None and player is not None:
            player["x"], player["y"], player["w"], player["h"] = hit["x"], hit["y"], hit["w"], hit["h"]
            blocks.remove(hit)
            rename(blocks)
        return ([player] if player else []) + blocks + others
    if action not in DIRS or player is None or frame is None:
        return state
    ux, uy = DIRS[action]
    pr = rect(player)
    nxt = shifted(pr, ux * STEP, uy * STEP)
    brs = [rect(b) for b in blocks]
    touched = [i for i, b in enumerate(brs) if overlap(nxt, b)]
    if touched:
        sockets = socket_interiors(state)
        for i in touched:
            brs = slide(i, brs, ux, uy, frame, sockets, pr)
        for b, r in zip(blocks, brs):
            b["x"], b["y"] = r[0], r[1]
        rename(blocks)
    elif not hits_colour(nxt, frame, WALL_PLAYER):
        player["x"], player["y"] = nxt[0], nxt[1]
    return [player] + blocks + others
