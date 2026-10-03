# Mechanics: arrows (1-4) move the player 3 cells; a move is blocked if the next rect leaves the portal
# bbox or covers a frame cell of colour 15 (player-only barrier) or 2 (solid wall). If the next rect hits a
# block, the player stays and kicks it: the push chain slides 1 cell per tick, absorbing contacted blocks,
# until a member would hit colour 2, the portal bound, the player, or a non-fitting socket interior (inset 1).
# Click (6) on a block: player takes that block's rect, block removed. Blocks re-named token_i by (y,x).
# Unconfirmed: exact-fit socket docking, actions 5/7 (no-op). No hidden state needed (all visible).
import copy

DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 3
PLAYER_WALLS = (2, 15)
BLOCK_WALLS = (2,)


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def inside(r, bound):
    return (r[0] >= bound[0] and r[1] >= bound[1]
            and r[0] + r[2] <= bound[0] + bound[2] and r[1] + r[3] <= bound[1] + bound[3])


def hits_colour(r, frame, colours):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            if 0 <= y < len(frame) and 0 <= x < len(frame[0]) and frame[y][x] in colours:
                return True
    return False


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_blocks(r, sockets):
    """Named guard: a block may not enter a socket interior it does not exactly fit."""
    for s in sockets:
        inner = socket_interior(s)
        if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def push_chain(blocks, start, dx, dy, player_r, bound, frame, sockets):
    """Slide the chain one cell per tick; returns new rects per block index."""
    pos = [rect(b) for b in blocks]
    moving = {start}
    while True:
        changed = True
        while changed:
            changed = False
            for i in list(moving):
                nr = shift(pos[i], dx, dy)
                for j in range(len(pos)):
                    if j not in moving and overlaps(nr, pos[j]):
                        moving.add(j)
                        changed = True
        ok = True
        for i in moving:
            nr = shift(pos[i], dx, dy)
            if (not inside(nr, bound) or hits_colour(nr, frame, BLOCK_WALLS)
                    or overlaps(nr, player_r) or socket_blocks(nr, sockets)):
                ok = False
                break
        if not ok:
            return pos
        for i in moving:
            pos[i] = shift(pos[i], dx, dy)


def rename_blocks(blocks):
    for k, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % k


def transition_function(state, action, frame=None):
    st = copy.deepcopy(state)
    player = next((o for o in st if o["type"] == "player"), None)
    blocks = [o for o in st if o["type"] == "block"]
    sockets = [o for o in st if o["type"] == "target"]
    portal = next((o for o in st if o["type"] == "portal"), None)
    bound = rect(portal) if portal else (0, 0, 64, 64)
    frame = frame or []
    if player is None:
        return st

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        hit = next((b for b in blocks if overlaps((cx, cy, 1, 1), rect(b))), None)
        if hit is not None:
            player["x"], player["y"], player["w"], player["h"] = hit["x"], hit["y"], hit["w"], hit["h"]
            st = [o for o in st if o is not hit]
            blocks = [b for b in blocks if b is not hit]
            rename_blocks(blocks)
        return st

    if action not in DIRS:
        return st
    ux, uy = DIRS[action]
    pr = rect(player)
    nr = shift(pr, ux * STEP, uy * STEP)
    kicked = [i for i, b in enumerate(blocks) if overlaps(nr, rect(b))]
    if kicked:
        moved = list(map(rect, blocks))
        for i in kicked:
            if rect(blocks[i]) == moved[i]:
                tmp = [dict(b, x=moved[k][0], y=moved[k][1]) for k, b in enumerate(blocks)]
                moved = push_chain(tmp, i, ux, uy, pr, bound, frame, sockets)
        for b, r in zip(blocks, moved):
            b["x"], b["y"] = r[0], r[1]
        rename_blocks(blocks)
        return st
    if inside(nr, bound) and not hits_colour(nr, frame, PLAYER_WALLS):
        player["x"], player["y"] = nr[0], nr[1]
    return st
