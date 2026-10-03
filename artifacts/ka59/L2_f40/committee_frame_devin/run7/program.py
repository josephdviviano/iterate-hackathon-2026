# Mechanics: arrows 1-4 move the player 3 cells; blocked if its next rect leaves the portal bbox or covers
# frame colour 2 or 15 (PLAYER_WALLS). If the next rect overlaps a block, the player stays and kicks it:
# the push chain slides 1 cell per tick (contacted blocks join) until any member hits colour 2, the bound,
# the player or a socket interior (inset 1) it does not fit. Click on a block = player takes its rect, block gone.
# Blocks re-named token_i by (y,x). Unconfirmed: exact-fit socket filling (tags) and actions 5/7 (no-ops).
import copy

STEP = 3
PLAYER_WALLS = {2, 15}
BLOCK_WALLS = {2}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shifted(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def inside(r, bound):
    return r[0] >= bound[0] and r[1] >= bound[1] and r[0] + r[2] <= bound[0] + bound[2] and r[1] + r[3] <= bound[1] + bound[3]


def hits_colour(r, frame, colours):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            if 0 <= y < len(frame) and 0 <= x < len(frame[0]) and frame[y][x] in colours:
                return True
    return False


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


# --- pairwise guards -------------------------------------------------------
def player_blocked(r, bound, frame):
    return not inside(r, bound) or hits_colour(r, frame, PLAYER_WALLS)


def socket_rejects(r, sockets):
    for s in sockets:
        inner = socket_interior(s)
        if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, bound, frame, player_r, sockets):
    return (not inside(r, bound) or hits_colour(r, frame, BLOCK_WALLS)
            or overlap(r, player_r) or socket_rejects(r, sockets))


# --- block update: push chain ---------------------------------------------
def push_chain(blocks, start, dx, dy, bound, frame, player_r, sockets):
    pos = {i: rect(b) for i, b in enumerate(blocks)}
    while True:
        chain = {start}
        grew = True
        while grew:
            grew = False
            for i in list(chain):
                nr = shifted(pos[i], dx, dy)
                for j in pos:
                    if j not in chain and overlap(nr, pos[j]):
                        chain.add(j)
                        grew = True
        if any(block_blocked(shifted(pos[i], dx, dy), bound, frame, player_r, sockets) for i in chain):
            break
        for i in chain:
            pos[i] = shifted(pos[i], dx, dy)
    for i, b in enumerate(blocks):
        b["x"], b["y"] = pos[i][0], pos[i][1]


def rename_blocks(blocks):
    for k, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % k


def transition_function(state, action, frame=None):
    state = copy.deepcopy(state)
    if frame is None:
        frame = [[1] * 64 for _ in range(64)]
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    portal = next((o for o in state if o["type"] == "portal"), None)
    bound = rect(portal) if portal else (0, 0, 64, 64)
    others = [o for o in state if o["type"] not in ("player", "block")]
    if player is None:
        return state

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        hit = next((b for b in blocks if overlap((cx, cy, 1, 1), rect(b))), None)
        if hit is not None:
            player["x"], player["y"], player["w"], player["h"] = hit["x"], hit["y"], hit["w"], hit["h"]
            blocks = [b for b in blocks if b is not hit]
    elif action in DIRS:
        dx, dy = DIRS[action]
        nr = shifted(rect(player), dx * STEP, dy * STEP)
        kicked = next((i for i, b in enumerate(blocks) if overlap(nr, rect(b))), None)
        if kicked is not None:
            push_chain(blocks, kicked, dx, dy, bound, frame, rect(player), sockets)
        elif not player_blocked(nr, bound, frame):
            player["x"], player["y"] = nr[0], nr[1]

    rename_blocks(blocks)
    return [player] + blocks + others
