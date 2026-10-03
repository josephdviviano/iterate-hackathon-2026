# Mechanics: arrows move the player 3 cells; a move is blocked if its next rect leaves the portal bbox or covers
# frame colour 2 (solid wall) or 15 (player-only barrier). If the next rect overlaps a block, the player stays and
# kicks it: the block slides 1 cell per tick, absorbing every block it touches (push chain), until any member would
# hit colour 2, leave the portal bbox, overlap the player, or enter a socket interior (bbox inset 1) it does not fit.
# Click (6) on a block: player takes the block's rect, block consumed. Blocks re-named token_i by (y,x). Unconfirmed: A5/A7.
import copy

DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 3
PLAYER_WALLS = {2, 15}
BLOCK_WALLS = {2}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def inside(r, bound):
    return r[0] >= bound[0] and r[1] >= bound[1] and r[0] + r[2] <= bound[0] + bound[2] and r[1] + r[3] <= bound[1] + bound[3]


def hits_colour(r, frame, colours):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            if frame[y][x] in colours:
                return True
    return False


def socket_interiors(state):
    return [(t["x"] + 1, t["y"] + 1, t["w"] - 2, t["h"] - 2) for t in state if t["type"] == "target"]


def block_blocked(r, bound, frame, player_r, sockets):
    """Guard for a block's next rect: terrain, bound, player, non-fitting socket."""
    if not inside(r, bound) or hits_colour(r, frame, BLOCK_WALLS):
        return True
    if player_r is not None and overlap(r, player_r):
        return True
    for s in sockets:
        if overlap(r, s) and not (r == s):
            return True
    return False


def push_chain(blocks, start, d, bound, frame, player_r, sockets):
    """Slide a chain of blocks one cell per tick until blocked; touched blocks join the chain."""
    dx, dy = d
    rects = [rect(b) for b in blocks]
    chain = {start}
    while True:
        # closure: blocks overlapped by any chain member's next rect join the chain
        changed = True
        while changed:
            changed = False
            for i in range(len(rects)):
                if i in chain:
                    continue
                if any(overlap(shift(rects[j], dx, dy), rects[i]) for j in chain):
                    chain.add(i)
                    changed = True
        if any(block_blocked(shift(rects[j], dx, dy), bound, frame, player_r, sockets) for j in chain):
            break
        for j in chain:
            rects[j] = shift(rects[j], dx, dy)
    for b, r in zip(blocks, rects):
        b["x"], b["y"] = r[0], r[1]


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def transition_function(state, action, frame=None):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    portal = next((o for o in state if o["type"] == "portal"), None)
    bound = rect(portal) if portal else (0, 0, 64, 63)
    sockets = socket_interiors(state)
    others = [o for o in state if o["type"] not in ("player", "block")]

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        hit = next((b for b in blocks if overlap(rect(b), (cx, cy, 1, 1))), None)
        if hit is not None and player is not None:
            player["x"], player["y"], player["w"], player["h"] = hit["x"], hit["y"], hit["w"], hit["h"]
            blocks.remove(hit)
    elif action in DIRS and player is not None and frame is not None:
        dx, dy = DIRS[action]
        nxt = shift(rect(player), dx * STEP, dy * STEP)
        kicked = [i for i, b in enumerate(blocks) if overlap(nxt, rect(b))]
        if kicked:
            push_chain(blocks, kicked[0], (dx, dy), bound, frame, rect(player), sockets)
        elif inside(nxt, bound) and not hits_colour(nxt, frame, PLAYER_WALLS):
            player["x"], player["y"] = nxt[0], nxt[1]

    rename_blocks(blocks)
    return ([player] if player else []) + blocks + others
