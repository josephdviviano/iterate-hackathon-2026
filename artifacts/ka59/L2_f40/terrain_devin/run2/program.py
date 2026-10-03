# Mechanics: arrows move the player 3px; if its next rect overlaps a block it KICKS instead (player stays):
#   the hit block slides 3px/tick, gathering every block its shifted rect would overlap (push chain), until any
#   chain member would leave the board, hit a block-solid terrain pixel, the player, or a non-fitting socket interior.
# Terrain (pixels row-major, -1 = empty): every terrain pixel is solid for the player; colour-15 terrain is passable
#   for blocks (they slide over it), other colours are solid. Click on a block = player takes its rect, block removed.
# Blocks re-ranked token_i by (y,x). Unconfirmed: fitting-socket entry (tags stay 'empty'), player vs colour 2, A5/A7.
import copy

STEP = 3
BOARD = 63
BLOCK_PASSABLE_COLOURS = {15}
DELTA = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shifted(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def terrain_cells(state):
    cells = {}
    for o in state:
        if o.get("type") != "terrain":
            continue
        for j, row in enumerate(o.get("pixels") or []):
            for i, v in enumerate(row):
                if v >= 0:
                    cells.setdefault((o["x"] + i, o["y"] + j), set()).add(v)
    return cells


def cells_of(r):
    return [(x, y) for x in range(r[0], r[0] + r[2]) for y in range(r[1], r[1] + r[3])]


def on_board(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= BOARD and r[1] + r[3] <= BOARD


def player_blocked_by_terrain(r, terrain):
    return any(c in terrain for c in cells_of(r))


def block_blocked_by_terrain(r, terrain):
    return any(c in terrain and terrain[c] - BLOCK_PASSABLE_COLOURS for c in cells_of(r))


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


def socket_rejects(r, s):
    inner = socket_interior(s)
    return overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3])


def block_blocked(r, terrain, player_r, sockets):
    if not on_board(r) or block_blocked_by_terrain(r, terrain) or overlaps(r, player_r):
        return True
    return any(socket_rejects(r, s) for s in sockets)


def push_chain(seed, blocks, d):
    chain = {seed}
    grew = True
    while grew:
        grew = False
        for i in list(chain):
            nr = shifted(rect(blocks[i]), d)
            for j, b in enumerate(blocks):
                if j not in chain and overlaps(nr, rect(b)):
                    chain.add(j)
                    grew = True
    return chain


def kick(seed, blocks, d, terrain, player_r, sockets):
    moved = False
    while True:
        chain = push_chain(seed, blocks, d)
        if any(block_blocked(shifted(rect(blocks[i]), d), terrain, player_r, sockets) for i in chain):
            return moved
        for i in chain:
            blocks[i]["x"] += d[0]
            blocks[i]["y"] += d[1]
        moved = True


def rename_blocks(blocks):
    for k, b in enumerate(sorted(blocks, key=lambda b: (b["y"], b["x"]))):
        b["name"] = "token_%d" % k


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o.get("type") == "player"), None)
    blocks = [o for o in state if o.get("type") == "block"]
    sockets = [o for o in state if o.get("type") == "target"]
    others = [o for o in state if o.get("type") not in ("player", "block")]
    if player is None:
        return state
    terrain = terrain_cells(state)

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        hit = [b for b in blocks if b["x"] <= cx < b["x"] + b["w"] and b["y"] <= cy < b["y"] + b["h"]]
        if hit:
            b = hit[0]
            player["x"], player["y"], player["w"], player["h"] = b["x"], b["y"], b["w"], b["h"]
            blocks.remove(b)
            rename_blocks(blocks)
        return [player] + blocks + others

    if action not in DELTA:
        return state
    d = DELTA[action]
    nr = shifted(rect(player), d)
    hit = [j for j, b in enumerate(blocks) if overlaps(nr, rect(b))]
    if hit:
        kick(hit[0], blocks, d, terrain, rect(player), sockets)
        rename_blocks(blocks)
    elif on_board(nr) and not player_blocked_by_terrain(nr, terrain):
        player["x"], player["y"] = nr[0], nr[1]
    return [player] + blocks + others
