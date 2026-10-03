# Mechanics: player moves 3 cells per arrow (1 up,2 down,3 left,4 right) unless blocked by a hidden wall.
# If a block touches the player on the move side, the player stays and kicks it: the block (plus any blocks
# it runs into, chained) slides 3 cells/step until something is blocked (wall, player, socket interior, bounds).
# Click (6) on a block turns it into the player (player takes its rect); the old player disappears.
# Blocks are renamed token_i by (y, x). Walls are invisible (portal has no pixels): wall rects are inferred.
import copy

STEP = 3
BOUNDS = (0, 0, 63, 63)
# Hidden wall rects (x, y, w, h) inferred from blocked moves; level-specific, not visible in the state.
HIDDEN_WALLS = [(27, 54, 3, 3), (33, 27, 12, 3)]
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shifted(r, dx, dy):
    return (r[0] + dx * STEP, r[1] + dy * STEP, r[2], r[3])


def in_bounds(r):
    return r[0] >= BOUNDS[0] and r[1] >= BOUNDS[1] and r[0] + r[2] <= BOUNDS[2] and r[1] + r[3] <= BOUNDS[3]


def hits_wall(r):
    return not in_bounds(r) or any(overlap(r, w) for w in HIDDEN_WALLS)


def socket_interior(o):
    return (o["x"] + 1, o["y"] + 1, o["w"] - 2, o["h"] - 2)


def blocked_by_socket(block_rect, sockets):
    # Guard block/target: a block cannot slide into a socket interior unless it fits it exactly.
    for s in sockets:
        inner = socket_interior(s)
        if overlap(block_rect, inner) and block_rect != inner:
            return True
    return False


def touching_in_dir(a, b, dx, dy):
    return overlap(shifted(a, dx, dy), b)


def push_chain(first, blocks, dx, dy):
    chain, frontier = [first], [first]
    while frontier:
        cur = frontier.pop()
        for b in blocks:
            if b not in chain and touching_in_dir(rect(cur), rect(b), dx, dy):
                chain.append(b)
                frontier.append(b)
    return chain


def slide_blocks(first, blocks, player, sockets, dx, dy):
    while True:
        chain = push_chain(first, blocks, dx, dy)
        ok = True
        for b in chain:
            r = shifted(rect(b), dx, dy)
            if hits_wall(r) or (player is not None and overlap(r, rect(player))) or blocked_by_socket(r, sockets):
                ok = False
                break
        if not ok:
            return
        for b in chain:
            b["x"] += dx * STEP
            b["y"] += dy * STEP
        if any(rect(b) == socket_interior(s) for b in chain for s in sockets):
            return


def update_player_move(player, blocks, sockets, action):
    dx, dy = DIRS[action]
    for b in blocks:
        if touching_in_dir(rect(player), rect(b), dx, dy):
            slide_blocks(b, blocks, player, sockets, dx, dy)
            return
    r = shifted(rect(player), dx, dy)
    if not hits_wall(r):
        player["x"], player["y"] = r[0], r[1]


def update_click(state, player, blocks, x, y):
    for b in blocks:
        if b["x"] <= x < b["x"] + b["w"] and b["y"] <= y < b["y"] + b["h"]:
            new_player = dict(player) if player is not None else {"name": "player", "type": "player", "layer": 1,
                                                                    "visible": True, "tags": ["token", "white_center"]}
            new_player.update({"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]})
            state = [o for o in state if o is not b and o is not player]
            state.append(new_player)
            return state
    return state


def rename_blocks(state):
    blocks = sorted((o for o in state if o["type"] == "block"), key=lambda o: (o["y"], o["x"]))
    for i, b in enumerate(blocks):
        b["name"] = "token_%d" % i
    return state


def transition_function(state, action):
    state = copy.deepcopy(state)
    player = next((o for o in state if o["type"] == "player"), None)
    blocks = [o for o in state if o["type"] == "block"]
    sockets = [o for o in state if o["type"] == "target"]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            state = update_click(state, player, blocks, action["x"], action["y"])
    elif action in DIRS and player is not None:
        update_player_move(player, blocks, sockets, action)
    return rename_blocks(state)
