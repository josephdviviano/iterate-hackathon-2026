# Mechanics: arrows (1 up,2 down,3 left,4 right) move the player 3 cells if its next rect stays on the
# portal's walkable floor; if the next rect overlaps a block, the player stays and kicks it: the push chain
# (closure of blocks hit by shifted rects) slides 3/tick until any member would leave the floor, hit the player
# or enter a socket interior it does not fit. Click on a block = player takes that block's rect, block removed.
# Blocks re-rank token_i by (y,x). Unconfirmed: portal floor layout (undrawn, incl. upper corridor x>=18), 5/7.
import copy

STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
# Walkable area carved inside the portal wall bbox, inclusive (x0, y0, x1, y1) relative to the portal origin.
PORTAL_FLOOR = [(30, 30, 59, 59), (15, 42, 29, 50), (6, 42, 14, 47), (18, 30, 29, 35)]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shifted(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def contains_point(r, px, py):
    return r[0] <= px < r[0] + r[2] and r[1] <= py < r[1] + r[3]


# guard: portal bounds a mover -- every cell of the rect must lie on the portal's floor
def portal_allows(portal, r):
    ox, oy = (portal["x"], portal["y"]) if portal else (0, 0)
    for px in range(r[0], r[0] + r[2]):
        for py in range(r[1], r[1] + r[3]):
            if not any(x0 <= px - ox <= x1 and y0 <= py - oy <= y1 for x0, y0, x1, y1 in PORTAL_FLOOR):
                return False
    return True


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


# guard: target blocks a block when the block would overlap the socket interior without fitting it
def target_blocks_block(target, r):
    inner = socket_interior(target)
    return overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3])


# guard: player blocks a block
def player_blocks_block(player, r):
    return player is not None and overlaps(r, rect(player))


def block_blocked(r, state):
    portal = next((o for o in state if o["type"] == "portal"), None)
    if not portal_allows(portal, r):
        return True
    for o in state:
        if o["type"] == "target" and target_blocks_block(o, r):
            return True
        if o["type"] == "player" and player_blocks_block(o, r):
            return True
    return False


# interaction block-block: pushing -- closure of blocks hit by the shifted group
def push_chain(blocks, start, d):
    group = set(start)
    changed = True
    while changed:
        changed = False
        for i in list(group):
            nr = shifted(rect(blocks[i]), d)
            for j, b in enumerate(blocks):
                if j not in group and overlaps(nr, rect(b)):
                    group.add(j)
                    changed = True
    return group


# block rule: a kicked block slides until its push chain is blocked
def slide(state, blocks, start, d):
    moved = False
    while True:
        group = push_chain(blocks, start, d)
        if any(block_blocked(shifted(rect(blocks[i]), d), state) for i in group):
            return moved
        for i in group:
            blocks[i]["x"] += d[0]
            blocks[i]["y"] += d[1]
        moved = True


def rename_blocks(state):
    blocks = sorted((o for o in state if o["type"] == "block"), key=lambda o: (o["y"], o["x"]))
    for i, b in enumerate(blocks):
        b["name"] = "token_%d" % i


# player rule
def move(state, action):
    d = DIRS[action]
    player = next((o for o in state if o["type"] == "player"), None)
    if player is None:
        return state
    blocks = [o for o in state if o["type"] == "block"]
    nr = shifted(rect(player), d)
    hit = [i for i, b in enumerate(blocks) if overlaps(nr, rect(b))]
    if hit:
        slide(state, blocks, hit, d)
        return state
    portal = next((o for o in state if o["type"] == "portal"), None)
    if portal_allows(portal, nr):
        player["x"], player["y"] = nr[0], nr[1]
    return state


# click rule: clicked block becomes the player
def click(state, cx, cy):
    player = next((o for o in state if o["type"] == "player"), None)
    block = next((o for o in state if o["type"] == "block" and contains_point(rect(o), cx, cy)), None)
    if player is None or block is None:
        return state
    player["x"], player["y"], player["w"], player["h"] = block["x"], block["y"], block["w"], block["h"]
    state = [o for o in state if o is not block]
    rename_blocks(state)
    return state


def transition_function(state, action):
    state = copy.deepcopy(state)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            return click(state, action["x"], action["y"])
        action = action.get("action_id")
    if action in DIRS:
        state = move(state, action)
        rename_blocks(state)
    return state
