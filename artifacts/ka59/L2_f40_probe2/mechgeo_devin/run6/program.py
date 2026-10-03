# Mechanics: arrows (1-4) move the player 3 cells; a move into a block KICKS it instead (player stays):
#   the push chain (blocks hit by moving blocks, recomputed every tick) slides 3/tick until any member
#   would leave the floor, hit the player, or enter a socket interior it does not exactly fit.
# Click (6) on a block: player takes that block's rect, block and old player sprite vanish. 5/7: no-op.
# Blocks are renamed token_i by (y, x). Stateless. UNCONFIRMED: the floor is the portal/wall sprite's
#   undrawn layout (extractor gives no pixels); its 4 rects are inferred, incl. the upper corridor (t78).
import copy

DIRS = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}
STEP = 3


def wall_floor(portal):
    """Walkable layout of the wall sprite, relative to its origin (inclusive x0, y0, x1, y1)."""
    ox, oy = portal["x"], portal["y"]
    rects = [(30, 30, 59, 59),   # main room
             (15, 42, 29, 50),   # lower corridor
             (6, 42, 14, 47),    # neck into the left socket
             (18, 30, 29, 35)]   # upper corridor
    return [(ox + a, oy + b, ox + c, oy + d) for a, b, c, d in rects]


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def overlaps(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1


def on_floor(r, floor):
    x, y, w, h = r
    for px in range(x, x + w):
        for py in range(y, y + h):
            if not any(a <= px <= c and b <= py <= d for a, b, c, d in floor):
                return False
    return True


def socket_interior(s):
    return (s["x"] + 1, s["y"] + 1, s["w"] - 2, s["h"] - 2)


# ---- named guards -------------------------------------------------------
def socket_blocks(block_rect, sockets):
    """A block may not enter a socket interior whose size it does not match."""
    for s in sockets:
        inner = socket_interior(s)
        if overlaps(block_rect, inner) and (block_rect[2], block_rect[3]) != (inner[2], inner[3]):
            return True
    return False


def block_blocked(r, player, sockets, floor):
    return (not on_floor(r, floor)) or overlaps(r, rect(player)) or socket_blocks(r, sockets)


def push_chain(group, blocks, dx, dy):
    """Close the moving group under 'next rect overlaps another block'."""
    group = set(group)
    changed = True
    while changed:
        changed = False
        for i, b in enumerate(blocks):
            if i in group:
                continue
            if any(overlaps(rect(blocks[g], dx, dy), rect(b)) for g in group):
                group.add(i)
                changed = True
    return group


# ---- per-type update rules ---------------------------------------------
def kick(start, blocks, player, sockets, floor, dx, dy):
    moved = False
    while True:
        group = push_chain(start, blocks, dx, dy)
        if any(block_blocked(rect(blocks[g], dx, dy), player, sockets, floor) for g in group):
            return moved
        for g in group:
            blocks[g]["x"] += dx
            blocks[g]["y"] += dy
        moved = True


def player_move(player, blocks, sockets, floor, action):
    dx, dy = DIRS[action]
    nxt = rect(player, dx, dy)
    hit = [i for i, b in enumerate(blocks) if overlaps(nxt, rect(b))]
    if hit:
        kick(hit, blocks, player, sockets, floor, dx, dy)
        return
    if on_floor(nxt, floor):
        player["x"] += dx
        player["y"] += dy


def click(objs, player, blocks, x, y):
    for b in blocks:
        bx, by, bw, bh = rect(b)
        if bx <= x < bx + bw and by <= y < by + bh:
            player["x"], player["y"], player["w"], player["h"] = bx, by, bw, bh
            blocks.remove(b)
            return True
    return False


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def transition_function(state, action):
    objs = copy.deepcopy(state)
    players = [o for o in objs if o["type"] == "player"]
    blocks = [o for o in objs if o["type"] == "block"]
    sockets = [o for o in objs if o["type"] == "target"]
    portals = [o for o in objs if o["type"] == "portal"]
    others = [o for o in objs if o["type"] not in ("player", "block")]
    if not players or not portals:
        return objs
    player = players[0]
    floor = wall_floor(portals[0])
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(objs, player, blocks, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        player_move(player, blocks, sockets, floor, action)
    rename_blocks(blocks)
    return [player] + blocks + others
