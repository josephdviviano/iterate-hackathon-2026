# Mechanics: arrows (1-4 = up/down/left/right) move the 3-step player; colour 2 (wall) and colour 15
# (terrain) cells and the portal bbox bound block the player. If the player's next rect overlaps a block it
# kicks it instead (player stays): the block slides 3/step, collecting blocks it bumps (push chain), until a
# member would leave the portal bbox, touch colour 2, hit the player, or enter a non-fitting socket interior
# (bbox inset 1). Colour 15 is passable for blocks. Click on a block = player takes its rect, block removed;
# tokens re-ranked token_i by (y,x). Unconfirmed: fitting socket entry (tags), actions 5/7 (no-op here).
import copy

STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALL = 2
TERRAIN = 15


def rect(o, dx=0, dy=0):
    return (o["x"] + dx, o["y"] + dy, o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            yield x, y


def in_bounds(r, bounds):
    return (r[0] >= bounds[0] and r[1] >= bounds[1]
            and r[0] + r[2] <= bounds[0] + bounds[2] and r[1] + r[3] <= bounds[1] + bounds[3])


def has_colour(r, frame, colours):
    return any(frame[y][x] in colours for x, y in cells(r))


def interior(sock):
    return (sock["x"] + 1, sock["y"] + 1, sock["w"] - 2, sock["h"] - 2)


def socket_rejects(r, sockets):
    for s in sockets:
        i = interior(s)
        if overlap(r, i) and (r[2], r[3]) != (i[2], i[3]):
            return True
    return False


def player_blocked(r, frame, bounds):
    return not in_bounds(r, bounds) or has_colour(r, frame, (WALL, TERRAIN))


def block_blocked(r, frame, bounds, player, sockets):
    return (not in_bounds(r, bounds) or has_colour(r, frame, (WALL,))
            or overlap(r, rect(player)) or socket_rejects(r, sockets))


def push_chain(group, blocks, dx, dy):
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if b in group:
                continue
            if any(overlap(rect(g, dx, dy), rect(b)) for g in group):
                group.append(b)
                changed = True
    return group


def slide(first, blocks, dx, dy, frame, bounds, player, sockets):
    group = [first]
    while True:
        group = push_chain(group, blocks, dx, dy)
        if any(block_blocked(rect(g, dx, dy), frame, bounds, player, sockets) for g in group):
            return
        for g in group:
            g["x"] += dx
            g["y"] += dy


def rename_blocks(blocks):
    for i, b in enumerate(sorted(blocks, key=lambda o: (o["y"], o["x"]))):
        b["name"] = "token_%d" % i


def transition_function(state, action, frame=None):
    objs = copy.deepcopy(state)
    player = next((o for o in objs if o["type"] == "player"), None)
    blocks = [o for o in objs if o["type"] == "block"]
    sockets = [o for o in objs if o["type"] == "target"]
    portal = next((o for o in objs if o["type"] == "portal"), None)
    bounds = rect(portal) if portal else (0, 0, 64, 64)
    if frame is None:
        frame = [[0] * 64 for _ in range(64)]
    if player is None:
        return objs
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            cx, cy = action["x"], action["y"]
            hit = next((b for b in blocks if overlap((cx, cy, 1, 1), rect(b))), None)
            if hit is not None:
                player["x"], player["y"], player["w"], player["h"] = hit["x"], hit["y"], hit["w"], hit["h"]
                objs.remove(hit)
                blocks.remove(hit)
                rename_blocks(blocks)
        return objs
    if action not in DIRS:
        return objs
    dx, dy = DIRS[action]
    nxt = rect(player, dx, dy)
    kicked = next((b for b in blocks if overlap(nxt, rect(b))), None)
    if kicked is not None:
        slide(kicked, blocks, dx, dy, frame, bounds, player, sockets)
        rename_blocks(blocks)
    elif not player_blocked(nxt, frame, bounds):
        player["x"] += dx
        player["y"] += dy
    return objs
