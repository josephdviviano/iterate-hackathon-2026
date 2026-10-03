# Mechanics: arrows move the player 3 cells; if the next player rect overlaps a block the block is kicked
# and the push chain slides 1 cell/tick until a member hits colour 2, the HUD row, the player or a
# non-fitting socket interior (colour 15 terrain blocks only the player). Click on a block: player takes
# its rect, old player rect is left painted as an empty token (centre 4). Player edges touching a block are
# drawn 0. HUD: row 63 shows ceil(actions/2) zeros from the right. Unconfirmed: what the empty token blocks.
H, W = 64, 64
HUD_ROW = H - 1
WALL_ALL, WALL_PLAYER = {2}, {2, 15}
BORDER, PLAYER_CENTRE, BLOCK_CENTRE, EMPTY_CENTRE, RIM, HUD_ON, HUD_OFF = 4, 0, 5, 4, 14, 0, 4
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 3
_mem = {}


def rect(o):
    return [o["x"], o["y"], o["w"], o["h"]]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shifted(r, dx, dy, k=1):
    return [r[0] + dx * k, r[1] + dy * k, r[2], r[3]]


def cells(r):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            yield x, y


def interior(t):
    return [t["x"] + 1, t["y"] + 1, t["w"] - 2, t["h"] - 2]


def in_bounds(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= W and r[1] + r[3] <= HUD_ROW


def terrain_free(bg, r, walls):
    return in_bounds(r) and all(bg[y][x] not in walls for x, y in cells(r))


def socket_allows(targets, r):
    for t in targets:
        i = interior(t)
        if overlap(i, r) and not (r == i):
            return False
    return True


def background(state, frame):
    floor = max(range(16), key=lambda c: sum(row.count(c) for row in frame))
    bg = [row[:] for row in frame]
    targets = [o for o in state if o["type"] == "target"]
    for o in state:
        if o["type"] in ("player", "block"):
            for x, y in cells(rect(o)):
                on_rim = any(overlap([x, y, 1, 1], rect(t)) and not overlap([x, y, 1, 1], interior(t)) for t in targets)
                bg[y][x] = BORDER if on_rim else floor
    return bg


def hud_count(frame):
    row, n = frame[HUD_ROW], 0
    while n < W and row[W - 1 - n] == HUD_ON:
        n += 1
    return n


def slide(bg, targets, blocks, player, start, dx, dy):
    moved = False
    while True:
        group, grew = {start}, True
        while grew:
            grew = False
            for i, b in enumerate(blocks):
                if i not in group and any(overlap(shifted(blocks[j], dx, dy), b) for j in group):
                    group.add(i); grew = True
        ok = True
        for j in group:
            n = shifted(blocks[j], dx, dy)
            if not terrain_free(bg, n, WALL_ALL) or overlap(n, player) or not socket_allows(targets, n):
                ok = False
        if not ok:
            return moved
        for j in group:
            blocks[j] = shifted(blocks[j], dx, dy)
        moved = True


def draw_token(f, r, centre):
    cx = range(r[0] + (r[2] - 1) // 2, r[0] + r[2] // 2 + 1)
    cy = range(r[1] + (r[3] - 1) // 2, r[1] + r[3] // 2 + 1)
    for x, y in cells(r):
        f[y][x] = centre if (x in cx and y in cy) else RIM


def draw_player(f, p, blocks):
    draw_token(f, p, PLAYER_CENTRE)
    x0, y0, w, h = p
    for b in blocks:
        if overlap(shifted(p, -1, 0), b):
            for y in range(y0, y0 + h): f[y][x0] = PLAYER_CENTRE
        if overlap(shifted(p, 1, 0), b):
            for y in range(y0, y0 + h): f[y][x0 + w - 1] = PLAYER_CENTRE
        if overlap(shifted(p, 0, -1), b):
            for x in range(x0, x0 + w): f[y0][x] = PLAYER_CENTRE
        if overlap(shifted(p, 0, 1), b):
            for x in range(x0, x0 + w): f[y0 + h - 1][x] = PLAYER_CENTRE


def transition_function(state, action, frame):
    if _mem.get("frame") == frame:
        bg, actions = _mem["bg"], _mem["actions"]
    else:
        bg = background(state, frame)
        z = hud_count(frame)
        actions = max(0, 2 * z - 1)
    targets = [o for o in state if o["type"] == "target"]
    players = [rect(o) for o in state if o["type"] == "player"]
    blocks = [rect(o) for o in state if o["type"] == "block"]
    player = players[0] if players else None
    bg = [row[:] for row in bg]
    aid = action["action_id"] if isinstance(action, dict) else action
    if aid in DIRS and player:
        dx, dy = DIRS[aid]
        nxt = shifted(player, dx, dy, STEP)
        hit = [i for i, b in enumerate(blocks) if overlap(nxt, b)]
        if hit:
            for i in hit:
                slide(bg, targets, blocks, player, i, dx, dy)
        elif terrain_free(bg, nxt, WALL_PLAYER):
            player = nxt
    elif aid == 6 and player:
        cx, cy = action["x"], action["y"]
        for i, b in enumerate(blocks):
            if overlap(b, [cx, cy, 1, 1]):
                draw_token(bg, player, EMPTY_CENTRE)
                player = blocks.pop(i)
                break
    actions += 1
    out = [row[:] for row in bg]
    for b in blocks:
        draw_token(out, b, BLOCK_CENTRE)
    if player:
        draw_player(out, player, blocks)
    z = (actions + 1) // 2
    for x in range(W):
        out[HUD_ROW][x] = HUD_ON if x >= W - z else HUD_OFF
    _mem.update(frame=[row[:] for row in out], bg=bg, actions=actions)
    return out
