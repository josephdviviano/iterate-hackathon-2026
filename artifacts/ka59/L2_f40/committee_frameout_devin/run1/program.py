# Mechanics: bg colours 1 floor, 15 wall for player only, 2 wall for all, 4 socket border/HUD row 63. Player moves 3;
# if its next rect hits a block it kicks it (player stays): block slides 3/tick chaining blocks it hits until a member
# would hit colour 2/edge/player/a socket interior (target inset 1) it doesn't fit. Click block = player takes its rect,
# old player left drawn with centre 4. Player edges touching a block drawn 0; HUD = ceil(n/2) zeros from right (n hidden, continuity-gated).
# Unconfirmed: fitting-socket entry, ghost collisions, actions 5/7 (treated as no-op ticks), fallback n=2*zeros.
import copy

STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
PLAYER_WALLS = (2, 15)
BLOCK_WALLS = (2,)
HUD_Y = 63
_mem = {"frame": None, "bg": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def cells(r):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            yield x, y


def socket_interiors(targets):
    return [(t["x"] + 1, t["y"] + 1, t["w"] - 2, t["h"] - 2) for t in targets]


def on_socket_border(x, y, targets):
    for t in targets:
        r = rect(t)
        if overlap((x, y, 1, 1), r) and not overlap((x, y, 1, 1), (r[0] + 1, r[1] + 1, r[2] - 2, r[3] - 2)):
            return True
    return False


def hits_wall(bg, r, walls):
    for x, y in cells(r):
        if not (0 <= x < 64 and 0 <= y < HUD_Y) or bg[y][x] in walls:
            return True
    return False


def socket_blocks(r, interiors):
    return any(overlap(r, s) and (r[2], r[3]) != (s[2], s[3]) for s in interiors)


def block_can_move(bg, r, player, interiors):
    return not (hits_wall(bg, r, BLOCK_WALLS) or overlap(r, player) or socket_blocks(r, interiors))


def push_chain(blocks, group, d):
    group = set(group)
    grew = True
    while grew:
        grew = False
        for i, b in enumerate(blocks):
            if i not in group and any(overlap(shift(blocks[j], d), b) for j in group):
                group.add(i)
                grew = True
    return group


def slide(bg, blocks, start, d, player, interiors):
    blocks = list(blocks)
    group = {start}
    while True:
        group = push_chain(blocks, group, d)
        if not all(block_can_move(bg, shift(blocks[i], d), player, interiors) for i in group):
            return blocks
        for i in group:
            blocks[i] = shift(blocks[i], d)


def centre_range(n):
    return range((n - 1) // 2, n // 2 + 1) if n % 2 else range(n // 2 - 1, n // 2 + 1)


def draw_token(fr, r, centre):
    x0, y0, w, h = r
    for x, y in cells(r):
        if 0 <= x < 64 and 0 <= y < 64:
            fr[y][x] = centre if (x - x0) in centre_range(w) and (y - y0) in centre_range(h) else 14


def touching_sides(p, blocks):
    sides = set()
    for b in blocks:
        if p[1] < b[1] + b[3] and b[1] < p[1] + p[3]:
            if b[0] + b[2] == p[0]:
                sides.add("L")
            if p[0] + p[2] == b[0]:
                sides.add("R")
        if p[0] < b[0] + b[2] and b[0] < p[0] + p[2]:
            if b[1] + b[3] == p[1]:
                sides.add("T")
            if p[1] + p[3] == b[1]:
                sides.add("B")
    return sides


def draw_player(fr, p, blocks):
    draw_token(fr, p, 0)
    x0, y0, w, h = p
    sides = touching_sides(p, blocks)
    for x, y in cells(p):
        if (("L" in sides and x == x0) or ("R" in sides and x == x0 + w - 1)
                or ("T" in sides and y == y0) or ("B" in sides and y == y0 + h - 1)):
            fr[y][x] = 0


def hud_zeros(frame):
    z = 0
    for x in range(63, -1, -1):
        if frame[HUD_Y][x] != 0:
            break
        z += 1
    return z


def erase_background(frame, sprites, targets):
    bg = copy.deepcopy(frame)
    for r in sprites:
        for x, y in cells(r):
            if 0 <= x < 64 and 0 <= y < 64:
                bg[y][x] = 4 if on_socket_border(x, y, targets) else 1
    return bg


def transition_function(state, action, frame):
    players = [o for o in state if o["type"] == "player"]
    block_objs = [o for o in state if o["type"] == "block"]
    targets = [o for o in state if o["type"] == "target"]
    blocks = [rect(o) for o in block_objs]
    player = rect(players[0]) if players else None
    sprites = blocks + ([player] if player else [])

    if _mem["frame"] is not None and frame == _mem["frame"]:
        bg, n = copy.deepcopy(_mem["bg"]), _mem["n"]
    else:
        bg = erase_background(frame, sprites, targets)
        n = 2 * hud_zeros(frame)
    interiors = socket_interiors(targets)

    if isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        hit = [i for i, b in enumerate(blocks) if overlap((cx, cy, 1, 1), b)]
        if hit and player:
            ghost = copy.deepcopy(frame)
            for x, y in cells(player):
                bg[y][x] = ghost[y][x]
            x0, y0, w, h = player
            for x, y in cells(player):
                if (x - x0) in centre_range(w) and (y - y0) in centre_range(h):
                    bg[y][x] = 4
            player = blocks.pop(hit[0])
    elif action in DIRS and player:
        d = DIRS[action]
        nxt = shift(player, d)
        kicked = [i for i, b in enumerate(blocks) if overlap(nxt, b)]
        if kicked:
            blocks = slide(bg, blocks, kicked[0], d, player, interiors)
        elif not hits_wall(bg, nxt, PLAYER_WALLS):
            player = nxt

    n += 1
    out = copy.deepcopy(bg)
    for b in blocks:
        draw_token(out, b, 5)
    if player:
        draw_player(out, player, blocks)
    z = min(64, (n + 1) // 2)
    for x in range(64):
        out[HUD_Y][x] = 0 if x >= 64 - z else 4
    _mem.update(frame=copy.deepcopy(out), bg=bg, n=n)
    return out
