# Mechanics: arrows move the player 3 cells over floor colours {1,4}; if its next rect hits a block, that block is
# kicked instead (player stays) and slides 3/tick with a push chain until any member would leave {1,4,15},
# overlap the player, or enter a socket interior it does not fit. Click on a block: player takes its rect, old
# player stays as an inert remnant (core 4). Player sprite: e ring, core 0, sides touching a block drawn 0.
# HUD row 63: zeros from the right = (actions+1)//2 (continuity-tracked). Unconfirmed: fitting socket entry.
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
E, CORE_BLOCK, CORE_PLAYER, CORE_REMNANT, BG = 14, 5, 0, 4, 1
_mem = {"frame": None, "bg": None, "acts": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d, k=3):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def interior(s):
    return (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)


def on_board(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= 64 and r[1] + r[3] <= 63


def floor_ok(bg, r, allowed):
    return on_board(r) and all(bg[y][x] in allowed for x, y in cells(r))


def socket_ok(r, sockets):
    for s in sockets:
        i = interior(s)
        if overlap(r, i) and (r[2], r[3]) != (i[2], i[3]):
            return False
    return True


def block_can_enter(bg, r, player, sockets):
    return floor_ok(bg, r, BLOCK_FLOOR) and not overlap(r, player) and socket_ok(r, sockets)


def push_chain(blocks, start, d):
    group = set(start)
    grew = True
    while grew:
        grew = False
        for j, b in enumerate(blocks):
            if j not in group and any(overlap(shift(blocks[i], d), b) for i in group):
                group.add(j)
                grew = True
    return group


def slide(bg, blocks, start, d, player, sockets):
    blocks = list(blocks)
    while True:
        group = push_chain(blocks, start, d)
        moved = [shift(blocks[i], d) for i in group]
        if not all(block_can_enter(bg, r, player, sockets) for r in moved):
            return blocks
        for i in group:
            blocks[i] = shift(blocks[i], d)


def core(r):
    xs = range((r[2] - 1) // 2, r[2] // 2 + 1)
    ys = range((r[3] - 1) // 2, r[3] // 2 + 1)
    return {(r[0] + i, r[1] + j) for i in xs for j in ys}


def draw_token(fr, r, core_col):
    c = core(r)
    for x, y in cells(r):
        fr[y][x] = core_col if (x, y) in c else E


def touching_sides(p, blocks):
    sides = set()
    for b in blocks:
        yo = p[1] < b[1] + b[3] and b[1] < p[1] + p[3]
        xo = p[0] < b[0] + b[2] and b[0] < p[0] + p[2]
        if yo and b[0] + b[2] == p[0]:
            sides.add("L")
        if yo and p[0] + p[2] == b[0]:
            sides.add("R")
        if xo and b[1] + b[3] == p[1]:
            sides.add("U")
        if xo and p[1] + p[3] == b[1]:
            sides.add("D")
    return sides


def draw_player(fr, p, blocks):
    draw_token(fr, p, CORE_PLAYER)
    sides = touching_sides(p, blocks)
    for x, y in cells(p):
        if ("L" in sides and x == p[0]) or ("R" in sides and x == p[0] + p[2] - 1) or \
           ("U" in sides and y == p[1]) or ("D" in sides and y == p[1] + p[3] - 1):
            fr[y][x] = CORE_PLAYER


def background(frame, objs, sockets):
    bg = [row[:] for row in frame]
    for o in objs:
        for x, y in cells(rect(o)):
            on_border = any(s[0] <= x < s[0] + s[2] and s[1] <= y < s[1] + s[3] and not
                            (interior(s)[0] <= x < interior(s)[0] + interior(s)[2] and
                             interior(s)[1] <= y < interior(s)[1] + interior(s)[3]) for s in sockets)
            bg[y][x] = 4 if on_border else BG
    return bg


def hud_zeros(frame):
    n = 0
    for x in range(63, -1, -1):
        if frame[63][x] != 0:
            break
        n += 1
    return n


def transition_function(state, action, frame):
    sockets = [rect(o) for o in state if o["type"] == "target"]
    players = [o for o in state if o["type"] == "player"]
    blocks_o = [o for o in state if o["type"] == "block"]
    zeros = hud_zeros(frame)
    if _mem["frame"] == frame and _mem["bg"] is not None:
        bg, acts = _mem["bg"], _mem["acts"]
    else:
        bg = background(frame, players + blocks_o, sockets)
        acts = 0 if zeros == 0 else 2 * zeros - 1
    player = rect(players[0]) if players else None
    blocks = [rect(o) for o in blocks_o]
    bg = [row[:] for row in bg]

    if isinstance(action, dict) and player is not None:
        cx, cy = action.get("x", -1), action.get("y", -1)
        hit = [i for i, b in enumerate(blocks) if b[0] <= cx < b[0] + b[2] and b[1] <= cy < b[1] + b[3]]
        if hit:
            draw_token(bg, player, CORE_REMNANT)
            player = blocks.pop(hit[0])
    elif action in DIRS and player is not None:
        d = DIRS[action]
        nxt = shift(player, d)
        kicked = [i for i, b in enumerate(blocks) if overlap(nxt, b)]
        if kicked:
            blocks = slide(bg, blocks, kicked, d, player, sockets)
        elif floor_ok(bg, nxt, PLAYER_FLOOR):
            player = nxt

    acts += 1
    out = [row[:] for row in bg]
    for b in blocks:
        draw_token(out, b, CORE_BLOCK)
    if player is not None:
        draw_player(out, player, blocks)
    z = (acts + 1) // 2
    for x in range(64):
        out[63][x] = 0 if x >= 64 - z else 4
    _mem.update(frame=[row[:] for row in out], bg=bg, acts=acts)
    return out
