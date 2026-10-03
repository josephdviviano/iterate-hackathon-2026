# Mechanics: arrows (1-4) step the player 3 cells over floor (bg colour 1, socket cells); colour 15/2 and the HUD row block it.
# If the player's next rect hits a block, the block is kicked: the pushed chain slides 1 cell at a time over colours 1/4/15
# until any member would hit a wall, the player, or a socket interior it does not exactly fit; the player stays.
# Click on a block: player takes its rect, block vanishes, old body stays as a shell (centre 4, not an object). Player edges facing a touching block draw 0.
# HUD row 63 fills with 0 from the right: zeros = (n+1)//2, n = actions since level start (hidden, continuity-gated). Unconfirmed: fitting-socket fill, player on 15.
E, BLACK, WHITE, FLOOR, PAD, SAND = 14, 5, 0, 1, 4, 15
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_last = {"frame": None, "n": 0}


def rect(o):
    return [o["x"], o["y"], o["w"], o["h"]]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def interior(t):
    return [t[0] + 1, t[1] + 1, t[2] - 2, t[3] - 2]


def on_border(x, y, t):
    return (t[0] <= x < t[0] + t[2] and t[1] <= y < t[1] + t[3]
            and (x in (t[0], t[0] + t[2] - 1) or y in (t[1], t[1] + t[3] - 1)))


def in_rect(x, y, t):
    return t[0] <= x < t[0] + t[2] and t[1] <= y < t[1] + t[3]


def background(frame, movers, sockets):
    bg = [row[:] for row in frame]
    for r in movers:
        for x, y in cells(r):
            if 0 <= x < 64 and 0 <= y < 64:
                bg[y][x] = PAD if any(on_border(x, y, t) for t in sockets) else FLOOR
    return bg


def hud_row(bg):
    return len(bg) - 1


def cell_ok(bg, x, y, allowed, sockets):
    if not (0 <= x < 64 and 0 <= y < hud_row(bg)):
        return False
    return bg[y][x] in allowed or any(in_rect(x, y, t) for t in sockets)


def player_can_stand(bg, r, sockets):
    return all(cell_ok(bg, x, y, (FLOOR,), sockets) for x, y in cells(r))


def block_can_stand(bg, r, sockets):
    if not all(cell_ok(bg, x, y, (FLOOR, SAND), sockets) for x, y in cells(r)):
        return False
    for t in sockets:
        i = interior(t)
        if overlap(r, i) and (r[2], r[3]) != (i[2], i[3]):
            return False
    return True


def shift(r, d):
    return [r[0] + d[0], r[1] + d[1], r[2], r[3]]


def slide_chain(bg, blocks, start, d, player, sockets):
    group = set(start)
    while True:
        while True:
            moved = [shift(blocks[i], d) for i in group]
            joins = {j for j in range(len(blocks)) if j not in group
                     and any(overlap(m, blocks[j]) for m in moved)}
            if not joins:
                break
            group |= joins
        if any(overlap(m, player) or not block_can_stand(bg, m, sockets) for m in moved):
            return
        for i in group:
            blocks[i] = shift(blocks[i], d)


def step(bg, player, blocks, action, sockets):
    if not isinstance(action, dict) and action in DIRS:
        d = DIRS[action]
        nxt = shift(player, (3 * d[0], 3 * d[1]))
        hit = [i for i, b in enumerate(blocks) if overlap(nxt, b)]
        if hit:
            slide_chain(bg, blocks, hit, d, player, sockets)
        elif player_can_stand(bg, nxt, sockets):
            player = nxt
    elif isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        for i, b in enumerate(blocks):
            if in_rect(cx, cy, b):
                draw_block(bg, player, PAD)
                player = b
                blocks = blocks[:i] + blocks[i + 1:]
                break
    return player, blocks


def centre(n):
    return {(n - 1) // 2, n // 2}


def draw_block(f, r, fill):
    cx, cy = centre(r[2]), centre(r[3])
    for x, y in cells(r):
        if 0 <= x < 64 and 0 <= y < 64:
            f[y][x] = fill if (x - r[0]) in cx and (y - r[1]) in cy else E


def touching(r, blocks, d):
    x0, y0, w, h = r
    if d == (-1, 0):
        edge = [x0 - 1, y0, 1, h]
    elif d == (1, 0):
        edge = [x0 + w, y0, 1, h]
    elif d == (0, -1):
        edge = [x0, y0 - 1, w, 1]
    else:
        edge = [x0, y0 + h, w, 1]
    return any(overlap(edge, b) for b in blocks)


def draw_player(f, r, blocks):
    draw_block(f, r, WHITE)
    x0, y0, w, h = r
    for d in DIRS.values():
        if not touching(r, blocks, d):
            continue
        if d[0]:
            xs = [x0 if d[0] < 0 else x0 + w - 1]
            ys = range(y0, y0 + h)
        else:
            ys = [y0 if d[1] < 0 else y0 + h - 1]
            xs = range(x0, x0 + w)
        for y in ys:
            for x in xs:
                f[y][x] = WHITE


def draw_hud(f, n):
    y = hud_row(f)
    zeros = (n + 1) // 2
    for x in range(64):
        f[y][x] = WHITE if x >= 64 - zeros else PAD


def prior_actions(frame):
    zeros = sum(1 for v in frame[hud_row(frame)] if v == WHITE)
    if _last["frame"] == frame:
        return _last["n"]
    return 0 if zeros == 0 else 2 * zeros - 1


def transition_function(state, action, frame):
    n = prior_actions(frame) + 1
    player = next(rect(o) for o in state if o["type"] == "player")
    blocks = [rect(o) for o in state if o["type"] == "block"]
    sockets = [rect(o) for o in state if o["type"] == "target"]
    bg = background(frame, [player] + blocks, sockets)
    player, blocks = step(bg, player, blocks, action, sockets)
    out = [row[:] for row in bg]
    for b in blocks:
        draw_block(out, b, BLACK)
    draw_player(out, player, blocks)
    draw_hud(out, n)
    _last["frame"] = [row[:] for row in out]
    _last["n"] = n
    return out
