# Mechanics: arrows move the 3-cell-step player over floor colours {1,4}; if its next rect hits a block
# the block is kicked instead (player stays) and slides 3/step, chaining blocks it runs into, over {1,4,15}
# until any member would hit another colour, the player, or a non-fitting socket interior. Click on a block
# = it becomes the player; old player is left as an inert e/4 token. Player draws a 0 edge on each side touching a block.
# HUD row 63: one more 0 from the right every 2 actions (phase hidden; continuity-gated, else odd assumed).
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {"frame": None, "k": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def shift(r, d):
    return (r[0] + d[0] * STEP, r[1] + d[1] * STEP, r[2], r[3])


def background(frame, movers, sockets):
    """Frame with movable objects erased: socket cells from socket bboxes, else majority of nearest uncovered cells."""
    bg = [row[:] for row in frame]
    covered = set()
    for r in movers:
        covered.update(cells(r))
    for (x, y) in covered:
        val = None
        for s in sockets:
            if s[0] <= x < s[0] + s[2] and s[1] <= y < s[1] + s[3]:
                border = x in (s[0], s[0] + s[2] - 1) or y in (s[1], s[1] + s[3] - 1)
                val = 4 if border else 1
        if val is None:
            seen = []
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                xx, yy = x + dx, y + dy
                while 0 <= xx < 64 and 0 <= yy < 64 and (xx, yy) in covered:
                    xx, yy = xx + dx, yy + dy
                if 0 <= xx < 64 and 0 <= yy < 64:
                    seen.append(frame[yy][xx])
            val = max(seen, key=seen.count) if seen else 1
        bg[y][x] = val
    return bg


def interior(s):
    return (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)


def socket_blocks(r, sockets):
    for s in sockets:
        inn = interior(s)
        if overlap(r, inn) and (r[2], r[3]) != (inn[2], inn[3]):
            return True
    return False


def on_floor(bg, r, allowed):
    for (x, y) in cells(r):
        if not (0 <= x < 64 and 0 <= y < 64) or bg[y][x] not in allowed:
            return False
    return True


def block_blocked(bg, r, sockets, player):
    return (not on_floor(bg, r, BLOCK_FLOOR)) or socket_blocks(r, sockets) or overlap(r, player)


def push_chain(bg, blocks, start, d, sockets, player):
    """Slide the kicked block (and every block it runs into) until any member is blocked."""
    group = {start}
    while True:
        while True:
            grown = {j for j, b in enumerate(blocks) if j not in group
                     and any(overlap(shift(blocks[i], d), b) for i in group)}
            if not grown:
                break
            group |= grown
        if any(block_blocked(bg, shift(blocks[i], d), sockets, player) for i in group):
            return blocks
        blocks = [shift(b, d) if j in group else b for j, b in enumerate(blocks)]


def touching_sides(p, b):
    sides = set()
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


def centre_span(n):
    return (n // 2, n // 2) if n % 2 else (n // 2 - 1, n // 2)


def draw_token(f, r, centre_colour, zero_sides=()):
    x0, y0, w, h = r
    cx, cy = centre_span(w), centre_span(h)
    for (x, y) in cells(r):
        i, j = x - x0, y - y0
        c = centre_colour if (cx[0] <= i <= cx[1] and cy[0] <= j <= cy[1]) else 14
        if ("L" in zero_sides and i == 0) or ("R" in zero_sides and i == w - 1) \
                or ("U" in zero_sides and j == 0) or ("D" in zero_sides and j == h - 1):
            c = 0
        if 0 <= x < 64 and 0 <= y < 64:
            f[y][x] = c


def hud_zeros(frame):
    n = 0
    while n < 64 and frame[63][63 - n] == 0:
        n += 1
    return n


def draw_hud(f, k):
    z = (k + 1) // 2
    for x in range(64):
        f[63][x] = 0 if x >= 64 - z else 4


def transition_function(state, action, frame):
    players = [rect(o) for o in state if o["type"] == "player"]
    blocks = [rect(o) for o in state if o["type"] == "block"]
    sockets = [rect(o) for o in state if o["type"] == "target"]
    z = hud_zeros(frame)
    if _mem["frame"] is not None and frame == _mem["frame"]:
        k = _mem["k"]
    else:
        k = 2 * z - 1 if z > 0 else 0
    if not players:
        return frame
    player = players[0]
    bg = background(frame, players + blocks, sockets)

    if isinstance(action, dict):
        cx, cy = action.get("x"), action.get("y")
        hit = [j for j, b in enumerate(blocks) if b[0] <= cx < b[0] + b[2] and b[1] <= cy < b[1] + b[3]]
        if hit:
            draw_token(bg, player, 4)
            player = blocks[hit[0]]
            blocks = [b for j, b in enumerate(blocks) if j != hit[0]]
    elif action in DIRS:
        d = DIRS[action]
        nxt = shift(player, d)
        hit = [j for j, b in enumerate(blocks) if overlap(nxt, b)]
        if hit:
            blocks = push_chain(bg, blocks, hit[0], d, sockets, player)
        elif on_floor(bg, nxt, PLAYER_FLOOR):
            player = nxt

    out = [row[:] for row in bg]
    for b in blocks:
        draw_token(out, b, 5)
    sides = set()
    for b in blocks:
        sides |= touching_sides(player, b)
    draw_token(out, player, 0, sides)
    k += 1
    draw_hud(out, k)
    _mem["frame"] = [row[:] for row in out]
    _mem["k"] = k
    return out
