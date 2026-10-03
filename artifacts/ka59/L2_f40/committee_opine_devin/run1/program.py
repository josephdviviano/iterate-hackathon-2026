# Mechanics: arrows move the player 3 cells over floor colours {1,4} inside the portal bbox; a move whose
# next rect hits a block kicks it instead (player stays): the hit chain slides 1 cell per tick (closure over
# touched blocks) over {1,4,15} until terrain, bounds, the player or a non-fitting socket interior stops it.
# Click on a block: the player takes its rect, the old player stays as an inert husk (e + centre 4). HUD row 63
# = ceil(n/2) zeros from the right, n = actions this level (parity hidden: continuity-gated, fallback n=2z).
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
_memo = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def cells(r):
    x, y, w, h = r
    return [(i, j) for j in range(y, y + h) for i in range(x, x + w)]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def interior(r):
    return (r[0] + 1, r[1] + 1, r[2] - 2, r[3] - 2)


def background(frame, movers, sockets, bound):
    bg = [row[:] for row in frame]
    covered = set()
    for r in movers:
        covered.update(cells(r))
    H, W = len(frame), len(frame[0])
    for (x, y) in covered:
        sock = next((s for s in sockets if overlap((x, y, 1, 1), s)), None)
        if sock:
            bg[y][x] = 1 if overlap((x, y, 1, 1), interior(sock)) else 4
            continue
        votes = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            i, j = x + dx, y + dy
            while 0 <= i < W and 0 <= j < H - 1 and (i, j) in covered:
                i, j = i + dx, j + dy
            if 0 <= i < W and 0 <= j < H - 1:
                votes.append(frame[j][i])
        if votes:
            bg[y][x] = max(votes, key=lambda c: (votes.count(c), -votes.index(c)))
    return bg


def inside(r, bound):
    return bound[0] <= r[0] and bound[1] <= r[1] and r[0] + r[2] <= bound[0] + bound[2] \
        and r[1] + r[3] <= bound[1] + bound[3]


def floor_ok(bg, r, floor):
    return all(bg[y][x] in floor for x, y in cells(r))


def socket_rejects(r, sockets):
    for s in sockets:
        inn = interior(s)
        if overlap(r, inn) and (r[2], r[3]) != (inn[2], inn[3]):
            return True
    return False


def block_blocked(r, bg, bound, player, sockets):
    return (not inside(r, bound) or not floor_ok(bg, r, BLOCK_FLOOR)
            or overlap(r, player) or socket_rejects(r, sockets))


def push_chain(blocks, seed, dx, dy, bg, bound, player, sockets):
    while True:
        chain = set(seed)
        grew = True
        while grew:
            grew = False
            for k in range(len(blocks)):
                if k not in chain and any(overlap(shift(blocks[c], dx, dy), blocks[k]) for c in chain):
                    chain.add(k)
                    grew = True
        if any(block_blocked(shift(blocks[c], dx, dy), bg, bound, player, sockets) for c in chain):
            return blocks
        blocks = [shift(b, dx, dy) if k in chain else b for k, b in enumerate(blocks)]


def paint(out, r, centre):
    x, y, w, h = r
    for i, j in cells(r):
        out[j][i] = 14
    for j in range(y + (h - 1) // 2, y + h // 2 + 1):
        for i in range(x + (w - 1) // 2, x + w // 2 + 1):
            out[j][i] = centre


def touching_sides(p, b):
    px, py, pw, ph = p
    bx, by, bw, bh = b
    sides = []
    vert = py < by + bh and by < py + ph
    horz = px < bx + bw and bx < px + pw
    if vert and bx + bw == px:
        sides.append("L")
    if vert and bx == px + pw:
        sides.append("R")
    if horz and by + bh == py:
        sides.append("U")
    if horz and by == py + ph:
        sides.append("D")
    return sides


def paint_player(out, p, blocks):
    paint(out, p, 0)
    x, y, w, h = p
    sides = set()
    for b in blocks:
        sides.update(touching_sides(p, b))
    for s in sides:
        if s in "LR":
            i = x if s == "L" else x + w - 1
            for j in range(y, y + h):
                out[j][i] = 0
        else:
            j = y if s == "U" else y + h - 1
            for i in range(x, x + w):
                out[j][i] = 0


def hud_count(frame):
    row = frame[-1]
    z = 0
    while z < len(row) and row[len(row) - 1 - z] == 0:
        z += 1
    return 2 * z


def transition_function(state, action, frame):
    player = next(rect(o) for o in state if o["type"] == "player")
    blocks = sorted((rect(o) for o in state if o["type"] == "block"), key=lambda r: (r[1], r[0]))
    sockets = [rect(o) for o in state if o["type"] == "target"]
    portal = next((rect(o) for o in state if o["type"] == "portal"), (0, 0, 63, 63))
    n = _memo["n"] if frame == _memo["frame"] else hud_count(frame)
    bg = background(frame, blocks + [player], sockets, portal)

    if isinstance(action, dict):
        cx, cy = action["x"], action["y"]
        hit = [k for k, b in enumerate(blocks) if overlap((cx, cy, 1, 1), b)]
        if hit:
            paint(bg, player, 4)
            player = blocks.pop(hit[0])
    elif action in DIRS:
        dx, dy = DIRS[action]
        nxt = shift(player, 3 * dx, 3 * dy)
        seed = [k for k, b in enumerate(blocks) if overlap(nxt, b)]
        if seed:
            blocks = push_chain(blocks, seed, dx, dy, bg, portal, player, sockets)
        elif inside(nxt, portal) and floor_ok(bg, nxt, PLAYER_FLOOR):
            player = nxt

    out = [row[:] for row in bg]
    for b in blocks:
        paint(out, b, 5)
    paint_player(out, player, blocks)
    n += 1
    z = (n + 1) // 2
    row = out[-1]
    for i in range(len(row)):
        if i >= len(row) - z:
            row[i] = 0
    _memo["frame"], _memo["n"] = out, n
    return out
