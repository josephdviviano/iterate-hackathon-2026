# Mechanics: arrows move the player 3 cells over floor colours {1,4} inside the portal bbox; if the
# destination overlaps a block it is a kick: the player stays and the block slides 1 cell per tick,
# absorbing every block it touches, until any member would hit a non-{1,4,15} cell, the bbox, the
# player or a socket interior (inset 1) it does not exactly fit. Click on a block: the player takes it,
# the old player stays as an inert husk (centre 4). HUD row 63: ceil(n/2) zeros from the right, n =
# actions this level (hidden parity, continuity-gated; fallback n=2z-1). Unconfirmed: socket docking.
import copy

PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 3
MEMO = {}


def rect(o):
    return [o["x"], o["y"], o["w"], o["h"]]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def shifted(r, d, k):
    return [r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3]]


def inside(r, bound):
    return (r[0] >= bound[0] and r[1] >= bound[1] and r[0] + r[2] <= bound[0] + bound[2]
            and r[1] + r[3] <= bound[1] + bound[3])


def interior(s):
    return [s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2]


def socket_guard(r, sockets):
    for s in sockets:
        i = interior(s)
        if overlap(r, i) and (r[2], r[3]) != (i[2], i[3]):
            return False
    return True


def terrain_ok(r, bg, floor, bound):
    return inside(r, bound) and all(bg[y][x] in floor for x, y in cells(r))


def infer_background(frame, movers, sockets):
    bg = [row[:] for row in frame]
    covered = set()
    for r in movers:
        covered.update(cells(r))
    H, W = len(frame), len(frame[0])
    for x, y in covered:
        sock = [s for s in sockets if overlap([x, y, 1, 1], s)]
        if sock:
            bg[y][x] = 1 if overlap([x, y, 1, 1], interior(sock[0])) else 4
            continue
        votes = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cx, cy = x + dx, y + dy
            while 0 <= cx < W and 0 <= cy < H - 1 and (cx, cy) in covered:
                cx, cy = cx + dx, cy + dy
            if 0 <= cx < W and 0 <= cy < H - 1:
                votes.append(frame[cy][cx])
        bg[y][x] = max(votes, key=votes.count) if votes else 1
    return bg


def slide_chain(first, blocks, player, d, bg, bound, sockets):
    chain = {first}
    while True:
        grew = True
        while grew:
            grew = False
            nxt = [shifted(blocks[i], d, 1) for i in chain]
            for j, b in enumerate(blocks):
                if j not in chain and any(overlap(n, b) for n in nxt):
                    chain.add(j)
                    grew = True
        nxt = {i: shifted(blocks[i], d, 1) for i in chain}
        if not all(terrain_ok(n, bg, BLOCK_FLOOR, bound) and not overlap(n, player)
                   and socket_guard(n, sockets) for n in nxt.values()):
            return
        for i, n in nxt.items():
            blocks[i] = n
        if any(n == interior(s) for n in nxt.values() for s in sockets):
            return


def step_player(action, player, blocks, bg, bound, sockets):
    d = DIRS[action]
    dest = shifted(player, d, STEP)
    hit = [i for i, b in enumerate(blocks) if overlap(dest, b)]
    if hit:
        slide_chain(hit[0], blocks, player, d, bg, bound, sockets)
        return player
    if terrain_ok(dest, bg, PLAYER_FLOOR, bound):
        return dest
    return player


def centre_cells(r):
    xs = range(r[0] + (r[2] - 1) // 2, r[0] + r[2] // 2 + 1)
    ys = range(r[1] + (r[3] - 1) // 2, r[1] + r[3] // 2 + 1)
    return [(x, y) for y in ys for x in xs]


def draw_token(out, r, centre):
    for x, y in cells(r):
        out[y][x] = 14
    for x, y in centre_cells(r):
        out[y][x] = centre


def touching_sides(p, b):
    sides = []
    xspan = p[0] < b[0] + b[2] and b[0] < p[0] + p[2]
    yspan = p[1] < b[1] + b[3] and b[1] < p[1] + p[3]
    if yspan and b[0] + b[2] == p[0]:
        sides.append("L")
    if yspan and p[0] + p[2] == b[0]:
        sides.append("R")
    if xspan and b[1] + b[3] == p[1]:
        sides.append("U")
    if xspan and p[1] + p[3] == b[1]:
        sides.append("D")
    return sides


def draw_player(out, p, blocks):
    draw_token(out, p, 0)
    x0, y0, x1, y1 = p[0], p[1], p[0] + p[2] - 1, p[1] + p[3] - 1
    for b in blocks:
        for s in touching_sides(p, b):
            for x, y in cells(p):
                if (s == "L" and x == x0) or (s == "R" and x == x1) or \
                   (s == "U" and y == y0) or (s == "D" and y == y1):
                    out[y][x] = 0


def hud_zeros(frame):
    row = frame[-1]
    z = 0
    while z < len(row) and row[len(row) - 1 - z] == 0:
        z += 1
    return z


def draw_hud(out, n):
    z = (n + 1) // 2
    W = len(out[-1])
    for x in range(W):
        out[-1][x] = 0 if x >= W - z else 4


def transition_function(state, action, frame):
    player = None
    blocks, sockets = [], []
    bound = [0, 0, len(frame[0]) - 1, len(frame) - 1]
    for o in state:
        if o["type"] == "player":
            player = rect(o)
        elif o["type"] == "block":
            blocks.append(rect(o))
        elif o["type"] == "target":
            sockets.append(rect(o))
        elif o["type"] == "portal":
            bound = rect(o)
    movers = blocks + ([player] if player else [])
    if MEMO.get("frame") == frame:
        bg, n = copy.deepcopy(MEMO["bg"]), MEMO["n"]
    else:
        z = hud_zeros(frame)
        bg, n = infer_background(frame, movers, sockets), (2 * z - 1 if z else 0)
    n += 1
    if player is not None:
        if isinstance(action, dict):
            cx, cy = action.get("x"), action.get("y")
            hit = [i for i, b in enumerate(blocks) if overlap([cx, cy, 1, 1], b)]
            if hit:
                draw_token(bg, player, 4)
                player = blocks.pop(hit[0])
        elif action in DIRS:
            player = step_player(action, player, blocks, bg, bound, sockets)
    out = [row[:] for row in bg]
    for b in blocks:
        draw_token(out, b, 5)
    if player is not None:
        draw_player(out, player, blocks)
    draw_hud(out, n)
    MEMO.clear()
    MEMO.update(frame=[row[:] for row in out], bg=bg, n=n)
    return out
