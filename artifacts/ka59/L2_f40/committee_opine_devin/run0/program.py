# Mechanics: kick/slide token game. A1-4 move the player 3 cells over floor colours {1,4} inside the portal bbox.
# If the player's next rect overlaps a block, the player stays and the hit blocks slide 1 cell per tick as a chain
# (contacted blocks join) over {1,4,15} until a member hits a wall, the player, or a non-fitting socket interior.
# A6 click on a block: player takes its rect; old player remains as an inert husk (core 4). HUD row 63 shows
# ceil(n/2) zeros from the right, n = actions since level start (parity hidden; continuity-gated). Unconfirmed: fitting socket.
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
HUD_ROW = 63
_memo = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def cells(r):
    x, y, w, h = r
    return [(i, j) for j in range(y, y + h) for i in range(x, x + w)]


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def inside(r, b):
    return r[0] >= b[0] and r[1] >= b[1] and r[0] + r[2] <= b[0] + b[2] and r[1] + r[3] <= b[1] + b[3]


def interior(s):
    return (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)


def background(frame, movers, sockets):
    covered = set()
    for r in movers:
        covered.update(cells(r))
    bg = [row[:] for row in frame]
    H, W = len(frame), len(frame[0])
    for (x, y) in covered:
        val = None
        for s in sockets:
            if inside((x, y, 1, 1), s):
                val = 1 if inside((x, y, 1, 1), interior(s)) else 4
        if val is None:
            votes = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                i, j = x + dx, y + dy
                while 0 <= i < W and 0 <= j < H and (i, j) in covered:
                    i, j = i + dx, j + dy
                if 0 <= i < W and 0 <= j < H:
                    votes.append(frame[j][i])
            val = max(votes, key=votes.count) if votes else 1
        bg[y][x] = val
    return bg


def on_floor(r, bg, bound, floor):
    return inside(r, bound) and all(bg[j][i] in floor for i, j in cells(r))


def socket_rejects(r, sockets):
    for s in sockets:
        inn = interior(s)
        if overlaps(r, inn) and (r[2], r[3]) != (inn[2], inn[3]):
            return True
    return False


def push_chain(blocks, seeds, player, d, bg, bound, sockets):
    dx, dy = d
    chain = set(seeds)
    moved = False
    while True:
        while True:
            new = {k: shift(blocks[k], dx, dy) for k in chain}
            extra = {k for k in blocks if k not in chain and any(overlaps(new[c], blocks[k]) for c in chain)}
            if not extra:
                break
            chain |= extra
        if any(not on_floor(r, bg, bound, BLOCK_FLOOR) or overlaps(r, player) or socket_rejects(r, sockets)
               for r in new.values()):
            return moved
        for k in chain:
            blocks[k] = new[k]
        moved = True
        if any(inside(blocks[k], interior(s)) and blocks[k][2:] == interior(s)[2:] for k in chain for s in sockets):
            return moved


def core(n):
    return range((n - 1) // 2, n // 2 + 1)


def draw_token(out, r, centre):
    x, y, w, h = r
    for i, j in cells(r):
        out[j][i] = 14
    for j in core(h):
        for i in core(w):
            out[y + j][x + i] = centre


def touching_sides(p, b):
    px, py, pw, ph = p
    bx, by, bw, bh = b
    sides = []
    yspan = py < by + bh and by < py + ph
    xspan = px < bx + bw and bx < px + pw
    if yspan and bx + bw == px:
        sides.append("L")
    if yspan and px + pw == bx:
        sides.append("R")
    if xspan and by + bh == py:
        sides.append("U")
    if xspan and py + ph == by:
        sides.append("D")
    return sides


def draw_player(out, p, blocks):
    draw_token(out, p, 0)
    x, y, w, h = p
    for b in blocks:
        for s in touching_sides(p, b):
            if s in "LR":
                i = x if s == "L" else x + w - 1
                for j in range(y, y + h):
                    out[j][i] = 0
            else:
                j = y if s == "U" else y + h - 1
                for i in range(x, x + w):
                    out[j][i] = 0


def hud_zeros(frame):
    row = frame[HUD_ROW]
    z = 0
    while z < len(row) and row[len(row) - 1 - z] == 0:
        z += 1
    return z


def draw_hud(out, n):
    row = out[HUD_ROW]
    W = len(row)
    z = min((n + 1) // 2, W)
    for i in range(W):
        if row[i] in (0, 4):
            row[i] = 0 if i >= W - z else 4


def transition_function(state, action, frame):
    players = [o for o in state if o.get("type") == "player"]
    blocks = {o["name"]: rect(o) for o in state if o.get("type") == "block"}
    sockets = [rect(o) for o in state if o.get("type") == "target"]
    portals = [rect(o) for o in state if o.get("type") == "portal"]
    bound = portals[0] if portals else (0, 0, len(frame[0]), len(frame) - 1)
    n = _memo["n"] if _memo["frame"] == frame else 2 * hud_zeros(frame)
    if not players:
        return [row[:] for row in frame]
    player = rect(players[0])
    movers = [player] + list(blocks.values())
    bg = background(frame, movers, sockets)
    husk = None
    if isinstance(action, int) and action in DIRS:
        dx, dy = DIRS[action]
        nxt = shift(player, 3 * dx, 3 * dy)
        hit = [k for k, b in blocks.items() if overlaps(nxt, b)]
        if hit:
            push_chain(blocks, hit, player, (dx, dy), bg, bound, sockets)
        elif on_floor(nxt, bg, bound, PLAYER_FLOOR):
            player = nxt
    elif isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action["x"], action["y"]
        for k, b in list(blocks.items()):
            if overlaps((cx, cy, 1, 1), b):
                husk = player
                player = b
                del blocks[k]
                break
    n += 1
    out = [row[:] for row in bg]
    if husk:
        draw_token(out, husk, 4)
    for b in blocks.values():
        draw_token(out, b, 5)
    draw_player(out, player, blocks.values())
    draw_hud(out, n)
    _memo["frame"] = [row[:] for row in out]
    _memo["n"] = n
    return out
