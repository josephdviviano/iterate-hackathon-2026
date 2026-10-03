# Mechanics: arrows 1-4 move the player 3 cells; if its next rect overlaps blocks it KICKS them instead
# (player stays) and the push chain slides 1 cell/tick, absorbing blocks it hits, until any member meets
# a non-floor cell, the board edge, the player, or a socket interior (target bbox inset 1) it does not fit.
# Floors read from the frame: player walks {1,4}, blocks slide over {1,4,15}. Click on a block = player
# takes its rect, old player stays as an inert husk (e + centre 4). HUD row 63: ceil(n/2) zeros, n = actions.
# Unconfirmed: fitting-socket completion, actions 5/7, husk collisions (treated as wall), non-block clicks.
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
N = 64
_memo = {"frame": None, "bg": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def cells(r):
    x, y, w, h = r
    return [(i, j) for j in range(y, y + h) for i in range(x, x + w)]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d, k):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def in_board(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= N - 1 and r[1] + r[3] <= N - 1


def interior(t):
    return (t["x"] + 1, t["y"] + 1, t["w"] - 2, t["h"] - 2)


def socket_rejects(r, sockets):
    for t in sockets:
        inn = interior(t)
        if overlap(r, inn) and (r[2], r[3]) != (inn[2], inn[3]):
            return True
    return False


def on_floor(r, bg, floor):
    return in_board(r) and all(bg[j][i] in floor for i, j in cells(r))


def guess_bg(frame, covered, sockets):
    bg = [row[:] for row in frame]
    for (i, j) in covered:
        val = None
        for t in sockets:
            x, y, w, h = rect(t)
            if x <= i < x + w and y <= j < y + h:
                border = i in (x, x + w - 1) or j in (y, y + h - 1)
                val = 4 if border else 1
        if val is None:
            seen = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                a, b = i + dx, j + dy
                while 0 <= a < N and 0 <= b < N:
                    if (a, b) not in covered and frame[b][a] in (1, 2, 15):
                        seen.append(frame[b][a])
                        break
                    a, b = a + dx, b + dy
            val = max(seen, key=seen.count) if seen else 1
        bg[j][i] = val
    return bg


def push_chain(seed, blocks, player, d, bg, sockets):
    moving = set(seed)
    pos = {k: r for k, r in blocks.items()}
    moved = False
    while True:
        grow = True
        while grow:
            grow = False
            for k in list(moving):
                nr = shift(pos[k], d, 1)
                for o, r in pos.items():
                    if o not in moving and overlap(nr, r):
                        moving.add(o)
                        grow = True
        nxt = {k: shift(pos[k], d, 1) for k in moving}
        if any(not on_floor(r, bg, BLOCK_FLOOR) or overlap(r, player) or socket_rejects(r, sockets)
               for r in nxt.values()):
            return pos, moved
        pos.update(nxt)
        moved = True
        if any(any(r == interior(t) for t in sockets) for r in nxt.values()):
            return pos, moved


def step(player, blocks, action, bg, sockets):
    """Returns (player rect, blocks dict, husk rect or None)."""
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            cx, cy = action.get("x"), action.get("y")
            for k, r in blocks.items():
                if r[0] <= cx < r[0] + r[2] and r[1] <= cy < r[1] + r[3]:
                    nb = dict(blocks)
                    del nb[k]
                    return r, nb, player
        return player, blocks, None
    if action not in DIRS:
        return player, blocks, None
    d = DIRS[action]
    nr = shift(player, d, 3)
    hit = [k for k, r in blocks.items() if overlap(nr, r)]
    if hit:
        pos, _ = push_chain(hit, blocks, player, d, bg, sockets)
        return player, pos, None
    if on_floor(nr, bg, PLAYER_FLOOR):
        return nr, blocks, None
    return player, blocks, None


def centre_cells(r):
    x, y, w, h = r
    return [(x + i, y + j) for j in range((h - 1) // 2, h // 2 + 1) for i in range((w - 1) // 2, w // 2 + 1)]


def draw_token(out, r, centre):
    for i, j in cells(r):
        out[j][i] = 14
    for i, j in centre_cells(r):
        out[j][i] = centre


def draw_player(out, r, blocks):
    draw_token(out, r, 0)
    x, y, w, h = r
    for b in blocks:
        bx, by, bw, bh = b
        xs = bx < x + w and x < bx + bw
        ys = by < y + h and y < by + bh
        if xs and by + bh == y:
            for i in range(x, x + w): out[y][i] = 0
        if xs and by == y + h:
            for i in range(x, x + w): out[y + h - 1][i] = 0
        if ys and bx + bw == x:
            for j in range(y, y + h): out[j][x] = 0
        if ys and bx == x + w:
            for j in range(y, y + h): out[j][x + w - 1] = 0


def hud_zeros(frame):
    z = 0
    for i in range(N - 1, -1, -1):
        if frame[N - 1][i] != 0:
            break
        z += 1
    return z


def transition_function(state, action, frame):
    frame = [[int(v) for v in row] for row in frame]
    players = [o for o in state if o.get("type") == "player"]
    blocks = {o["name"]: rect(o) for o in state if o.get("type") == "block"}
    sockets = [o for o in state if o.get("type") == "target"]
    if not players:
        return frame
    player = rect(players[0])
    covered = set(cells(player))
    for r in blocks.values():
        covered.update(cells(r))
    continuous = _memo["frame"] == frame
    if continuous:
        bg = [row[:] for row in _memo["bg"]]
        n = _memo["n"]
    else:
        bg = guess_bg(frame, covered, sockets)
        z = hud_zeros(frame)
        n = 0 if z == 0 else 2 * z - 1
    for j in range(N):
        for i in range(N):
            if (i, j) not in covered:
                bg[j][i] = frame[j][i]
    np_, nb, husk = step(player, blocks, action, bg, sockets)
    if husk is not None:
        draw_token(bg, husk, 4)
    out = [row[:] for row in frame]
    for i, j in covered:
        out[j][i] = bg[j][i]
    for r in nb.values():
        draw_token(out, r, 5)
    draw_player(out, np_, list(nb.values()))
    n += 1
    z = (n + 1) // 2
    for i in range(N):
        out[N - 1][i] = 0 if i >= N - z else bg[N - 1][i] if bg[N - 1][i] != 0 else 4
    _memo.update(frame=[row[:] for row in out], bg=bg, n=n)
    return out
