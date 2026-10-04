# Mechanics: arrows move the player 3 cells over floor colours {1,4}; a move whose rect hits a block
# kicks it: the push chain slides 1 cell per tick until a member hits colour 2/edge/player or a
# socket interior (inset 1) it does not fit. Click on a block = player possesses it; old player stays
# as an inert husk (rim e, centre 4). HUD row 63: ceil(n/2) zeros from x=63, n = actions this level.
# Unconfirmed: fitting-socket docking, husk collisions, actions 5/7 (treated as no-ops but counted).
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 3
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
SIZE = 64
_memo = {"frame": None, "n": 0}


def rect(o):
    return [o["x"], o["y"], o["w"], o["h"]]


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def shifted(r, dx, dy, k=1):
    return [r[0] + dx * k, r[1] + dy * k, r[2], r[3]]


def in_board(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= SIZE - 1 and r[1] + r[3] <= SIZE - 1


def interior(s):
    return [s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2]


def terrain(frame, movers, sockets):
    """Background frame with movable sprites erased (socket rim 4, interior 1, else nearest uncovered)."""
    covered = set()
    for r in movers:
        covered.update(cells(r))
    bg = [row[:] for row in frame]
    for (x, y) in covered:
        val = None
        for s in sockets:
            if s[0] <= x < s[0] + s[2] and s[1] <= y < s[1] + s[3]:
                border = x in (s[0], s[0] + s[2] - 1) or y in (s[1], s[1] + s[3] - 1)
                val = 4 if border else 1
        if val is None:
            votes = {}
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cx, cy = x + dx, y + dy
                while 0 <= cx < SIZE and 0 <= cy < SIZE and (cx, cy) in covered:
                    cx, cy = cx + dx, cy + dy
                if 0 <= cx < SIZE and 0 <= cy < SIZE:
                    c = frame[cy][cx]
                    votes[c] = votes.get(c, 0) + 1
            val = max(sorted(votes), key=lambda c: votes[c]) if votes else 1
        bg[y][x] = val
    return bg


def floor_ok(bg, r, allowed):
    return in_board(r) and all(bg[y][x] in allowed for (x, y) in cells(r))


def socket_rejects(r, sockets):
    for s in sockets:
        inn = interior(s)
        if overlaps(r, inn) and (r[2], r[3]) != (inn[2], inn[3]):
            return True
    return False


def push_chain(seed, blocks, player, bg, sockets, dx, dy):
    """Slide the blocks in `seed` (indices) 1 cell per tick, absorbing blocks they run into."""
    moving = set(seed)
    while True:
        changed = True
        while changed:
            changed = False
            for i in list(moving):
                nr = shifted(blocks[i], dx, dy)
                for j, b in enumerate(blocks):
                    if j not in moving and overlaps(nr, b):
                        moving.add(j)
                        changed = True
        nexts = {i: shifted(blocks[i], dx, dy) for i in moving}
        for i, nr in nexts.items():
            if not floor_ok(bg, nr, BLOCK_FLOOR) or overlaps(nr, player) or socket_rejects(nr, sockets):
                return
        docked = False
        for i, nr in nexts.items():
            blocks[i] = nr
            docked = docked or any(nr == interior(s) for s in sockets)
        if docked:
            return


def draw_token(out, r, centre):
    for (x, y) in cells(r):
        out[y][x] = 14
    xs = range(r[0] + (r[2] - 1) // 2, r[0] + r[2] // 2 + 1)
    ys = range(r[1] + (r[3] - 1) // 2, r[1] + r[3] // 2 + 1)
    for y in ys:
        for x in xs:
            out[y][x] = centre


def touching_edges(p, b):
    """Player edge cells facing block b when they are edge-adjacent with overlapping span."""
    px0, py0, px1, py1 = p[0], p[1], p[0] + p[2] - 1, p[1] + p[3] - 1
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2] - 1, b[1] + b[3] - 1
    span_y = py0 <= by1 and by0 <= py1
    span_x = px0 <= bx1 and bx0 <= px1
    out = []
    if span_y and bx0 == px1 + 1:
        out += [(px1, y) for y in range(py0, py1 + 1)]
    if span_y and bx1 == px0 - 1:
        out += [(px0, y) for y in range(py0, py1 + 1)]
    if span_x and by0 == py1 + 1:
        out += [(x, py1) for x in range(px0, px1 + 1)]
    if span_x and by1 == py0 - 1:
        out += [(x, py0) for x in range(px0, px1 + 1)]
    return out


def draw_hud(out, n):
    zeros = (n + 1) // 2
    for x in range(SIZE):
        out[SIZE - 1][x] = 0 if x >= SIZE - zeros else 4


def hud_count(frame):
    z = sum(1 for v in frame[SIZE - 1] if v == 0)
    return 2 * z - 1 if z > 0 else 0


def transition_function(state, action, frame):
    player = None
    blocks, sockets = [], []
    for o in state:
        if o.get("type") == "player":
            player = rect(o)
        elif o.get("type") == "block":
            blocks.append(rect(o))
        elif o.get("type") == "target":
            sockets.append(rect(o))
    n = _memo["n"] if _memo["frame"] == frame else hud_count(frame)
    if player is None:
        return [row[:] for row in frame]
    bg = terrain(frame, [player] + blocks, sockets)

    if isinstance(action, int) and action in DIRS:
        dx, dy = DIRS[action]
        nxt = shifted(player, dx, dy, STEP)
        hit = [i for i, b in enumerate(blocks) if overlaps(nxt, b)]
        if hit:
            push_chain(hit, blocks, player, bg, sockets, dx, dy)
        elif floor_ok(bg, nxt, PLAYER_FLOOR):
            player = nxt
    elif isinstance(action, dict) and action.get("action_id") == 6:
        cx, cy = action.get("x", -1), action.get("y", -1)
        for i, b in enumerate(blocks):
            if b[0] <= cx < b[0] + b[2] and b[1] <= cy < b[1] + b[3]:
                draw_token(bg, player, 4)
                player = blocks.pop(i)
                break

    out = [row[:] for row in bg]
    for b in blocks:
        draw_token(out, b, 5)
    draw_token(out, player, 0)
    for b in blocks:
        for (x, y) in touching_edges(player, b):
            out[y][x] = 0
    n += 1
    draw_hud(out, n)
    _memo["frame"], _memo["n"] = [row[:] for row in out], n
    return out
