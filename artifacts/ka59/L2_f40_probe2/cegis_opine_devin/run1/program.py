# Mechanics: arrows move the player 3px over floor colours {1,4}; if the player's next rect overlaps a block it
# kicks it instead (player stays): the block slides 1px/tick, chaining every block its shift overlaps, until any
# member leaves floor {1,4,15}, hits the player, or a socket guard rejects it. Click on a block possesses it (old
# player left as husk, centre 4). HUD row 63: (n+1)//2 zeros from the right, n = actions since level start.
# Hypothesis (step 101, one sighting): a block that fits a socket stops on that socket's line unless aligned with it.
WALK, SLIDE = {1, 4}, {1, 4, 15}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_memo = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def interior(s):
    return (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)


def fits(r, s):
    i = interior(s)
    return (r[2], r[3]) == (i[2], i[3])


def infer_bg(frame, covered, sockets):
    bg = [row[:] for row in frame]
    vis = lambda x, y: 0 <= x < 64 and 0 <= y < 63 and (x, y) not in covered

    def ray(x, y, dx, dy):
        x, y = x + dx, y + dy
        while 0 <= x < 64 and 0 <= y < 63:
            if (x, y) not in covered:
                return x, y
            x, y = x + dx, y + dy
        return None

    def across(x, y, horiz):
        a = ray(x, y, -1, 0) if horiz else ray(x, y, 0, -1)
        b = ray(x, y, 1, 0) if horiz else ray(x, y, 0, 1)
        return a, b

    def boundary_lookup(x, y, horiz, a, b):
        ca, cb = frame[a[1]][a[0]], frame[b[1]][b[0]]
        for d in range(1, 63):
            for s in (-d, d):
                if horiz:
                    yy = y + s
                    if vis(a[0], yy) and vis(b[0], yy) and vis(x, yy) and \
                            frame[yy][a[0]] == ca and frame[yy][b[0]] == cb:
                        return frame[yy][x]
                else:
                    xx = x + s
                    if vis(xx, a[1]) and vis(xx, b[1]) and vis(xx, y) and \
                            frame[a[1]][xx] == ca and frame[b[1]][xx] == cb:
                        return frame[y][xx]
        return None

    for (x, y) in covered:
        if y >= 63:
            continue
        sock = [s for s in sockets if overlap((x, y, 1, 1), s)]
        if sock:
            bg[y][x] = 1 if overlap((x, y, 1, 1), interior(sock[0])) else 4
            continue
        pairs = [across(x, y, True), across(x, y, False)]
        val = None
        for a, b in pairs:
            if a and b and frame[a[1]][a[0]] == frame[b[1]][b[0]]:
                val = frame[a[1]][a[0]]
                break
        if val is None:
            for horiz, (a, b) in zip((True, False), pairs):
                if a and b:
                    val = boundary_lookup(x, y, horiz, a, b)
                    if val is not None:
                        break
        if val is None:
            votes = [frame[p[1]][p[0]] for ab in pairs for p in ab if p]
            val = max(set(votes), key=votes.count) if votes else 1
        bg[y][x] = val
    return bg


def socket_rejects(r, nr, s, dx, dy):
    i = interior(s)
    if not fits(nr, s):
        return overlap(nr, i)
    if r[:2] == i[:2]:
        return True
    if dx:
        aligned = i[1] <= nr[1] and nr[1] + nr[3] <= i[1] + i[3]
        lead_now, lead_next = (r[0], nr[0]) if dx < 0 else (r[0] + r[2] - 1, nr[0] + nr[2] - 1)
        lo, hi = i[0], i[0] + i[2] - 1
    else:
        aligned = i[0] <= nr[0] and nr[0] + nr[2] <= i[0] + i[2]
        lead_now, lead_next = (r[1], nr[1]) if dy < 0 else (r[1] + r[3] - 1, nr[1] + nr[3] - 1)
        lo, hi = i[1], i[1] + i[3] - 1
    if aligned:
        return False
    return not (lo <= lead_now <= hi) and lo <= lead_next <= hi


def block_blocked(r, nr, bg, player, sockets, dx, dy):
    for (x, y) in cells(nr):
        if not (0 <= x < 63 and 0 <= y < 63) or bg[y][x] not in SLIDE:
            return True
    if overlap(nr, player):
        return True
    return any(socket_rejects(r, nr, s, dx, dy) for s in sockets)


def push_chain(blocks, start, bg, player, sockets, dx, dy):
    while True:
        group, todo = set(), list(start)
        while todo:
            i = todo.pop()
            if i in group:
                continue
            group.add(i)
            nr = shift(blocks[i], dx, dy)
            todo += [j for j in range(len(blocks)) if j not in group and overlap(nr, blocks[j])]
        if any(block_blocked(blocks[i], shift(blocks[i], dx, dy), bg, player, sockets, dx, dy) for i in group):
            return blocks
        blocks = [shift(b, dx, dy) if i in group else b for i, b in enumerate(blocks)]


def centre(r):
    xs = range(r[0] + (r[2] - 1) // 2, r[0] + r[2] // 2 + 1)
    ys = range(r[1] + (r[3] - 1) // 2, r[1] + r[3] // 2 + 1)
    return [(x, y) for y in ys for x in xs]


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


def draw_block(out, r):
    for (x, y) in cells(r):
        out[y][x] = 14
    for (x, y) in centre(r):
        out[y][x] = 5


def draw_player(out, p, blocks):
    for (x, y) in cells(p):
        out[y][x] = 14
    sides = touching_sides(p, blocks)
    for (x, y) in cells(p):
        if ("L" in sides and x == p[0]) or ("R" in sides and x == p[0] + p[2] - 1) or \
                ("U" in sides and y == p[1]) or ("D" in sides and y == p[1] + p[3] - 1):
            out[y][x] = 0
    for (x, y) in centre(p):
        out[y][x] = 0


def action_count(frame):
    if _memo["frame"] is not None and frame == _memo["frame"]:
        return _memo["n"]
    z = 0
    while z < 64 and frame[63][63 - z] == 0:
        z += 1
    return 2 * z - 1 if z else 0


def transition_function(state, action, frame):
    frame = [[int(v) for v in row] for row in frame]
    players = [o for o in state if o.get("type") == "player"]
    blocks = [rect(o) for o in state if o.get("type") == "block"]
    blocks.sort(key=lambda r: (r[1], r[0]))
    sockets = [rect(o) for o in state if o.get("type") == "target"]
    n = action_count(frame)
    if not players:
        return frame
    player = rect(players[0])
    covered = set(cells(player))
    for b in blocks:
        covered |= set(cells(b))
    bg = infer_bg(frame, covered, sockets)

    aid = action["action_id"] if isinstance(action, dict) else action
    if aid in DIRS:
        dx, dy = DIRS[aid]
        nxt = shift(player, 3 * dx, 3 * dy)
        hit = [i for i, b in enumerate(blocks) if overlap(nxt, b)]
        if hit:
            blocks = push_chain(blocks, hit, bg, player, sockets, dx, dy)
        elif all(0 <= x < 63 and 0 <= y < 63 and bg[y][x] in WALK for (x, y) in cells(nxt)):
            player = nxt
    elif aid == 6:
        cx, cy = action.get("x"), action.get("y")
        hit = [i for i, b in enumerate(blocks) if overlap((cx, cy, 1, 1), b)]
        if hit:
            for (x, y) in cells(player):
                bg[y][x] = frame[y][x]
            for (x, y) in centre(player):
                bg[y][x] = 4
            player = blocks.pop(hit[0])

    out = [row[:] for row in bg]
    for b in blocks:
        draw_block(out, b)
    draw_player(out, player, blocks)
    n += 1
    z = (n + 1) // 2
    out[63] = [4] * (64 - z) + [0] * z if z < 64 else [0] * 64
    out[63] = out[63][:64]
    _memo["frame"], _memo["n"] = [row[:] for row in out], n
    return out
