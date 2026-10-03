# Mechanics: arrows move the player 3 cells over floor (colours 1/4 inside the portal bbox); if its next rect overlaps
# a block, the block is kicked instead: the push chain slides 1 cell per tick (absorbing blocks it touches) until any
# member hits a wall (colour 2 etc.; colour 15 is passable for blocks only), the player, or a non-fitting socket interior.
# Click on a block: the player takes its rect, the old player stays as an inert husk (centre 4). Player sides touching
# a block render 0. HUD row below the board fills 0 from the right, ceil(actions/2). Unconfirmed: fitted socket entry, A5/A7.
WALL_FREE_PLAYER = {1, 4}
WALL_FREE_BLOCK = {1, 4, 15}
BORDER, PLAYER_CENTRE, TOKEN_BODY, TOKEN_CENTRE, HUSK_CENTRE, FLOOR = 4, 0, 14, 5, 4, 1
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_memory = {"frame": None, "n": 0}


def cells(r):
    x, y, w, h = r
    return [(x + i, y + j) for j in range(h) for i in range(w)]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d, k=1):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def centre_idx(n):
    return [n // 2] if n % 2 else [n // 2 - 1, n // 2]


def background(frame, covers, sockets):
    covered = set()
    for r in covers:
        covered.update(cells(r))
    H, W = len(frame), len(frame[0])
    bg = [row[:] for row in frame]

    def probe(x, y, dx, dy):
        x, y = x + dx, y + dy
        while 0 <= x < W and 0 <= y < H:
            if (x, y) not in covered:
                return frame[y][x]
            x, y = x + dx, y + dy
        return None

    for (x, y) in covered:
        l, r, u, d = probe(x, y, -1, 0), probe(x, y, 1, 0), probe(x, y, 0, -1), probe(x, y, 0, 1)
        bg[y][x] = l if l is not None and l == r else u if u is not None and u == d else FLOOR
    for s in sockets:
        for (x, y) in cells(s):
            if (x, y) in covered:
                inner = s[0] < x < s[0] + s[2] - 1 and s[1] < y < s[1] + s[3] - 1
                bg[y][x] = FLOOR if inner else BORDER
    return bg


def in_bounds(r, board):
    return board[0] <= r[0] and r[0] + r[2] <= board[0] + board[2] and board[1] <= r[1] and r[1] + r[3] <= board[1] + board[3]


def player_can_enter(r, bg, board):
    return in_bounds(r, board) and all(bg[y][x] in WALL_FREE_PLAYER for x, y in cells(r))


def blocked_by_socket(r, sockets):
    for s in sockets:
        inner = (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)
        if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
            return True
    return False


def block_can_enter(r, bg, board, player, sockets):
    return (in_bounds(r, board) and all(bg[y][x] in WALL_FREE_BLOCK for x, y in cells(r))
            and not overlap(r, player) and not blocked_by_socket(r, sockets))


def push_chain(blocks, group, d):
    group = set(group)
    while True:
        moved = [shift(blocks[i], d) for i in group]
        extra = {j for j, b in enumerate(blocks) if j not in group and any(overlap(m, b) for m in moved)}
        if not extra:
            return group
        group |= extra


def slide(blocks, start, d, bg, board, player, sockets):
    blocks = list(blocks)
    group = set(start)
    while True:
        group = push_chain(blocks, group, d)
        if not all(block_can_enter(shift(blocks[i], d), bg, board, player, sockets) for i in group):
            return blocks
        for i in group:
            blocks[i] = shift(blocks[i], d)


def step(player, blocks, action, bg, board, sockets):
    husk = None
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            px, py = action["x"], action["y"]
            for i, b in enumerate(blocks):
                if overlap((px, py, 1, 1), b):
                    husk = player
                    player = b
                    blocks = blocks[:i] + blocks[i + 1:]
                    break
        return player, blocks, husk
    d = DIRS.get(action)
    if d is None:
        return player, blocks, husk
    nxt = shift(player, d, 3)
    hit = [i for i, b in enumerate(blocks) if overlap(nxt, b)]
    if hit:
        blocks = slide(blocks, hit, d, bg, board, player, sockets)
    elif player_can_enter(nxt, bg, board):
        player = nxt
    return player, blocks, husk


def paint_sprite(out, r, body, centre_colour):
    x, y, w, h = r
    for (cx, cy) in cells(r):
        out[cy][cx] = body
    for i in centre_idx(w):
        for j in centre_idx(h):
            out[y + j][x + i] = centre_colour


def paint_player(out, r, blocks):
    paint_sprite(out, r, TOKEN_BODY, PLAYER_CENTRE)
    x, y, w, h = r
    sides = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in sides:
        if dx:
            edge = [(x if dx < 0 else x + w - 1, y + j) for j in range(h)]
        else:
            edge = [(x + i, y if dy < 0 else y + h - 1) for i in range(w)]
        touching = any(overlap((ex + dx, ey + dy, 1, 1), b) for ex, ey in edge for b in blocks)
        if touching:
            for ex, ey in edge:
                out[ey][ex] = PLAYER_CENTRE


def transition_function(state, action, frame=None):
    player = next(rect(o) for o in state if o["type"] == "player")
    blocks = [rect(o) for o in state if o["type"] == "block"]
    sockets = [rect(o) for o in state if o["type"] == "target"]
    portal = next((rect(o) for o in state if o["type"] == "portal"), (0, 0, len(frame[0]) - 1, len(frame) - 1))
    if _memory["frame"] is not None and _memory["frame"] == frame:
        n = _memory["n"]
    else:
        hud_row = portal[1] + portal[3]
        z = sum(1 for v in frame[hud_row] if v == 0) if hud_row < len(frame) else 0
        n = max(0, 2 * z - 1)
    bg = background(frame, [player] + blocks, sockets)
    new_player, new_blocks, husk = step(player, blocks, action, bg, portal, sockets)
    out = [row[:] for row in bg]
    if husk is not None:
        paint_sprite(out, husk, TOKEN_BODY, HUSK_CENTRE)
    for b in new_blocks:
        paint_sprite(out, b, TOKEN_BODY, TOKEN_CENTRE)
    paint_player(out, new_player, new_blocks)
    n += 1
    hud_row = portal[1] + portal[3]
    if hud_row < len(out):
        W = len(out[hud_row])
        for x in range(W - (n + 1) // 2, W):
            out[hud_row][x] = 0
    _memory["frame"], _memory["n"] = [row[:] for row in out], n
    return out
