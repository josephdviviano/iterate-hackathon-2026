# Mechanics: two colour-10 4x4 blocks move together (A1 up, A2 down, A3 apart, A4 together, 4 cells);
# a move succeeds iff the destination 4x4 is all floor colour 5 (own cells excepted). Clicking a
# colour-9 marker arms it (11) and turns blocks into colour-1 players; then A1-A4 move the marker's
# 4x4 cell under the same floor rule; clicking a player disarms; other clicks and A5 are no-ops.
# HUD: zero bars at row 0 (from right) and row 63 (from left), width 3(n+1)//7, n = actions this level.
FLOOR, BLOCK, PLAYER, MARKER, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_memo = {"frame": None, "n": 0}


def squares(frame, colours, size):
    claimed, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            c = frame[y][x]
            if c in colours and (x, y) not in claimed:
                cells = {(x + i, y + j) for i in range(size) for j in range(size)}
                claimed |= cells
                out.append((x, y, c))
    return out


def can_place(frame, x, y, size, own):
    for j in range(size):
        for i in range(size):
            cx, cy = x + i, y + j
            if not (0 <= cx < 64 and 1 <= cy < 63):
                return False
            if (cx, cy) not in own and frame[cy][cx] != FLOOR:
                return False
    return True


def cells(x, y, size):
    return {(x + i, y + j) for i in range(size) for j in range(size)}


def paint(frame, x, y, size, colour):
    for cx, cy in cells(x, y, size):
        frame[cy][cx] = colour


def bar_width(n):
    return 3 * (n + 1) // 7


def actions_before(frame):
    if _memo["frame"] is not None and _memo["frame"] == frame:
        return _memo["n"]
    w = sum(1 for v in frame[0] if v == BAR)
    return 0 if w == 0 else (7 * w + 2) // 3 - 1


def step_blocks(frame, out, blocks, action):
    dx, dy = DIRS[action]
    blocks = sorted(blocks)
    moves = []
    for k, (x, y, c) in enumerate(blocks):
        mx = dx if action in (1, 2) else (dx if k == 0 else -dx)
        moves.append((x, y, x + mx, y + dy, c))
    ok = [can_place(frame, nx, ny, 4, cells(x, y, 4)) for x, y, nx, ny, c in moves]
    for (x, y, nx, ny, c), good in zip(moves, ok):
        if good:
            paint(out, x, y, 4, FLOOR)
    for (x, y, nx, ny, c), good in zip(moves, ok):
        paint(out, nx if good else x, ny if good else y, 4, c)


def step_marker(frame, out, marker, action):
    dx, dy = DIRS[action]
    cx, cy = marker[0] - 1, marker[1] - 1
    if can_place(frame, cx + dx, cy + dy, 4, cells(cx, cy, 4)):
        paint(out, marker[0], marker[1], 2, FLOOR)
        paint(out, marker[0] + dx, marker[1] + dy, 2, ACTIVE)


def click(frame, out, blocks, markers, x, y):
    if not (0 <= x < 64 and 0 <= y < 64):
        return
    hit = frame[y][x]
    if hit == MARKER:
        for bx, by, c in blocks:
            paint(out, bx, by, 4, PLAYER)
        for mx, my, c in markers:
            if c == ACTIVE:
                paint(out, mx, my, 2, MARKER)
            if (x, y) in cells(mx, my, 2):
                paint(out, mx, my, 2, ACTIVE)
    elif hit == PLAYER:
        for bx, by, c in blocks:
            paint(out, bx, by, 4, BLOCK)
        for mx, my, c in markers:
            if c == ACTIVE:
                paint(out, mx, my, 2, MARKER)


def draw_hud(out, n):
    w = bar_width(n)
    for x in range(64):
        out[0][x] = BAR if x >= 64 - w else FLOOR
        out[63][x] = BAR if x < w else FLOOR


def transition_function(state, action, frame):
    n = actions_before(frame)
    out = [list(r) for r in frame]
    blocks = squares(frame, (BLOCK, PLAYER), 4)
    markers = squares(frame, (MARKER, ACTIVE), 2)
    armed = any(c == PLAYER for _, _, c in blocks)
    if isinstance(action, dict):
        click(frame, out, blocks, markers, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        active = [m for m in markers if m[2] == ACTIVE]
        if armed and active:
            step_marker(frame, out, active[0], action)
        elif not armed:
            step_blocks(frame, out, blocks, action)
    draw_hud(out, n + 1)
    out = [[int(v) for v in r] for r in out]
    _memo["frame"] = [list(r) for r in out]
    _memo["n"] = n + 1
    return out
