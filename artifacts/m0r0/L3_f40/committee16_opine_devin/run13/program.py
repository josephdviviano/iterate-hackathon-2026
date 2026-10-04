# Mechanics: blocks (colour-10 4x4) move together: A1/A2 y-/+4, A3 apart / A4 together (mirrored by side).
# Clicking an inactive marker (9) arms: blocks -> players (1), that marker -> active (11); clicking a player disarms.
# While armed, A1-A4 move the active marker's 4x4 cell (marker xy-1) by 4. A move succeeds iff the destination
# 4x4 is floor colour 5 (own cells excepted) inside rows 1-62. HUD: zeros in row 0 from the right and row 63 from
# the left, width (128a+150)//300, a = actions this level (hidden, continuity-gated). Unconfirmed: level-reset trigger.
FLOOR, BLOCK, PLAYER, MARKER, ACTIVE, HUD = 5, 10, 1, 9, 11, 0
_memo = []  # (frame, a) for frames we returned


def bar_width(a):
    return (128 * a + 150) // 300


def read_bar(frame):
    return sum(1 for v in frame[0] if v == HUD)


def find_squares(frame, colour, size):
    seen, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in seen:
                cells = [(x + i, y + j) for j in range(size) for i in range(size)]
                if all(0 <= cx < 64 and 1 <= cy < 63 and frame[cy][cx] == colour for cx, cy in cells):
                    seen.update(cells)
                    out.append((x, y))
    return out


def cell_free(frame, x, y, own):
    for j in range(4):
        for i in range(4):
            cx, cy = x + i, y + j
            if not (0 <= cx < 64 and 1 <= cy < 63):
                return False
            if (cx, cy) not in own and frame[cy][cx] != FLOOR:
                return False
    return True


def cells_of(x, y, size):
    return {(x + i, y + j) for j in range(size) for i in range(size)}


def paint(out, x, y, size, colour):
    for j in range(size):
        for i in range(size):
            out[y + j][x + i] = colour


def move_blocks(frame, out, blocks, action):
    blocks = sorted(blocks)
    moves = []
    for k, (x, y) in enumerate(blocks):
        if action in (1, 2):
            dx, dy = 0, (-4 if action == 1 else 4)
        else:
            side = -1 if x + 2 < 32 else 1
            dx, dy = (side * 4 if action == 3 else -side * 4), 0
        own = cells_of(x, y, 4)
        if cell_free(frame, x + dx, y + dy, own):
            moves.append((x, y, x + dx, y + dy))
    for x, y, _, _ in moves:
        paint(out, x, y, 4, FLOOR)
    for _, _, nx, ny in moves:
        paint(out, nx, ny, 4, BLOCK)


def move_marker(frame, out, marker, action):
    mx, my = marker
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    cx, cy = mx - 1, my - 1
    if cell_free(frame, cx + dx, cy + dy, cells_of(mx, my, 2)):
        paint(out, mx, my, 2, FLOOR)
        paint(out, mx + dx, my + dy, 2, ACTIVE)


def click(frame, out, x, y):
    if not (0 <= x < 64 and 0 <= y < 64):
        return
    c = frame[y][x]
    if c == MARKER:
        for mx, my in find_squares(frame, ACTIVE, 2):
            paint(out, mx, my, 2, MARKER)
        for mx, my in find_squares(frame, MARKER, 2):
            if mx <= x < mx + 2 and my <= y < my + 2:
                paint(out, mx, my, 2, ACTIVE)
        for bx, by in find_squares(frame, BLOCK, 4):
            paint(out, bx, by, 4, PLAYER)
    elif c == PLAYER:
        for mx, my in find_squares(frame, ACTIVE, 2):
            paint(out, mx, my, 2, MARKER)
        for bx, by in find_squares(frame, PLAYER, 4):
            paint(out, bx, by, 4, BLOCK)


def actions_so_far(frame):
    for f, a in reversed(_memo):
        if f == frame:
            return a
    w = read_bar(frame)
    if w == 0:
        return 0
    return max(a for a in range(400) if bar_width(a) == w)


def transition_function(state, action, frame):
    a = actions_so_far(frame)
    out = [list(row) for row in frame]
    if isinstance(action, dict):
        click(frame, out, action.get("x", -1), action.get("y", -1))
    elif action in (1, 2, 3, 4):
        players = find_squares(frame, PLAYER, 4)
        if players:
            for m in find_squares(frame, ACTIVE, 2)[:1]:
                move_marker(frame, out, m, action)
        else:
            move_blocks(frame, out, find_squares(frame, BLOCK, 4), action)
    a += 1
    w = bar_width(a)
    for x in range(64):
        out[0][x] = HUD if x >= 64 - w else FLOOR
        out[63][x] = HUD if x < w else FLOOR
    out = [[int(v) for v in row] for row in out]
    _memo.append(([list(r) for r in out], a))
    del _memo[:-16]
    return out
