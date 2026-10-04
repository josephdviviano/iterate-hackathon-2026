# Mechanics: twin cyan blocks (10) move together: A1/A2 up/down 4, A3 apart / A4 together (mirrored by half).
# A move of a 4x4 square succeeds iff its dest cells (outside its own) are floor colour 5 in rows 1-62.
# Click on an inactive marker (9) arms: blocks -> players (1), that marker active (11); click a player disarms.
# While armed, A1-A4 move the active marker's 4x4 cell (marker xy-1) by 4; A5 and other clicks are no-ops.
# HUD: zero bars in row 0 (from right) and row 63 (from left), width round(64a/150); a recovered via frame memo.
BLOCK, PLAYER, INACTIVE, ACTIVE, FLOOR, BAR = 10, 1, 9, 11, 5, 0
STEP = 4
_memo = []


def find_squares(frame, colours, size):
    claimed = set()
    found = []
    for y in range(1, 63):
        for x in range(64):
            c = frame[y][x]
            if c not in colours or (x, y) in claimed:
                continue
            if x + size > 64 or y + size > 63:
                continue
            cells = [(x + i, y + j) for j in range(size) for i in range(size)]
            if all(frame[cy][cx] == c and (cx, cy) not in claimed for cx, cy in cells):
                claimed.update(cells)
                found.append((x, y, c))
    return found


def can_place(frame, x0, y0, nx, ny, size):
    own = {(x0 + i, y0 + j) for i in range(size) for j in range(size)}
    for j in range(size):
        for i in range(size):
            x, y = nx + i, ny + j
            if (x, y) in own:
                continue
            if not (0 <= x < 64 and 1 <= y <= 62) or frame[y][x] != FLOOR:
                return False
    return True


def fill(out, x, y, size, c):
    for j in range(size):
        for i in range(size):
            out[y + j][x + i] = c


def bar_width(a):
    return (128 * a + 150) // 300


def read_bar(frame):
    w = 0
    while w < 64 and frame[63][w] == BAR:
        w += 1
    return w


def recover_a(frame):
    for f, a in reversed(_memo):
        if f == frame:
            return a
    w = read_bar(frame)
    if w == 0:
        return 0
    best = 0
    for a in range(1000):
        if bar_width(a) == w:
            best = a
        elif bar_width(a) > w:
            break
    return best


DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}


def step_blocks(frame, out, blocks, action):
    dx, dy = DIRS[action]
    moves = []
    for x, y, c in blocks:
        sx = dx if x + 2 < 32 else -dx
        if can_place(frame, x, y, x + sx, y + dy, 4):
            moves.append((x, y, x + sx, y + dy, c))
    for x, y, nx, ny, c in moves:
        fill(out, x, y, 4, FLOOR)
    for x, y, nx, ny, c in moves:
        fill(out, nx, ny, 4, c)


def step_marker(frame, out, marker, action):
    dx, dy = DIRS[action]
    x, y, _ = marker
    if can_place(frame, x - 1, y - 1, x - 1 + dx, y - 1 + dy, 4):
        fill(out, x, y, 2, FLOOR)
        fill(out, x + dx, y + dy, 2, ACTIVE)


def click(frame, out, blocks, markers, cx, cy):
    if not (0 <= cx < 64 and 0 <= cy < 64):
        return
    c = frame[cy][cx]
    if c == INACTIVE:
        hit = [m for m in markers if m[0] <= cx < m[0] + 2 and m[1] <= cy < m[1] + 2]
        for x, y, mc in markers:
            if mc == ACTIVE:
                fill(out, x, y, 2, INACTIVE)
        for x, y, _ in hit:
            fill(out, x, y, 2, ACTIVE)
        for x, y, _ in blocks:
            fill(out, x, y, 4, PLAYER)
    elif c == PLAYER:
        for x, y, mc in markers:
            if mc == ACTIVE:
                fill(out, x, y, 2, INACTIVE)
        for x, y, _ in blocks:
            fill(out, x, y, 4, BLOCK)


def transition_function(state, action, frame):
    a = recover_a(frame) + 1
    out = [list(r) for r in frame]
    blocks = find_squares(frame, (BLOCK, PLAYER), 4)
    markers = find_squares(frame, (INACTIVE, ACTIVE), 2)
    armed = any(c == PLAYER for _, _, c in blocks)
    if isinstance(action, dict):
        click(frame, out, blocks, markers, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        if armed:
            act = [m for m in markers if m[2] == ACTIVE]
            if act:
                step_marker(frame, out, act[0], action)
        else:
            step_blocks(frame, out, blocks, action)
    for x in range(64):
        out[0][x] = FLOOR if out[0][x] == BAR else out[0][x]
        out[63][x] = FLOOR if out[63][x] == BAR else out[63][x]
    w = bar_width(a)
    for i in range(w):
        out[0][63 - i] = BAR
        out[63][i] = BAR
    _memo.append(([list(r) for r in out], a))
    del _memo[:-16]
    return out
