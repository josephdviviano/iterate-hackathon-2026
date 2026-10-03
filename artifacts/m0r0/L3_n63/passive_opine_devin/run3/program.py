# Mechanics: two 4x4 cyan (10) blocks move on floor colour 5; A1/A2 move both y-/+4, A3/A4 move them
# mirrored (left half x-/+4, right half x+/-4); a move happens only if every dest cell is floor 5.
# Click on an inactive 2x2 marker (9) arms it: marker -> 11, blocks -> colour 1; while armed arrows move
# the 11 marker by 4 under the same floor rule; click on an armed block (1) disarms; other clicks / A5 no-op.
# HUD: zero bars on row 0 (right-aligned) and row 63 (left-aligned), width round(64a/150), a = actions this level.
FLOOR, BLOCK, ARMED, MARKER, ACTIVE = 5, 10, 1, 9, 11
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_memo = []


def bar_width(a):
    return (128 * a + 150) // 300


def read_bar(frame):
    w = 0
    while w < 64 and frame[0][63 - w] == 0:
        w += 1
    return w


def squares(frame, colours, size):
    claimed, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            if (x, y) in claimed or frame[y][x] not in colours:
                continue
            cells = [(x + i, y + j) for j in range(size) for i in range(size)]
            claimed.update(cells)
            out.append((x, y, frame[y][x]))
    return out


def cells_of(x, y, size):
    return [(x + i, y + j) for j in range(size) for i in range(size)]


def can_place(frame, x, y, size, own):
    for cx, cy in cells_of(x, y, size):
        if (cx, cy) in own:
            continue
        if not (0 <= cx < 64 and 1 <= cy < 63) or frame[cy][cx] != FLOOR:
            return False
    return True


def move_sprite(out, frame, x, y, size, nx, ny):
    colour = frame[y][x]
    for cx, cy in cells_of(x, y, size):
        out[cy][cx] = FLOOR
    for cx, cy in cells_of(nx, ny, size):
        out[cy][cx] = colour


def recolour(out, x, y, size, colour):
    for cx, cy in cells_of(x, y, size):
        out[cy][cx] = colour


def step_blocks(frame, out, blocks, action):
    dx, dy = DIRS[action]
    for x, y, _ in blocks:
        mx = dx if x + 2 < 32 else -dx
        if can_place(frame, x + mx * 4, y + dy * 4, 4, set(cells_of(x, y, 4))):
            move_sprite(out, frame, x, y, 4, x + mx * 4, y + dy * 4)


def step_marker(frame, out, markers, action):
    dx, dy = DIRS[action]
    for x, y, c in markers:
        if c == ACTIVE and can_place(frame, x + dx * 4, y + dy * 4, 2, set(cells_of(x, y, 2))):
            move_sprite(out, frame, x, y, 2, x + dx * 4, y + dy * 4)


def click(frame, out, blocks, markers, cx, cy):
    if not (0 <= cx < 64 and 0 <= cy < 64):
        return
    hit = frame[cy][cx]
    if hit == MARKER:
        for x, y, c in markers:
            inside = x <= cx < x + 2 and y <= cy < y + 2
            recolour(out, x, y, 2, ACTIVE if inside else MARKER)
        for x, y, _ in blocks:
            recolour(out, x, y, 4, ARMED)
    elif hit == ARMED:
        for x, y, c in markers:
            recolour(out, x, y, 2, MARKER)
        for x, y, _ in blocks:
            recolour(out, x, y, 4, BLOCK)


def draw_bar(out, w):
    for x in range(64):
        out[0][x] = 0 if x >= 64 - w else FLOOR
        out[63][x] = 0 if x < w else FLOOR


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
    out = [list(r) for r in frame]
    blocks = squares(frame, (BLOCK, ARMED), 4)
    markers = squares(frame, (MARKER, ACTIVE), 2)
    armed = any(c == ARMED for _, _, c in blocks)
    if isinstance(action, dict):
        click(frame, out, blocks, markers, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        if armed:
            step_marker(frame, out, markers, action)
        else:
            step_blocks(frame, out, blocks, action)
    a += 1
    draw_bar(out, bar_width(a))
    out = [[int(v) for v in r] for r in out]
    _memo.append(([list(r) for r in out], a))
    del _memo[:-16]
    return out
