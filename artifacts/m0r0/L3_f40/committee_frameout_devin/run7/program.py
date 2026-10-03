# Mechanics: two mirrored 4x4 blocks (colour 10) move by 4: A1 up, A2 down, A3 apart, A4 together;
# each moves only if its destination 4x4 is all floor colour 5 (own cells excepted). Clicking an
# inactive marker (9) arms the blocks (10->1) and makes it active (11, old active->9); clicking an
# armed player (1) disarms. While armed A1-4 move the active marker's 4x4 cell (marker xy-1). A5 no-op.
# HUD: hidden action counter a (continuity-gated), zero bars w=3(a+1)//7 on row 0 (right) and row 63 (left).
FLOOR, BLOCK, ARMED, INACT, ACT = 5, 10, 1, 9, 11
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_last = {"frame": None, "a": 0}


def squares(frame, colours, size):
    seen, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] in colours and (x, y) not in seen:
                cells = {(x + i, y + j) for i in range(size) for j in range(size)}
                seen |= cells
                out.append((x, y))
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


def fill(frame, x, y, size, colour):
    for j in range(size):
        for i in range(size):
            frame[y + j][x + i] = colour


def cells(x, y, size):
    return {(x + i, y + j) for i in range(size) for j in range(size)}


def bar_width(a):
    return 3 * (a + 1) // 7


def counter_from_frame(frame):
    w = sum(1 for x in range(64) if frame[0][x] == 0)
    if w == 0:
        return 0
    a = 0
    while bar_width(a) < w:
        a += 1
    return a


def move_blocks(frame, out, action):
    blocks = sorted(squares(frame, {BLOCK}, 4))
    dx, dy = DIRS[action]
    moves = []
    for k, (x, y) in enumerate(blocks):
        mdx = dx if (k == 0 or action in (1, 2)) else -dx
        if can_place(frame, x + mdx, y + dy, 4, cells(x, y, 4)):
            moves.append((x, y, x + mdx, y + dy))
    for x, y, nx, ny in moves:
        fill(out, x, y, 4, FLOOR)
    for x, y, nx, ny in moves:
        fill(out, nx, ny, 4, BLOCK)


def move_marker(frame, out, action):
    act = squares(frame, {ACT}, 2)
    if not act:
        return
    mx, my = act[0]
    dx, dy = DIRS[action]
    if can_place(frame, mx - 1 + dx, my - 1 + dy, 4, cells(mx - 1, my - 1, 4)):
        fill(out, mx, my, 2, FLOOR)
        fill(out, mx + dx, my + dy, 2, ACT)


def recolour(out, src, dst):
    for y in range(1, 63):
        for x in range(64):
            if out[y][x] == src:
                out[y][x] = dst


def click(frame, out, cx, cy):
    if not (0 <= cx < 64 and 0 <= cy < 64):
        return
    c = frame[cy][cx]
    if c == INACT:
        for mx, my in squares(frame, {INACT}, 2):
            if (cx, cy) in cells(mx, my, 2):
                recolour(out, ACT, INACT)
                recolour(out, BLOCK, ARMED)
                fill(out, mx, my, 2, ACT)
                return
    elif c == ARMED:
        recolour(out, ACT, INACT)
        recolour(out, ARMED, BLOCK)


def transition_function(state, action, frame):
    a = _last["a"] if frame == _last["frame"] else counter_from_frame(frame)
    out = [row[:] for row in frame]
    armed = any(v == ARMED for row in frame[1:63] for v in row)
    if isinstance(action, dict):
        click(frame, out, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        if armed:
            move_marker(frame, out, action)
        else:
            move_blocks(frame, out, action)
    a += 1
    w = bar_width(a)
    for x in range(64):
        out[0][x] = 0 if x >= 64 - w else FLOOR
        out[63][x] = 0 if x < w else FLOOR
    _last["frame"], _last["a"] = [row[:] for row in out], a
    return out
