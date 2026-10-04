# Mechanics: two 4x4 blocks (colour 10) move together: A1 y-4, A2 y+4, A3/A4 x-4/x+4 mirrored by
# half (left block A3 -> left, right block A3 -> right); each moves only if its dest 4x4 is all floor 5
# within rows 1..62. Click a 9 marker: arm (blocks -> 1) / switch, that marker -> active 11. Click a 1
# player: disarm (players -> 10, active 11 -> 9). Armed: arrows move the active marker's 4x4 cell by 4.
# HUD: a = calls this level (all count); bar of 0s width round(64a/150), row 0 from right, row 63 from left.
FLOOR, BLOCK, ARMED, MARKER, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
BUDGET = 150
_memo = []  # recent (returned frame, a), most recent first


def bar_width(a):
    return (128 * a + 150) // 300


def read_bar(frame):
    w = 0
    while w < 64 and frame[0][63 - w] == BAR:
        w += 1
    return w


def infer_actions(frame):
    key = tuple(tuple(r) for r in frame)
    for f, a in _memo:
        if f == key:
            return a
    w = read_bar(frame)
    if w == 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w:
        a += 1
    return a


def squares(frame, colour, size):
    """Top-left corners of solid size x size squares of a colour, claimed in reading order."""
    seen, out = set(), []
    for y in range(64 - size + 1):
        for x in range(64 - size + 1):
            if (x, y) in seen or frame[y][x] != colour:
                continue
            cells = [(x + i, y + j) for j in range(size) for i in range(size)]
            if all(frame[cy][cx] == colour and (cx, cy) not in seen for cx, cy in cells):
                seen.update(cells)
                out.append((x, y))
    return out


def cell_free(frame, x, y, size=4):
    if x < 0 or x + size > 64 or y < 1 or y + size > 63:
        return False
    return all(frame[y + j][x + i] == FLOOR for j in range(size) for i in range(size))


def fill(out, x, y, w, h, colour):
    for j in range(h):
        for i in range(w):
            out[y + j][x + i] = colour


def recolour(out, frame, src, dst):
    for y in range(64):
        for x in range(64):
            if frame[y][x] == src:
                out[y][x] = dst


def step_vector(action):
    return {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(action)


def move_blocks(frame, out, action):
    d = step_vector(action)
    if d is None:
        return
    moves = []
    for bx, by in squares(frame, BLOCK, 4):
        dx, dy = d
        if dx and bx + 2 >= 32:
            dx = -dx  # right-half block mirrors horizontal moves
        if cell_free(frame, bx + dx, by + dy):
            moves.append((bx, by, bx + dx, by + dy))
    for bx, by, _, _ in moves:
        fill(out, bx, by, 4, 4, FLOOR)
    for _, _, nx, ny in moves:
        fill(out, nx, ny, 4, 4, BLOCK)


def move_marker(frame, out, action):
    d = step_vector(action)
    if d is None:
        return
    for mx, my in squares(frame, ACTIVE, 2):
        cx, cy = mx - 1, my - 1
        nx, ny = cx + d[0], cy + d[1]
        if cell_free(frame, nx, ny):
            fill(out, mx, my, 2, 2, FLOOR)
            fill(out, nx + 1, ny + 1, 2, 2, ACTIVE)


def marker_at(frame, x, y, colour):
    for mx, my in squares(frame, colour, 2):
        if mx <= x < mx + 2 and my <= y < my + 2:
            return mx, my
    return None


def click(frame, out, x, y):
    if not (0 <= x < 64 and 0 <= y < 64):
        return
    c = frame[y][x]
    if c == MARKER:
        m = marker_at(frame, x, y, MARKER)
        if m is None:
            return
        recolour(out, frame, ACTIVE, MARKER)
        recolour(out, frame, BLOCK, ARMED)
        fill(out, m[0], m[1], 2, 2, ACTIVE)
    elif c == ARMED:
        recolour(out, frame, ARMED, BLOCK)
        recolour(out, frame, ACTIVE, MARKER)


def armed(frame):
    return any(ARMED in row for row in frame[1:63])


def draw_bar(out, a):
    w = min(64, bar_width(min(a, BUDGET)))
    for i in range(64):
        out[0][63 - i] = BAR if i < w else FLOOR
        out[63][i] = BAR if i < w else FLOOR


def transition_function(state, action, frame):
    a = infer_actions(frame) + 1
    out = [list(map(int, row)) for row in frame]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(frame, out, int(action.get("x", -1)), int(action.get("y", -1)))
    elif armed(frame):
        move_marker(frame, out, action)
    else:
        move_blocks(frame, out, action)
    draw_bar(out, a)
    _memo.insert(0, (tuple(tuple(r) for r in out), a))
    del _memo[16:]
    return out
