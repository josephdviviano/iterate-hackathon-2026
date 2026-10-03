# Mechanics: frame-level mirror-blocks/marker game. Floor = colour 5; walls = any other colour.
# Block mode (4x4 colour-10 blocks): A1/A2 move both blocks y-4/y+4, A3/A4 mirror x (left dx, right -dx, A3 dx=-4);
#   each block moves alone iff its destination 4x4 is all floor (markers, walls, other block block it).
# Click inactive marker (9) -> blocks turn 1 (armed), marker 11; armed A1-4 move the active marker's 4x4 cell
#   (marker xy-1) by 4 if all floor; click other marker = switch, click player = disarm, else no-op. HUD bars row 0 (right) and row 63 (left), w=3(a+1)//7 with a = actions; hidden a is continuity-gated (fallback: 0 if no bar else largest a for that width).
FLOOR, BLOCK, ARMED, INACT, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_mem = {"frame": None, "a": 0}


def squares(frame, colour, size):
    claimed, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in claimed:
                out.append((x, y))
                claimed.update((x + i, y + j) for i in range(size) for j in range(size))
    return out


def fill(frame, x, y, size, colour):
    for j in range(size):
        for i in range(size):
            frame[y + j][x + i] = colour


def free(frame, x, y, size=4):
    if x < 0 or y < 1 or x + size > 64 or y + size > 63:
        return False
    return all(frame[y + j][x + i] == FLOOR for j in range(size) for i in range(size))


def bar_width(frame):
    return sum(1 for v in frame[0] if v == BAR)


def fallback_count(frame):
    w = bar_width(frame)
    if w == 0:
        return 0
    a = 0
    while 3 * (a + 2) // 7 <= w:
        a += 1
    return a


def move_blocks(frame, act):
    blocks = sorted(squares(frame, BLOCK, 4))
    dx, dy = DIRS[act]
    deltas = [(dx, dy), (-dx, dy)] if len(blocks) == 2 else [(dx, dy)] * len(blocks)
    moves = []
    for (bx, by), (ddx, ddy) in zip(blocks, deltas):
        nx, ny = bx + ddx, by + ddy
        work = [row[:] for row in frame]
        fill(work, bx, by, 4, FLOOR)
        if free(work, nx, ny):
            moves.append((bx, by, nx, ny))
    for bx, by, _, _ in moves:
        fill(frame, bx, by, 4, FLOOR)
    for _, _, nx, ny in moves:
        fill(frame, nx, ny, 4, BLOCK)


def move_marker(frame, act):
    act_m = squares(frame, ACTIVE, 2)
    if not act_m:
        return
    mx, my = act_m[0]
    dx, dy = DIRS[act]
    work = [row[:] for row in frame]
    fill(work, mx, my, 2, FLOOR)
    if free(work, mx - 1 + dx, my - 1 + dy):
        fill(frame, mx, my, 2, FLOOR)
        fill(frame, mx + dx, my + dy, 2, ACTIVE)


def marker_origin(frame, x, y, colour):
    mx = x - 1 if x > 0 and frame[y][x - 1] == colour else x
    my = y - 1 if y > 0 and frame[y - 1][x] == colour else y
    return mx, my


def recolour(frame, src, dst):
    for row in frame:
        for i, v in enumerate(row):
            if v == src:
                row[i] = dst


def click(frame, x, y):
    if not (0 <= x < 64 and 1 <= y < 63):
        return
    c = frame[y][x]
    armed = bool(squares(frame, ARMED, 4))
    if c == INACT:
        mx, my = marker_origin(frame, x, y, INACT)
        if armed:
            recolour(frame, ACTIVE, INACT)
        else:
            recolour(frame, BLOCK, ARMED)
        fill(frame, mx, my, 2, ACTIVE)
    elif c == ARMED:
        recolour(frame, ACTIVE, INACT)
        recolour(frame, ARMED, BLOCK)


def draw_hud(frame, a):
    w = min(64, 3 * (a + 1) // 7)
    for x in range(64):
        frame[0][x] = BAR if x >= 64 - w else FLOOR
        frame[63][x] = BAR if x < w else FLOOR


def transition_function(state, action, frame):
    a = _mem["a"] if _mem["frame"] == frame else fallback_count(frame)
    out = [row[:] for row in frame]
    if isinstance(action, dict):
        click(out, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        if squares(out, ARMED, 4):
            move_marker(out, action)
        else:
            move_blocks(out, action)
    a += 1
    draw_hud(out, a)
    _mem["frame"], _mem["a"] = [row[:] for row in out], a
    return out
