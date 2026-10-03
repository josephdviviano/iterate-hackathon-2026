# Mechanics: floor = colour 5; blocks (10) / armed players (1) / markers (9 inactive, 11 active) on a 4px grid.
# Block mode: A1/A2 move both blocks y-4/y+4, A3 left x-4 & right x+4, A4 the reverse; each block moves iff its
# dest 4x4 is all floor. Click inactive marker -> arm (blocks recolour 1, marker 11, old active 9); click armed
# player -> disarm; A1-4 then move the active marker's 4x4 cell iff dest cell is floor. HUD: colour-0 bar width
# 3(a+1)//7 at row 0 (right) and row 63 (left), a = actions this level (continuity-gated). Unconfirmed: A7.
FLOOR, BLOCK, ARMED, INACT, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_memo = {"frame": None, "a": 0}


def cells_of(frame, colour):
    return [(x, y) for y in range(1, 63) for x in range(64) if frame[y][x] == colour]


def squares(frame, colour):
    """Top-left corners of same-colour squares."""
    pts = set(cells_of(frame, colour))
    return sorted((x, y) for x, y in pts if (x - 1, y) not in pts and (x, y - 1) not in pts)


def region_free(frame, x0, y0, own):
    for y in range(y0, y0 + 4):
        for x in range(x0, x0 + 4):
            if not (1 <= y <= 62 and 0 <= x < 64):
                return False
            if (x, y) not in own and frame[y][x] != FLOOR:
                return False
    return True


def fill(out, x0, y0, size, colour):
    for y in range(y0, y0 + size):
        for x in range(x0, x0 + size):
            out[y][x] = colour


def move_square(frame, out, x, y, dx, dy, size, colour, cell_off):
    """Move a size-square at (x,y) whose 4x4 cell starts at (x-cell_off, y-cell_off)."""
    own = {(x + i, y + j) for i in range(size) for j in range(size)}
    if not region_free(frame, x - cell_off + dx, y - cell_off + dy, own):
        return False
    fill(out, x, y, size, FLOOR)
    fill(out, x + dx, y + dy, size, colour)
    return True


def step_blocks(frame, out, action):
    blocks = sorted(squares(frame, BLOCK))
    if action not in DIRS or len(blocks) != 2:
        return
    dx, dy = DIRS[action]
    (lx, ly), (rx, ry) = blocks
    for (x, y), mdx in (((lx, ly), dx), ((rx, ry), -dx)):
        move_square(frame, out, x, y, mdx, dy, 4, BLOCK, 0)


def step_marker(frame, out, action):
    act = squares(frame, ACTIVE)
    if action in DIRS and act:
        dx, dy = DIRS[action]
        x, y = act[0]
        move_square(frame, out, x, y, dx, dy, 2, ACTIVE, 1)


def click(frame, out, x, y):
    if not (0 <= x < 64 and 0 <= y < 64):
        return
    c = frame[y][x]
    if c == INACT:
        mx, my = x, y
        while frame[my][mx - 1] == INACT:
            mx -= 1
        while frame[my - 1][mx] == INACT:
            my -= 1
        for ax, ay in cells_of(frame, ACTIVE):
            out[ay][ax] = INACT
        for bx, by in cells_of(frame, BLOCK):
            out[by][bx] = ARMED
        fill(out, mx, my, 2, ACTIVE)
    elif c == ARMED:
        for ax, ay in cells_of(frame, ACTIVE):
            out[ay][ax] = INACT
        for bx, by in cells_of(frame, ARMED):
            out[by][bx] = BLOCK


def bar_width(a):
    return 3 * (a + 1) // 7


def infer_actions(frame):
    w = sum(1 for v in frame[63] if v == BAR)
    if w == 0:
        return 0
    a = 0
    while bar_width(a) < w:
        a += 1
    return a


def transition_function(state, action, frame):
    frame = [[int(v) for v in row] for row in frame]
    a = _memo["a"] if _memo["frame"] == frame else infer_actions(frame)
    out = [row[:] for row in frame]
    aid = action["action_id"] if isinstance(action, dict) else int(action)
    if aid == 6:
        click(frame, out, int(action["x"]), int(action["y"]))
    elif cells_of(frame, BLOCK):
        step_blocks(frame, out, aid)
    else:
        step_marker(frame, out, aid)
    a += 1
    w = bar_width(a)
    for x in range(w):
        out[0][63 - x] = BAR
        out[63][x] = BAR
    _memo["frame"], _memo["a"] = [row[:] for row in out], a
    return out
