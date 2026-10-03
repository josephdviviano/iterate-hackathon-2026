# Mechanics: mirror-blocks/marker game on a 4x4 grid; floor = colour 5, every other colour blocks.
# Block mode (colour-10 blocks): A1/A2 move both blocks up/down 4; A3 left block x-4 / right x+4, A4 reverse.
# Click colour 9 marker = arm/switch (blocks->1, marker->11); click colour 1 = disarm; armed A1-4 move the 11 marker.
# HUD: colour-0 bars on row 0 (right-aligned) and row 63 (left-aligned), width round(64a/150), a = actions this level.
# Hidden a is resumed from a memo of frames we returned (most recent first), else the largest a with that width.
FLOOR, BLOCK, ARMED, MARKER, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
STEP = 4
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
_memo = []


def bar_width(a):
    return (128 * a + 150) // 300


def action_count(frame):
    key = tuple(tuple(r) for r in frame)
    for k, a in reversed(_memo):
        if k == key:
            return a
    w = frame[0].count(BAR)
    if w == 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w:
        a += 1
    return a


def squares(frame, colour, size):
    seen, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in seen:
                cells = {(x + i, y + j) for i in range(size) for j in range(size)}
                seen |= cells
                out.append((x, y))
    return out


def fits(frame, x, y, size, own):
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


def paint(out, x, y, size, colour):
    for cx, cy in cells(x, y, size):
        out[cy][cx] = colour


def move_square(frame, out, x, y, size, colour, dx, dy, own_box):
    """Move a size x size sprite whose 4x4 cell is own_box=(bx,by); returns True if moved."""
    bx, by = own_box
    own = cells(bx, by, STEP)
    if not fits(frame, bx + dx, by + dy, STEP, own):
        return False
    paint(out, x, y, size, FLOOR)
    return True


def block_step(frame, out, action):
    blocks = sorted(squares(frame, BLOCK, STEP))
    if len(blocks) != 2:
        return
    dx, dy = DIRS[action]
    deltas = [(dx, dy), (-dx, dy)] if dx else [(0, dy), (0, dy)]
    moved = []
    for (x, y), (ddx, ddy) in zip(blocks, deltas):
        if move_square(frame, out, x, y, STEP, BLOCK, ddx, ddy, (x, y)):
            moved.append((x + ddx, y + ddy))
        else:
            moved.append((x, y))
    for x, y in moved:
        paint(out, x, y, STEP, BLOCK)


def marker_step(frame, out, action):
    act = squares(frame, ACTIVE, 2)
    if not act:
        return
    x, y = act[0]
    dx, dy = DIRS[action]
    if move_square(frame, out, x, y, 2, ACTIVE, dx, dy, (x - 1, y - 1)):
        paint(out, x + dx, y + dy, 2, ACTIVE)


def recolour(out, src, dst):
    for y in range(1, 63):
        for x in range(64):
            if out[y][x] == src:
                out[y][x] = dst


def click(frame, out, cx, cy):
    if not (0 <= cx < 64 and 0 <= cy < 64):
        return
    hit = frame[cy][cx]
    if hit == MARKER:
        recolour(out, ACTIVE, MARKER)
        recolour(out, BLOCK, ARMED)
        for x, y in squares(frame, MARKER, 2):
            if (cx, cy) in cells(x, y, 2):
                paint(out, x, y, 2, ACTIVE)
    elif hit == ARMED:
        recolour(out, ARMED, BLOCK)
        recolour(out, ACTIVE, MARKER)


def render_hud(out, a):
    w = bar_width(a)
    out[0] = [FLOOR] * (64 - w) + [BAR] * w
    out[63] = [BAR] * w + [FLOOR] * (64 - w)


def transition_function(state, action, frame):
    a = action_count(frame)
    out = [list(r) for r in frame]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(frame, out, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        if squares(frame, BLOCK, STEP):
            block_step(frame, out, action)
        else:
            marker_step(frame, out, action)
    a += 1
    render_hud(out, a)
    out = [[int(v) for v in r] for r in out]
    _memo.append((tuple(tuple(r) for r in out), a))
    del _memo[:-16]
    return out
