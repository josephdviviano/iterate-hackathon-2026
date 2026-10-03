# Mechanics: two colour-10 4x4 blocks on a visible maze (floor colour 5). A1/A2 move both y-/+4, A3 moves them
# apart (left x-4, right x+4), A4 together; a 4x4 move succeeds iff every destination cell outside the mover is 5.
# Click a colour-9 marker: arm (blocks 10->1), clicked marker -> 11 (active), old active -> 9; click a 1 player: disarm.
# Armed: arrows move active marker's 4x4 cell (2x2 + margin) by 4, same floor rule; A5 no-op. HUD: n = actions this level (hidden); 0-bars row 0 (right)/63 (left), w=round(64n/150).
# n resumes from a recent returned frame (memo of 8), else largest n for the width. Unconfirmed: 150-action budget reading.
FLOOR, BLOCK, PLAYER, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_seen = []  # recent (returned frame, n), most recent last


def bar_width(n):
    return (128 * n + 150) // 300


def squares(frame, colour, size):
    seen, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in seen:
                cells = {(x + i, y + j) for i in range(size) for j in range(size)}
                if all(0 <= cx < 64 and 1 <= cy < 63 and frame[cy][cx] == colour for cx, cy in cells):
                    seen |= cells
                    out.append((x, y))
    return out


def can_place(frame, own, x, y, size):
    for j in range(size):
        for i in range(size):
            cx, cy = x + i, y + j
            if not (0 <= cx < 64 and 1 <= cy < 63):
                return False
            if (cx, cy) not in own and frame[cy][cx] != FLOOR:
                return False
    return True


def move_square(frame, x, y, size, dx, dy, colour, sprite=None):
    """Move a size x size cell; sprite = (offset, sprite size) drawn inside it."""
    own = {(x + i, y + j) for i in range(size) for j in range(size)}
    if not can_place(frame, own, x + dx, y + dy, size):
        return False
    off, ss = sprite if sprite else (0, size)
    for j in range(ss):
        for i in range(ss):
            frame[y + off + j][x + off + i] = FLOOR
    for j in range(ss):
        for i in range(ss):
            frame[y + dy + off + j][x + dx + off + i] = colour
    return True


def recolour(frame, src, dst):
    for row in frame:
        for i, v in enumerate(row):
            if v == src:
                row[i] = dst


def step_blocks(frame, action):
    blocks = sorted(squares(frame, BLOCK, 4))
    dx, dy = DIRS[action]
    for idx, (x, y) in enumerate(blocks):
        bx = dx
        if action in (3, 4) and idx == len(blocks) - 1 and len(blocks) > 1:
            bx = -dx
        move_square(frame, x, y, 4, bx, dy, BLOCK)


def step_marker(frame, action):
    act = squares(frame, ACTIVE, 2)
    if not act:
        return
    x, y = act[0]
    dx, dy = DIRS[action]
    move_square(frame, x - 1, y - 1, 4, dx, dy, ACTIVE, sprite=(1, 2))


def click(frame, cx, cy):
    if not (0 <= cx < 64 and 0 <= cy < 64):
        return
    c = frame[cy][cx]
    if c == MARK:
        recolour(frame, ACTIVE, MARK)
        stack = [(cx, cy)]
        while stack:
            px, py = stack.pop()
            if 0 <= px < 64 and 0 <= py < 64 and frame[py][px] == MARK:
                frame[py][px] = ACTIVE
                stack += [(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)]
        recolour(frame, BLOCK, PLAYER)
    elif c == PLAYER:
        recolour(frame, PLAYER, BLOCK)
        recolour(frame, ACTIVE, MARK)


def draw_hud(frame, n):
    w = bar_width(n)
    for x in range(64):
        frame[0][x] = BAR if x >= 64 - w else FLOOR
        frame[63][x] = BAR if x < w else FLOOR


def fallback_n(frame):
    w = sum(1 for v in frame[0] if v == BAR)
    if w == 0:
        return 0
    return max(a for a in range(400) if bar_width(a) == w)


def transition_function(state, action, frame):
    frame = [[int(v) for v in row] for row in frame]
    n = next((k for f, k in reversed(_seen) if f == frame), None)
    if n is None:
        n = fallback_n(frame)
    out = [row[:] for row in frame]
    if isinstance(action, dict):
        click(out, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        if squares(out, PLAYER, 4):
            step_marker(out, action)
        else:
            step_blocks(out, action)
    n += 1
    draw_hud(out, n)
    _seen.append(([row[:] for row in out], n))
    del _seen[:-8]
    return out
