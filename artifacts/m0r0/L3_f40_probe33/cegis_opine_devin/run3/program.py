# Mechanics: blocks (4x4 colour 10) move together: A1/A2 vertical +-4, A3 apart / A4 together (mirrored about x=32);
# each moves only if its destination 4x4 is all floor colour 5 (own cells excepted). Clicking an inactive marker
# (colour 9) arms: blocks become players (colour 1), that marker turns active (11), old active -> 9; clicking a player
# disarms. While armed, A1-A4 move the active marker's 4x4 cell by 4 under the same floor rule. A5/A7 are no-ops.
# HUD: a = actions this level; zeros in row 0 (right-aligned) and row 63 (left-aligned), width round(64a/150). Unconfirmed: budget 150.
FLOOR, BLOCK, PLAYER, MARKER, ACTIVE, HUDC = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_memo = []


def squares(frame, colour, size):
    seen, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            if (x, y) in seen or frame[y][x] != colour:
                continue
            cells = [(x + i, y + j) for j in range(size) for i in range(size)]
            if all(0 <= cx < 64 and 0 <= cy < 64 and frame[cy][cx] == colour for cx, cy in cells):
                seen.update(cells)
                out.append((x, y))
    return out


def region_free(frame, x, y, size, own):
    for j in range(size):
        for i in range(size):
            cx, cy = x + i, y + j
            if not (0 <= cx < 64 and 1 <= cy <= 62):
                return False
            if (cx, cy) not in own and frame[cy][cx] != FLOOR:
                return False
    return True


def fill(frame, x, y, size, colour):
    for j in range(size):
        for i in range(size):
            frame[y + j][x + i] = colour


def cells(x, y, size):
    return {(x + i, y + j) for j in range(size) for i in range(size)}


def move_block(frame, out, pos, dx, dy, colour):
    x, y = pos
    if region_free(frame, x + dx, y + dy, 4, cells(x, y, 4)):
        fill(out, x, y, 4, FLOOR)
        return (x + dx, y + dy)
    return pos


def step_blocks(frame, out, action):
    blocks = squares(frame, BLOCK, 4)
    dx, dy = DIRS[action]
    new = []
    for b in blocks:
        bdx = dx
        if dx and b[0] + 2 >= 32:
            bdx = -dx
        new.append(move_block(frame, out, b, bdx, dy, BLOCK))
    for x, y in new:
        fill(out, x, y, 4, BLOCK)


def step_marker(frame, out, action):
    dx, dy = DIRS[action]
    for x, y in squares(frame, ACTIVE, 2):
        cx, cy = x - 1, y - 1
        if region_free(frame, cx + dx, cy + dy, 4, cells(x, y, 2)):
            fill(out, x, y, 2, FLOOR)
            fill(out, x + dx, y + dy, 2, ACTIVE)


def recolour(frame, out, src, dst, size):
    for x, y in squares(frame, src, size):
        fill(out, x, y, size, dst)


def click(frame, out, cx, cy):
    if not (0 <= cx < 64 and 0 <= cy < 64):
        return
    hit = frame[cy][cx]
    if hit == MARKER:
        recolour(frame, out, BLOCK, PLAYER, 4)
        recolour(frame, out, ACTIVE, MARKER, 2)
        for x, y in squares(frame, MARKER, 2):
            if x <= cx < x + 2 and y <= cy < y + 2:
                fill(out, x, y, 2, ACTIVE)
    elif hit == PLAYER:
        recolour(frame, out, PLAYER, BLOCK, 4)
        recolour(frame, out, ACTIVE, MARKER, 2)


def bar_width(a):
    return min(64, (128 * a + 150) // 300)


def hud_count(frame):
    w = sum(1 for v in frame[0] if v == HUDC)
    for prev, a in reversed(_memo):
        if prev == frame:
            return a
    if w == 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w:
        a += 1
    return a


def draw_hud(out, a):
    w = bar_width(a)
    for x in range(64):
        out[0][x] = HUDC if x >= 64 - w else FLOOR
        out[63][x] = HUDC if x < w else FLOOR


def transition_function(state, action, frame):
    frame = [[int(v) for v in row] for row in frame]
    out = [row[:] for row in frame]
    a = hud_count(frame) + 1
    armed = bool(squares(frame, PLAYER, 4))
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(frame, out, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        if armed:
            step_marker(frame, out, action)
        else:
            step_blocks(frame, out, action)
    draw_hud(out, a)
    _memo.append(([row[:] for row in out], a))
    del _memo[:-4]
    return out
