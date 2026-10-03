# Mechanics: mirror-blocks/marker game, purely on frame colours (floor 5, walls 15/8 static scenery).
# Blocks = 4x4 colour 10; A1/A2 move both y-/+4, A3 moves them apart, A4 together; each moves iff dest 4x4 is all 5.
# Click a 9 marker -> blocks become 1 (armed), marker 11 (old 11 -> 9); click a 1 -> disarm; armed A1-4 move the 11 marker cell.
# HUD: rows 0 (right-aligned) and 63 (left-aligned) 0-bars, width round(64a/150) with a = actions this level (all count).
# Hidden a is kept per returned frame (recent memo); fallback = largest a giving the bar width. Unconfirmed: what ends a level.

FLOOR, BLOCK, ARMED, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_memo = []


def bar_width(a):
    return (128 * a + 150) // 300


def squares(frame, colour, size):
    seen, out = set(), []
    for y in range(64):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in seen:
                ok = all(0 <= x + i < 64 and 0 <= y + j < 64 and frame[y + j][x + i] == colour
                         for i in range(size) for j in range(size))
                if ok:
                    for i in range(size):
                        for j in range(size):
                            seen.add((x + i, y + j))
                    out.append((x, y))
    return out


def read_bar(frame):
    return sum(1 for x in range(64) if frame[63][x] == BAR)


def actions_so_far(frame):
    for f, a in reversed(_memo):
        if f == frame:
            return a
    w = read_bar(frame)
    if w == 0:
        return 0
    return max(a for a in range(400) if bar_width(a) == w)


def cells(x, y, size):
    return [(x + i, y + j) for i in range(size) for j in range(size)]


def can_place(frame, own, x, y):
    for cx, cy in cells(x, y, 4):
        if not (0 <= cx < 64 and 1 <= cy < 63):
            return False
        if (cx, cy) not in own and frame[cy][cx] != FLOOR:
            return False
    return True


def move_sprite(out, frame, old_cells, new_xy, size, colour, offset=0):
    for cx, cy in old_cells:
        out[cy][cx] = FLOOR
    nx, ny = new_xy
    for cx, cy in cells(nx + offset, ny + offset, size):
        out[cy][cx] = colour


def step_blocks(frame, out, dx, dy, action):
    blocks = sorted(squares(frame, BLOCK, 4))
    if len(blocks) < 2:
        return
    left, right = blocks[0], blocks[-1]
    moves = []
    for (x, y), side in ((left, -1), (right, 1)):
        mdx = dx if action in (1, 2) else (4 * side if action == 3 else -4 * side)
        moves.append(((x, y), (x + mdx, y + dy)))
    for (x, y), (nx, ny) in moves:
        own = set(cells(x, y, 4))
        if can_place(out, own, nx, ny):
            move_sprite(out, frame, own, (nx, ny), 4, BLOCK)


def step_marker(frame, out, dx, dy):
    act = squares(frame, ACTIVE, 2)
    if not act:
        return
    mx, my = act[0]
    cx, cy = mx - 1, my - 1
    own = set(cells(cx, cy, 4))
    if can_place(frame, own, cx + dx, cy + dy):
        move_sprite(out, frame, set(cells(mx, my, 2)), (cx + dx, cy + dy), 2, ACTIVE, 1)


def recolour(out, old, new):
    for y in range(1, 63):
        for x in range(64):
            if out[y][x] == old:
                out[y][x] = new


def click(frame, out, x, y):
    if not (0 <= x < 64 and 0 <= y < 64):
        return
    c = frame[y][x]
    if c == MARK:
        recolour(out, ACTIVE, MARK)
        recolour(out, BLOCK, ARMED)
        stack = [(x, y)]
        while stack:
            px, py = stack.pop()
            if 0 <= px < 64 and 0 <= py < 64 and out[py][px] == MARK:
                out[py][px] = ACTIVE
                stack += [(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)]
    elif c == ARMED:
        recolour(out, ARMED, BLOCK)
        recolour(out, ACTIVE, MARK)


def draw_hud(out, a):
    w = bar_width(a)
    for x in range(64):
        out[0][x] = BAR if x >= 64 - w else FLOOR
        out[63][x] = BAR if x < w else FLOOR


def transition_function(state, action, frame):
    a = actions_so_far(frame)
    out = [list(r) for r in frame]
    if isinstance(action, dict):
        click(frame, out, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        dx, dy = DIRS[action]
        if squares(frame, ACTIVE, 2):
            step_marker(frame, out, dx, dy)
        else:
            step_blocks(frame, out, dx, dy, action)
    a += 1
    draw_hud(out, a)
    out = [[int(v) for v in r] for r in out]
    _memo.append(([list(r) for r in out], a))
    del _memo[:-6]
    return out
