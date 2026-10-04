# Blocks move four cells: A1/A2 up/down, A3 apart, A4 together, mirrored by board half.
# Clicking an inactive marker arms it and recolours blocks; clicking an armed player disarms.
# Armed arrows move only the active marker; its enclosing 4x4 destination must be floor colour 5.
# HUD bars count every action with width round(64*a/150); hidden count requires last-frame continuity.
# Walls and hazards are unchanged terrain; unconfirmed: reset trigger and behaviour after budget exhaustion.
FLOOR, BLOCK, PLAYER, MARKER, ACTIVE, HUD = 5, 10, 1, 9, 11, 0
_last_frame = None
_last_count = 0


def bar_width(count):
    return (128 * count + 150) // 300


def find_squares(frame, colour, size):
    claimed, squares = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] != colour or (x, y) in claimed:
                continue
            cells = cells_of(x, y, size)
            if all(0 <= cx < 64 and 1 <= cy < 63
                   and frame[cy][cx] == colour for cx, cy in cells):
                claimed.update(cells)
                squares.append((x, y))
    return squares


def cells_of(x, y, size):
    return {(x + i, y + j) for j in range(size) for i in range(size)}


def destination_is_floor(frame, x, y, own):
    for cx, cy in cells_of(x, y, 4):
        if not (0 <= cx < 64 and 1 <= cy < 63):
            return False
        if (cx, cy) not in own and frame[cy][cx] != FLOOR:
            return False
    return True


def paint(out, x, y, size, colour):
    for cy in range(y, y + size):
        for cx in range(x, x + size):
            out[cy][cx] = colour


def update_blocks(frame, out, action):
    moves = []
    for x, y in find_squares(frame, BLOCK, 4):
        if action in (1, 2):
            dx, dy = 0, (-4 if action == 1 else 4)
        else:
            side = -1 if x + 2 < len(frame[0]) // 2 else 1
            dx, dy = (side * 4 if action == 3 else -side * 4), 0
        if destination_is_floor(frame, x + dx, y + dy, cells_of(x, y, 4)):
            moves.append((x, y, x + dx, y + dy))
    for x, y, _, _ in moves:
        paint(out, x, y, 4, FLOOR)
    for _, _, x, y in moves:
        paint(out, x, y, 4, BLOCK)


def update_active_marker(frame, out, action):
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    for x, y in find_squares(frame, ACTIVE, 2)[:1]:
        if destination_is_floor(frame, x - 1 + dx, y - 1 + dy, cells_of(x, y, 2)):
            paint(out, x, y, 2, FLOOR)
            paint(out, x + dx, y + dy, 2, ACTIVE)


def update_selection(frame, out, x, y):
    if not (0 <= x < 64 and 0 <= y < 64):
        return
    clicked = frame[y][x]
    if clicked == MARKER:
        for mx, my in find_squares(frame, ACTIVE, 2):
            paint(out, mx, my, 2, MARKER)
        for mx, my in find_squares(frame, MARKER, 2):
            if mx <= x < mx + 2 and my <= y < my + 2:
                paint(out, mx, my, 2, ACTIVE)
        for bx, by in find_squares(frame, BLOCK, 4):
            paint(out, bx, by, 4, PLAYER)
    elif clicked == PLAYER:
        for mx, my in find_squares(frame, ACTIVE, 2):
            paint(out, mx, my, 2, MARKER)
        for bx, by in find_squares(frame, PLAYER, 4):
            paint(out, bx, by, 4, BLOCK)


def actions_so_far(frame):
    if frame == _last_frame:
        return _last_count
    width = sum(value == HUD for value in frame[0])
    if width == 0:
        return 0
    return (300 * (width + 1) - 151) // 128


def update_hud(out, count):
    width = bar_width(count)
    for x in range(64):
        out[0][x] = HUD if x >= 64 - width else FLOOR
        out[63][x] = HUD if x < width else FLOOR


def transition_function(state, action, frame):
    global _last_frame, _last_count
    count = actions_so_far(frame)
    out = [list(row) for row in frame]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            update_selection(frame, out, action.get("x", -1), action.get("y", -1))
    elif action in (1, 2, 3, 4):
        if find_squares(frame, PLAYER, 4):
            update_active_marker(frame, out, action)
        else:
            update_blocks(frame, out, action)
    count += 1
    update_hud(out, count)
    out = [[int(value) for value in row] for row in out]
    _last_frame = [row[:] for row in out]
    _last_count = count
    return out
