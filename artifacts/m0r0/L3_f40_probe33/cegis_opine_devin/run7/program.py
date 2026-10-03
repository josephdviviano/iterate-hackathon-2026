# Mechanics: floor 5; two cyan 4x4 blocks (10) move together: A1/A2 y-/+4, A3 apart / A4 together (mirrored by x order).
# A sprite moves iff every destination cell not already its own is floor 5 (walls 15, hazard 8, markers, players block).
# Click on an inactive marker (9): blocks turn into players (1) and that marker becomes active (11); A1-A4 then move the
# active 2x2 marker by 4. Click on a player (1) disarms: players -> 10, active -> 9. Other clicks and A5/A7 are no-ops.
# HUD: rows 0 (from right) / 63 (from left) bars of 0, width = round(64a/150) with a = actions this level (hidden, memo-gated).
FLOOR, BLOCK, PLAYER, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
MOVES = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_memo = []


def bar_width(a):
    return (128 * a + 150) // 300


def squares(frame, colour, size):
    """Top-left corners of size x size squares of `colour` in the play rows (scan-claim)."""
    seen, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in seen:
                cells = [(x + i, y + j) for j in range(size) for i in range(size)]
                if all(0 <= cx < 64 and 1 <= cy < 63 and frame[cy][cx] == colour for cx, cy in cells):
                    seen.update(cells)
                    out.append((x, y))
    return out


def cells_of(pos, size):
    x, y = pos
    return [(x + i, y + j) for j in range(size) for i in range(size)]


def can_place(frame, pos, size, own):
    for cx, cy in cells_of(pos, size):
        if not (0 <= cx < 64 and 1 <= cy < 63):
            return False
        if (cx, cy) not in own and frame[cy][cx] != FLOOR:
            return False
    return True


def paint(out, pos, size, colour):
    for cx, cy in cells_of(pos, size):
        out[cy][cx] = colour


def move_sprite(frame, out, pos, size, colour, dx, dy):
    new = (pos[0] + dx, pos[1] + dy)
    if not can_place(frame, new, size, set(cells_of(pos, size))):
        return
    paint(out, pos, size, FLOOR)
    paint(out, new, size, colour)


def step_blocks(frame, out, blocks, action):
    """Block mode: both blocks move, horizontal moves mirrored about the pair."""
    dx, dy = MOVES[action]
    order = sorted(blocks)
    for k, pos in enumerate(order):
        sx = -dx if k == len(order) - 1 else dx
        move_sprite(frame, out, pos, 4, BLOCK, sx, dy)


def step_marker(frame, out, active, action):
    dx, dy = MOVES[action]
    move_sprite(frame, out, active, 2, ACTIVE, dx, dy)


def hit(pos, size, x, y):
    return pos[0] <= x < pos[0] + size and pos[1] <= y < pos[1] + size


def click(frame, out, x, y):
    blocks = squares(frame, BLOCK, 4)
    players = squares(frame, PLAYER, 4)
    marks = squares(frame, MARK, 2)
    actives = squares(frame, ACTIVE, 2)
    for m in marks:
        if hit(m, 2, x, y):
            for a in actives:
                paint(out, a, 2, MARK)
            for b in blocks:
                paint(out, b, 4, PLAYER)
            paint(out, m, 2, ACTIVE)
            return
    for p in players:
        if hit(p, 4, x, y):
            for q in players:
                paint(out, q, 4, BLOCK)
            for a in actives:
                paint(out, a, 2, MARK)
            return


def render_hud(out, a):
    w = bar_width(a)
    for x in range(64):
        out[0][x] = BAR if x >= 64 - w else FLOOR
        out[63][x] = BAR if x < w else FLOOR


def actions_so_far(frame):
    for f, a in reversed(_memo):
        if f == frame:
            return a
    w = sum(1 for v in frame[63] if v == BAR)
    if w == 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w:
        a += 1
    return a


def transition_function(state, action, frame):
    a = actions_so_far(frame) + 1
    out = [list(row) for row in frame]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(frame, out, action["x"], action["y"])
    elif action in MOVES:
        actives = squares(frame, ACTIVE, 2)
        if actives:
            step_marker(frame, out, actives[0], action)
        else:
            step_blocks(frame, out, squares(frame, BLOCK, 4), action)
    render_hud(out, a)
    out = [[int(v) for v in row] for row in out]
    _memo.append(([list(r) for r in out], a))
    del _memo[:-8]
    return out
