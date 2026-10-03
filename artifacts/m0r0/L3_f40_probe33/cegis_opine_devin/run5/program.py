# Mechanics: floor = colour 5; blocks = 4x4 colour 10 (armed players colour 1); markers = 2x2 colour 9 (active 11) centred in a 4x4 cell.
# Block mode: A1/A2 move both blocks y-/+4, A3 apart (left x-4, right x+4), A4 together; each piece moves iff its dest cell is all floor.
# Click on 9 arms/switches (blocks->1, that marker->11), click on 1 disarms; armed mode A1-A4 move the active marker cell by 4 under the same floor rule.
# HUD: colour-0 bars on row 0 (right-aligned) and row 63 (left-aligned), width round(64a/150), a = actions this level incl. no-ops/clicks/A5.
# Hidden counter a is continuity-gated by a memo of my returned frames; fallback = largest a with the before width (0 if no bar). Unconfirmed: A7, budget end.
FLOOR, BLOCK, ARMED, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
STEP = 4
_memo = []


def bar_width(a):
    return (128 * a + 150) // 300


def read_width(frame):
    return sum(1 for v in frame[0] if v == BAR)


def prior_actions(frame):
    key = tuple(map(tuple, frame))
    for k, a in reversed(_memo):
        if k == key:
            return a
    w = read_width(frame)
    if w == 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w:
        a += 1
    return a


def squares(frame, colour, size):
    seen, out = set(), []
    for y in range(64):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in seen:
                if all(0 <= y + j < 64 and 0 <= x + i < 64 and frame[y + j][x + i] == colour
                       for j in range(size) for i in range(size)):
                    out.append((x, y))
                    for j in range(size):
                        for i in range(size):
                            seen.add((x + i, y + j))
    return out


def cell_cells(x, y):
    return {(x + i, y + j) for j in range(STEP) for i in range(STEP)}


def can_place(frame, own, x, y):
    for (cx, cy) in cell_cells(x, y):
        if not (0 <= cx < 64 and 1 <= cy < 63):
            return False
        if (cx, cy) not in own and frame[cy][cx] != FLOOR:
            return False
    return True


def move_piece(frame, out, origin, sprite_off, size, colour, dx, dy):
    """origin = 4x4 cell top-left; sprite of given size at origin+sprite_off."""
    ox, oy = origin
    own = cell_cells(ox, oy)
    if not can_place(frame, own, ox + dx, oy + dy):
        return False
    for j in range(size):
        for i in range(size):
            out[oy + sprite_off + j][ox + sprite_off + i] = FLOOR
    for j in range(size):
        for i in range(size):
            out[oy + dy + sprite_off + j][ox + dx + sprite_off + i] = colour
    return True


def recolour(out, cells_list, size, colour, off=0):
    for (x, y) in cells_list:
        for j in range(size):
            for i in range(size):
                out[y + off + j][x + off + i] = colour


def step_blocks(frame, out, blocks, act):
    blocks = sorted(blocks)
    deltas = {1: [(0, -STEP), (0, -STEP)], 2: [(0, STEP), (0, STEP)],
              3: [(-STEP, 0), (STEP, 0)], 4: [(STEP, 0), (-STEP, 0)]}[act]
    if len(blocks) != 2:
        deltas = [deltas[0]] * len(blocks)
    for b, (dx, dy) in zip(blocks, deltas):
        move_piece(frame, out, b, 0, STEP, BLOCK, dx, dy)


def step_marker(frame, out, marker, act):
    dx, dy = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}[act]
    mx, my = marker
    move_piece(frame, out, (mx - 1, my - 1), 1, 2, ACTIVE, dx, dy)


def click(frame, out, x, y, blocks, players, markers, active):
    if not (0 <= x < 64 and 0 <= y < 64):
        return
    c = frame[y][x]
    if c == MARK:
        hit = [m for m in markers if m[0] <= x < m[0] + 2 and m[1] <= y < m[1] + 2]
        if not hit:
            return
        recolour(out, blocks, STEP, ARMED)
        recolour(out, active, 2, MARK)
        recolour(out, hit[:1], 2, ACTIVE)
    elif c == ARMED:
        recolour(out, players, STEP, BLOCK)
        recolour(out, active, 2, MARK)


def draw_hud(out, a):
    w = min(64, bar_width(a))
    for x in range(64):
        out[0][x] = BAR if x >= 64 - w else FLOOR
        out[63][x] = BAR if x < w else FLOOR


def transition_function(state, action, frame):
    a = prior_actions(frame)
    out = [list(map(int, row)) for row in frame]
    blocks = squares(frame, BLOCK, STEP)
    players = squares(frame, ARMED, STEP)
    markers = squares(frame, MARK, 2)
    active = squares(frame, ACTIVE, 2)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(frame, out, action.get("x", -1), action.get("y", -1),
                  blocks, players, markers, active)
    elif action in (1, 2, 3, 4):
        if blocks:
            step_blocks(frame, out, blocks, action)
        elif active:
            step_marker(frame, out, active[0], action)
    a += 1
    draw_hud(out, a)
    _memo.append((tuple(map(tuple, out)), a))
    del _memo[:-8]
    return out
