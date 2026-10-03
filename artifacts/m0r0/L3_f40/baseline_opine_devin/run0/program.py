# Mechanics: blocks (colour 10) / armed players (1) / markers (9 inactive, 11 active) on a 4-cell grid; floor = colour 5.
# A1/A2 move both blocks up/down 4; A3 moves left block left & right block right (A4 reverse); a 4x4 move succeeds iff
# every dest cell is floor inside rows 1..62. Click a 9 marker: blocks arm (10->1), it becomes active 11 (old 11->9);
# armed: A1-A4 move the active marker's 4x4 cell (marker bbox grown by 1); click a 1 player: disarm. Other clicks/A5 no-op.
# HUD: bar of 0s on row 0 (from right) and row 63 (from left), width 3(a+1)//7, a = actions this level (hidden, gated).
FLOOR, BLOCK, ARMED, MARK, ACTIVE, HUD = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_last = {"frame": None, "a": 0}


def cells_of(frame, colour):
    return {(x, y) for y in range(64) for x in range(64) if frame[y][x] == colour}


def components(cells):
    cells, comps = set(cells), []
    while cells:
        stack, comp = [cells.pop()], set()
        while stack:
            x, y = stack.pop()
            comp.add((x, y))
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def tiles(comp):
    """Split a component of 4x4 pieces into 4x4 tiles (top-left corners)."""
    x0 = min(x for x, _ in comp)
    y0 = min(y for _, y in comp)
    return sorted({(x0 + (x - x0) // 4 * 4, y0 + (y - y0) // 4 * 4) for x, y in comp})


def square(x, y, n=4):
    return [(x + i, y + j) for j in range(n) for i in range(n)]


def can_move(frame, x, y, dx, dy):
    """Guard: destination 4x4 cell is floor inside the playfield rows."""
    for cx, cy in square(x + dx, y + dy):
        if not (0 <= cx < 64 and 1 <= cy <= 62) or frame[cy][cx] != FLOOR:
            return False
    return True


def bar_width(frame):
    w = 0
    while w < 64 and frame[0][63 - w] == HUD:
        w += 1
    return w


def actions_from_bar(w):
    if w == 0:
        return 0
    return max(a for a in range(0, 200) if 3 * (a + 1) // 7 == w)


def step_blocks(frame, out, colour, action):
    pieces = [t for c in components(cells_of(frame, colour)) for t in tiles(c)]
    if len(pieces) < 2 or action not in DIRS:
        return
    pieces.sort()
    left, right = pieces[0], pieces[-1]
    dx, dy = DIRS[action]
    moves = [(left, dx, dy), (right, -dx, dy)]
    targets = [(p, ddx, ddy) for p, ddx, ddy in moves if can_move(frame, p[0], p[1], ddx, ddy)]
    for (x, y), _, _ in targets:
        for cx, cy in square(x, y):
            out[cy][cx] = FLOOR
    for (x, y), ddx, ddy in targets:
        for cx, cy in square(x + ddx, y + ddy):
            out[cy][cx] = colour


def step_marker(frame, out, action):
    act = cells_of(frame, ACTIVE)
    if not act or action not in DIRS:
        return
    mx = min(x for x, _ in act)
    my = min(y for _, y in act)
    dx, dy = DIRS[action]
    if not can_move(frame, mx - 1, my - 1, dx, dy):
        return
    for cx, cy in act:
        out[cy][cx] = FLOOR
    for cx, cy in act:
        out[cy + dy][cx + dx] = ACTIVE


def click(frame, out, x, y):
    if not (0 <= x < 64 and 0 <= y < 64):
        return
    hit = frame[y][x]
    if hit == MARK:
        comp = next(c for c in components(cells_of(frame, MARK)) if (x, y) in c)
        recolour(frame, out, {ACTIVE: MARK, BLOCK: ARMED})
        for cx, cy in comp:
            out[cy][cx] = ACTIVE
    elif hit == ARMED:
        recolour(frame, out, {ACTIVE: MARK, ARMED: BLOCK})


def recolour(frame, out, mapping):
    for yy in range(64):
        for xx in range(64):
            if frame[yy][xx] in mapping:
                out[yy][xx] = mapping[frame[yy][xx]]


def draw_hud(out, a):
    w = 3 * (a + 1) // 7
    for i in range(min(w, 64)):
        out[0][63 - i] = HUD
        out[63][i] = HUD


def transition_function(state, action, frame):
    frame = [[int(v) for v in row] for row in frame]
    if _last["frame"] is not None and frame == _last["frame"]:
        a = _last["a"]
    else:
        a = actions_from_bar(bar_width(frame))
    out = [row[:] for row in frame]
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(frame, out, int(action.get("x", -1)), int(action.get("y", -1)))
    elif cells_of(frame, ARMED):
        step_marker(frame, out, action)
    else:
        step_blocks(frame, out, BLOCK, action)
    a += 1
    draw_hud(out, a)
    _last["frame"], _last["a"] = [row[:] for row in out], a
    return out
