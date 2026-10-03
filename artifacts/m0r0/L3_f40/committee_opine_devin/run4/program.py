# Mechanics: two colour-10 4x4 blocks move together (A1 up, A2 down, A3 apart, A4 together, mirrored in x);
# each moves only if its destination 4x4 is all floor colour 5 (own cells excepted). Clicking a colour-9
# marker arms it (blocks -> colour 1 players, marker -> 11, old active -> 9); clicking a player disarms; then A1-A4
# move the active marker's 4x4 cell (marker drawn 2x2 at cell+1) under the same floor rule. A5 = no-op.
# HUD: colour-0 bars row 0 (right-aligned) and row 63 (left-aligned), width 3(a+1)//7, a = actions this level
# (hidden; continuity-gated, fallback = largest a matching the bar). Hypothesis 'marker A3 gone/no_change' = renaming only.
FLOOR, BLOCK, PLAYER, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_mem = {"frame": None, "a": 0}


def components(frame, colour):
    seen, comps = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in seen:
                stack, cells = [(x, y)], []
                seen.add((x, y))
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < 64 and 1 <= ny < 63 and (nx, ny) not in seen and frame[ny][nx] == colour:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                comps.append(cells)
    return comps


def tiles(frame, colour):
    """Top-left corners of 4x4 tiles covering components of the given colour."""
    out = []
    for cells in components(frame, colour):
        x0 = min(c[0] for c in cells)
        y0 = min(c[1] for c in cells)
        cs = set(cells)
        for (cx, cy) in sorted(cells, key=lambda c: (c[1], c[0])):
            if (cx - x0) % 4 == 0 and (cy - y0) % 4 == 0 and (cx, cy) in cs:
                out.append((cx, cy))
    return out


def cell_cells(x, y):
    return [(x + i, y + j) for j in range(4) for i in range(4)]


def free(frame, x, y, own):
    for cx, cy in cell_cells(x, y):
        if not (0 <= cx < 64 and 1 <= cy < 63):
            return False
        if (cx, cy) not in own and frame[cy][cx] != FLOOR:
            return False
    return True


def bar_width(frame):
    w = 0
    while w < 64 and frame[63][w] == BAR:
        w += 1
    return w


def count_from_bar(w):
    if w == 0:
        return 0
    a = 0
    while 3 * (a + 2) // 7 <= w:
        a += 1
    return a


def paint(out, cells, colour):
    for cx, cy in cells:
        out[cy][cx] = colour


def move_cell(out, frame, x, y, dx, dy, draw):
    own = set(cell_cells(x, y))
    if not free(frame, x + dx, y + dy, own):
        return False
    paint(out, own, FLOOR)
    draw(x + dx, y + dy)
    return True


def step_blocks(frame, out, colour, act):
    blocks = sorted(tiles(frame, colour))
    if len(blocks) != 2 or act not in DIRS:
        return
    dx, dy = DIRS[act]
    deltas = [(dx, dy), (-dx, dy)]
    targets = []
    for (bx, by), (ddx, ddy) in zip(blocks, deltas):
        ok = free(frame, bx + ddx, by + ddy, set(cell_cells(bx, by)))
        targets.append((bx + ddx, by + ddy) if ok else (bx, by))
    for b in blocks:
        paint(out, cell_cells(*b), FLOOR)
    for t in targets:
        paint(out, cell_cells(*t), colour)


def draw_marker(out, x, y, colour):
    paint(out, cell_cells(x, y), FLOOR)
    paint(out, [(x + 1 + i, y + 1 + j) for j in range(2) for i in range(2)], colour)


def step_marker(frame, out, act):
    act_cells = [c for c in components(frame, ACTIVE)]
    if not act_cells or act not in DIRS:
        return
    mx = min(c[0] for c in act_cells[0]) - 1
    my = min(c[1] for c in act_cells[0]) - 1
    dx, dy = DIRS[act]
    move_cell(out, frame, mx, my, dx, dy, lambda x, y: draw_marker(out, x, y, ACTIVE))


def recolour(out, frame, src, dst):
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == src:
                out[y][x] = dst


def click(frame, out, x, y):
    if not (0 <= x < 64 and 1 <= y < 63):
        return
    hit = frame[y][x]
    if hit == MARK:
        recolour(out, frame, BLOCK, PLAYER)
        recolour(out, frame, ACTIVE, MARK)
        for cells in components(frame, MARK):
            if (x, y) in cells:
                paint(out, cells, ACTIVE)
    elif hit == PLAYER:
        recolour(out, frame, PLAYER, BLOCK)
        recolour(out, frame, ACTIVE, MARK)


def render_hud(out, a):
    w = min(64, 3 * (a + 1) // 7)
    for x in range(64):
        out[0][x] = BAR if x >= 64 - w else FLOOR
        out[63][x] = BAR if x < w else FLOOR


def transition_function(state, action, frame):
    frame = [[int(v) for v in row] for row in frame]
    if _mem["frame"] is not None and _mem["frame"] == frame:
        a = _mem["a"]
    else:
        a = count_from_bar(bar_width(frame))
    out = [row[:] for row in frame]
    armed = any(PLAYER in row for row in frame[1:63])
    if isinstance(action, dict):
        click(frame, out, int(action.get("x", -1)), int(action.get("y", -1)))
    elif armed:
        step_marker(frame, out, action)
    else:
        step_blocks(frame, out, BLOCK, action)
    a += 1
    render_hud(out, a)
    _mem["frame"] = [row[:] for row in out]
    _mem["a"] = a
    return out
