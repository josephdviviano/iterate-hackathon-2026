# Mechanics: maze is drawn in the frame; floor = colour 5, everything else blocks. Two 4x4 blocks
# (colour 10; colour 1 when armed). A1/A2 move both up/down 4; A3 moves left block -4 and right +4
# (apart), A4 together; each moves iff its destination 4x4 is all floor (own cells excepted).
# Click on an inactive marker (9) arms: blocks->1, marker->11 (old active->9); click on an armed block
# disarms; A1-A4 in armed mode move the active marker's 4x4 cell. HUD: 0-bars row 0 (right) / row 63 (left), w=3(a+1)//7.
FLOOR, BLOCK, ARMED, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_mem = {"frame": None, "a": 0}


def components(frame, colours):
    seen, comps = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] in colours and (x, y) not in seen:
                stack, cells = [(x, y)], []
                seen.add((x, y))
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 1 <= ny < 63 and 0 <= nx < 64 and (nx, ny) not in seen and frame[ny][nx] in colours:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                comps.append((min(c[0] for c in cells), min(c[1] for c in cells), frame[y][x]))
    return comps


def cell(x, y):
    return [(x + i, y + j) for j in range(4) for i in range(4)]


def free(frame, x, y, own):
    for cx, cy in cell(x, y):
        if (cx, cy) in own:
            continue
        if not (0 <= cx < 64 and 1 <= cy < 63) or frame[cy][cx] != FLOOR:
            return False
    return True


def paint_block(out, x, y, col):
    for cx, cy in cell(x, y):
        out[cy][cx] = col


def paint_marker(out, x, y, col):
    for cx, cy in cell(x, y):
        out[cy][cx] = FLOOR
    for j in (1, 2):
        for i in (1, 2):
            out[y + j][x + i] = col


def bar_width(frame):
    return sum(1 for v in frame[0] if v == BAR)


def actions_so_far(frame):
    if _mem["frame"] == frame:
        return _mem["a"]
    w = bar_width(frame)
    if w == 0:
        return 0
    return max(a for a in range(200) if 3 * (a + 1) // 7 == w)


def step_blocks(frame, out, blocks, action):
    dx, dy = DIRS[action]
    blocks = sorted(blocks)
    moves = [(dx, dy), (-dx, dy)] if dx else [(0, dy), (0, dy)]
    plans = []
    for (x, y, col), (mx, my) in zip(blocks, moves):
        ok = free(frame, x + mx, y + my, set(cell(x, y)))
        plans.append((x, y, col, x + mx, y + my) if ok else None)
    for p in plans:
        if p:
            paint_block(out, p[0], p[1], FLOOR)
    for p in plans:
        if p:
            paint_block(out, p[3], p[4], p[2])


def step_marker(frame, out, active, action):
    dx, dy = DIRS[action]
    x, y = active[0] - 1, active[1] - 1
    if free(frame, x + dx, y + dy, set(cell(x, y))):
        paint_block(out, x, y, FLOOR)
        paint_marker(out, x + dx, y + dy, ACTIVE)


def click(frame, out, blocks, markers, cx, cy):
    hit = frame[cy][cx] if 0 <= cx < 64 and 0 <= cy < 64 else None
    if hit == MARK:
        for x, y, _ in blocks:
            paint_block(out, x, y, ARMED)
        for x, y, col in markers:
            if col == ACTIVE:
                paint_marker(out, x - 1, y - 1, MARK)
        tx, ty, _ = next(m for m in markers if m[0] <= cx < m[0] + 2 and m[1] <= cy < m[1] + 2)
        paint_marker(out, tx - 1, ty - 1, ACTIVE)
    elif hit == ARMED:
        for x, y, _ in blocks:
            paint_block(out, x, y, BLOCK)
        for x, y, col in markers:
            if col == ACTIVE:
                paint_marker(out, x - 1, y - 1, MARK)


def render_bar(out, a):
    w = 3 * (a + 1) // 7
    for x in range(64):
        if out[0][x] == BAR:
            out[0][x] = FLOOR
        if out[63][x] == BAR:
            out[63][x] = FLOOR
    for i in range(w):
        out[0][63 - i] = BAR
        out[63][i] = BAR


def transition_function(state, action, frame):
    a = actions_so_far(frame) + 1
    out = [list(row) for row in frame]
    blocks = components(frame, (BLOCK, ARMED))
    markers = components(frame, (MARK, ACTIVE))
    armed = any(c == ARMED for _, _, c in blocks)
    active = next((m for m in markers if m[2] == ACTIVE), None)
    if isinstance(action, dict):
        click(frame, out, blocks, markers, action["x"], action["y"])
    elif action in DIRS:
        if armed and active:
            step_marker(frame, out, active, action)
        else:
            step_blocks(frame, out, blocks, action)
    render_bar(out, a)
    _mem["frame"], _mem["a"] = [list(r) for r in out], a
    return out
