# Mechanics: frame-native mirror-blocks game. Floor = colour 5; any non-5 cell (walls 15/8, markers, blocks) blocks a 4x4 move.
# Block mode (colour-10 tiles): A1/A2 move both blocks y-4/y+4, A3 left x-4 & right x+4, A4 the mirror; each moves iff its dest 4x4 is all 5.
# Click colour 9 marker -> arm (blocks -> colour 1, that marker -> 11 active); click colour 1 -> disarm; other clicks/A5/A7 no-op.
# Armed: A1-A4 move the active marker's 4x4 cell by 4 iff dest is all floor. HUD bars rows 0 (right) / 63 (left): w = round(64*a/150), a = actions this level.
# Hypothesis: a is hidden, carried via continuity with my own recent outputs; fallback a = largest count giving the observed width (0 if no bar).
BUDGET = 150
FLOOR, BLOCK, ARMED, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_memory = []


def key(frame):
    return tuple(tuple(r) for r in frame)


def bar_width(a):
    return (64 * a + BUDGET // 2) // BUDGET


def fallback_count(frame):
    w = sum(1 for v in frame[0] if v == BAR)
    if w == 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w:
        a += 1
    return a


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
    out = []
    for cells in components(frame, colour):
        x0, y0 = min(c[0] for c in cells), min(c[1] for c in cells)
        x1, y1 = max(c[0] for c in cells), max(c[1] for c in cells)
        for ty in range(y0, y1 + 1, 4):
            for tx in range(x0, x1 + 1, 4):
                if frame[ty][tx] == colour:
                    out.append((tx, ty))
    return out


def markers(frame, colour):
    return [(min(c[0] for c in cells), min(c[1] for c in cells)) for cells in components(frame, colour)]


def cell_free(frame, x, y, size=4):
    if x < 0 or x + size > 64 or y < 1 or y + size > 63:
        return False
    return all(frame[y + j][x + i] == FLOOR for j in range(size) for i in range(size))


def paint(frame, x, y, size, colour):
    for j in range(size):
        for i in range(size):
            frame[y + j][x + i] = colour


def order_blocks(blocks, left_pos):
    blocks = sorted(blocks)
    if left_pos in blocks:
        blocks.remove(left_pos)
        blocks.insert(0, left_pos)
    return blocks


def step_blocks(frame, out, blocks, action):
    dx, dy = DIRS[action]
    moved = []
    for idx, (bx, by) in enumerate(blocks):
        mx = dx if idx == 0 else -dx
        nx, ny = bx + mx, by + dy
        if cell_free(frame, nx, ny):
            moved.append((idx, bx, by, nx, ny))
    for idx, bx, by, nx, ny in moved:
        paint(out, bx, by, 4, FLOOR)
    for idx, bx, by, nx, ny in moved:
        paint(out, nx, ny, 4, BLOCK)
        blocks[idx] = (nx, ny)
    return blocks


def step_active_marker(frame, out, action):
    act = markers(frame, ACTIVE)
    if not act:
        return
    mx, my = act[0]
    dx, dy = DIRS[action]
    if cell_free(frame, mx - 1 + dx, my - 1 + dy):
        paint(out, mx, my, 2, FLOOR)
        paint(out, mx + dx, my + dy, 2, ACTIVE)


def click(frame, out, x, y):
    if not (0 <= x < 64 and 1 <= y < 63):
        return
    hit = frame[y][x]
    if hit == MARK:
        for tx, ty in tiles(frame, BLOCK):
            paint(out, tx, ty, 4, ARMED)
        for mx, my in markers(frame, ACTIVE):
            paint(out, mx, my, 2, MARK)
        for mx, my in markers(frame, MARK):
            if mx <= x < mx + 2 and my <= y < my + 2:
                paint(out, mx, my, 2, ACTIVE)
    elif hit == ARMED:
        for tx, ty in tiles(frame, ARMED):
            paint(out, tx, ty, 4, BLOCK)
        for mx, my in markers(frame, ACTIVE):
            paint(out, mx, my, 2, MARK)


def draw_bars(out, a):
    w = bar_width(a)
    for x in range(64):
        out[0][x] = BAR if x >= 64 - w else FLOOR
        out[63][x] = BAR if x < w else FLOOR


def transition_function(state, action, frame):
    global _memory
    frame = [[int(v) for v in row] for row in frame]
    k = key(frame)
    known = [m for m in _memory if m[0] == k]
    if known:
        a, left_pos = known[-1][1], known[-1][2]
    else:
        a, left_pos = fallback_count(frame), None
    out = [row[:] for row in frame]
    blocks = tiles(frame, BLOCK) or tiles(frame, ARMED)
    if left_pos not in blocks:
        left_pos = min(blocks) if blocks else None
    blocks = order_blocks(blocks, left_pos)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(frame, out, int(action.get("x", -1)), int(action.get("y", -1)))
    elif action in DIRS:
        if tiles(frame, BLOCK):
            blocks = step_blocks(frame, out, blocks, action)
        else:
            step_active_marker(frame, out, action)
    a += 1
    draw_bars(out, a)
    left_pos = blocks[0] if blocks else None
    _memory = (_memory + [(key(out), a, left_pos)])[-4:]
    return out
