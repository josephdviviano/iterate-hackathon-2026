# Mechanics: two 4x4 blocks (colour 10; 1 when armed) and 2x2 markers (9; active 11) on a visible maze (floor 5).
# Block mode: A1/A2 move both blocks y-4/y+4; A3 left x-4 & right x+4, A4 the reverse; each moves iff its dest is floor.
# Click 9 marker = arm (blocks->1) / switch active marker (->11); click 1 = disarm; A1-4 armed = active marker +-4 in its 4x4 cell.
# HUD: colour-0 bars, row 0 from the right and row 63 from the left, width 3(a+1)//7, a = actions this level (incl. no-ops).
# Hypothesis: a is hidden (continuity-gated on frame equality); fallback a = 0 without bar, else largest a giving that width.
FLOOR, BLOCK, ARMED, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
_memo = {"frame": None, "a": 0}


def squares(frame, colours, size):
    seen, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            if (x, y) in seen or frame[y][x] not in colours:
                continue
            cells = [(x + i, y + j) for j in range(size) for i in range(size)]
            if all(0 <= cx < 64 and 1 <= cy < 63 and frame[cy][cx] in colours for cx, cy in cells):
                seen.update(cells)
                out.append([x, y, frame[y][x]])
    return out


def can_place(grid, x, y, own):
    for j in range(4):
        for i in range(4):
            cx, cy = x + i, y + j
            if not (0 <= cx < 64 and 1 <= cy < 63):
                return False
            if (cx, cy) not in own and grid[cy][cx] != FLOOR:
                return False
    return True


def cells_of(x, y, size):
    return {(x + i, y + j) for j in range(size) for i in range(size)}


def paint(grid, x, y, size, colour):
    for cx, cy in cells_of(x, y, size):
        grid[cy][cx] = colour


def bar_width(a):
    return min(64, 3 * (a + 1) // 7)


def read_actions(frame):
    w = sum(1 for v in frame[0] if v == BAR)
    if w == 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w and a < 1000:
        a += 1
    return a


def transition_function(state, action, frame):
    grid = [list(r) for r in frame]
    a = _memo["a"] if _memo["frame"] == frame else read_actions(frame)
    blocks = squares(frame, (BLOCK, ARMED), 4)
    markers = squares(frame, (MARK, ACTIVE), 2)
    armed = any(c == ARMED for _, _, c in blocks)
    aid = action["action_id"] if isinstance(action, dict) else action
    if aid == 6:
        cx, cy = action["x"], action["y"]
        hit = frame[cy][cx] if 0 <= cx < 64 and 0 <= cy < 64 else None
        if hit == MARK:
            for b in blocks:
                paint(grid, b[0], b[1], 4, ARMED)
            for m in markers:
                inside = m[0] <= cx < m[0] + 2 and m[1] <= cy < m[1] + 2
                paint(grid, m[0], m[1], 2, ACTIVE if inside else MARK)
        elif hit == ARMED:
            for b in blocks:
                paint(grid, b[0], b[1], 4, BLOCK)
            for m in markers:
                paint(grid, m[0], m[1], 2, MARK)
    elif aid in (1, 2, 3, 4):
        dy = {1: -4, 2: 4}.get(aid, 0)
        dx = {3: -4, 4: 4}.get(aid, 0)
        if armed:
            for m in markers:
                if m[2] != ACTIVE:
                    continue
                own = cells_of(m[0] - 1, m[1] - 1, 4)
                if can_place(grid, m[0] - 1 + dx, m[1] - 1 + dy, own):
                    paint(grid, m[0], m[1], 2, FLOOR)
                    paint(grid, m[0] + dx, m[1] + dy, 2, ACTIVE)
        else:
            blocks.sort(key=lambda b: b[0])
            for k, b in enumerate(blocks):
                sx = dx if k == 0 else -dx
                own = cells_of(b[0], b[1], 4)
                if can_place(grid, b[0] + sx, b[1] + dy, own):
                    paint(grid, b[0], b[1], 4, FLOOR)
                    paint(grid, b[0] + sx, b[1] + dy, 4, b[2])
    a += 1
    w = bar_width(a)
    for x in range(64):
        if frame[0][x] in (FLOOR, BAR):
            grid[0][x] = BAR if x >= 64 - w else FLOOR
        if frame[63][x] in (FLOOR, BAR):
            grid[63][x] = BAR if x < w else FLOOR
    _memo["frame"], _memo["a"] = grid, a
    return [[int(v) for v in r] for r in grid]
