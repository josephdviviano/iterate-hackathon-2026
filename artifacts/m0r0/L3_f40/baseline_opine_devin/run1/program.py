# Mechanics: twin cyan (10) 4x4 blocks on a visible maze; floor colour 5, every other colour blocks.
# A1/A2 move both blocks up/down 4; A3 left x-4 / right x+4, A4 the reverse; each moves iff dest 4x4 is floor.
# Click a colour-9 marker: arm (blocks -> colour 1) and make it active (11); click colour 1: disarm; else no-op.
# Armed: A1-A4 steer the active marker's 4x4 cell (2x2 sprite at +1). A5 no-op. HUD 0-bars rows 0/63, w=3(a+1)//7.
# Hypothesis: a = all actions since level start (continuity-gated); fallback from bar width. Hazard 8 = plain wall.
FLOOR, BLOCK, ARMED, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_last = {"frame": None, "a": 0}


def comps(frame, colour):
    seen, out = set(), []
    for y in range(64):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in seen:
                stack, cells = [(x, y)], []
                seen.add((x, y))
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in seen and frame[ny][nx] == colour:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                out.append((min(c[0] for c in cells), min(c[1] for c in cells), cells))
    return out


def blocks_of(frame, colour):
    res = []
    for _, _, cells in comps(frame, colour):
        for (x, y) in cells:
            if (x - 1, y) not in cells and (x, y - 1) not in cells and \
                    all((x + i, y + j) in cells for i in range(4) for j in range(4)):
                if not any(abs(x - bx) < 4 and abs(y - by) < 4 for bx, by in res):
                    res.append((x, y))
    return sorted(res)


def cell_cells(x, y):
    return [(x + i, y + j) for i in range(4) for j in range(4)]


def can_move(frame, x, y, dx, dy):
    own = set(cell_cells(x, y))
    for (cx, cy) in cell_cells(x + dx, y + dy):
        if (cx, cy) in own:
            continue
        if not (0 <= cx < 64 and 1 <= cy < 63) or frame[cy][cx] != FLOOR:
            return False
    return True


def paint_block(out, x, y, colour):
    for (cx, cy) in cell_cells(x, y):
        out[cy][cx] = colour


def paint_marker(out, x, y, colour):
    for (cx, cy) in cell_cells(x, y):
        out[cy][cx] = FLOOR
    for i in range(2):
        for j in range(2):
            out[y + 1 + j][x + 1 + i] = colour


def bar_width(a):
    return 3 * (a + 1) // 7


def read_counter(frame):
    w = sum(1 for x in range(64) if frame[0][x] == BAR)
    if w == 0:
        return 0
    return max(a for a in range(200) if bar_width(a) == w)


def transition_function(state, action, frame):
    a = _last["a"] if _last["frame"] == frame else read_counter(frame)
    out = [list(row) for row in frame]
    blocks = blocks_of(frame, BLOCK)
    players = blocks_of(frame, ARMED)
    inactive = [(x - 1, y - 1) for x, y, c in comps(frame, MARK) if len(c) == 4]
    active = [(x - 1, y - 1) for x, y, c in comps(frame, ACTIVE) if len(c) == 4]
    aid = action["action_id"] if isinstance(action, dict) else action

    if aid in DIRS:
        dx, dy = DIRS[aid]
        if blocks and not players:
            for i, (bx, by) in enumerate(blocks):
                mx = dx if i == 0 or dy else -dx
                if can_move(frame, bx, by, mx, dy):
                    paint_block(out, bx, by, FLOOR)
            for i, (bx, by) in enumerate(blocks):
                mx = dx if i == 0 or dy else -dx
                if can_move(frame, bx, by, mx, dy):
                    paint_block(out, bx + mx, by + dy, BLOCK)
        elif players and active:
            mx, my = active[0]
            if can_move(frame, mx, my, dx, dy):
                paint_marker(out, mx, my, FLOOR)
                for (cx, cy) in cell_cells(mx, my):
                    out[cy][cx] = FLOOR
                paint_marker(out, mx + dx, my + dy, ACTIVE)
    elif aid == 6:
        cx, cy = action["x"], action["y"]
        hit = frame[cy][cx] if 0 <= cx < 64 and 0 <= cy < 64 else None
        if hit == MARK:
            for (mx, my) in active:
                paint_marker(out, mx, my, MARK)
            for (mx, my) in inactive:
                if mx <= cx < mx + 4 and my <= cy < my + 4:
                    paint_marker(out, mx, my, ACTIVE)
            for (bx, by) in blocks:
                paint_block(out, bx, by, ARMED)
        elif hit == ARMED:
            for (mx, my) in active:
                paint_marker(out, mx, my, MARK)
            for (bx, by) in players:
                paint_block(out, bx, by, BLOCK)

    a += 1
    w = bar_width(a)
    for x in range(64):
        out[0][x] = BAR if x >= 64 - w else FLOOR
        out[63][x] = BAR if x < w else FLOOR
    _last["frame"], _last["a"] = out, a
    return out
