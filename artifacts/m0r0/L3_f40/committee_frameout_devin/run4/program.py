# Mechanics: 4x4-cell maze on a frame; floor colour 5, everything else (15, 8, markers, blocks) blocks motion.
# Block mode: two colour-10 blocks; A1/A2 move both y-4/y+4, A3 left x-4/right x+4, A4 the reverse; each moves alone if its target cell is all floor.
# Click a colour-9 marker: blocks turn colour 1 (armed), that marker turns 11 (active); A1-A4 then steer the active marker; click a colour-1 player = disarm.
# HUD: w = 3(a+1)//7 zero cells right-aligned on row 0 and left-aligned on row 63, a = actions since level start (hidden, continuity-gated).
# Unconfirmed: A5/A7 effects (treated as no-ops that still tick the counter); fallback a = 0 if no bar else (7w+2)//3.
FLOOR, BLOCK, ARMED, MARK, ACTIVE, HUD = 5, 10, 1, 9, 11, 0
_mem = {"frame": None, "a": 0}


def cells_of(frame, colour, size):
    seen, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == colour and (x, y) not in seen:
                for dy in range(size):
                    for dx in range(size):
                        seen.add((x + dx, y + dy))
                out.append((x, y))
    return out


def bar_width(frame):
    return sum(1 for v in frame[63] if v == HUD)


def free(frame, x, y, own):
    for dy in range(4):
        for dx in range(4):
            px, py = x + dx, y + dy
            if not (0 <= px < 64 and 1 <= py < 63):
                return False
            if frame[py][px] != FLOOR and (px, py) not in own:
                return False
    return True


def cell_set(x, y):
    return {(x + dx, y + dy) for dx in range(4) for dy in range(4)}


def paint_cell(frame, x, y, colour):
    for dy in range(4):
        for dx in range(4):
            frame[y + dy][x + dx] = colour


def paint_marker(frame, cx, cy, colour):
    paint_cell(frame, cx, cy, FLOOR)
    for dy in range(2):
        for dx in range(2):
            frame[cy + 1 + dy][cx + 1 + dx] = colour


DELTA = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}


def move_blocks(frame, out, blocks, colour, act):
    dx, dy = DELTA[act]
    blocks = sorted(blocks)
    moves = []
    for i, (x, y) in enumerate(blocks):
        mdx = dx if i == 0 else -dx
        nx, ny = x + mdx, y + dy
        if free(frame, nx, ny, cell_set(x, y)):
            moves.append(((x, y), (nx, ny)))
    for (x, y), _ in moves:
        paint_cell(out, x, y, FLOOR)
    for _, (nx, ny) in moves:
        paint_cell(out, nx, ny, colour)


def move_marker(frame, out, cell, act):
    dx, dy = DELTA[act]
    x, y = cell
    nx, ny = x + dx, y + dy
    if free(frame, nx, ny, cell_set(x, y)):
        paint_cell(out, x, y, FLOOR)
        paint_marker(out, nx, ny, ACTIVE)


def click(frame, out, cx, cy):
    if not (0 <= cx < 64 and 0 <= cy < 64):
        return
    c = frame[cy][cx]
    if c == MARK:
        sx, sy = cx, cy
        while frame[sy][sx - 1] == MARK:
            sx -= 1
        while frame[sy - 1][sx] == MARK:
            sy -= 1
        for (x, y) in cells_of(frame, ACTIVE, 2):
            paint_marker(out, x - 1, y - 1, MARK)
        for (x, y) in cells_of(frame, BLOCK, 4):
            paint_cell(out, x, y, ARMED)
        paint_marker(out, sx - 1, sy - 1, ACTIVE)
    elif c == ARMED:
        for (x, y) in cells_of(frame, ACTIVE, 2):
            paint_marker(out, x - 1, y - 1, MARK)
        for (x, y) in cells_of(frame, ARMED, 4):
            paint_cell(out, x, y, BLOCK)


def render_hud(out, a):
    w = 3 * (a + 1) // 7
    for x in range(64):
        out[0][x] = HUD if x >= 64 - w else FLOOR
        out[63][x] = HUD if x < w else FLOOR


def transition_function(state, action, frame):
    if _mem["frame"] is not None and frame == _mem["frame"]:
        a = _mem["a"]
    else:
        w = bar_width(frame)
        a = 0 if w == 0 else (7 * w + 2) // 3
    a += 1
    out = [row[:] for row in frame]
    if isinstance(action, dict):
        click(frame, out, action.get("x", -1), action.get("y", -1))
    elif action in DELTA:
        actives = cells_of(frame, ACTIVE, 2)
        if actives:
            x, y = actives[0]
            move_marker(frame, out, (x - 1, y - 1), action)
        else:
            blocks = cells_of(frame, BLOCK, 4)
            move_blocks(frame, out, blocks, BLOCK, action)
    render_hud(out, a)
    _mem["frame"] = [row[:] for row in out]
    _mem["a"] = a
    return out
