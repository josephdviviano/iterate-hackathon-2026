# Mechanics: two colour-10 blocks (left/right half) move in mirrored 4-cell steps (A1 up, A2 down, A3 apart, A4 together);
# a move succeeds iff the destination 4x4 cell is all floor colour 5 (own cells excepted) within rows 1-62.
# Click on a colour-9 marker arms it (blocks -> colour 1, marker -> 11, old active -> 9); click on colour 1 disarms; else no-op.
# While armed, A1-A4 move the active marker's 4x4 cell under the same floor rule. A5 = no-op. HUD: zeros in row 0 (from right)
# and row 63 (from left), width 3(a+1)//7, a = actions this level (hidden, continuity-gated). Unconfirmed: wider HUD phases.
FLOOR, BLOCK, ARMED, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
MOVES = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_memo = {"frame": None, "a": 0}


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
                comps.append((min(c[0] for c in cells), min(c[1] for c in cells)))
    return sorted(comps)


def cell(x, y):
    return {(x + i, y + j) for i in range(4) for j in range(4)}


def dest_free(frame, own, x, y):
    for cx, cy in cell(x, y):
        if not (0 <= cx < 64 and 1 <= cy < 63):
            return False
        if (cx, cy) not in own and frame[cy][cx] != FLOOR:
            return False
    return True


def paint(out, cells, colour):
    for cx, cy in cells:
        out[cy][cx] = colour


def move_cell(frame, out, x, y, dx, dy, draw):
    """Move the 4x4 cell at (x,y) by (dx,dy) if the destination is floor; draw(out, cx, cy) renders it."""
    own = cell(x, y)
    if not dest_free(frame, own, x + dx, y + dy):
        return False
    paint(out, own, FLOOR)
    draw(out, x + dx, y + dy)
    return True


def step_blocks(frame, out, action):
    blocks = components(frame, BLOCK)
    if action not in MOVES or len(blocks) != 2:
        return
    dx, dy = MOVES[action]
    (lx, ly), (rx, ry) = blocks
    draw = lambda o, x, y: paint(o, cell(x, y), BLOCK)
    move_cell(frame, out, lx, ly, dx, dy, draw)
    move_cell(frame, out, rx, ry, -dx, dy, draw)


def step_marker(frame, out, action):
    act = components(frame, ACTIVE)
    if action not in MOVES or not act:
        return
    dx, dy = MOVES[action]
    mx, my = act[0]

    def draw(o, x, y):
        paint(o, cell(x, y), FLOOR)
        paint(o, {(x + 1 + i, y + 1 + j) for i in range(2) for j in range(2)}, ACTIVE)
    move_cell(frame, out, mx - 1, my - 1, dx, dy, draw)


def recolour(frame, out, src, dst):
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == src:
                out[y][x] = dst


def click(frame, out, x, y):
    if not (0 <= x < 64 and 0 <= y < 64):
        return
    hit = frame[y][x]
    if hit == MARK:
        recolour(frame, out, BLOCK, ARMED)
        recolour(frame, out, ACTIVE, MARK)
        stack, seen = [(x, y)], {(x, y)}
        while stack:
            cx, cy = stack.pop()
            out[cy][cx] = ACTIVE
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in seen and frame[ny][nx] == MARK:
                    seen.add((nx, ny))
                    stack.append((nx, ny))
    elif hit == ARMED:
        recolour(frame, out, ARMED, BLOCK)
        recolour(frame, out, ACTIVE, MARK)


def bar_width(a):
    return 3 * (a + 1) // 7


def actions_before(frame):
    if _memo["frame"] == frame:
        return _memo["a"]
    w = sum(1 for v in frame[0] if v == BAR)
    if w == 0:
        return 0
    return max(a for a in range(7 * w + 7) if bar_width(a) == w)


def render_hud(out, a):
    w = bar_width(a)
    for x in range(64):
        out[0][x] = BAR if x >= 64 - w else FLOOR
        out[63][x] = BAR if x < w else FLOOR


def transition_function(state, action, frame):
    out = [list(r) for r in frame]
    a = actions_before(frame)
    if isinstance(action, dict):
        click(frame, out, action.get("x", -1), action.get("y", -1))
    elif any(frame[y][x] == ARMED for y in range(1, 63) for x in range(64)):
        step_marker(frame, out, action)
    else:
        step_blocks(frame, out, action)
    render_hud(out, a + 1)
    _memo["frame"] = [list(r) for r in out]
    _memo["a"] = a + 1
    return out
