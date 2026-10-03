# Mechanics: two colour-10 4x4 blocks; A1/A2 move both up/down 4, A3/A4 move them apart/together (mirrored in x);
# each block moves alone iff its destination 4x4 is all floor (colour 5) apart from itself. Click inactive marker (9)
# -> arm: blocks become colour-1 players, marker active (11); armed A1-4 steer the active marker by 4 (dest cell =
# marker grown by 1, must be floor); click player = disarm, click active marker/block = no-op. HUD bars row0 (from
# right) / row63 (from left) width 3(a+1)//7, a = actions this level (hidden; continuity-gated, fallback (7w+3)//3).
FLOOR, BLOCK, PLAYER, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
STEP = 4
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
_memo = {"frame": None, "a": 0}


def squares(f, colour, size):
    seen, out = set(), []
    for y in range(64):
        for x in range(64):
            if f[y][x] == colour and (x, y) not in seen:
                out.append((x, y))
                for j in range(size):
                    for i in range(size):
                        seen.add((x + i, y + j))
    return out


def cells(x, y, size):
    return [(x + i, y + j) for j in range(size) for i in range(size)]


def rect_free(f, x, y, size, own):
    for cx, cy in cells(x, y, size):
        if not (1 <= cy <= 62 and 0 <= cx <= 63):
            return False
        if (cx, cy) not in own and f[cy][cx] != FLOOR:
            return False
    return True


def paint(f, x, y, size, colour):
    for cx, cy in cells(x, y, size):
        f[cy][cx] = colour


def hit(x, y, px, py, size):
    return px <= x < px + size and py <= y < py + size


def move_blocks(f, blocks, action):
    dx, dy = MOVES[action]
    left, right = sorted(blocks)[:2] if len(blocks) >= 2 else (blocks + [None])[:2]
    plan = [(left, dx, dy), (right, -dx, dy)] if right else [(left, dx, dy)]
    for b, mx, my in plan:
        nx, ny = b[0] + mx, b[1] + my
        own = set(cells(b[0], b[1], 4))
        if rect_free(f, nx, ny, 4, own):
            paint(f, b[0], b[1], 4, FLOOR)
            paint(f, nx, ny, 4, BLOCK)


def move_marker(f, m, action):
    dx, dy = MOVES[action]
    off = (STEP - 2) // 2
    nx, ny = m[0] + dx, m[1] + dy
    if rect_free(f, nx - off, ny - off, STEP, set(cells(m[0], m[1], 2))):
        paint(f, m[0], m[1], 2, FLOOR)
        paint(f, nx, ny, 2, ACTIVE)


def recolour(f, colour_from, colour_to):
    for y in range(64):
        for x in range(64):
            if f[y][x] == colour_from:
                f[y][x] = colour_to


def click(f, x, y):
    blocks, players = squares(f, BLOCK, 4), squares(f, PLAYER, 4)
    marks, active = squares(f, MARK, 2), squares(f, ACTIVE, 2)
    target = next((m for m in marks if hit(x, y, m[0], m[1], 2)), None)
    if target is not None:
        if blocks:
            recolour(f, BLOCK, PLAYER)
        for a in active:
            paint(f, a[0], a[1], 2, MARK)
        paint(f, target[0], target[1], 2, ACTIVE)
    elif players and any(hit(x, y, p[0], p[1], 4) for p in players):
        recolour(f, PLAYER, BLOCK)
        recolour(f, ACTIVE, MARK)


def bar_width(f):
    w = 0
    while w < 64 and f[0][63 - w] == BAR:
        w += 1
    return w


def transition_function(state, action, frame):
    f = [list(r) for r in frame]
    if _memo["frame"] is not None and f == _memo["frame"]:
        a = _memo["a"]
    else:
        w = bar_width(f)
        a = 0 if w == 0 else (7 * w + 3) // 3
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(f, action["x"], action["y"])
    elif action in MOVES:
        blocks, active = squares(f, BLOCK, 4), squares(f, ACTIVE, 2)
        if blocks:
            move_blocks(f, blocks, action)
        elif active:
            move_marker(f, active[0], action)
    a += 1
    w = min(64, 3 * (a + 1) // 7)
    for i in range(w):
        f[0][63 - i] = BAR
        f[63][i] = BAR
    _memo["frame"], _memo["a"] = [list(r) for r in f], a
    return f
