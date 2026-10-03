# Mechanics: maze visible (floor 5, walls 15/8); two 4x4 cyan (10) blocks, 2x2 markers (9 inactive, 11 active).
# Block mode: A1/A2 move both blocks y-/+4; A3 moves left block x-4 and right x+4, A4 the reverse. A move happens
# per block iff its 4x4 destination (outside its own cells) is all floor 5 within rows 1..62. Click a 9 marker:
# blocks turn 1 (armed players), that marker -> 11 (old 11 -> 9); click a 1 player: disarm; else no-op. Armed: arrows
# move active marker by 4 (same floor rule). HUD bars rows 0/63 w=round(64a/150), a=actions this level; a resumes if frame = a recent output, else largest a for w.
FLOOR, BLOCK, PLAYER, MARK, ACTIVE = 5, 10, 1, 9, 11
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_mem = {"frame": None, "a": 0, "seen": []}


def squares(frame, colours, size):
    seen, out = set(), []
    for y in range(1, 63):
        for x in range(64):
            c = frame[y][x]
            if c in colours and (x, y) not in seen:
                cells = [(x + i, y + j) for j in range(size) for i in range(size)]
                if all(0 <= cx < 64 and 1 <= cy < 63 and frame[cy][cx] == c for cx, cy in cells):
                    seen.update(cells)
                    out.append((x, y, c))
    return out


def can_place(frame, x, y, own):
    for j in range(4):
        for i in range(4):
            cx, cy = x + i, y + j
            if (cx, cy) in own:
                continue
            if not (0 <= cx < 64 and 1 <= cy < 63) or frame[cy][cx] != FLOOR:
                return False
    return True


def cells(x, y, n):
    return {(x + i, y + j) for j in range(n) for i in range(n)}


def paint(out, x, y, n, c):
    for cx, cy in cells(x, y, n):
        out[cy][cx] = c


def bar_width(a):
    return (128 * a + 150) // 300


def counter_before(frame):
    if _mem["frame"] is not None and frame == _mem["frame"]:
        return _mem["a"]
    for f, a in _mem["seen"]:
        if f == frame:
            return a
    w = sum(1 for v in frame[0] if v == 0)
    if w == 0:
        return 0
    return max(a for a in range(0, 1000) if bar_width(a) == w)


def move_blocks(frame, out, blocks, act):
    dx, dy = DIRS[act]
    blocks = sorted(blocks)
    for idx, (x, y, c) in enumerate(blocks):
        bdx = dx if idx == 0 else -dx
        if len(blocks) == 1:
            bdx = dx
        nx, ny = x + bdx, y + dy
        if can_place(frame, nx, ny, cells(x, y, 4)):
            paint(out, x, y, 4, FLOOR)
    for idx, (x, y, c) in enumerate(blocks):
        bdx = dx if idx == 0 else -dx
        if len(blocks) == 1:
            bdx = dx
        nx, ny = x + bdx, y + dy
        if can_place(frame, nx, ny, cells(x, y, 4)):
            paint(out, nx, ny, 4, c)


def move_marker(frame, out, marker, act):
    dx, dy = DIRS[act]
    x, y = marker[0] - 1, marker[1] - 1
    if can_place(frame, x + dx, y + dy, cells(x, y, 4)):
        paint(out, marker[0], marker[1], 2, FLOOR)
        paint(out, marker[0] + dx, marker[1] + dy, 2, ACTIVE)


def click(frame, out, blocks, markers, cx, cy):
    if not (0 <= cx < 64 and 0 <= cy < 64):
        return
    c = frame[cy][cx]
    hit = [m for m in markers if m[0] <= cx < m[0] + 2 and m[1] <= cy < m[1] + 2]
    if c == MARK and hit:
        for m in markers:
            if m[2] == ACTIVE:
                paint(out, m[0], m[1], 2, MARK)
        paint(out, hit[0][0], hit[0][1], 2, ACTIVE)
        for b in blocks:
            paint(out, b[0], b[1], 4, PLAYER)
    elif c == PLAYER:
        for m in markers:
            if m[2] == ACTIVE:
                paint(out, m[0], m[1], 2, MARK)
        for b in blocks:
            paint(out, b[0], b[1], 4, BLOCK)


def transition_function(state, action, frame):
    a = counter_before(frame) + 1
    out = [list(row) for row in frame]
    blocks = squares(frame, (BLOCK, PLAYER), 4)
    markers = squares(frame, (MARK, ACTIVE), 2)
    armed = any(b[2] == PLAYER for b in blocks)
    if isinstance(action, dict):
        click(frame, out, blocks, markers, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        active = [m for m in markers if m[2] == ACTIVE]
        if armed and active:
            move_marker(frame, out, active[0], action)
        elif not armed:
            move_blocks(frame, out, blocks, action)
    w = bar_width(a)
    for x in range(64):
        out[0][x] = 0 if x >= 64 - w else FLOOR
        out[63][x] = 0 if x < w else FLOOR
    _mem["frame"] = [list(r) for r in out]
    _mem["a"] = a
    _mem["seen"] = (_mem["seen"] + [(_mem["frame"], a)])[-8:]
    return out
