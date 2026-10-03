# Mechanics: two colour-10 4x4 blocks move together (A1 up, A2 down by 4; A3 apart, A4 together, mirrored).
# Clicking an inactive marker (colour 9) arms: blocks -> colour 1, that marker -> 11, previous active -> 9.
# Armed mode: A1-A4 move the active marker's 4x4 cell (marker bbox grown by 1) by 4; clicking a player disarms.
# Guard: a mover moves only if every destination cell outside its own cells is floor (colour 5). Others no-op.
# HUD: colour-0 bars at row 0 (right-aligned) and row 63 (left-aligned), width 3(n+1)//7, n = actions so far
# (continuity-gated hidden counter; fallback = smallest n for the shown width). Unconfirmed: unseen-layout bounds.
FLOOR, BLOCK, PLAYER, MARKER, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_mem = {"frame": None, "n": 0}


def bar_width(n):
    return 3 * (n + 1) // 7


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


def square(x, y, size):
    return {(x + i, y + j) for i in range(size) for j in range(size)}


def can_move(frame, own, dest):
    for (x, y) in dest - own:
        if not (0 <= x < 64 and 1 <= y < 63) or frame[y][x] != FLOOR:
            return False
    return True


def read_counter(frame):
    w = sum(1 for v in frame[63] if v == BAR)
    if _mem["frame"] is not None and _mem["frame"] == frame and bar_width(_mem["n"]) == w:
        return _mem["n"]
    n = 0
    while bar_width(n) < w:
        n += 1
    return n


def transition_function(state, action, frame):
    n = read_counter(frame)
    out = [list(map(int, row)) for row in frame]
    blocks = components(frame, BLOCK)
    players = components(frame, PLAYER)
    markers = components(frame, MARKER)
    actives = components(frame, ACTIVE)
    armed = bool(players)
    aid = action if isinstance(action, int) else action.get("action_id")

    def paint(cells, colour):
        for (x, y) in cells:
            out[y][x] = colour

    if aid in DIRS:
        dx, dy = DIRS[aid]
        if not armed and len(blocks) == 2:
            (lx, ly), (rx, ry) = blocks
            moves = [((lx, ly), (lx + dx, ly + dy)),
                     ((rx, ry), (rx - dx, ry + dy))]
            movers = []
            for (sx, sy), (tx, ty) in moves:
                if can_move(frame, square(sx, sy, 4), square(tx, ty, 4)):
                    movers.append(((sx, sy), (tx, ty)))
            for (sx, sy), _ in movers:
                paint(square(sx, sy, 4), FLOOR)
            for _, (tx, ty) in movers:
                paint(square(tx, ty, 4), BLOCK)
        elif armed and actives:
            mx, my = actives[0]
            cx, cy = mx - 1, my - 1
            if can_move(frame, square(cx, cy, 4), square(cx + dx, cy + dy, 4)):
                paint(square(mx, my, 2), FLOOR)
                paint(square(mx + dx, my + dy, 2), ACTIVE)
    elif aid == 6:
        hit = frame[action["y"]][action["x"]]
        if hit == MARKER:
            for (bx, by) in blocks + players:
                paint(square(bx, by, 4), PLAYER)
            for (ax, ay) in actives:
                paint(square(ax, ay, 2), MARKER)
            for (mx, my) in markers:
                if square(mx, my, 2) & {(action["x"], action["y"])}:
                    paint(square(mx, my, 2), ACTIVE)
        elif hit == PLAYER:
            for (bx, by) in players:
                paint(square(bx, by, 4), BLOCK)
            for (ax, ay) in actives:
                paint(square(ax, ay, 2), MARKER)

    n += 1
    w = bar_width(n)
    for x in range(64):
        out[0][x] = BAR if x >= 64 - w else FLOOR
        out[63][x] = BAR if x < w else FLOOR
    _mem["frame"], _mem["n"] = [row[:] for row in out], n
    return out
