# Mechanics: frame-level mirror-blocks game. Floor = colour 5; walls (15/8) and anything non-5 block.
# Unarmed: A1/A2 move both blocks (10) y-/+4, A3 apart / A4 together (left x-4/+4, right mirrored),
# each block only into an all-floor 4x4 cell on its own half. Click inactive marker (9) arms: blocks
# -> players (1), marker -> active (11); armed A1-4 move the active marker 4 into an all-floor cell
# (its 4x4 cell = marker xy-1); click player disarms; other clicks and A5 are no-ops.
# HUD: row 0 right-aligned and row 63 left-aligned 0 bars of width floor(3(a+1)/7), a = actions since
# level start (hidden, continuity-gated; fallback = smallest a for the shown width). Unconfirmed:
# what a full bar does, and A5 having any effect beyond the timer tick.
import copy

FLOOR, BLOCK, PLAYER, INACT, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
N = 64
_last = {"frame": None, "a": 0}


def bar_width(frame):
    w = 0
    while w < N and frame[0][N - 1 - w] == BAR:
        w += 1
    return w


def width_for(a):
    return (3 * (a + 1)) // 7


def boxes(frame, colour, size):
    seen, out = set(), []
    for y in range(1, N - 1):
        for x in range(N):
            if frame[y][x] == colour and (x, y) not in seen:
                for dy in range(size):
                    for dx in range(size):
                        seen.add((x + dx, y + dy))
                out.append((x, y))
    return out


def fill(frame, x, y, size, colour):
    for dy in range(size):
        for dx in range(size):
            frame[y + dy][x + dx] = colour


def cell_free(frame, x, y):
    if x < 0 or y < 1 or x + 4 > N or y + 4 > N - 1:
        return False
    return all(frame[y + dy][x + dx] == FLOOR for dy in range(4) for dx in range(4))


def step_blocks(frame, action, colour):
    blocks = sorted(boxes(frame, colour, 4))
    if len(blocks) != 2:
        return
    (lx, ly), (rx, ry) = blocks
    if action in (1, 2):
        d = -4 if action == 1 else 4
        moves = [((lx, ly), (lx, ly + d)), ((rx, ry), (rx, ry + d))]
    else:
        d = -4 if action == 3 else 4
        moves = [((lx, ly), (lx + d, ly)), ((rx, ry), (rx - d, ry))]
    mid = N // 2
    for i, ((ox, oy), (nx, ny)) in enumerate(moves):
        own_half = nx + 4 <= mid if i == 0 else nx >= mid
        if own_half and cell_free(frame, nx, ny):
            fill(frame, ox, oy, 4, FLOOR)
            fill(frame, nx, ny, 4, colour)


def step_marker(frame, action):
    act = boxes(frame, ACTIVE, 2)
    if not act:
        return
    mx, my = act[0]
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    nx, ny = mx + dx, my + dy
    fill(frame, mx, my, 2, FLOOR)
    if cell_free(frame, nx - 1, ny - 1):
        fill(frame, nx, ny, 2, ACTIVE)
    else:
        fill(frame, mx, my, 2, ACTIVE)


def recolour(frame, src, dst):
    for row in frame:
        for x in range(N):
            if row[x] == src:
                row[x] = dst


def click(frame, x, y):
    if not (0 <= x < N and 0 <= y < N):
        return
    c = frame[y][x]
    armed = any(PLAYER in row for row in frame[1:N - 1])
    if c == INACT:
        recolour(frame, ACTIVE, INACT)
        for mx, my in boxes(frame, INACT, 2):
            if mx <= x < mx + 2 and my <= y < my + 2:
                fill(frame, mx, my, 2, ACTIVE)
        if not armed:
            recolour(frame, BLOCK, PLAYER)
    elif c == PLAYER:
        recolour(frame, PLAYER, BLOCK)
        recolour(frame, ACTIVE, INACT)


def draw_bar(frame, w):
    for i in range(min(w, N)):
        frame[0][N - 1 - i] = BAR
        frame[N - 1][i] = BAR


def transition_function(state, action, frame):
    if _last["frame"] is not None and frame == _last["frame"]:
        a = _last["a"]
    else:
        w = bar_width(frame)
        a = 0 if w == 0 else (7 * w - 1) // 3
    out = copy.deepcopy(frame)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(out, action.get("x", -1), action.get("y", -1))
    elif action in (1, 2, 3, 4):
        if any(PLAYER in row for row in out[1:N - 1]):
            step_marker(out, action)
        else:
            step_blocks(out, action, BLOCK)
    a += 1
    draw_bar(out, width_for(a))
    _last["frame"] = copy.deepcopy(out)
    _last["a"] = a
    return out
