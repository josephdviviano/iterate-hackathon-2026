# Mechanics: mirror-blocks/marker game from frame colours; floor 5, any other colour blocks; rows 0/63 are HUD.
# Blocks (10, 4x4): A1/A2 y-/+4, A3 apart (left x-4, right x+4), A4 together; each moves iff dest cells are floor.
# Click colour 9 marker: arm (blocks->1, it->11, old active->9); click armed player (1): disarm; else no-op. A5 no-op.
# Armed: arrows move the active 2x2 marker by 4 iff its bbox grown by 1 is floor. HUD w=3(a+1)//7 zeros row0 R/row63 L.
# Unconfirmed: hidden action count a (continuity-gated; fallback a=0 if no bar else max a with width w).

FLOOR, BLOCK, ARMED, MARK, ACTIVE = 5, 10, 1, 9, 11
_memo = {"frame": None, "a": 0}


def components(frame, colour):
    seen, comps = set(), []
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
                comps.append((min(c[0] for c in cells), min(c[1] for c in cells)))
    return sorted(comps)


def bar_width(a):
    return 3 * (a + 1) // 7


def fallback_actions(frame):
    w = 0
    while w < 64 and frame[0][63 - w] == 0:
        w += 1
    return 0 if w == 0 else (7 * w + 3) // 3


def free(frame, x0, y0, size, own):
    for y in range(y0, y0 + size):
        for x in range(x0, x0 + size):
            if not (1 <= y <= 62 and 0 <= x <= 63):
                return False
            if (x, y) not in own and frame[y][x] != FLOOR:
                return False
    return True


def cells(x0, y0, size):
    return {(x, y) for y in range(y0, y0 + size) for x in range(x0, x0 + size)}


def move_sprite(frame, x0, y0, dx, dy, size, colour, pad=0):
    own = cells(x0, y0, size)
    if not free(frame, x0 + dx - pad, y0 + dy - pad, size + 2 * pad, own):
        return False
    for x, y in own:
        frame[y][x] = FLOOR
    for x, y in cells(x0 + dx, y0 + dy, size):
        frame[y][x] = colour
    return True


def recolour(frame, x0, y0, size, colour):
    for x, y in cells(x0, y0, size):
        frame[y][x] = colour


def step_blocks(frame, action):
    blocks = components(frame, BLOCK)
    if len(blocks) != 2:
        return
    (lx, ly), (rx, ry) = blocks
    moves = {1: ((0, -4), (0, -4)), 2: ((0, 4), (0, 4)), 3: ((-4, 0), (4, 0)), 4: ((4, 0), (-4, 0))}
    (ldx, ldy), (rdx, rdy) = moves[action]
    move_sprite(frame, lx, ly, ldx, ldy, 4, BLOCK)
    move_sprite(frame, rx, ry, rdx, rdy, 4, BLOCK)


def step_marker(frame, action):
    active = components(frame, ACTIVE)
    if not active:
        return
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    mx, my = active[0]
    move_sprite(frame, mx, my, dx, dy, 2, ACTIVE, pad=1)


def click(frame, cx, cy):
    if not (0 <= cx < 64 and 0 <= cy < 64):
        return
    hit = frame[cy][cx]
    if hit == MARK:
        for mx, my in components(frame, ACTIVE):
            recolour(frame, mx, my, 2, MARK)
        for mx, my in components(frame, MARK):
            if mx <= cx < mx + 2 and my <= cy < my + 2:
                recolour(frame, mx, my, 2, ACTIVE)
        for bx, by in components(frame, BLOCK):
            recolour(frame, bx, by, 4, ARMED)
    elif hit == ARMED:
        for mx, my in components(frame, ACTIVE):
            recolour(frame, mx, my, 2, MARK)
        for bx, by in components(frame, ARMED):
            recolour(frame, bx, by, 4, BLOCK)


def draw_hud(frame, a):
    w = bar_width(a)
    for i in range(w):
        frame[0][63 - i] = 0
        frame[63][i] = 0


def transition_function(state, action, frame):
    out = [list(row) for row in frame]
    a = _memo["a"] if _memo["frame"] == frame else fallback_actions(frame)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(out, action["x"], action["y"])
    elif action in (1, 2, 3, 4):
        if components(out, ACTIVE):
            step_marker(out, action)
        else:
            step_blocks(out, action)
    a += 1
    draw_hud(out, a)
    _memo["frame"] = [list(row) for row in out]
    _memo["a"] = a
    return out
