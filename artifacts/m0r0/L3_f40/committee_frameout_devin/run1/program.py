# Mechanics: mirror-blocks/marker game read and rendered purely from frame colours (5 floor, f/8 walls,
# 10 blocks, 1 armed players, 9 inactive / 11 active 2x2 markers centred in 4x4 cells). A1/A2 move both blocks
# y-/+4, A3 apart / A4 together (mirrored x); in armed mode A1-4 move the active marker's cell by 4. A move
# succeeds only if the destination 4x4 cell is all floor (walls, markers, players, other block block it).
# Click 9 = arm/switch active marker, click 1 = disarm, else no-op. HUD bars w=3(a+1)//7 (a hidden, continuity-gated).
FLOOR, BLOCK, PLAYER, INACTIVE, ACTIVE, HUD = 5, 10, 1, 9, 11, 0
_mem = {"frame": None, "a": 0}


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
                        if 1 <= ny < 63 and 0 <= nx < 64 and (nx, ny) not in seen and frame[ny][nx] == colour:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                comps.append(cells)
    return comps


def blocks_of(frame, colour):
    """Split a colour into 4x4 squares (top-left corners), sorted left to right."""
    cells = set(c for comp in components(frame, colour) for c in comp)
    out = []
    for (x, y) in sorted(cells, key=lambda c: (c[1], c[0])):
        if (x, y) in cells and all((x + i, y + j) in cells for i in range(4) for j in range(4)):
            out.append((x, y))
            for i in range(4):
                for j in range(4):
                    cells.discard((x + i, y + j))
    return sorted(out)


def cell_free(frame, x, y, own):
    for j in range(4):
        for i in range(4):
            px, py = x + i, y + j
            if not (0 <= px < 64 and 1 <= py < 63):
                return False
            if (px, py) not in own and frame[py][px] != FLOOR:
                return False
    return True


def square(x, y, size):
    return set((x + i, y + j) for i in range(size) for j in range(size))


def paint(frame, cells, colour):
    for x, y in cells:
        frame[y][x] = colour


DELTA = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}


def move_blocks(frame, act):
    blocks = blocks_of(frame, BLOCK)
    if len(blocks) != 2:
        return
    dx, dy = DELTA[act]
    deltas = [(dx, dy), (-dx, dy)]
    moved = []
    for k, (bx, by) in enumerate(blocks):
        ex, ey = deltas[k]
        nx, ny = bx + ex, by + ey
        if cell_free(frame, nx, ny, square(bx, by, 4)):
            moved.append(((bx, by), (nx, ny)))
    for (bx, by), _ in moved:
        paint(frame, square(bx, by, 4), FLOOR)
    for _, (nx, ny) in moved:
        paint(frame, square(nx, ny, 4), BLOCK)


def move_marker(frame, act):
    comps = components(frame, ACTIVE)
    if not comps:
        return
    mx, my = min(c[0] for c in comps[0]), min(c[1] for c in comps[0])
    cx, cy = mx - 1, my - 1
    dx, dy = DELTA[act]
    if cell_free(frame, cx + dx, cy + dy, square(cx, cy, 4)):
        paint(frame, square(mx, my, 2), FLOOR)
        paint(frame, square(mx + dx, my + dy, 2), ACTIVE)


def click(frame, x, y):
    if not (0 <= x < 64 and 1 <= y < 63):
        return
    c = frame[y][x]
    if c == INACTIVE:
        target = next(comp for comp in components(frame, INACTIVE) if (x, y) in comp)
        for comp in components(frame, ACTIVE):
            paint(frame, comp, INACTIVE)
        paint(frame, target, ACTIVE)
        for comp in components(frame, BLOCK):
            paint(frame, comp, PLAYER)
    elif c == PLAYER:
        for comp in components(frame, ACTIVE):
            paint(frame, comp, INACTIVE)
        for comp in components(frame, PLAYER):
            paint(frame, comp, BLOCK)


def bar_width(frame):
    return sum(1 for v in frame[63] if v == HUD)


def render_hud(frame, a):
    w = 3 * (a + 1) // 7
    for x in range(64):
        frame[0][x] = HUD if x >= 64 - w else FLOOR
        frame[63][x] = HUD if x < w else FLOOR


def transition_function(state, action, frame):
    if _mem["frame"] is not None and frame == _mem["frame"]:
        a = _mem["a"]
    else:
        w = bar_width(frame)
        a = 0 if w == 0 else (7 * w + 3) // 3
    out = [list(row) for row in frame]
    if isinstance(action, dict):
        click(out, action.get("x", -1), action.get("y", -1))
    elif action in DELTA:
        if blocks_of(out, BLOCK):
            move_blocks(out, action)
        else:
            move_marker(out, action)
    a += 1
    render_hud(out, a)
    _mem["frame"] = [list(row) for row in out]
    _mem["a"] = a
    return out
