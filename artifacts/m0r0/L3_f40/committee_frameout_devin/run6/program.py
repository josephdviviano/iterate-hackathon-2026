# Mechanics: mirror-blocks/marker game, rule over frame colours only. Floor = colour 5; every other colour blocks.
# Blocks (10, armed 1) are 4x4; sorted by x = left/right. A1/A2 move both y-+4; A3 left x-4/right x+4, A4 mirrored.
# A move succeeds iff the destination 4x4 cell (rows 1..62) is all floor apart from the mover's own cells.
# Click colour 9 = arm/switch (marker -> 11, blocks -> 1), click colour 1 = disarm; armed arrows steer the active marker.
# HUD: zeros right-aligned in row 0 and left-aligned in row 63, width 3(a+1)//7, a = actions since level start (hypothesis; continuity-gated).
FLOOR, BLOCK, ARMED, MARK, ACTIVE, BAR = 5, 10, 1, 9, 11, 0
_memo = {"frame": None, "a": 0}


def components(frame, colours):
    seen, comps = set(), []
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] in colours and (x, y) not in seen:
                stack, cells = [(x, y)], []
                seen.add((x, y))
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < 64 and 1 <= ny < 63 and (nx, ny) not in seen and frame[ny][nx] in colours:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                comps.append(cells)
    return comps


def block_origins(frame):
    out = []
    for cells in components(frame, (BLOCK, ARMED)):
        x0, y0 = min(c[0] for c in cells), min(c[1] for c in cells)
        s = set(cells)
        for (x, y) in sorted(cells):
            if (x - x0) % 4 == 0 and (y - y0) % 4 == 0 and (x, y) in s:
                out.append((x, y))
    return sorted(out)


def marker_origins(frame, colour):
    return [(min(c[0] for c in cells), min(c[1] for c in cells)) for cells in components(frame, (colour,))]


def cell_free(frame, x, y, size, own):
    for yy in range(y, y + size):
        for xx in range(x, x + size):
            if not (0 <= xx < 64 and 1 <= yy < 63):
                return False
            if (xx, yy) not in own and frame[yy][xx] != FLOOR:
                return False
    return True


def move_square(frame, x, y, size, dx, dy, colour):
    # square at (x,y) of given size; returns True if moved
    own = {(xx, yy) for yy in range(y, y + size) for xx in range(x, x + size)}
    if not cell_free(frame, x + dx, y + dy, size, own):
        return False
    for (xx, yy) in own:
        frame[yy][xx] = FLOOR
    for yy in range(y + dy, y + dy + size):
        for xx in range(x + dx, x + dx + size):
            frame[yy][xx] = colour
    return True


def move_marker(frame, mx, my, dx, dy):
    # marker 2x2 occupies the centre of its 4x4 cell (cell origin = marker - 1)
    own = {(xx, yy) for yy in range(my - 1, my + 3) for xx in range(mx - 1, mx + 3)}
    if not cell_free(frame, mx - 1 + dx, my - 1 + dy, 4, own):
        return
    for yy in range(my, my + 2):
        for xx in range(mx, mx + 2):
            frame[yy][xx] = FLOOR
    for yy in range(my + dy, my + dy + 2):
        for xx in range(mx + dx, mx + dx + 2):
            frame[yy][xx] = ACTIVE


def recolour(frame, src, dst):
    for y in range(1, 63):
        for x in range(64):
            if frame[y][x] == src:
                frame[y][x] = dst


def arrow(frame, action):
    dy = {1: -4, 2: 4}.get(action, 0)
    sx = {3: -4, 4: 4}.get(action, 0)
    active = marker_origins(frame, ACTIVE)
    if active:
        mx, my = active[0]
        move_marker(frame, mx, my, sx, dy)
        return
    blocks = block_origins(frame)
    if not blocks:
        return
    left, right = blocks[0], blocks[-1]
    colour = frame[left[1]][left[0]]
    move_square(frame, left[0], left[1], 4, sx, dy, colour)
    if right != left:
        move_square(frame, right[0], right[1], 4, -sx, dy, colour)


def click(frame, x, y):
    if not (0 <= x < 64 and 1 <= y < 63):
        return
    hit = frame[y][x]
    if hit == MARK:
        recolour(frame, ACTIVE, MARK)
        for cells in components(frame, (MARK,)):
            if (x, y) in cells:
                for (cx, cy) in cells:
                    frame[cy][cx] = ACTIVE
        recolour(frame, BLOCK, ARMED)
    elif hit == ARMED:
        recolour(frame, ARMED, BLOCK)
        recolour(frame, ACTIVE, MARK)


def bar_width(a):
    return 3 * (a + 1) // 7


def actions_from_bar(frame):
    w = sum(1 for v in frame[0] if v == BAR)
    if w == 0:
        return 0
    a = 0
    while bar_width(a + 1) <= w:
        a += 1
    return a


def draw_hud(frame, a):
    w = min(64, bar_width(a))
    for x in range(64):
        frame[0][x] = BAR if x >= 64 - w else FLOOR
        frame[63][x] = BAR if x < w else FLOOR


def transition_function(state, action, frame):
    if _memo["frame"] is not None and frame == _memo["frame"]:
        a = _memo["a"]
    else:
        a = actions_from_bar(frame)
    out = [list(r) for r in frame]
    if isinstance(action, dict):
        click(out, action.get("x", -1), action.get("y", -1))
    elif action in (1, 2, 3, 4):
        arrow(out, action)
    a += 1
    draw_hud(out, a)
    _memo["frame"] = [list(r) for r in out]
    _memo["a"] = a
    return out
