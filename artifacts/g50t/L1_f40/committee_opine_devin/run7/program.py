# Mechanics: player 5x5 (colour 9, centre 5) moves 6 cells iff dest box is all floor 5, or holds the
# plate centre and is all {5,8}. A body (player/ghost) on the plate presses it: the 7x7 key tile at the
# arm end slides 6 toward the rope line, vacated cells -> 5; release slides it back (vacated cells copy
# the column beyond). A5: on spawn no-op; else record (ghost = player trail lag 2 from [start, spawn])
# or clear ghost; HUD ring/bar swap. Counter row 63: a 9 -> 1 on even calls (hidden call count).
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
FLOOR, ROPE, PCOL, GCOL = 5, 8, 9, 2
MEM = {"last": None, "n": 0, "trail": None, "start": None, "plate": None}


def cp(f):
    return [list(r) for r in f]


def box(x, y):
    return [(x + i, y + j) for j in range(5) for i in range(5)]


def covers(pos, pt):
    return pos is not None and pos[0] <= pt[0] < pos[0] + 5 and pos[1] <= pt[1] < pos[1] + 5


def find_plate(g):
    _, arm = rope_axes(g)
    for y in range(1, 63):
        for x in range(1, 63):
            if abs(y - arm) > 3 and all(g[y + j][x + i] == ROPE for j in (-1, 0, 1) for i in (-1, 0, 1)):
                return (x, y)
    return None


def rope_axes(g):
    cols = [sum(1 for y in range(64) if g[y][x] == ROPE) for x in range(64)]
    rows = [sum(1 for x in range(64) if g[y][x] == ROPE) for y in range(64)]
    return cols.index(max(cols)), rows.index(max(rows))


def key_end(g, line, arm):
    xs = [x for y in range(arm - 2, arm + 3) for x in range(64) if g[y][x] == ROPE]
    return min(xs) if line > min(xs) else max(xs)


def slide_key(g, line, arm, press):
    kx = key_end(g, line, arm)
    s = 1 if line > kx else -1
    d = s if press else -s
    lo, hi = (kx - 1, kx + 5) if s > 0 else (kx - 5, kx + 1)
    out = cp(g)
    for y in range(arm - 3, arm + 4):
        seg = [g[y][x] for x in range(lo, hi + 1)]
        beyond = g[y][hi + 1] if s > 0 else g[y][lo - 1]
        for x in range(lo, hi + 1):
            out[y][x] = FLOOR if press else beyond
        for i, v in enumerate(seg):
            out[y][lo + 6 * d + i] = v
    return out


def unpressed_world(frame, bodies, pressed, plate):
    g = cp(frame)
    for p in bodies:
        for x, y in box(*p):
            g[y][x] = FLOOR
    line, arm = rope_axes(g)
    if pressed:
        g = slide_key(g, line, arm, False)
    if find_plate(g) is None and plate is not None:
        cx, cy = plate
        for j in (-1, 0, 1):
            for i in (-1, 0, 1):
                g[cy + j][cx + i] = ROPE
        y = cy + 2
        while y < 63 and g[y][cx] != ROPE:
            g[y][cx] = ROPE
            y += 1
    return g, line, arm


def guess_plate(frame, bodies):
    g = cp(frame)
    for p in bodies:
        for x, y in box(*p):
            g[y][x] = FLOOR
    pl = find_plate(g)
    if pl:
        return pl
    line, _ = rope_axes(g)
    top = min(y for y in range(64) if g[y][line] == ROPE)
    for p in bodies:
        if p[0] <= line < p[0] + 5 and p[1] + 5 >= top - 1:
            return (p[0] + 2, p[1] + 2)
    return (line, top - 3)


def can_enter(world, pos, plate):
    cells = [world[y][x] if 0 <= x < 64 and 0 <= y < 64 else 0 for x, y in box(*pos)]
    if all(c == FLOOR for c in cells):
        return True
    return covers(pos, plate) and all(c in (FLOOR, ROPE) for c in cells)


def find_spawn(world):
    for y in range(2, 59, 6):
        for x in range(2, 59, 6):
            if all(world[b][a] == FLOOR for a, b in box(x, y)):
                return (x, y)
    return None


def draw_body(g, pos, col):
    for x, y in box(*pos):
        g[y][x] = col
    g[pos[1] + 2][pos[0] + 2] = FLOOR


def ring(g, x0, col):
    for j in range(3):
        for i in range(3):
            g[1 + j][x0 + i] = col
    g[2][x0 + 1] = 0


def hud_record(g):
    ring(g, 1, GCOL)
    ring(g, 5, PCOL)
    for i in range(3):
        g[5][1 + i] = 0
        g[5][5 + i] = PCOL


def hud_clear(g):
    ring(g, 1, PCOL)
    for j in range(3):
        for i in range(3):
            g[1 + j][5 + i] = 1
    for i in range(3):
        g[5][1 + i] = PCOL
        g[5][5 + i] = 0


def tick_counter(g, n):
    if n % 2 == 0:
        for x in range(63, -1, -1):
            if g[63][x] == PCOL:
                g[63][x] = 1
                break


def transition_function(state, action, frame):
    player = ghost = None
    for o in state:
        if o.get("type") == "player":
            player = (o["x"], o["y"])
        elif o.get("type") == "ghost" and o.get("visible", True):
            ghost = (o["x"], o["y"])
    cont = MEM["last"] is not None and MEM["last"] == frame
    ones = sum(1 for v in frame[63] if v == 1)
    n = MEM["n"] + 1 if cont else 2 * ones
    if ones == 0 and not cont:
        n = 0
    if n == 0 or MEM["start"] is None:
        MEM["start"] = player
    bodies = [p for p in (player, ghost) if p is not None]
    plate = guess_plate(frame, bodies)
    pressed_before = any(covers(p, plate) for p in bodies)
    U, line, arm = unpressed_world(frame, bodies, pressed_before, plate)
    spawn = find_spawn(U) or player
    ghost_mode = frame[1][1] == GCOL
    trail = MEM["trail"] if cont and MEM["trail"] is not None else None
    if ghost_mode and trail is None:
        trail = [ghost or spawn, ghost or spawn, player]
    world_before = slide_key(U, line, arm, True) if pressed_before else U
    out_frame = cp(frame)
    new_p, new_g = player, ghost
    if isinstance(action, int) and action in DIRS:
        dx, dy = DIRS[action]
        cand = (player[0] + 6 * dx, player[1] + 6 * dy)
        if can_enter(world_before, cand, plate):
            new_p = cand
            if ghost_mode:
                trail = trail + [cand]
                new_g = trail[-3] if len(trail) >= 3 else None
    elif action == 5 and player != spawn:
        new_p = spawn
        if ghost_mode:
            ghost_mode, new_g, trail = False, None, None
            hud_clear(out_frame)
        else:
            ghost_mode, new_g = True, None
            trail = [MEM["start"] or spawn, spawn]
            hud_record(out_frame)
    after_bodies = [p for p in (new_p, new_g) if p is not None]
    pressed_after = any(covers(p, plate) for p in after_bodies)
    W = slide_key(U, line, arm, True) if pressed_after else U
    for y in range(7, 63):
        out_frame[y] = list(W[y])
    if new_g is not None:
        draw_body(out_frame, new_g, GCOL)
    draw_body(out_frame, new_p, PCOL)
    tick_counter(out_frame, n)
    MEM.update(last=cp(out_frame), n=n, trail=trail, plate=plate)
    return out_frame
