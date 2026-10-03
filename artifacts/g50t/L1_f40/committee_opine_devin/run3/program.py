# Mechanics: player 5x5 ring 9 (centre shows 5) moves 6 per A1-4 iff its destination box is all floor 5;
# wall-8 cells are allowed when the box holds the plate centre (plate = topmost 3x3 square of 8).
# Body on plate: door block at the arm's key end shifts 6 toward the line column (vacated cells copy the
# outer column); leaving shifts it back. A5 off spawn: record life path as 2-ghost at path[k-1] + HUD ring, or clear.
# Counter row 63: one cell -> colour 1 on even calls (hidden parity). Unconfirmed: ghost past path end, gate.
FLOOR, WALL, PLAYER, GHOST, SLOT = 5, 8, 9, 2, 1
STEP, SIZE = 6, 5
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {}


def box_cells(x, y):
    return [(x + i, y + j) for j in range(SIZE) for i in range(SIZE)]


def in_box(px, py, bx, by):
    return bx <= px < bx + SIZE and by <= py < by + SIZE


def find_body(frame, colour):
    for y in range(7, 63):
        for x in range(64 - SIZE + 1):
            if all(frame[y + j][x + i] == colour for j in range(SIZE) for i in range(SIZE)
                   if not (i == 2 and j == 2)):
                return (x, y)
    return None


def find_plate(bg):
    for y in range(62):
        for x in range(62):
            if all(bg[y + j][x + i] == WALL for j in range(3) for i in range(3)):
                return (x + 1, y + 1)
    return None


def door_geometry(bg):
    cols = [sum(1 for y in range(64) if bg[y][x] == WALL) for x in range(64)]
    rows = [sum(1 for x in range(64) if bg[y][x] == WALL) for y in range(64)]
    line, arm = cols.index(max(cols)), rows.index(max(rows))
    xs = [x for x in range(64) if bg[arm][x] == WALL]
    if line >= (xs[0] + xs[-1]) / 2:
        return arm, xs[0], 1
    return arm, xs[-1] - STEP + 1, -1


def shift_door(bg, sign):
    """sign=+1 press (toward line), -1 release."""
    arm, kx, d = door_geometry(bg)
    s = d * sign
    src = list(range(kx, kx + STEP))
    rows = range(max(0, arm - 3), min(63, arm + 4))
    for y in rows:
        block = [bg[y][x] for x in src]
        dst = [x + s * STEP for x in src]
        fill_col = min(src) - 1 if s > 0 else max(src) + 1
        fill = bg[y][fill_col] if 0 <= fill_col < 64 else 0
        for x in src:
            bg[y][x] = fill
        for x, v in zip(dst, block):
            if 0 <= x < 64:
                bg[y][x] = v


def pressed(plate, bodies):
    return plate is not None and any(b and in_box(plate[0], plate[1], b[0], b[1]) for b in bodies)


def can_enter(bg, plate, x, y):
    if x < 0 or y < 0 or x + SIZE > 64 or y + SIZE > 63:
        return False
    on_plate = plate is not None and in_box(plate[0], plate[1], x, y)
    for cx, cy in box_cells(x, y):
        v = bg[cy][cx]
        if v != FLOOR and not (on_plate and v == WALL):
            return False
    return True


def spawn_cell(bg, plate, ref):
    ox, oy = ref[0] % STEP, ref[1] % STEP
    for y in range(oy, 64, STEP):
        for x in range(ox, 64, STEP):
            if can_enter(bg, plate, x, y):
                return (x, y)
    return ref


def draw_body(out, pos, colour):
    if pos is None:
        return
    for cx, cy in box_cells(*pos):
        out[cy][cx] = colour
    out[pos[1] + 2][pos[0] + 2] = FLOOR


def draw_ring(out, x, y, colour):
    for j in range(3):
        for i in range(3):
            out[y + j][x + i] = 0 if (i == 1 and j == 1) else colour


def draw_hud(out, ghost_on):
    for y in range(1, 6):
        for x in range(1, 12):
            out[y][x] = 0
    g = 1 if ghost_on else 0
    for i in range(g):
        draw_ring(out, 1 + 4 * i, 1, GHOST)
    draw_ring(out, 1 + 4 * g, 1, PLAYER)
    for i in range(3):
        out[5][1 + 4 * g + i] = PLAYER
    for s in range(g + 1, 2):
        for j in range(3):
            for i in range(3):
                out[1 + j][1 + 4 * s + i] = SLOT


def draw_counter(out, n):
    zeros = n // 2 + 1
    for x in range(64):
        out[63][x] = SLOT if x >= 64 - zeros else PLAYER


def obj_pos(state, typ):
    for o in state:
        if o.get("type") == typ and o.get("visible", True):
            return (o["x"], o["y"])
    return None


def transition_function(state, action, frame):
    player = obj_pos(state, "player") or find_body(frame, PLAYER)
    ghost = obj_pos(state, "ghost")
    covered = set(box_cells(*player)) | (set(box_cells(*ghost)) if ghost else set())
    cont = _mem.get("last") == frame
    old_bg = _mem.get("bg")
    if old_bg is not None and not all(old_bg[y][x] == frame[y][x] for y in range(7, 63)
                                      for x in range(64) if (x, y) not in covered):
        old_bg = None
    bg = [[(old_bg[y][x] if old_bg else FLOOR) if (x, y) in covered else frame[y][x]
           for x in range(64)] for y in range(64)]
    ghost_on = frame[1][1] == GHOST
    if cont:
        n, life, rec, k = _mem["n"], list(_mem["life"]), list(_mem["rec"]), _mem["k"]
    else:
        z = sum(1 for v in frame[63] if v == SLOT)
        n, life = 2 * z, [player]
        rec, k = ([ghost], 1) if ghost else ([], 0)
    plate = find_plate(bg)
    was_pressed = pressed(plate, [player, ghost])
    spawn = spawn_cell(bg, plate, player)

    new_player = player
    if action in MOVES:
        dx, dy = MOVES[action]
        nx, ny = player[0] + dx * STEP, player[1] + dy * STEP
        if can_enter(bg, plate, nx, ny):
            new_player = (nx, ny)
            life.append(new_player)
            if ghost_on:
                k += 1
    elif action == 5 and player != spawn:
        if ghost_on:
            ghost_on, rec, k = False, [], 0
        else:
            ghost_on, rec, k = True, life, 0
        new_player, life = spawn, [spawn]

    new_ghost = rec[min(k - 1, len(rec) - 1)] if ghost_on and k >= 1 and rec else None
    now_pressed = pressed(plate, [new_player, new_ghost])
    if now_pressed != was_pressed:
        shift_door(bg, 1 if now_pressed else -1)

    out = [row[:] for row in bg]
    draw_hud(out, ghost_on)
    draw_counter(out, n)
    draw_body(out, new_ghost, GHOST)
    draw_body(out, new_player, PLAYER)
    _mem.update(last=out, bg=bg, n=n + 1, life=life, rec=rec, k=k)
    return [row[:] for row in out]
