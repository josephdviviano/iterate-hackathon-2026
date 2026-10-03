# Mechanics: player (5x5) steps 6 cells on A1-4 iff the dest box is all floor colour 5 (wall-8 cells allowed when the box holds the plate).
# Plate = 3x3 of 8; a player/ghost box on it presses it: the 7x7 door block at the arm's end slides 6 toward the line (vacated -> 5);
# leaving restores plate + line stub, door slides back (vacated cells copy the corridor column beyond). A5 off spawn toggles ghost mode:
# record (life path -> ghost, red HUD ring, 9-HUD +4) or clear; ghost shows path[k-1] after k moves. Counter row 63 loses a 9 on even calls.
# Hypotheses: spawn = top-left free lattice cell; A5 on spawn = no-op; ghost clamps at path end; fallback n = 2*(64-c)-1 (unconfirmed).
FLOOR, WALL, BG = 5, 8, 0
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP, SZ = 6, 5
_mem = {}


def sprite(state, typ):
    for o in state:
        if o.get("type") == typ and o.get("visible", True):
            return o
    return None


def box_has(pos, pt):
    return pos is not None and pt is not None and pos[0] <= pt[0] < pos[0] + SZ and pos[1] <= pt[1] < pos[1] + SZ


def find_plate(fr, boxes):
    for b in boxes:
        if b is None:
            continue
        cx, cy = b[0] + 2, b[1] + 2
        if fr[cy][cx] == FLOOR:
            for dx, dy in MOVES.values():
                x, y = cx + dx * 3, cy + dy * 3
                if 0 <= x < 64 and 0 <= y < 64 and fr[y][x] == WALL:
                    return (cx, cy)
    for y in range(1, 63):
        for x in range(1, 63):
            if all(fr[y + j][x + i] == WALL for i in (-1, 0, 1) for j in (-1, 0, 1)):
                return (x, y)
    return None


def erase(fr, pos):
    for y in range(pos[1], pos[1] + SZ):
        for x in range(pos[0], pos[0] + SZ):
            fr[y][x] = FLOOR


def restore_plate(fr, plate, pos):
    cx, cy = plate
    for dx, dy in MOVES.values():
        x, y = cx + dx * 3, cy + dy * 3
        if 0 <= x < 64 and 0 <= y < 64 and fr[y][x] == WALL:
            for t in range(4):
                if box_has(pos, (cx + dx * t, cy + dy * t)):
                    fr[cy + dy * t][cx + dx * t] = WALL
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            fr[cy + j][cx + i] = WALL


def door_geometry(fr, plate):
    lx, y = plate[0], plate[1] + 2
    while y + 1 < 64 and fr[y + 1][lx] == WALL:
        y += 1
    ay = y
    s = 1 if fr[ay][lx - 1] == WALL else -1      # s = direction key -> line
    x = lx
    while 0 < x and x < 63 and fr[ay][x - s] == WALL:
        x -= s
    return lx, ay, x + 2 * s, s


def shift_door(fr, plate, toward):
    lx, ay, kx, s = door_geometry(fr, plate)
    d = STEP * s * (1 if toward else -1)
    cells = {(x, y): fr[y][x] for y in range(ay - 3, ay + 4) for x in range(kx - 3, kx + 4)}
    beyond = kx + 4 * s
    new = {(x + d, y): v for (x, y), v in cells.items()}
    for (x, y) in cells:
        if (x, y) not in new:
            fr[y][x] = FLOOR if toward else fr[y][beyond if 0 <= beyond < 64 else x]
    for (x, y), v in new.items():
        if 0 <= x < 64 and (x - lx) * s < 0:
            fr[y][x] = v


def draw(fr, pos, pix):
    for j, row in enumerate(pix):
        for i, v in enumerate(row):
            if v >= 0:
                fr[pos[1] + j][pos[0] + i] = v


def free(fr, pos, plate):
    x0, y0 = pos
    if x0 < 0 or y0 < 0 or x0 + SZ > 64 or y0 + SZ > 63:
        return False
    on_plate = box_has(pos, plate)
    for y in range(y0, y0 + SZ):
        for x in range(x0, x0 + SZ):
            c = fr[y][x]
            if c != FLOOR and not (on_plate and c == WALL):
                return False
    return True


def spawn_cell(fr, pos, plate):
    for y in range(pos[1] % STEP, 60, STEP):
        for x in range(pos[0] % STEP, 60, STEP):
            if free(fr, (x, y), None) or (x, y) == tuple(pos):
                return (x, y)
    return tuple(pos)


def hud_rows(fr):
    top = next((y for y in range(64) if FLOOR in fr[y]), 7)
    return range(0, top)


def hud_toggle(fr, record):
    rows = hud_rows(fr)
    nine = [(x, y) for y in rows for x in range(64) if fr[y][x] == 9]
    if not nine:
        return
    mx, my = min(x for x, _ in nine), min(y for _, y in nine)
    if record:
        ring = [(x, y) for x, y in nine if x < mx + 3 and y < my + 3]
        for x, y in nine:
            fr[y][x] = BG
        for j in range(3):
            for i in range(3):
                fr[my + j][mx + 4 + i] = BG
        for x, y in nine:
            fr[y][x + 4] = 9
        for x, y in ring:
            fr[y][x] = 2
    else:
        for y in rows:
            for x in range(64):
                if fr[y][x] == 2:
                    fr[y][x] = BG
        for x, y in nine:
            fr[y][x] = BG
        for x, y in nine:
            fr[y][x - 4] = 9
        for j in range(3):
            for i in range(3):
                fr[my + j][mx + i] = 1


def transition_function(state, action, frame):
    fr = [list(r) for r in frame]
    p = sprite(state, "player")
    g = sprite(state, "ghost")
    if p is None:
        return fr
    pos = (p["x"], p["y"])
    gpos = (g["x"], g["y"]) if g else None
    ppix = p["pixels"]
    gpix = g["pixels"] if g else [[2 if v >= 0 else -1 for v in r] for r in ppix]
    c = sum(1 for v in frame[63] if v == 9)
    if _mem.get("out") == frame:
        n, life, path, k = _mem["n"], _mem["life"], _mem["path"], _mem["k"]
        plate = _mem["plate"]
    else:
        n = 0 if c == 64 else 2 * (64 - c) - 1
        life, path, k = [pos], ([gpos] if gpos else []), (1 if gpos else 0)
        plate = None
    if plate is None:
        plate = find_plate(fr, [pos, gpos])
    mode = any(fr[y][x] == 2 for y in hud_rows(fr) for x in range(64))
    pressed0 = box_has(pos, plate) or box_has(gpos, plate)
    for b in (pos, gpos):
        if b:
            erase(fr, b)
    if plate and pressed0:
        restore_plate(fr, plate, pos if box_has(pos, plate) else gpos)     # unpressed background for collision reads
    npos, ngpos = pos, gpos
    if action in MOVES:
        dx, dy = MOVES[action]
        cand = (pos[0] + dx * STEP, pos[1] + dy * STEP)
        if free(fr, cand, plate):
            npos = cand
            life = life + [npos]
            if mode and path:
                k += 1
                ngpos = path[min(k - 1, len(path) - 1)]
    elif action == 5:
        sp = spawn_cell(fr, pos, plate)
        if pos != sp:
            hud_toggle(fr, not mode)
            path = [] if mode else life
            life, k, npos, ngpos = [sp], 0, sp, None
    pressed1 = box_has(npos, plate) or box_has(ngpos, plate)
    if plate and pressed0 != pressed1:
        shift_door(fr, plate, pressed1)
    if plate and pressed1:
        for j in (-1, 0, 1):
            for i in (-1, 0, 1):
                fr[plate[1] + j][plate[0] + i] = FLOOR
    if ngpos and ngpos != npos:
        draw(fr, ngpos, gpix)
    draw(fr, npos, ppix)
    if n % 2 == 0 and c > 0:
        fr[63][c - 1] = 1
    _mem.update(out=[list(r) for r in fr], n=n + 1, life=life, path=path, k=k, plate=plate)
    return fr
