# Mechanics: player 5x5 ring (colour 9) steps 6 cells; a move succeeds iff the destination box is all floor (5),
# or it holds the plate centre (then wall 8 is allowed). Plate covered by player/ghost -> plate hidden and the key
# tile (key bbox +1) slides 6 toward the wall line; vacated cells become floor. A5 off spawn: record ghost mode
# (red HUD ring prepended) or clear it, respawn at the first cell moved to; ghost (colour 2) = trail 2 moves behind,
# trail reset to [level start, spawn]. Counter row 63 gains a 1 from the right on every even call. Unconfirmed: ghost lag-2 vs re-run of the recorded path.
FLOOR, WALL, PCOL, GCOL, TICK = 5, 8, 9, 2, 1
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {}


def box(x, y):
    return [(x + i, y + j) for j in range(5) for i in range(5)]


def find_obj(state, typ):
    for o in state:
        if o.get("type") == typ and o.get("visible", True):
            return (o["x"], o["y"])
    return None


def wall_parts(S):
    """Plate = topmost 3x3 wall square, line = fullest wall column, arm = fullest wall row."""
    cells = [(x, y) for y in range(7, 63) for x in range(64) if S[y][x] == WALL]
    if not cells:
        return None
    plate = None
    for y in range(7, 61):
        for x in range(62):
            if all(S[y + j][x + i] == WALL for i in range(3) for j in range(3)):
                plate = (x + 1, y + 1)
                break
        if plate:
            break
    if plate is None:
        return None
    cols, rows = {}, {}
    for x, y in cells:
        cols[x] = cols.get(x, 0) + 1
        rows[y] = rows.get(y, 0) + 1
    line = max(cols, key=lambda c: cols[c])
    arm = max(rows, key=lambda r: rows[r])
    keyxs = [x for x, y in cells if abs(y - arm) <= 2 and x != line]
    if not keyxs:
        return None
    s = 1 if line > min(keyxs) else -1
    e = min(keyxs) if s > 0 else max(keyxs)
    tile = (e - 1, e + 5) if s > 0 else (e - 5, e + 1)
    return {"plate": plate, "arm": arm, "tile": tile, "s": s}


def scenery(S, parts, pressed):
    out = [row[:] for row in S]
    if not (parts and pressed):
        return out
    px, py = parts["plate"]
    for j in range(-1, 2):
        for i in range(-1, 2):
            out[py + j][px + i] = FLOOR
    x0, x1 = parts["tile"]
    d = 6 * parts["s"]
    for y in range(parts["arm"] - 3, parts["arm"] + 4):
        for x in range(x0, x1 + 1):
            if not (x0 <= x - d <= x1):
                out[y][x] = FLOOR
        for x in range(x0, x1 + 1):
            out[y][x + d] = S[y][x]
    return out


def covers(pos, pt):
    return pos is not None and pt is not None and pos[0] <= pt[0] < pos[0] + 5 and pos[1] <= pt[1] < pos[1] + 5


def is_pressed(parts, P, G):
    return bool(parts) and (covers(P, parts["plate"]) or covers(G, parts["plate"]))


def can_enter(scen, parts, pos):
    cells = box(*pos)
    if any(not (0 <= x < 64 and 0 <= y < 63) for x, y in cells):
        return False
    vals = [scen[y][x] for x, y in cells]
    if all(v == FLOOR for v in vals):
        return True
    return bool(parts) and covers(pos, parts["plate"]) and all(v in (FLOOR, WALL) for v in vals)


def draw_ring(out, x, y, size, col):
    for j in range(size):
        for i in range(size):
            if (i, j) != (size // 2, size // 2):
                out[y + j][x + i] = col


def render_hud(out, mode):
    for y in range(1, 6):
        for x in range(1, 12):
            out[y][x] = 0
    if mode:
        draw_ring(out, 1, 1, 3, GCOL)
        draw_ring(out, 5, 1, 3, PCOL)
        bar = 5
    else:
        draw_ring(out, 1, 1, 3, PCOL)
        for j in range(3):
            for i in range(3):
                out[1 + j][5 + i] = TICK
        bar = 1
    for i in range(3):
        out[5][bar + i] = PCOL


def stateless(state, frame):
    P = find_obj(state, "player")
    G = find_obj(state, "ghost")
    S = [row[:] for row in frame]
    for pos in (P, G):
        if pos:
            for x, y in box(*pos):
                S[y][x] = FLOOR
    ones = frame[63].count(TICK)
    n = 0 if ones == 0 else 2 * ones - 1
    return {"S": S, "P": P, "G": G, "mode": frame[1][1] == GCOL, "n": n,
            "start": P if ones == 0 else None, "spawn": None, "trail": [P]}


def transition_function(state, action, frame):
    st = _mem.get("st") if _mem.get("last") == frame else None
    if st is None:
        st = stateless(state, frame)
    st = dict(st, trail=list(st["trail"]))
    S, parts = st["S"], wall_parts(st["S"])
    P, G = st["P"], st["G"]
    a = action if isinstance(action, int) else 6
    if a in DIRS and P is not None:
        dx, dy = DIRS[a]
        dest = (P[0] + 6 * dx, P[1] + 6 * dy)
        if can_enter(scenery(S, parts, is_pressed(parts, P, G)), parts, dest):
            P = dest
            if st["spawn"] is None:
                st["spawn"] = P
            st["trail"].append(P)
            if st["mode"] and len(st["trail"]) >= 3:
                G = st["trail"][-3]
    elif a == 5 and P is not None and st["spawn"] is not None and P != st["spawn"]:
        st["mode"] = not st["mode"]
        G = None
        P = st["spawn"]
        st["trail"] = [st["start"] if st["start"] else P, P]
    st["P"], st["G"] = P, G
    out = scenery(S, parts, is_pressed(parts, P, G))
    out[63] = frame[63][:]
    for y in range(1, 6):
        out[y] = frame[y][:12] + out[y][12:]
    ones = frame[63].count(TICK)
    if st["n"] % 2 == 0 and ones < 64:
        out[63][63 - ones] = TICK
    st["n"] += 1
    if (frame[1][1] == GCOL) != st["mode"]:
        render_hud(out, st["mode"])
    if G is not None and G != P:
        draw_ring(out, G[0], G[1], 5, GCOL)
    if P is not None:
        draw_ring(out, P[0], P[1], 5, PCOL)
    _mem["st"], _mem["last"] = st, [row[:] for row in out]
    return out

