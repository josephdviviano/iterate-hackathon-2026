# Mechanics: player 5x5 ring (colour 9, centre 5) steps 6 cells; a move succeeds iff the dest box is all floor 5, or holds the plate centre.
# Plate = topmost 3x3 of rope colour 8; a body (player/ghost) on it pushes the door block (arm rows +-3, 6 cols from arm start) 6 toward the line.
# A5 off spawn: 1st toggles ghost mode (HUD ring 2 prepended, ghost colour 2 trails the player 2 moves behind, trail reset [start, spawn]),
# 2nd clears it; both respawn the player at spawn = top-left lattice cell with an all-floor box. A5 on spawn is a no-op.
# Counter row 63: after call n, n//2+1 cells from the right are colour 1 (call parity hidden; continuity-gated, fallback guesses n).
FLOOR, ROPE, VOID = 5, 8, 0
DIRS = {1: (0, -6), 2: (0, 6), 3: (-6, 0), 4: (6, 0)}
_mem = {}


def body_pos(state, typ):
    for o in state:
        if o.get("type") == typ and o.get("visible", True):
            return (o["x"], o["y"])
    return None


def box(x, y):
    return [(x + i, y + j) for j in range(5) for i in range(5)]


def find_plate(S):
    for y in range(8, 61):
        for x in range(1, 62):
            if all(S[y + j][x + i] == ROPE for j in range(3) for i in range(3)):
                return (x + 1, y + 1)
    return None


def rope_axes(S):
    cols = [sum(1 for y in range(64) if S[y][x] == ROPE) for x in range(64)]
    rows = [sum(1 for x in range(64) if S[y][x] == ROPE) for y in range(64)]
    line = max(range(64), key=lambda x: cols[x])
    arm = max(range(64), key=lambda y: rows[y])
    return line, arm


def door_shift(S, press):
    """Shift the door block 6 toward the line (press) or back (release)."""
    line, arm = rope_axes(S)
    xs = [x for x in range(64) if S[arm][x] == ROPE]
    if not xs:
        return S
    a = min(xs)
    side = 1 if a < line else -1
    S = [r[:] for r in S]
    old = [r[:] for r in S]
    for y in range(max(0, arm - 3), min(64, arm + 4)):
        for k in range(6):
            if press:
                src = a + side * k if side == 1 else a - k
                dst = src + side * 6
                fill = a - side
                S[y][dst] = old[y][src]
                S[y][src] = old[y][fill]
            else:
                src = a + k if side == 1 else a - k
                dst = src - side * 6
                fill = a + side * 6
                S[y][dst] = old[y][src]
                S[y][src] = old[y][fill]
    return S


def derive_unpressed(frame, bodies):
    S = [r[:] for r in frame]
    pressed = False
    for b in bodies:
        if b is None:
            continue
        x, y = b
        under = y + 5 < 64 and frame[y + 5][x + 2] == ROPE
        for cx, cy in box(x, y):
            S[cy][cx] = FLOOR
        if under:
            pressed = True
            for j in range(1, 4):
                for i in range(1, 4):
                    S[y + j][x + i] = ROPE
            S[y + 4][x + 2] = ROPE
    if pressed:
        S = door_shift(S, False)
    return S


def find_spawn(U):
    for y in range(8, 59, 6):
        for x in range(2, 59, 6):
            if all(U[cy][cx] == FLOOR for cx, cy in box(x, y)):
                return (x, y)
    return None


def can_enter(U, plate, x, y):
    if x < 0 or y < 0 or x + 5 > 64 or y + 5 > 63:
        return False
    if plate and x <= plate[0] < x + 5 and y <= plate[1] < y + 5:
        return all(U[cy][cx] in (FLOOR, ROPE) for cx, cy in box(x, y))
    return all(U[cy][cx] == FLOOR for cx, cy in box(x, y))


def covers(b, plate):
    return b is not None and plate is not None and b[0] <= plate[0] < b[0] + 5 and b[1] <= plate[1] < b[1] + 5


def draw_ring(F, x, y, c, centre):
    for j in range(3):
        for i in range(3):
            F[y + j][x + i] = c
    F[y + 1][x + 1] = centre


def render_hud(F, mode):
    for y in range(1, 6):
        for x in range(1, 8):
            F[y][x] = 0
    if mode:
        draw_ring(F, 1, 1, 2, 0)
        draw_ring(F, 5, 1, 9, 0)
        for x in range(5, 8):
            F[5][x] = 9
    else:
        draw_ring(F, 1, 1, 9, 0)
        for x in range(1, 4):
            F[5][x] = 9
        for j in range(3):
            for i in range(3):
                F[1 + j][5 + i] = 1


def render_counter(F, n):
    ones = n // 2 + 1
    for x in range(64):
        F[63][x] = 1 if x >= 64 - ones else 9


def render_body(F, b, c):
    x, y = b
    for cx, cy in box(x, y):
        F[cy][cx] = c
    F[y + 2][x + 2] = FLOOR


def fallback_model(state, frame):
    p = body_pos(state, "player")
    g = body_pos(state, "ghost")
    U = derive_unpressed(frame, [p, g])
    z = sum(1 for x in range(64) if frame[63][x] == 1)
    n = 0 if z == 0 else 2 * z - 1
    mode = frame[1][1] == 2
    start = p if n == 0 else None
    trail = [g, g, p] if g else [p]
    return {"U": U, "n": n, "mode": mode, "trail": trail, "start": start, "p": p}


def transition_function(state, action, frame):
    global _mem
    if _mem.get("frame") == frame and _mem.get("model"):
        m = _mem["model"]
        m["p"] = body_pos(state, "player") or m["p"]
    else:
        m = fallback_model(state, frame)
        if _mem.get("start") and m["start"] is None:
            m["start"] = _mem["start"]
    U = m["U"]
    plate = find_plate(U)
    spawn = find_spawn(U) or m["p"]
    if m["n"] == 0:
        m["start"] = m["p"]
    start = m["start"] or spawn
    p = m["p"]
    aid = action if isinstance(action, int) else action.get("action_id")
    if aid in DIRS:
        dx, dy = DIRS[aid]
        nx, ny = p[0] + dx, p[1] + dy
        if p is not None and can_enter(U, plate, nx, ny):
            p = (nx, ny)
            m["trail"].append(p)
    elif aid == 5 and p != spawn:
        m["mode"] = not m["mode"]
        m["trail"] = [start, spawn]
        p = spawn
    m["p"] = p
    ghost = None
    if m["mode"] and len(m["trail"]) >= 3:
        ghost = m["trail"][-3]
        if ghost == p:
            ghost = None
    F = [r[:] for r in U]
    if covers(p, plate) or covers(ghost, plate):
        F = door_shift(F, True)
    render_hud(F, m["mode"])
    render_counter(F, m["n"])
    if ghost:
        render_body(F, ghost, 2)
    render_body(F, p, 9)
    m["n"] += 1
    _mem = {"frame": [r[:] for r in F], "model": m, "start": m["start"]}
    return F

