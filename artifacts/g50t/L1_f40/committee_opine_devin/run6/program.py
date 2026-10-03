# Mechanics: 5x5 bodies on a 6-cell lattice; a move succeeds iff the dest box is all floor 5
# (rope 8 allowed when the box holds the plate centre). Plate = topmost 3x3 of 8; covered by a body
# -> plate drawn as floor and the 7x7 key tile at the arm end slides 6 toward the rope line.
# A5 (off spawn) toggles record mode: life path is stored, player returns to spawn, ghost (colour 2)
# then shows record[k-1] after k successful moves; HUD swaps. Row-63 counter: one 9->1 per even call.
# Unconfirmed: ghost pressing the plate, ghost past the end of its record, ghost blocking moves.
import copy

FLOOR, ROPE, HUD_RING, GHOST, SLOT = 5, 8, 9, 2, 1
STEP = 6
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

_mem = {}


def body_cells(x, y):
    return [(x + i, y + j) for j in range(5) for i in range(5)]


def find_plate(base):
    for y in range(1, 62):
        for x in range(64 - 2):
            if all(base[y + j][x + i] == ROPE for j in range(3) for i in range(3)):
                return (x + 1, y + 1)
    return None


def find_key_tile(base, plate):
    """Return (tile rect x0,y0,x1,y1 inclusive, dx, dy) of the key tile the plate controls."""
    if plate is None:
        return None
    px, py = plate
    for sy in (1, -1):
        y = py + 2 * sy
        if not (0 <= y < 64 and base[y][px] == ROPE):
            continue
        while 0 <= y + sy < 64 and base[y + sy][px] == ROPE:
            y += sy
        arm_y = y
        for sx in (-1, 1):
            if not (0 <= px + sx < 64 and base[arm_y][px + sx] == ROPE):
                continue
            x = px
            while 0 <= x + sx < 64 and base[arm_y][x + sx] == ROPE:
                x += sx
            kx0 = x if sx < 0 else x - 4
            return (kx0 - 1, arm_y - 3, kx0 + 5, arm_y + 3, -sx * STEP, 0)
    return None


def covers(pos, pt):
    return pos is not None and pt is not None and pos[0] <= pt[0] < pos[0] + 5 and pos[1] <= pt[1] < pos[1] + 5


def scenery(base, plate, tile, pressed):
    g = copy.deepcopy(base)
    if not pressed:
        return g
    px, py = plate
    for j in range(-1, 2):
        for i in range(-1, 2):
            g[py + j][px + i] = FLOOR
    if tile:
        x0, y0, x1, y1, dx, dy = tile
        patch = {(x, y): base[y][x] for y in range(y0, y1 + 1) for x in range(x0, x1 + 1) if 0 <= x < 64 and 0 <= y < 64}
        for (x, y) in patch:
            g[y][x] = FLOOR
        for (x, y), v in patch.items():
            if 0 <= x + dx < 64 and 0 <= y + dy < 64:
                g[y + dy][x + dx] = v
    return g


def can_enter(scn, pos, plate):
    for (x, y) in body_cells(*pos):
        if not (0 <= x < 64 and 1 <= y < 63):
            return False
        v = scn[y][x]
        if v == FLOOR:
            continue
        if v == ROPE and covers(pos, plate):
            continue
        return False
    return True


def find_spawn(base, ref):
    ox, oy = ref[0] % STEP, ref[1] % STEP
    for y in range(oy, 60, STEP):
        for x in range(ox, 60, STEP):
            if all(base[cy][cx] == FLOOR for (cx, cy) in body_cells(x, y)):
                return (x, y)
    return ref


def draw_ring(g, pos, colour):
    for (x, y) in body_cells(*pos):
        g[y][x] = colour
    g[pos[1] + 2][pos[0] + 2] = FLOOR


def draw_hud(g, mode):
    for y in range(1, 6):
        for x in range(1, 8):
            g[y][x] = 0
    def ring(x0, c):
        for j in range(3):
            for i in range(3):
                g[1 + j][x0 + i] = c
        g[2][x0 + 1] = 0
    if mode:
        ring(1, GHOST)
        ring(5, HUD_RING)
        bar = 5
    else:
        ring(1, HUD_RING)
        for j in range(3):
            for i in range(3):
                g[1 + j][5 + i] = SLOT
        bar = 1
    for i in range(3):
        g[5][bar + i] = HUD_RING


def obj_pos(state, t):
    for o in state:
        if o.get("type") == t and o.get("visible", True):
            return (o["x"], o["y"])
    return None


def fresh(state, frame):
    player = obj_pos(state, "player")
    ghost = obj_pos(state, "ghost")
    base = [list(r) for r in frame]
    for pos in (player, ghost):
        if pos:
            for (x, y) in body_cells(*pos):
                base[y][x] = FLOOR
    ones = frame[63].count(SLOT)
    return {
        "base": base,
        "calls": max(0, 2 * ones - 1),
        "mode": frame[1][1] == GHOST,
        "path": [player],
        "record": [ghost] if ghost else [],
        "k": 1 if ghost else 0,
        "spawn": None,
    }


def transition_function(state, action, frame):
    global _mem
    if _mem.get("last") != frame:
        _mem = fresh(state, frame)
    m = _mem
    base = m["base"]
    plate = find_plate(base)
    tile = find_key_tile(base, plate)
    if m["spawn"] is None:
        m["spawn"] = find_spawn(base, obj_pos(state, "player") or (0, 0))
    player = obj_pos(state, "player")
    ghost = m["record"][min(m["k"] - 1, len(m["record"]) - 1)] if (m["mode"] and m["k"] > 0 and m["record"]) else None

    aid = action["action_id"] if isinstance(action, dict) else action
    if player is not None and aid in DIRS:
        dx, dy = DIRS[aid]
        dest = (player[0] + dx * STEP, player[1] + dy * STEP)
        scn = scenery(base, plate, tile, covers(player, plate) or covers(ghost, plate))
        if can_enter(scn, dest, plate):
            player = dest
            m["path"].append(dest)
            if m["mode"]:
                m["k"] += 1
                ghost = m["record"][min(m["k"] - 1, len(m["record"]) - 1)] if m["record"] else None
    elif player is not None and aid == 5 and player != m["spawn"]:
        if m["mode"]:
            m["mode"], m["record"] = False, []
        else:
            m["mode"], m["record"] = True, list(m["path"])
        player = m["spawn"]
        m["path"], m["k"], ghost = [player], 0, None

    pressed = covers(player, plate) or covers(ghost, plate)
    out = scenery(base, plate, tile, pressed)
    if ghost:
        draw_ring(out, ghost, GHOST)
    if player:
        draw_ring(out, player, HUD_RING)
    draw_hud(out, m["mode"])
    out[63] = list(frame[63])
    if m["calls"] % 2 == 0:
        row = out[63]
        for x in range(63, -1, -1):
            if row[x] != SLOT:
                row[x] = SLOT
                break
    m["calls"] += 1
    m["last"] = [list(r) for r in out]
    return out
