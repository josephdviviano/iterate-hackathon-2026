# Mechanics: frame-native ghost/rope game. Floor 5, void 0 (blocks), rope 8 blocks except a box holding the plate.
# Player (9) / ghost (2) are 5x5 sprites with a floor-coloured centre, moving 6 cells; A5 off spawn records the life
# trail as a ghost (HUD ring -> 2, next slot ring 9 + bar) or clears an existing ghost, then respawns; ghost shows
# trail[k-1] after k successful moves. Plate covered by player/ghost -> key 7x7 patch slides 6 toward the column.
# Counter row 63: 9->1 from the right on odd calls (parity hidden; continuity-gated, fallback assumes even).
FLOOR, VOID, ROPE, PLAYER_C, GHOST_C, HUD_C, SPARE_C = 5, 0, 8, 9, 2, 9, 1
STEP = 6
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
MEM = {}


def copy(f):
    return [list(r) for r in f]


def box_cells(x, y, w=5, h=5):
    return [(x + i, y + j) for j in range(h) for i in range(w)]


def find_obj(state, typ):
    for o in state:
        if o.get("type") == typ and o.get("visible", True):
            return (o["x"], o["y"])
    return None


# ---------- rope parsing (plate, column, key) ----------
def parse_rope(bg):
    rope = {(x, y) for y in range(64) for x in range(64) if bg[y][x] == ROPE}
    plate = None
    for (x, y) in sorted(rope, key=lambda p: (p[1], p[0])):
        if all((x + i, y + j) in rope for i in range(3) for j in range(3)):
            plate = (x + 1, y + 1)
            break
    cols = {}
    for (x, y) in rope:
        cols[x] = cols.get(x, 0) + 1
    colx = max(cols, key=lambda c: cols[c]) if cols else 0
    key, best = None, 17
    for (x, y) in rope:
        if plate and abs(x + 2 - plate[0]) <= 3 and abs(y + 2 - plate[1]) <= 3:
            continue
        n = sum((x + i, y + j) in rope for i in range(5) for j in range(5))
        if n > best:
            best, key = n, (x, y)
    d = 0
    if key:
        d = 1 if colx > key[0] + 2 else -1
    return plate, key, d


def strip_sprites(frame, state):
    bg = copy(frame)
    for typ in ("player", "ghost"):
        p = find_obj(state, typ)
        if p:
            for (x, y) in box_cells(*p):
                if 0 <= x < 64 and 0 <= y < 64:
                    bg[y][x] = FLOOR
    return bg


def terrain(base, rope, pressed):
    plate, key, d = rope
    t = copy(base)
    if pressed and key:
        kx, ky = key[0] - 1, key[1] - 1
        patch = {(x, y): base[y][x] for (x, y) in box_cells(kx, ky, 7, 7)}
        for (x, y) in patch:
            t[y][x] = FLOOR
        for (x, y), v in patch.items():
            nx = x + STEP * d
            if 0 <= nx < 64:
                t[y][nx] = v
    return t


def covers(pos, pt):
    return pos is not None and pt is not None and pos[0] <= pt[0] < pos[0] + 5 and pos[1] <= pt[1] < pos[1] + 5


def is_pressed(rope, player, ghost):
    return covers(player, rope[0]) or covers(ghost, rope[0])


def blocked(t, rope, dest):
    x0, y0 = dest
    if x0 < 0 or y0 < 0 or x0 + 5 > 64 or y0 + 5 > 63:
        return True
    plate_ok = covers(dest, rope[0])
    for (x, y) in box_cells(x0, y0):
        c = t[y][x]
        if c == VOID or (c == ROPE and not plate_ok):
            return True
    return False


def draw_sprite(f, pos, colour):
    if pos is None:
        return
    for (x, y) in box_cells(*pos):
        if 0 <= x < 64 and 0 <= y < 64:
            f[y][x] = FLOOR if (x - pos[0], y - pos[1]) == (2, 2) else colour


# ---------- HUD ----------
def hud_slots(frame):
    n = 0
    while 1 + 4 * n + 2 < 64 and frame[2][1 + 4 * n] != 0:
        n += 1
    return n


def draw_hud(f, nslots, cur):
    for i in range(nslots):
        x0 = 1 + 4 * i
        for j in range(3):
            for k in range(3):
                if i < cur:
                    c = GHOST_C
                elif i == cur:
                    c = HUD_C
                else:
                    c = SPARE_C
                if i <= cur and (j, k) == (1, 1):
                    c = 0
                f[1 + k][x0 + j] = c
            f[5][x0 + j] = HUD_C if i == cur else 0


def draw_counter(f, ones):
    for x in range(64):
        if f[63][x] in (HUD_C, SPARE_C):
            f[63][x] = SPARE_C if x >= 64 - ones else HUD_C


# ---------- state model ----------
def fresh_model(state, frame):
    player = find_obj(state, "player")
    ghost = find_obj(state, "ghost")
    base = strip_sprites(frame, state)
    rope = parse_rope(base)
    ones = sum(1 for v in frame[63] if v == SPARE_C)
    return {
        "base": base, "rope": rope, "n": 2 * ones - 1 if ones else 0,
        "player": player, "ghost": ghost, "spawn": None,
        "trail": [player] if player else [], "gtrail": [ghost] if ghost else [],
        "k": 1 if ghost else 0, "cur": 1 if ghost else 0, "nslots": hud_slots(frame),
    }


def render(m, frame):
    f = terrain(m["base"], m["rope"], is_pressed(m["rope"], m["player"], m["ghost"]))
    for y in list(range(0, 7)) + [63]:
        f[y] = list(frame[y])
    draw_sprite(f, m["ghost"], GHOST_C)
    draw_sprite(f, m["player"], PLAYER_C)
    draw_hud(f, m["nslots"], m["cur"])
    draw_counter(f, (m["n"] + 1) // 2)
    return f


def step(m, action):
    m["n"] += 1
    player = m["player"]
    if player is None:
        return
    if isinstance(action, int) and action in DIRS:
        t = terrain(m["base"], m["rope"], is_pressed(m["rope"], player, m["ghost"]))
        dx, dy = DIRS[action]
        dest = (player[0] + STEP * dx, player[1] + STEP * dy)
        if blocked(t, m["rope"], dest):
            return
        m["player"] = dest
        m["trail"].append(dest)
        if m["spawn"] is None:
            m["spawn"] = dest
        if m["gtrail"]:
            m["k"] += 1
            m["ghost"] = m["gtrail"][min(m["k"] - 1, len(m["gtrail"]) - 1)]
    elif action == 5:
        spawn = m["spawn"] or player
        if player == spawn:
            return
        if m["gtrail"]:
            m["gtrail"], m["cur"] = [], 0
        else:
            m["gtrail"], m["cur"] = list(m["trail"]), min(1, m["nslots"] - 1)
        m["ghost"], m["k"] = None, 0
        m["player"] = spawn
        m["trail"] = [spawn]


def transition_function(state, action, frame):
    m = MEM.get("model")
    if m is None or MEM.get("last") != frame:
        m = fresh_model(state, frame)
        if m is not None and MEM.get("model"):
            old = MEM["model"]
            if old["rope"][1] == m["rope"][1] or old["base"][40:56] == m["base"][40:56]:
                m["spawn"] = old["spawn"]
    step(m, action)
    out = render(m, frame)
    MEM["model"], MEM["last"] = m, copy(out)
    return out
