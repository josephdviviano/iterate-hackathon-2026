# Floor 5 passable, void 0 / rope 8 block; A1-4 move the 5x5 player 6 cells iff dest box all floor (rope ok on the plate = topmost 3x3 rope square).
# Plate covered by player/ghost -> plate drawn as floor, door (rope beside arm row, away from line column) + 1-cell rim slides 6 toward the line.
# A5 off spawn toggles ghost mode + respawn at spawn (first cell moved to); record shifts HUD icons +4, adds colour-2 ring; clear restores HUD.
# Ghost = player trail 2 successful moves behind, trail reset to [level start, spawn]; row 63: (n+1)//2 cells 9->1 from right, n = calls (continuity-gated).
# Unconfirmed: gate/target behaviour, A5 at spawn while ghost on, ghost pressing the plate, unpress with player standing in the door slot.
FLOOR, VOID, ROPE, GHOST_C, SPENT = 5, 0, 8, 2, 1
STEP = 6
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {}


def find(state, typ):
    return [o for o in state if o.get("type") == typ]


def box_cells(x, y, w=5, h=5):
    return [(x + i, y + j) for j in range(h) for i in range(w)]


def parse_rope(world):
    cells = {(x, y) for y in range(64) for x in range(64) if world[y][x] == ROPE}
    if not cells:
        return None
    plate = None
    for (x, y) in sorted(cells, key=lambda c: (c[1], c[0])):
        if all((x + i, y + j) in cells for i in range(3) for j in range(3)):
            plate = (x + 1, y + 1)
            break
    cols, rws = {}, {}
    for (x, y) in cells:
        cols[x] = cols.get(x, 0) + 1
        rws[y] = rws.get(y, 0) + 1
    line = max(cols, key=lambda k: cols[k])
    arm = max(rws, key=lambda k: rws[k])
    door = [(x, y) for (x, y) in cells if 1 <= abs(y - arm) <= 2 and abs(x - line) > 1]
    if plate is None or not door:
        return {"plate": plate, "rect": None}
    x0 = min(c[0] for c in door) - 1
    x1 = max(c[0] for c in door) + 1
    y0 = min(c[1] for c in door) - 1
    y1 = max(c[1] for c in door) + 1
    dx = STEP if line > x1 else -STEP
    return {"plate": plate, "rect": (x0, y0, x1, y1), "dx": dx}


def pressed_world(world, rope):
    out = [r[:] for r in world]
    if not rope or not rope["rect"]:
        return out
    px, py = rope["plate"]
    for (x, y) in box_cells(px - 1, py - 1, 3, 3):
        out[y][x] = FLOOR
    x0, y0, x1, y1 = rope["rect"]
    dx = rope["dx"]
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            out[y][x] = FLOOR
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if 0 <= x + dx < 64:
                out[y][x + dx] = world[y][x]
    return out


def covers(pos, pt):
    return pos is not None and pt is not None and pos[0] <= pt[0] < pos[0] + 5 and pos[1] <= pt[1] < pos[1] + 5


def is_pressed(m):
    plate = m["rope"]["plate"] if m["rope"] else None
    return covers(m["player"], plate) or covers(m["ghost"], plate)


def current_world(m):
    return pressed_world(m["world"], m["rope"]) if is_pressed(m) else m["world"]


def can_enter(m, pos):
    world = current_world(m)
    plate = m["rope"]["plate"] if m["rope"] else None
    on_plate = covers(pos, plate)
    for (x, y) in box_cells(*pos):
        if not (0 <= x < 64 and 0 <= y < 64):
            return False
        c = world[y][x]
        if c == FLOOR or (c == ROPE and on_plate):
            continue
        return False
    return True


def counter_ones(frame):
    row = frame[63]
    k = 0
    for v in reversed(row):
        if v != SPENT:
            break
        k += 1
    return k


def fresh_model(state, frame):
    world = [r[:] for r in frame]
    pl = find(state, "player")
    p = pl[0] if pl else None
    gh = find(state, "ghost")
    g = gh[0] if gh else None
    for o in (p, g):
        if o:
            for (x, y) in box_cells(o["x"], o["y"], o["w"], o["h"]):
                if 0 <= x < 64 and 0 <= y < 64:
                    world[y][x] = FLOOR
    ones = counter_ones(frame)
    n = 2 * ones - 1 if ones > 0 else 0
    ppos = (p["x"], p["y"]) if p else None
    gpos = (g["x"], g["y"]) if g else None
    prev = _mem.get("level")
    start = ppos if n == 0 else (prev["start"] if prev else ppos)
    spawn = prev["spawn"] if prev and n > 0 else None
    return {
        "world": world, "rope": parse_rope(world), "n": n,
        "player": ppos, "pix": p["pixels"] if p else [[9] * 5] * 5,
        "ghost": gpos, "mode": gpos is not None or frame[2][1] == GHOST_C,
        "trail": [gpos, ppos] if gpos else [ppos], "start": start, "spawn": spawn,
        "moved": True, "hud_saved": None,
    }


def hud_record(world, m):
    m["hud_saved"] = [r[:8] for r in world[0:7]]
    ring = [[world[1 + j][1 + i] for i in range(3)] for j in range(5)]
    for j in range(5):
        for i in range(3):
            world[1 + j][5 + i] = ring[j][i]
            world[1 + j][1 + i] = VOID
    for j in range(3):
        for i in range(3):
            if ring[j][i] != VOID:
                world[1 + j][1 + i] = GHOST_C


def hud_clear(world, m):
    saved = m["hud_saved"]
    if saved:
        for j in range(7):
            world[j][:8] = saved[j][:]
        return
    for j in range(5):
        for i in range(3):
            world[1 + j][1 + i] = world[1 + j][5 + i]
            world[1 + j][5 + i] = SPENT if j < 3 else VOID


def draw_sprite(out, pos, pix, colour=None):
    if pos is None:
        return
    for j, row in enumerate(pix):
        for i, v in enumerate(row):
            x, y = pos[0] + i, pos[1] + j
            if v >= 0 and 0 <= x < 64 and 0 <= y < 64:
                out[y][x] = colour if colour is not None else v


def step(m, action):
    m["n"] += 1
    if isinstance(action, int) and action in DIRS:
        dx, dy = DIRS[action]
        new = (m["player"][0] + dx * STEP, m["player"][1] + dy * STEP)
        if can_enter(m, new):
            m["player"] = new
            if m["spawn"] is None:
                m["spawn"] = new
            m["trail"].append(new)
            if m["mode"]:
                m["ghost"] = m["trail"][-3] if len(m["trail"]) >= 3 else None
    elif action == 5:
        spawn = m["spawn"] or m["start"]
        if m["player"] != spawn:
            m["player"] = spawn
            if m["mode"]:
                m["mode"], m["ghost"] = False, None
                hud_clear(m["world"], m)
                m["trail"] = [spawn]
            else:
                m["mode"] = True
                hud_record(m["world"], m)
                m["trail"] = [m["start"], spawn]
                m["ghost"] = None


def render(m):
    out = [r[:] for r in current_world(m)]
    if m["ghost"] != m["player"]:
        draw_sprite(out, m["ghost"], m["pix"], GHOST_C)
    draw_sprite(out, m["player"], m["pix"])
    ones = (m["n"] + 1) // 2
    for x in range(64):
        if out[63][x] == SPENT:
            out[63][x] = m["base"]
    for k in range(min(ones, 64)):
        out[63][63 - k] = SPENT
    return out


def transition_function(state, action, frame):
    global _mem
    m = _mem.get("model")
    if m is None or _mem.get("last") != frame:
        m = fresh_model(state, frame)
        cnt = find(state, "counter")
        m["base"] = cnt[0]["pixels"][0][0] if cnt and cnt[0]["pixels"] else 9
        if m["base"] == SPENT:
            m["base"] = 9
    step(m, action)
    out = render(m)
    _mem = {"model": m, "last": [r[:] for r in out],
            "level": {"start": m["start"], "spawn": m["spawn"]}}
    return out
