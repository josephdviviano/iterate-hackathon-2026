# Mechanics: player (5x5) moves 6px per A1-4; blocked by wall px in the destination box (plate excepted) and by y<8;
#   the very first input of a level (counter n==0) is a no-op that logs a 'stay' in the path. Counter loses 1px on odd calls (hidden parity).
# Plate (topmost 3x3 wall square) covered by player/ghost -> key+arm rows left of the line shift 6 toward the line.
# A5 off spawn: no ghost -> record this life path, red hud at (1,1), white huds x+4; ghost -> clear; player -> spawn.
# Ghost = recorded path (start + successful moves)[k], k = moves since record; hidden on the player. Unconfirmed: first-input no-op vs invisible cell.
import json

SPAWN = (14, 8)
STEP = 6
TOP = 8
GHOST_PX = 2
_mem = {"last": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def pix_set(o):
    out = set()
    for j, row in enumerate(o.get("pixels") or []):
        for i, v in enumerate(row):
            if v != -1:
                out.add((o["x"] + i, o["y"] + j))
    return out


def box(pos):
    return {(pos[0] + i, pos[1] + j) for i in range(5) for j in range(5)}


def find_plate(px):
    tops = sorted((y, x) for (x, y) in px
                  if all((x + i, y + j) in px for i in range(3) for j in range(3)))
    if not tops:
        return None
    y, x = tops[0]
    return {(x + i, y + j) for i in range(3) for j in range(3)}


def wall_geometry(px):
    cols = {}
    for x, y in px:
        cols[x] = cols.get(x, 0) + 1
    line = max(cols, key=lambda c: (cols[c], c))
    arm = max(y for x, y in px if x == line)
    return line, arm


def parse_wall(px, bodies):
    """Stateless recovery of the unpressed wall pixel set from a visible (possibly cropped/pressed) wall."""
    px = set(px)
    line, arm = wall_geometry(px)
    plate = find_plate(px)
    if plate is None:
        topy = min(y for x, y in px if x == line)
        cover = [b for b in bodies if b[0] <= line <= b[0] + 4 and b[1] <= topy <= b[1] + 5]
        if cover:
            by = cover[0][1]
            plate = {(line + i, by + 1 + j) for i in (-1, 0, 1) for j in range(3)}
            px |= plate | {(line, y) for y in range(by + 4, topy)}
            key = {(x, y) for x, y in px if x < line and abs(y - arm) <= 2}
            px -= key
            px |= {(x - STEP, y) for x, y in key}
            left = min(x for x, y in px if y == arm)
            px |= {(x, arm) for x in range(left, line)}
            plate = find_plate(px)
    return px, plate


def render_wall(full, plate, pressed):
    if not pressed:
        return set(full)
    line, arm = wall_geometry(full)
    key = {(x, y) for x, y in full if x < line and abs(y - arm) <= 2}
    shifted = {(x + STEP, y) for x, y in key if x + STEP < line}
    return (full - key) | shifted


def player_move(pos, action, wall, plate, n):
    if n == 0:
        return pos
    dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action]
    nx, ny = pos[0] + dx * STEP, pos[1] + dy * STEP
    if ny < TOP or nx < 0 or nx > 59 or ny > 58:
        return pos
    dest = box((nx, ny))
    if plate and plate & dest:
        dest = dest - box(_bbox_origin(plate))
    if dest & wall:
        return pos
    return (nx, ny)


def _bbox_origin(cells):
    return (min(x for x, y in cells) - 1, min(y for x, y in cells) - 1)


def model_from_state(state):
    by = {o["type"]: [] for o in state}
    for o in state:
        by[o["type"]].append(o)
    player = by["player"][0]
    pos = (player["x"], player["y"])
    ghosts = by.get("ghost", [])
    bodies = [pos] + [(g["x"], g["y"]) for g in ghosts]
    wall_px = set()
    for o in by.get("wall", []):
        wall_px |= pix_set(o)
    full, plate = parse_wall(wall_px, bodies)
    w = by["counter"][0]["w"]
    n = 0 if w >= 64 else 2 * (64 - w) - 1
    ghost_mode = any(o["pixels"][0][0] == GHOST_PX for o in by.get("hud", []))
    gpath = None
    if ghost_mode:
        g = (ghosts[0]["x"], ghosts[0]["y"]) if ghosts else pos
        gpath = [g, g]
    return {"pos": pos, "full": full, "plate": plate, "n": n, "path": [pos],
            "ghost_mode": ghost_mode, "gpath": gpath, "k": 1 if ghost_mode else 0}


def ghost_pos(m):
    if not m["ghost_mode"] or m["k"] == 0:
        return None
    p = m["gpath"]
    return p[min(m["k"], len(p) - 1)]


def step(m, action, state):
    m = dict(m)
    if isinstance(action, dict):
        action = action.get("action_id")
    pos = m["pos"]
    if action in (1, 2, 3, 4):
        g = ghost_pos(m)
        pressed = m["plate"] and any(m["plate"] & box(b) for b in [pos] + ([g] if g else []))
        wall = render_wall(m["full"], m["plate"], pressed)
        new = player_move(pos, action, wall, m["plate"], m["n"])
        if new != pos or m["n"] == 0:
            m["path"] = m["path"] + [new]
        if new != pos and m["ghost_mode"]:
            m["k"] += 1
        m["pos"] = new
    elif action == 5 and pos != SPAWN:
        if m["ghost_mode"]:
            m["ghost_mode"], m["gpath"], m["k"] = False, None, 0
        else:
            m["ghost_mode"], m["gpath"], m["k"] = True, list(m["path"]), 0
        m["pos"] = SPAWN
        m["path"] = [SPAWN]
    m["n"] += 1
    return m


def render(m, state):
    base = {o["type"]: o for o in state if o["type"] in ("player", "gate", "target", "counter")}
    huds = [o for o in state if o["type"] == "hud" and o["pixels"][0][0] != GHOST_PX]
    red = [o for o in state if o["type"] == "hud" and o["pixels"][0][0] == GHOST_PX]
    walls = [o for o in state if o["type"] == "wall"]
    ghosts = [o for o in state if o["type"] == "ghost"]
    shift = 4 if m["ghost_mode"] else 0
    cur_shift = 4 if red else 0
    out = []
    for h in sorted(huds, key=lambda o: (o["y"], o["x"])):
        h = dict(h)
        h["x"] = h["x"] - cur_shift + shift
        out.append(h)
    p = dict(base["player"])
    p["x"], p["y"] = m["pos"]
    out.append(p)
    out.append(dict(base["gate"]))
    out.append(dict(base["target"]))
    c = dict(base["counter"])
    c["w"] = 64 - (m["n"] + 1) // 2
    c["pixels"] = [[c["pixels"][0][0]] * c["w"]]
    out.append(c)
    bodies = [m["pos"]]
    if m["ghost_mode"]:
        r = dict(red[0]) if red else {"type": "hud", "tags": ["hud"], "x": 1, "y": 1, "w": 3, "h": 3,
                                      "pixels": [[2, 2, 2], [2, -1, 2], [2, 2, 2]], "layer": 0, "visible": True}
        out.append(r)
        g = ghost_pos(m)
        if g is not None:
            bodies.append(g)
            if g != m["pos"]:
                go = dict(ghosts[0]) if ghosts else {
                    "type": "ghost", "tags": ["ghost", "snake_tail", "follower"], "w": 5, "h": 5,
                    "pixels": [[2] * 5, [2] * 5, [2, 2, -1, 2, 2], [2] * 5, [2] * 5], "layer": 1, "visible": True}
                go["x"], go["y"] = g
                out.append(go)
    pressed = m["plate"] and any(m["plate"] & box(b) for b in bodies)
    wall = render_wall(m["full"], m["plate"], pressed)
    for b in bodies:
        wall -= box(b)
    if wall:
        tmpl = walls[0] if walls else {"type": "wall", "tags": ["wall", "obstacle"], "layer": 0, "visible": True}
        x0, y0 = min(x for x, y in wall), min(y for x, y in wall)
        x1, y1 = max(x for x, y in wall), max(y for x, y in wall)
        colour = next((v for row in (walls[0]["pixels"] if walls else []) for v in row if v != -1), 8)
        wo = dict(tmpl)
        wo.update(x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1,
                  pixels=[[colour if (x, y) in wall else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)])
        out.append(wo)
    for i, o in enumerate(out):
        o["name"] = "%s_%03d" % (o["type"], i)
    return out


def transition_function(state, action):
    if _mem["last"] is not None and _mem["model"] is not None and canon(state) == _mem["last"]:
        m = _mem["model"]
    else:
        m = model_from_state(state)
    m = step(m, action, state)
    out = render(m, state)
    _mem["last"], _mem["model"] = canon(out), m
    return out


_mem["model"] = None
