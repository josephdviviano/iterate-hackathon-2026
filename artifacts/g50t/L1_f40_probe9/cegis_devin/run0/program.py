# Mechanics: player 5x5 moves 6 (A1-4); blocked by y<8, screen edge, any wall px in dest box unless it holds the
# plate centre; first input of a fresh level (n==0) moves nothing but is logged in the life path as a 'stay' step.
# Plate covered by a body -> wall px left of the line column shift +6 toward it; layer-1 bodies crop the wall.
# A5 after moving: respawn at first reached cell; no red hud -> record path as ghost (red hud (1,1), huds +4), else
# clear. Ghost = path[k] after k moves (absent at k=0, hidden on player). Counter -1 on odd calls. Unconfirmed: stay-log vs lag.
import json

STEP = 6
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
FIELD_TOP = 8
DEFAULT_RESPAWN = (14, 8)
BODY = 5
GHOST_TAGS = ["ghost", "snake_tail", "follower"]
_M = {"last": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def body_pixels(c):
    return [[-1 if (i == 2 and j == 2) else c for i in range(BODY)] for j in range(BODY)]


def obj_cells(o):
    out = set()
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            if v != -1:
                out.add((o["x"] + i, o["y"] + j))
    return out


def box(p):
    return {(p[0] + i, p[1] + j) for i in range(BODY) for j in range(BODY)}


def is_red_hud(o):
    return o["type"] == "hud" and any(v == 2 for r in o["pixels"] for v in r)


# ---------------- wall model ----------------
def line_col(cells):
    cnt = {}
    for x, _ in cells:
        cnt[x] = cnt.get(x, 0) + 1
    return max(cnt, key=lambda x: (cnt[x], x))


def find_plate(cells):
    for (x, y) in sorted(cells, key=lambda c: (c[1], c[0])):
        if all((x + i, y + j) in cells for i in range(3) for j in range(3)):
            return (x + 1, y + 1)
    return None


def press(base, lx):
    left = {c for c in base if c[0] < lx}
    moved = {(x + STEP, y) for x, y in left if x + STEP < lx}
    return (base - left) | moved


def unpress(cells, lx):
    left = {c for c in cells if c[0] < lx}
    moved = {(x - STEP, y) for x, y in left}
    out = (cells - left) | moved
    arm_y = max(y for x, y in cells if x == lx)
    row = [x for x, y in moved if y == arm_y]
    if row:
        out |= {(x, arm_y) for x in range(min(row), lx)}
    return out


def covered(plate, bodies):
    return any(plate in box(b) for b in bodies)


def wall_now(W, bodies):
    return press(W["base"], W["lx"]) if covered(W["plate"], bodies) else W["base"]


def parse_wall(cells, bodies):
    lx = line_col(cells)
    plate = find_plate(cells)
    cells = set(cells)
    if plate is None:
        hit = [b for b in bodies if b[0] <= lx < b[0] + BODY] or bodies
        b = hit[0]
        plate = (b[0] + 2, b[1] + 2)
        cells |= {(plate[0] + i, plate[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1)}
        ys = [y for x, y in cells if x == lx and y > plate[1] + 1]
        if ys:
            cells |= {(lx, y) for y in range(plate[1] + 2, min(ys))}
    if covered(plate, bodies):
        cells = unpress(cells, lx)
    return {"base": cells, "lx": lx, "plate": plate}


# ---------------- state parsing ----------------
def parse(state):
    M = {}
    player = next(o for o in state if o["type"] == "player")
    M["p"] = (player["x"], player["y"])
    ghost = next((o for o in state if o["type"] == "ghost"), None)
    M["ghost_mode"] = any(is_red_hud(o) for o in state)
    if ghost is not None:
        M["gtrail"], M["k"] = [(ghost["x"], ghost["y"])], 1
    else:
        M["gtrail"], M["k"] = [], 0
    counter = next(o for o in state if o["type"] == "counter")
    M["n"] = max(0, 2 * (64 - counter["w"]) - 1)
    M["R"] = None if M["n"] == 0 else DEFAULT_RESPAWN
    M["path"] = [M["p"]]
    M["moved"] = M["R"] is not None and M["p"] != M["R"]
    bodies = [M["p"]] + ([(ghost["x"], ghost["y"])] if ghost else [])
    wcells = set()
    for o in state:
        if o["type"] == "wall":
            wcells |= obj_cells(o)
    M["W"] = parse_wall(wcells, bodies)
    M["W"]["tags"] = next((o["tags"] for o in state if o["type"] == "wall"), ["wall", "obstacle"])
    return M


def ghost_pos(M):
    if not M["gtrail"] or M["k"] == 0:
        return None
    return M["gtrail"][min(M["k"], len(M["gtrail"]) - 1)]


# ---------------- per-type updates ----------------
def try_move(M, action):
    dx, dy = DIRS[action]
    x, y = M["p"][0] + dx, M["p"][1] + dy
    if y < FIELD_TOP or x < 0 or x + BODY > 64 or y + BODY > 63:
        return False
    dest = box((x, y))
    walls = wall_now(M["W"], bodies_of(M))
    if M["W"]["plate"] not in dest and dest & walls:
        return False
    M["p"] = (x, y)
    return True


def bodies_of(M):
    g = ghost_pos(M)
    return [M["p"]] + ([g] if g is not None else [])


def step(M, action):
    events = None
    if action in DIRS and M["n"] == 0:
        M["path"].append(M["p"])
    elif action in DIRS:
        if try_move(M, action):
            if M["R"] is None:
                M["R"] = M["p"]
            M["path"].append(M["p"])
            M["moved"] = True
            if M["gtrail"]:
                M["k"] += 1
    elif action == 5 and M["moved"]:
        if M["ghost_mode"]:
            M["ghost_mode"], M["gtrail"], M["k"] = False, [], 0
            events = "clear"
        else:
            M["ghost_mode"], M["gtrail"], M["k"] = True, list(M["path"]), 0
            events = "record"
        M["p"] = M["R"]
        M["path"] = [M["R"]]
        M["moved"] = False
    M["n"] += 1
    return events


# ---------------- rendering ----------------
def render(state, M, events):
    base, extra = [], []
    for o in state:
        o = dict(o)
        t = o["type"]
        if t in ("ghost", "wall") or is_red_hud(o):
            if is_red_hud(o) and events != "clear":
                extra.append(o)
            continue
        if t == "hud" and events:
            o["x"] += 4 if events == "record" else -4
        if t == "player":
            o["x"], o["y"] = M["p"]
        if t == "counter":
            w = 64 - (M["n"] + 1) // 2
            o["w"], o["pixels"] = w, [[9] * w]
        base.append(o)
    if events == "record":
        extra.insert(0, {"name": "", "type": "hud", "tags": ["hud"], "x": 1, "y": 1, "w": 3, "h": 3,
                         "pixels": [[2, 2, 2], [2, -1, 2], [2, 2, 2]], "layer": 0, "visible": True})
    out = base + extra
    g = ghost_pos(M)
    bodies = [M["p"]]
    if g is not None and g != M["p"]:
        out.append({"name": "", "type": "ghost", "tags": list(GHOST_TAGS), "x": g[0], "y": g[1], "w": BODY,
                    "h": BODY, "pixels": body_pixels(2), "layer": 1, "visible": True})
        bodies.append(g)
    cells = set(wall_now(M["W"], bodies_of(M)))
    for b in bodies:
        cells -= box(b)
    if cells:
        x0, x1 = min(c[0] for c in cells), max(c[0] for c in cells)
        y0, y1 = min(c[1] for c in cells), max(c[1] for c in cells)
        px = [[8 if (x, y) in cells else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]
        out.append({"name": "", "type": "wall", "tags": list(M["W"]["tags"]), "x": x0, "y": y0, "w": x1 - x0 + 1,
                    "h": y1 - y0 + 1, "pixels": px, "layer": 0, "visible": True})
    for i, o in enumerate(out):
        o["name"] = "%s_%03d" % (o["type"], i)
    return out


def transition_function(state, action):
    aid = action["action_id"] if isinstance(action, dict) else action
    if _M["last"] is not None and canon(state) == _M["last"]:
        M = _M["model"]
    else:
        M = parse(state)
    events = step(M, aid)
    out = render(state, M, events)
    _M["last"], _M["model"] = canon(out), M
    return out
