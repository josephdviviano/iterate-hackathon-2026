# Mechanics: player 5x5 steps 6px (A1-4) inside the field, blocked by any wall pixel in its destination box unless the box holds
# the 3x3 pressure plate; first input of a fresh level (counter full) is a no-op. Counter bar w = 64-(n+1)//2, n = actions so far
# (hidden parity). Plate covered by player/ghost slides the door block 6px toward the arm. A5 (after a move this life): no ghost ->
# record life's path as ghost (red hud icon at (1,1), old huds x+4), else clear it; player respawns at (14,8). Ghost shows path[k-1]
# after k moves, hidden on the player. Unconfirmed: spawn constant (14,8), field bounds, ghost past end of path (clamped).
import json

STEP, SPAWN = 6, (14, 8)
FIELD = (2, 8, 63, 62)  # min x, min y, max x+w, max y+h (exclusive)
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {"out": None, "model": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def obj_pixels(o):
    return {(o["x"] + j, o["y"] + i) for i, r in enumerate(o["pixels"]) for j, v in enumerate(r) if v >= 0}


def box(p, w=5, h=5):
    return {(p[0] + i, p[1] + j) for i in range(w) for j in range(h)}


def find_plate(px):
    for (x, y) in sorted(px):
        sq = box((x, y), 3, 3)
        if sq <= px and len(px & box((x - 1, y - 1))) <= 15:
            return (x + 1, y + 1)
    return None


def find_door(px, plate):
    best, bp = 0, None
    for (x, y) in px:
        for ox in range(5):
            for oy in range(5):
                p = (x - ox, y - oy)
                b = box(p)
                if plate and plate in box((p[0] - 1, p[1] - 1), 7, 7):
                    continue
                c = len(px & b)
                if c > best or (c == best and p < bp):
                    best, bp = c, p
    if best < 18:
        return None
    dirn = 1 if (bp[0] + 5, bp[1] + 2) in px else -1
    pat = {(x - bp[0], y - bp[1]) for (x, y) in px & box(bp)}
    return {"pos": bp, "dir": dirn, "pat": pat}


def wall_pixels(m, pressed):
    px = set(m["wall"])
    d = m["door"]
    if pressed and d:
        dx, dy = d["pos"]
        s = d["dir"] * STEP
        px -= box((dx if s > 0 else dx - 1, dy), STEP, 5)
        px |= {(dx + s + a, dy + b) for (a, b) in d["pat"]}
    return px


def parse(state):
    m = {"base": [], "hud": [], "ghost_trail": None}
    occl = set()
    wall_vis = set()
    for o in state:
        t = o["type"]
        if t == "player":
            m["player"] = (o["x"], o["y"]); m["pobj"] = o; occl |= box(m["player"])
        elif t == "ghost":
            m["ghost_seen"] = (o["x"], o["y"]); m["gobj"] = o; occl |= box(m["ghost_seen"])
        elif t == "wall":
            wall_vis |= obj_pixels(o)
        elif t == "hud" and any(v == 2 for r in o["pixels"] for v in r):
            m["ghost_trail"] = []
        elif t == "counter":
            m["n"] = 0 if o["w"] >= 64 else 2 * (64 - o["w"]) - 1
            m["base"].append(o)
        else:
            m["base"].append(o)
    gm = m["ghost_trail"] is not None
    for o in m["base"]:
        if o["type"] == "hud" and gm:
            o["x"] -= 4
    plate = find_plate(wall_vis)
    wall = set(wall_vis)
    door = find_door(wall, plate)
    if plate is None and door:  # plate occluded: door is displaced, undo the slide
        dx, dy = door["pos"]; s = door["dir"] * STEP
        wall -= box((dx, dy))
        nx = dx - s
        wall |= {(nx + a, dy + b) for (a, b) in door["pat"]}
        for b in range(5):
            if (dx + 5 if s > 0 else dx - 1, dy + b) in wall_vis:
                wall |= {(x, dy + b) for x in (range(nx + 5, dx + 5) if s > 0 else range(dx, nx))}
        door["pos"] = (nx, dy)
    m.update(wall=wall, plate=plate, door=door, k=0, trail=[m["player"]],
             moved=m["player"] != SPAWN)
    if gm:
        g = m.get("ghost_seen", m["player"])
        m["ghost_trail"] = [g]; m["k"] = 1
    return m


def ghost_pos(m):
    tr = m["ghost_trail"]
    if not tr or m["k"] < 1:
        return None
    return tr[min(m["k"] - 1, len(tr) - 1)]


def pressed(m):
    if not m["plate"]:
        return False
    g = ghost_pos(m)
    return m["plate"] in box(m["player"]) or (g is not None and m["plate"] in box(g))


def blocked(m, dest):
    x, y = dest
    if x < FIELD[0] or y < FIELD[1] or x + 5 > FIELD[2] or y + 5 > FIELD[3]:
        return True
    b = box(dest)
    if m["plate"] and m["plate"] in b:
        return False
    return bool(b & wall_pixels(m, pressed(m)))


def step(m, action):
    fresh = m["n"] == 0
    m["n"] += 1
    a = action if isinstance(action, int) else action.get("action_id")
    if a in DIRS and not fresh:
        dx, dy = DIRS[a]
        dest = (m["player"][0] + dx * STEP, m["player"][1] + dy * STEP)
        if not blocked(m, dest):
            m["player"] = dest; m["trail"].append(dest); m["moved"] = True; m["k"] += 1
    elif a == 5 and m["moved"]:
        m["ghost_trail"] = list(m["trail"]) if m["ghost_trail"] is None else None
        m["player"] = SPAWN; m["trail"] = [SPAWN]; m["moved"] = False; m["k"] = 0
    return m


def render(m):
    out = []
    gm = m["ghost_trail"] is not None
    huds = [o for o in m["base"] if o["type"] == "hud"]
    rest = [o for o in m["base"] if o["type"] != "hud"]
    for o in huds:
        q = dict(o); q["x"] = o["x"] + (4 if gm else 0); out.append(q)
    p = dict(m["pobj"]); p["x"], p["y"] = m["player"]; out.append(p)
    for o in rest:
        q = dict(o)
        if o["type"] == "counter":
            w = max(0, 64 - (m["n"] + 1) // 2)
            q["w"] = w; q["pixels"] = [[9] * w]
        out.append(q)
    occl = box(m["player"])
    if gm:
        h = dict(huds[0]); h["x"] = huds[0]["x"]
        h["pixels"] = [[2 if v >= 0 else -1 for v in r] for r in huds[0]["pixels"]]
        out.append(h)
        g = ghost_pos(m)
        if g is not None and g != m["player"]:
            go = dict(m["pobj"]); go.update(type="ghost", tags=["ghost", "snake_tail", "follower"], x=g[0], y=g[1])
            go["pixels"] = [[2 if v >= 0 else -1 for v in r] for r in m["pobj"]["pixels"]]
            out.append(go); occl |= box(g)
    px = wall_pixels(m, pressed(m)) - occl
    if px:
        x0 = min(x for x, _ in px); y0 = min(y for _, y in px)
        x1 = max(x for x, _ in px); y1 = max(y for _, y in px)
        out.append({"type": "wall", "tags": ["wall", "obstacle"], "layer": 0, "visible": True,
                    "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
                    "pixels": [[8 if (x, y) in px else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]})
    for i, o in enumerate(out):
        o["name"] = "%s_%03d" % (o["type"], i)
    return out


def transition_function(state, action):
    import copy
    if _mem["out"] is not None and canon(state) == _mem["out"]:
        m = copy.deepcopy(_mem["model"])
    else:
        m = parse(copy.deepcopy(state))
    m = step(m, action)
    out = render(m)
    _mem["out"] = canon(out); _mem["model"] = copy.deepcopy(m)
    return out
