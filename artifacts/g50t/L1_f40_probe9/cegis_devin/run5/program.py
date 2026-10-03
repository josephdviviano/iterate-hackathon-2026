# Mechanics: player 5x5 moves 6px (A1-4); blocked by board/HUD band (y<8) or any non-plate wall px in the
# dest box; the first input of a fresh level (counter w=64) is a no-op. Counter loses 1px on odd calls.
# Wall = rope: plate (topmost 3x3, walkable) pressed by player/ghost box -> arm/key left of column slides +6 toward it.
# A5: player on spawn -> no-op; ghost mode -> clear; else record (red hud, huds x+4); both respawn at spawn.
# Ghost = player trail lag 2, trail reset to [level start, spawn]; hidden when on the player. Unconfirmed: invisible-cell alt for step 1.
import json

STEP = 6
SPAWN_DEFAULT = (14, 8)
GHOST_PX = [[2] * 5, [2] * 5, [2, 2, -1, 2, 2], [2] * 5, [2] * 5]
RED_HUD_PX = [[2, 2, 2], [2, -1, 2], [2, 2, 2]]
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}

_levels = {}
_last = {"out": None, "n": 0, "trail": None, "k": 0}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def level_key(state):
    return tuple(sorted((o["type"], o["x"], o["y"]) for o in state if o["type"] in ("gate", "target")))


def px_set(o):
    out = set()
    for j, row in enumerate(o["pixels"]):
        for i, v in enumerate(row):
            if v != -1:
                out.add((o["x"] + i, o["y"] + j))
    return out


def find_plate(walls):
    for y in sorted({p[1] for p in walls}):
        for x in sorted(p[0] for p in walls if p[1] == y):
            if all((x + i, y + j) in walls for i in range(3) for j in range(3)):
                return {(x + i, y + j) for i in range(3) for j in range(3)}
    return set()


class Rope:
    def __init__(self, walls):
        self.base = set(walls)
        self.plate = find_plate(walls)
        if self.plate:
            self.centre = (min(p[0] for p in self.plate) + 1, min(p[1] for p in self.plate) + 1)
            self.col = self.centre[0]
        else:
            self.centre, self.col = None, None

    def render(self, pressed):
        if not pressed or self.col is None:
            return set(self.base)
        arm = {p for p in self.base if p not in self.plate and p[0] != self.col}
        left = sum(1 for p in arm if p[0] < self.col) >= sum(1 for p in arm if p[0] > self.col)
        d = STEP if left else -STEP
        out = self.base - arm
        for (x, y) in arm:
            nx = x + d
            if (left and nx < self.col) or (not left and nx > self.col):
                out.add((nx, y))
        return out

    def pressed_by(self, boxes):
        return self.centre is not None and any(inside(self.centre, b) for b in boxes)


def inside(p, b):
    return b[0] <= p[0] < b[0] + 5 and b[1] <= p[1] < b[1] + 5


def get_level(state):
    key = level_key(state)
    lv = _levels.setdefault(key, {})
    walls = set()
    for o in state:
        if o["type"] == "wall":
            walls |= px_set(o)
    player = next(o for o in state if o["type"] == "player")
    counter = next(o for o in state if o["type"] == "counter")
    if counter["w"] == 64 and "start" not in lv:
        lv["start"] = (player["x"], player["y"])
    if "rope" not in lv:
        r = Rope(walls)
        boxes = [(o["x"], o["y"]) for o in state if o["layer"] == 1]
        if r.plate and not r.pressed_by(boxes):
            lv["rope"] = r
    return lv, walls


def make(name_type, tags, x, y, w, h, px, layer):
    return {"name": name_type, "type": name_type, "tags": tags, "x": x, "y": y, "w": w, "h": h,
            "pixels": px, "layer": layer, "visible": True}


def wall_obj(pix):
    if not pix:
        return None
    x0, y0 = min(p[0] for p in pix), min(p[1] for p in pix)
    x1, y1 = max(p[0] for p in pix), max(p[1] for p in pix)
    px = [[8 if (x, y) in pix else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]
    return make("wall", [], x0, y0, x1 - x0 + 1, y1 - y0 + 1, px, 0)


def transition_function(state, action):
    cont = _last["out"] is not None and canon(state) == _last["out"]
    lv, cur_walls = get_level(state)
    by = {}
    for o in state:
        by.setdefault(o["type"], []).append(o)
    player = by["player"][0]
    counter = by["counter"][0]
    huds = [o for o in by.get("hud", []) if o["pixels"][0][0] != 2]
    red = [o for o in by.get("hud", []) if o["pixels"][0][0] == 2]
    ghost_mode = bool(red)
    start = lv.get("start", SPAWN_DEFAULT)
    spawn = lv.get("spawn", SPAWN_DEFAULT)
    pos = (player["x"], player["y"])
    ghosts = by.get("ghost", [])

    if cont:
        n, trail, k = _last["n"], _last["trail"], _last["k"]
    else:
        n = 0 if counter["w"] == 64 else 2 * (64 - counter["w"]) - 1
        k = 0
        trail = None
        if ghost_mode:
            if ghosts:
                g = (ghosts[0]["x"], ghosts[0]["y"])
                trail, k = [g, g, pos], 3
            elif pos == spawn:
                trail = [start, spawn]
            else:
                trail, k = [pos, pos, pos], 3
    rope = lv.get("rope")

    def logical_walls(boxes):
        if rope is None:
            return cur_walls
        return rope.render(rope.pressed_by(boxes))

    act = action["action_id"] if isinstance(action, dict) else action
    if act in MOVES and n > 0:
        dx, dy = MOVES[act]
        nx, ny = pos[0] + dx, pos[1] + dy
        gpos = trail[-3] if ghost_mode and trail and k >= 1 else None
        walls_now = logical_walls([pos] + ([gpos] if gpos else []))
        on_plate = rope is not None and rope.pressed_by([(nx, ny)])
        blocked = not (0 <= nx and nx + 5 <= 64 and ny >= 8 and ny + 5 <= 63)
        if not blocked and not on_plate:
            blocked = any(inside(p, (nx, ny)) for p in walls_now)
        if not blocked:
            if "spawn" not in lv and lv.get("start") == pos:
                lv["spawn"] = (nx, ny)
            pos = (nx, ny)
            if ghost_mode and trail is not None:
                trail = trail + [pos]
                k += 1
    elif act == 5 and n > 0 and pos != spawn:
        if ghost_mode:
            ghost_mode, trail, k = False, None, 0
        else:
            ghost_mode, trail, k = True, [start, spawn], 0
        pos = spawn
    n += 1

    out = []
    shift = 4 if ghost_mode else 0
    for h in sorted(huds, key=lambda o: o["name"]):
        nh = dict(h)
        nh["x"] = 1 + shift
        out.append(nh)
    np_ = dict(player)
    np_["x"], np_["y"] = pos
    out.append(np_)
    for t in ("gate", "target"):
        out.extend(dict(o) for o in by.get(t, []))
    w = 64 - (n + 1) // 2
    nc = dict(counter)
    nc["w"], nc["x"], nc["pixels"] = w, 0, [[9] * w]
    out.append(nc)
    boxes = [pos]
    if ghost_mode:
        out.append(make("hud", ["hud"], 1, 1, 3, 3, [r[:] for r in RED_HUD_PX], 0))
        if trail is not None and k >= 1:
            g = trail[-3]
            boxes.append(g)
            if g != pos:
                out.append(make("ghost", ["ghost", "snake_tail", "follower"], g[0], g[1], 5, 5,
                                [r[:] for r in GHOST_PX], 1))
    walls = logical_walls(boxes)
    visible = {p for p in walls if not any(inside(p, b) for b in boxes)}
    wo = wall_obj(visible)
    if wo is not None:
        old = by.get("wall")
        if old:
            wo["tags"] = list(old[0]["tags"])
        out.append(wo)
    for i, o in enumerate(out):
        o["name"] = "%s_%03d" % (o["type"], i)
    _last.update(out=canon(out), n=n, trail=trail, k=k)
    return out
