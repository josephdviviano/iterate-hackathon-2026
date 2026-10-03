# Mechanics: player A1-4 sets cap dir and moves 4 unless bounds/wall/box/chaser block; an idle box
# directly ahead of the cap is 'faced' (3); A5 toggles faced<->grabbed (0); a grabbed box moves with the player
# (boxes may enter the wall). Chaser acts every step: release a seated carried box > pick up a free box whose
# right side it touches > step 4 toward nearest pickup point / nearest empty slot (+4,0), x first, y if blocked.
# Wall = period-4 dither masked by layer>0 objects. Unconfirmed: pulling grabbed boxes, chaser crossing the wall.
import copy

DIRS = {1: (0, -4, "cap_up"), 2: (0, 4, "cap_down"), 3: (-4, 0, "cap_left"), 4: (4, 0, "cap_right")}
BORDER = {"idle": 4, "carried": 5, "faced": 3, "grabbed": 0}
DITHER = [[2, -1, 2, 2], [-1, 2, 2, 2], [2, 2, 2, -1], [2, 2, -1, 2]]
SIZE = 64


def box_state(b):
    return next(t for t in b["tags"] if t in BORDER)


def set_box_state(b, s):
    seated = "seated" in b["tags"]
    b["tags"] = ["box", s] + (["seated"] if seated else [])


def box_pixels(s):
    c = BORDER[s]
    return [[c] * 4, [c, 9, 9, c], [c, 9, 9, c], [c] * 4]


def player_pixels(cap):
    p = [[14] * 4 for _ in range(4)]
    for i in range(4):
        if cap == "cap_up": p[0][i] = 0
        elif cap == "cap_down": p[3][i] = 0
        elif cap == "cap_left": p[i][0] = 0
        else: p[i][3] = 0
    return p


def overlaps(a, x, y, w=4, h=4):
    return a["x"] < x + w and x < a["x"] + a["w"] and a["y"] < y + h and y < a["y"] + a["h"]


def in_wall(wall, x, y):
    return wall is not None and overlaps(wall, x, y)


def in_bounds(x, y):
    return 0 <= x <= SIZE - 4 and 0 <= y <= SIZE - 4


def player_cap(p):
    return next(t for t in p["tags"] if t.startswith("cap_"))


def cap_delta(cap):
    return next((dx, dy) for dx, dy, c in DIRS.values() if c == cap)


def step_player(player, boxes, chaser, wall, action):
    grabbed = [b for b in boxes if box_state(b) == "grabbed"]
    if action == 5:
        for b in boxes:
            if box_state(b) == "faced": set_box_state(b, "grabbed")
            elif box_state(b) == "grabbed": set_box_state(b, "faced")
        return
    if action not in DIRS:
        return
    dx, dy, cap = DIRS[action]
    if not grabbed:
        player["tags"] = ["player", cap]
    nx, ny = player["x"] + dx, player["y"] + dy
    movers = [player] + grabbed
    others = [b for b in boxes if b not in grabbed] + ([chaser] if chaser else [])
    ok = in_bounds(nx, ny) and not in_wall(wall, nx, ny)
    for m in movers:
        mx, my = m["x"] + dx, m["y"] + dy
        if not in_bounds(mx, my) or any(overlaps(o, mx, my) for o in others):
            ok = False
    if ok:
        for m in movers:
            m["x"] += dx
            m["y"] += dy


def refresh_faced(player, boxes):
    dx, dy = cap_delta(player_cap(player))
    fx, fy = player["x"] + dx, player["y"] + dy
    for b in boxes:
        s = box_state(b)
        if s in ("idle", "faced") and "seated" not in b["tags"]:
            set_box_state(b, "faced" if (b["x"], b["y"]) == (fx, fy) else "idle")


def free(b):
    return box_state(b) in ("idle", "faced") and "seated" not in b["tags"]


def step_chaser(chaser, player, boxes, slots, wall):
    carried = next((b for b in boxes if box_state(b) == "carried" and "seated" not in b["tags"]), None)
    seated = next((b for b in boxes if box_state(b) == "carried" and "seated" in b["tags"]), None)
    if seated is not None:
        set_box_state(seated, "idle")
        return
    cx, cy = chaser["x"], chaser["y"]
    if carried is None:
        adj = [b for b in boxes if free(b) and (b["x"] + 4, b["y"]) == (cx, cy)]
        if adj:
            set_box_state(adj[0], "carried")
            return
        goals = [(b["x"] + 4, b["y"]) for b in boxes if free(b)]
        ref = (cx, cy)
    else:
        goals = [(s["x"] + 4, s["y"]) for s in slots if "empty" in s["tags"]]
        ref = (carried["x"] + 4, carried["y"])
    if not goals:
        return
    gx, gy = min(goals, key=lambda g: abs(g[0] - ref[0]) + abs(g[1] - ref[1]))
    blockers = [b for b in boxes if b is not carried] + [player]

    def can(x, y):
        if not in_bounds(x, y) or in_wall(wall, x, y) or any(overlaps(o, x, y) for o in blockers):
            return False
        if carried is not None:
            bx, by = x - 4, y
            if not in_bounds(bx, by) or any(overlaps(o, bx, by) for o in blockers):
                return False
        return True

    sx = (gx > cx) - (gx < cx)
    sy = (gy > cy) - (gy < cy)
    moves = ([(4 * sx, 0)] if sx else []) + ([(0, 4 * sy)] if sy else [])
    for dx, dy in moves:
        if can(cx + dx, cy + dy):
            chaser["x"] += dx
            chaser["y"] += dy
            if carried is not None:
                carried["x"] += dx
                carried["y"] += dy
            return


def update_slots(boxes, slots):
    for b in boxes:
        if box_state(b) == "carried" and "seated" not in b["tags"]:
            if any((s["x"], s["y"]) == (b["x"], b["y"]) and "empty" in s["tags"] for s in slots):
                b["tags"] = b["tags"] + ["seated"]
    for s in slots:
        filled = any("seated" in b["tags"] and (b["x"], b["y"]) == (s["x"], s["y"]) for b in boxes)
        s["tags"] = ["tray", "slot", "passable", "filled" if filled else "empty"]


def render_wall(wall, objs):
    occ = [o for o in objs if o["layer"] > 0 and o["type"] != "wall"]
    px = []
    for r in range(wall["h"]):
        row = []
        for c in range(wall["w"]):
            X, Y = wall["x"] + c, wall["y"] + r
            v = DITHER[Y % 4][X % 4]
            if any(o["x"] <= X < o["x"] + o["w"] and o["y"] <= Y < o["y"] + o["h"] for o in occ):
                v = -1
            row.append(v)
        px.append(row)
    wall["pixels"] = px


def transition_function(state, action):
    objs = copy.deepcopy(state)
    act = action["action_id"] if isinstance(action, dict) else action
    player = next((o for o in objs if o["type"] == "player"), None)
    chaser = next((o for o in objs if o["type"] == "chaser"), None)
    wall = next((o for o in objs if o["type"] == "wall"), None)
    boxes = [o for o in objs if o["type"] == "block"]
    slots = [o for o in objs if o["type"] == "target"]
    if player is not None:
        step_player(player, boxes, chaser, wall, act)
        refresh_faced(player, boxes)
    if chaser is not None:
        step_chaser(chaser, player, boxes, slots, wall)
    update_slots(boxes, slots)
    for b in boxes:
        b["pixels"] = box_pixels(box_state(b))
    for i, b in enumerate(sorted(boxes, key=lambda b: (b["y"], b["x"]))):
        b["name"] = "box_%d" % i
    if player is not None:
        player["pixels"] = player_pixels(player_cap(player))
    if wall is not None:
        render_wall(wall, objs)
    return objs
