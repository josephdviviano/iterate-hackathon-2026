# Mechanics: player A1-4 turns its cap and moves 4px unless the wall bbox, a box or the chaser blocks;
# an idle box right in front of the cap becomes 'faced', A5 toggles faced<->grabbed, a grabbed box moves rigidly
# with the player (boxes may enter the wall). Chaser acts every call: release a seated carried box > grab the idle
# unseated box at its left > step 4 (x first, y if blocked) toward nearest pickup point / nearest empty slot when carrying.
# Hidden state: none needed (stateless); unconfirmed: pulling a grabbed box, cap_down pixels, chaser tie-breaks.
import copy

DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
CAPS = {1: "cap_up", 2: "cap_down", 3: "cap_left", 4: "cap_right"}
BOX_COL = {"idle": 4, "faced": 3, "grabbed": 0, "carried": 5}
STATES = ("idle", "faced", "grabbed", "carried")


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def player_pixels(cap):
    px = [[14] * 4 for _ in range(4)]
    for i in range(4):
        if cap == "cap_up": px[0][i] = 0
        if cap == "cap_down": px[3][i] = 0
        if cap == "cap_left": px[i][0] = 0
        if cap == "cap_right": px[i][3] = 0
    return px


def box_state(b):
    return next(t for t in b["tags"] if t in STATES)


def in_bounds(r):
    return 0 <= r[0] and 0 <= r[1] and r[0] + r[2] <= 64 and r[1] + r[3] <= 64


def blocked(r, solids):
    return not in_bounds(r) or any(overlap(r, s) for s in solids)


def step_player(player, boxes, chaser, wall, action):
    cap = next(t for t in player["tags"] if t.startswith("cap_"))
    grabbed = [b for b in boxes if box_state(b) == "grabbed"]
    if action in DIRS:
        dx, dy = DIRS[action]
        if not grabbed:
            cap = CAPS[action]
        nr = (player["x"] + dx, player["y"] + dy, 4, 4)
        movers = [player] + grabbed
        others = [rect(b) for b in boxes if b not in grabbed] + [rect(chaser)]
        ok = not blocked(nr, others + [rect(wall)])
        for g in grabbed:
            ok = ok and not blocked((g["x"] + dx, g["y"] + dy, 4, 4), others)
        if ok:
            for m in movers:
                m["x"] += dx
                m["y"] += dy
    player["tags"] = ["player", cap]
    player["pixels"] = player_pixels(cap)
    fdx, fdy = DIRS[[k for k, v in CAPS.items() if v == cap][0]]
    front = (player["x"] + fdx, player["y"] + fdy)
    for b in boxes:
        s = box_state(b)
        at_front = (b["x"], b["y"]) == front
        if action == 5 and at_front and s in ("faced", "grabbed"):
            set_state(b, "grabbed" if s == "faced" else "faced")
        elif s == "idle" and at_front:
            set_state(b, "faced")
        elif s == "faced" and not at_front:
            set_state(b, "idle")


def set_state(b, s):
    b["tags"] = ["box", s] + (["seated"] if "seated" in b["tags"] else [])


def sign(v):
    return (v > 0) - (v < 0)


def try_move(chaser, cargo, tx, ty, solids):
    dx, dy = tx - (cargo or chaser)["x"], ty - (cargo or chaser)["y"]
    for mx, my in ((4 * sign(dx), 0), (0, 4 * sign(dy))):
        if mx == 0 and my == 0:
            continue
        units = [chaser] + ([cargo] if cargo else [])
        ok = not blocked((chaser["x"] + mx, chaser["y"] + my, 4, 4), solids["chaser"])
        if cargo:
            ok = ok and not blocked((cargo["x"] + mx, cargo["y"] + my, 4, 4), solids["box"])
        if ok:
            for u in units:
                u["x"] += mx
                u["y"] += my
            return


def step_chaser(chaser, boxes, player, wall, slots):
    carried = [b for b in boxes if box_state(b) == "carried"]
    others = [rect(b) for b in boxes if b not in carried]
    if carried:
        c = carried[0]
        if "seated" in c["tags"]:
            set_state(c, "idle")
            return
        empty = [s for s in slots if not any((b["x"], b["y"]) == (s["x"], s["y"]) for b in boxes)]
        if not empty:
            return
        t = min(empty, key=lambda s: (abs(s["x"] - c["x"]) + abs(s["y"] - c["y"]), s["y"], s["x"]))
        solids = {"chaser": others + [rect(player), rect(wall)], "box": others + [rect(player)]}
        try_move(chaser, c, t["x"], t["y"], solids)
        return
    free = [b for b in boxes if box_state(b) == "idle" and "seated" not in b["tags"]]
    for b in free:
        if (b["x"] + 4, b["y"]) == (chaser["x"], chaser["y"]):
            set_state(b, "carried")
            return
    if not free:
        return
    t = min(free, key=lambda b: (abs(b["x"] + 4 - chaser["x"]) + abs(b["y"] - chaser["y"]), b["y"], b["x"]))
    try_move(chaser, None, t["x"] + 4, t["y"], {"chaser": others + [rect(player), rect(wall)]})


def wall_texture(wall):
    rows = {}
    for r, row in enumerate(wall["pixels"]):
        if any(v != -1 for v in row):
            rows.setdefault(r % 4, row)
    return rows


def transition_function(state, action):
    st = copy.deepcopy(state)
    player = next(o for o in st if o["type"] == "player")
    chaser = next(o for o in st if o["type"] == "chaser")
    wall = next(o for o in st if o["type"] == "wall")
    boxes = [o for o in st if o["type"] == "block"]
    slots = [o for o in st if o["type"] == "target"]
    tex = wall_texture(wall)
    aid = action if isinstance(action, int) else action.get("action_id")
    step_player(player, boxes, chaser, wall, aid)
    step_chaser(chaser, boxes, player, wall, slots)
    for b in boxes:
        s = box_state(b)
        seated = any((b["x"], b["y"]) == (sl["x"], sl["y"]) for sl in slots)
        b["tags"] = ["box", s] + (["seated"] if seated else [])
        c = BOX_COL[s]
        b["pixels"] = [[c] * 4, [c, 9, 9, c], [c, 9, 9, c], [c] * 4]
    for sl in slots:
        filled = any((b["x"], b["y"]) == (sl["x"], sl["y"]) for b in boxes)
        sl["tags"] = ["tray", "slot", "passable", "filled" if filled else "empty"]
    if len(tex) == 4:
        upper = [rect(o) for o in st if o.get("layer", 0) > 0]
        px = []
        for r in range(wall["h"]):
            row = []
            for c in range(wall["w"]):
                cell = (wall["x"] + c, wall["y"] + r, 1, 1)
                row.append(-1 if any(overlap(cell, u) for u in upper) else tex[r % 4][c])
            px.append(row)
        wall["pixels"] = px
    boxes.sort(key=lambda b: (b["y"], b["x"]))
    for i, b in enumerate(boxes):
        b["name"] = "box_%d" % i
    return st
