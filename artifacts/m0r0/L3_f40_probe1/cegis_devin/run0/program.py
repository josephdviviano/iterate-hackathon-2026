# Mechanics: mirror-blocks game. Block mode: A1/A2 move both blocks y-4/y+4, A3 apart (left x-4, right x+4), A4 together;
# each block moves alone unless its target 4x4 cell is out of [2,58], a maze cell, a marker cell (marker xy-1) or the other block.
# Click marker -> armed (blocks -> color_1 players, marker active color_11); armed A1-A4 steer the active marker by 4 (same blocking);
# click other marker = switch, click player = disarm, other clicks/A5 no-op. Bars wall_0 (top right, bottom left) w=floor(3(k+1)/7), k=actions in level.
# Names type_color_rank by (y,x) (blocks reserve 0,1). Hypothesis: maze cells are invisible level geometry; (22,18) only seen blocking at step 124; fallback k=ceil(7w/3).
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
BLOCK_PIX = [[10] * 4 for _ in range(4)]
MOVES = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_last = {"out": None, "k": 0, "model": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def bar_width(k):
    return (3 * (k + 1)) // 7


def parse(state):
    m = {"mode": "block", "blocks": [], "markers": [], "scenery": [], "w": 0}
    for o in state:
        t = o["type"]
        if t == "player":
            m["blocks"].append([o["x"], o["y"], o["name"]])
            if "armed" in o.get("tags", []):
                m["mode"] = "armed"
        elif t == "marker":
            m["markers"].append([o["x"], o["y"], "active" in o.get("tags", [])])
        elif t == "wall" and "color_0" in o.get("tags", []):
            m["w"] = max(m["w"], o["w"])
        else:
            m["scenery"].append(o)
    m["blocks"].sort(key=lambda b: (b[0], b[1]))
    m["left"] = 0
    for i, b in enumerate(m["blocks"]):
        if b[2] == "block_left":
            m["left"] = i
    return m


def marker_cells(m, skip=None):
    return {(mk[0] - 1, mk[1] - 1) for i, mk in enumerate(m["markers"]) if i != skip}


def free(cell, blocked):
    x, y = cell
    return LO <= x <= HI and LO <= y <= HI and cell not in MAZE and cell not in blocked


def move_blocks(m, dx, dy):
    left = m["left"]
    occupied = marker_cells(m)
    targets = []
    for i, b in enumerate(m["blocks"]):
        sx = dx if i == left else -dx
        targets.append((b[0] + sx, b[1] + dy))
    ok = [free(t, occupied) for t in targets]
    final = [targets[i] if ok[i] else (b[0], b[1]) for i, b in enumerate(m["blocks"])]
    for i in range(len(final)):
        others = {final[j] for j in range(len(final)) if j != i}
        if ok[i] and final[i] in others:
            final[i] = (m["blocks"][i][0], m["blocks"][i][1])
    for i, b in enumerate(m["blocks"]):
        b[0], b[1] = final[i]


def move_marker(m, dx, dy):
    for i, mk in enumerate(m["markers"]):
        if mk[2]:
            cell = (mk[0] - 1 + dx, mk[1] - 1 + dy)
            blocked = marker_cells(m, skip=i) | {(b[0], b[1]) for b in m["blocks"]}
            if free(cell, blocked):
                mk[0], mk[1] = mk[0] + dx, mk[1] + dy


def hit(px, py, x, y, size):
    return x <= px < x + size and y <= py < y + size


def click(m, px, py):
    for i, mk in enumerate(m["markers"]):
        if hit(px, py, mk[0], mk[1], 2):
            if not mk[2]:
                for other in m["markers"]:
                    other[2] = False
                mk[2] = True
                m["mode"] = "armed"
            return
    for b in m["blocks"]:
        if hit(px, py, b[0], b[1], 4):
            if m["mode"] == "armed":
                m["mode"] = "block"
                for mk in m["markers"]:
                    mk[2] = False
            return


def render(m, k):
    items = []
    for o in m["scenery"]:
        items.append(dict(o))
    w = bar_width(k)
    if w > 0:
        base = {"type": "wall", "tags": ["color_0", "wall"], "h": 1, "w": w, "layer": 0, "visible": True}
        items.append(dict(base, x=64 - w, y=0, _p="wall_0"))
        items.append(dict(base, x=0, y=63, _p="wall_0"))
    for mk in m["markers"]:
        c = "11" if mk[2] else "9"
        items.append({"type": "marker", "tags": ["color_" + c, "active" if mk[2] else "inactive", "marker"],
                      "x": mk[0], "y": mk[1], "w": 2, "h": 2, "layer": 1, "visible": True, "_p": "marker_" + c})
    out = []
    if m["mode"] == "block":
        for i, b in enumerate(m["blocks"]):
            out.append({"name": "block_left" if i == m["left"] else "block_right", "type": "player",
                        "tags": ["cyan", "block", "player"], "x": b[0], "y": b[1], "w": 4, "h": 4,
                        "layer": 1, "visible": True, "pixels": [r[:] for r in BLOCK_PIX]})
        start = 2
    else:
        for b in m["blocks"]:
            items.append({"type": "player", "tags": ["color_1", "armed", "player"], "x": b[0], "y": b[1],
                          "w": 4, "h": 4, "layer": 1, "visible": True, "_p": "player_1"})
        start = 0
    items.sort(key=lambda o: (o["y"], o["x"]))
    for r, o in enumerate(items):
        if "_p" in o:
            prefix = o.pop("_p")
        else:
            prefix = o["name"].rsplit("_", 1)[0]
        o["name"] = "%s_%d" % (prefix, start + r)
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    if _last["out"] is not None and canon(state) == canon(_last["out"]):
        k = _last["k"]
    else:
        k = (7 * m["w"] + 2) // 3
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(m, action["x"], action["y"])
    elif action in MOVES:
        dx, dy = MOVES[action]
        if m["mode"] == "block":
            move_blocks(m, dx, dy)
        else:
            move_marker(m, dx, dy)
    k += 1
    out = render(m, k)
    _last["out"] = json.loads(json.dumps(out))
    _last["k"] = k
    return out
