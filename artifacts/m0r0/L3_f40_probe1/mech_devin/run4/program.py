# Mechanics: two cyan blocks (block mode) move together: A1/A2 y-4/+4, A3 apart (left x-4, right x+4), A4 together;
# each moves alone into a free 4x4 cell (bounds [2,58], not maze/marker cell/other block). Clicking an inactive
# marker arms (blocks -> color_1 players, marker active); armed A1-4 move the active marker 4 (blocked by maze/markers/players),
# click another marker switches, click a player disarms. Timer bars (color_0) w=3(a+1)//7, a = hidden action count.
# Hypothesis: invisible maze cells {(14,38),(6,18),(38,14),(22,18)}; (22,18) inferred only from step 124; names = (y,x) rank.
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
MOVES = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
BLOCK_PX = [[10] * 4 for _ in range(4)]
_mem = {"out": None, "a": 0, "left": None}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def color_of(o):
    for t in o.get("tags", []):
        if t.startswith("color_"):
            return t[6:]
    return "x"


def parse(state):
    m = {"scenery": [], "bar": 0, "blocks": None, "players": [], "markers": [], "active": None}
    blocks = {}
    for o in state:
        tags = o.get("tags", [])
        if o["type"] == "marker":
            m["markers"].append([o["x"], o["y"]])
            if "active" in tags:
                m["active"] = len(m["markers"]) - 1
        elif "block" in tags:
            blocks[o["name"]] = (o["x"], o["y"])
        elif "armed" in tags:
            m["players"].append((o["x"], o["y"]))
        elif o["type"] == "wall" and color_of(o) == "0":
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["scenery"].append(dict(o))
    if blocks:
        m["blocks"] = [blocks.get("block_left"), blocks.get("block_right")]
    return m


def marker_cell(mk):
    return (mk[0] - 1, mk[1] - 1)


def in_bounds(c):
    return LO <= c[0] <= HI and LO <= c[1] <= HI


def hit(x, y, ox, oy, size):
    return ox <= x < ox + size and oy <= y < oy + size


def step_blocks(m, act):
    dx, dy = MOVES[act]
    deltas = [(dx, dy), (dx, dy)] if dx == 0 else [(dx, 0), (-dx, 0)]
    mcells = {marker_cell(mk) for mk in m["markers"]}
    cur = list(m["blocks"])
    new = list(cur)
    for i in (0, 1):
        t = (cur[i][0] + deltas[i][0], cur[i][1] + deltas[i][1])
        other = cur[1 - i]
        if in_bounds(t) and t not in MAZE and t not in mcells and t != other:
            new[i] = t
    m["blocks"] = new


def step_marker(m, act):
    if m["active"] is None:
        return
    dx, dy = MOVES[act]
    mk = m["markers"][m["active"]]
    t = (mk[0] - 1 + dx, mk[1] - 1 + dy)
    others = {marker_cell(k) for j, k in enumerate(m["markers"]) if j != m["active"]}
    if in_bounds(t) and t not in MAZE and t not in others and t not in set(m["players"]):
        m["markers"][m["active"]] = [mk[0] + dx, mk[1] + dy]


def click(m, x, y, left):
    for j, mk in enumerate(m["markers"]):
        if hit(x, y, mk[0], mk[1], 2):
            if m["blocks"] is not None:
                left = m["blocks"][0]
                m["players"] = list(m["blocks"])
                m["blocks"] = None
                m["active"] = j
            elif j != m["active"]:
                m["active"] = j
            return left
    if m["blocks"] is None:
        for p in m["players"]:
            if hit(x, y, p[0], p[1], 4):
                ps = sorted(m["players"])
                if left in ps:
                    ps.remove(left)
                    ps = [left] + ps
                m["blocks"] = ps[:2]
                m["players"] = []
                m["active"] = None
                return left
    return left


def render(m, a):
    w = 3 * (a + 1) // 7
    objs = []
    for o in m["scenery"]:
        objs.append(dict(o))
    if w > 0:
        for (bx, by) in ((64 - w, 0), (0, 63)):
            objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                         "visible": True, "w": w, "x": bx, "y": by, "_c": "0"})
    for j, mk in enumerate(m["markers"]):
        act = m["blocks"] is None and j == m["active"]
        c = "11" if act else "9"
        objs.append({"h": 2, "layer": 1, "tags": ["color_" + c, "active" if act else "inactive", "marker"],
                     "type": "marker", "visible": True, "w": 2, "x": mk[0], "y": mk[1]})
    for p in m["players"]:
        objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                     "visible": True, "w": 4, "x": p[0], "y": p[1]})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    start = 2 if m["blocks"] is not None else 0
    out = []
    for i, o in enumerate(objs):
        o.pop("_c", None)
        o["name"] = "%s_%s_%d" % (o["type"], color_of(o), start + i)
        out.append(o)
    if m["blocks"] is not None:
        for nm, b in zip(("block_left", "block_right"), m["blocks"]):
            out.append({"h": 4, "layer": 1, "name": nm, "pixels": [r[:] for r in BLOCK_PX],
                        "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                        "w": 4, "x": b[0], "y": b[1]})
    return out


def transition_function(state, action):
    m = parse(state)
    if _mem["out"] is not None and canon(state) == _mem["out"]:
        a, left = _mem["a"], _mem["left"]
    else:
        a = (7 * m["bar"] + 2) // 3 if m["bar"] > 0 else 0
        left = min(m["players"]) if m["players"] else None
    if isinstance(action, dict):
        left = click(m, action["x"], action["y"], left)
    elif action in MOVES:
        if m["blocks"] is not None:
            step_blocks(m, action)
        else:
            step_marker(m, action)
    a += 1
    out = render(m, a)
    _mem["out"], _mem["a"], _mem["left"] = canon(out), a, left
    return out
