# Mechanics: two cyan blocks (block mode) move together: A1/A2 y-/+4, A3 apart, A4 together (mirrored x);
# each moves alone iff its target 4x4 cell is in bounds [2,58] and not a maze cell, marker cell (xy-1) or the other block.
# Click marker -> arm (blocks become color_1 players, marker active color_11); armed A1-4 steer the active marker;
# click other marker = switch, click player = disarm; names = type_color_rank by (y,x); timer bars w=3(a+1)//7.
# Hypothesis: invisible maze {(14,38),(6,18),(38,14),(22,18)} (22,18 inferred only from step 124); counter a hidden.
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
_mem = {"canon": None, "a": 0, "left": None}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _bar_w(a):
    return 3 * (a + 1) // 7


def _free(cell, blocked):
    x, y = cell
    return LO <= x <= HI and LO <= y <= HI and cell not in MAZE and cell not in blocked


def _hit(o, x, y):
    return o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def parse(state):
    m = {"blocks": None, "players": [], "markers": [], "scenery": [], "bar": 0}
    blocks = {}
    for o in state:
        t = o["type"]
        if t == "player" and "block" in o["tags"]:
            blocks["left" if o["name"] == "block_left" else "right"] = (o["x"], o["y"])
        elif t == "player":
            m["players"].append((o["x"], o["y"]))
        elif t == "marker":
            m["markers"].append([o["x"], o["y"], "active" in o["tags"]])
        elif t == "wall" and "color_0" in o["tags"]:
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["scenery"].append(dict(o))
    if blocks:
        m["blocks"] = blocks
    return m


def step_blocks(m, action):
    b = m["blocks"]
    marker_cells = {(mx - 1, my - 1) for mx, my, _ in m["markers"]}
    if action in (1, 2):
        d = {"left": (0, -4 if action == 1 else 4), "right": (0, -4 if action == 1 else 4)}
    else:
        dx = -4 if action == 3 else 4
        d = {"left": (dx, 0), "right": (-dx, 0)}
    new = {}
    for k in ("left", "right"):
        other = b["right" if k == "left" else "left"]
        tgt = (b[k][0] + d[k][0], b[k][1] + d[k][1])
        new[k] = tgt if _free(tgt, marker_cells | {other}) else b[k]
    m["blocks"] = new


def step_marker(m, action):
    d = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    for mk in m["markers"]:
        if not mk[2]:
            continue
        blocked = {(x - 1, y - 1) for x, y, act in m["markers"] if not act}
        blocked |= set(m["players"])
        tgt = (mk[0] - 1 + d[0], mk[1] - 1 + d[1])
        if _free(tgt, blocked):
            mk[0], mk[1] = tgt[0] + 1, tgt[1] + 1


def click(m, x, y, left):
    for mk in m["markers"]:
        if mk[0] <= x < mk[0] + 2 and mk[1] <= y < mk[1] + 2:
            if mk[2]:
                return left
            for other in m["markers"]:
                other[2] = other is mk
            if m["blocks"] is not None:
                m["players"] = [m["blocks"]["left"], m["blocks"]["right"]]
                left = m["blocks"]["left"]
                m["blocks"] = None
            return left
    if m["blocks"] is None:
        for px, py in m["players"]:
            if px <= x < px + 4 and py <= y < py + 4:
                ps = m["players"]
                lp = left if left in ps else min(ps)
                rp = [p for p in ps if p != lp][0]
                m["blocks"] = {"left": lp, "right": rp}
                m["players"] = []
                for mk in m["markers"]:
                    mk[2] = False
                return None
    return left


def obj(name, typ, tags, x, y, w, h, layer):
    return {"name": name, "type": typ, "tags": tags, "x": x, "y": y, "w": w, "h": h,
            "layer": layer, "visible": True}


def render(m, a):
    out, others = [], []
    w = _bar_w(a)
    if w > 0:
        others.append(("wall", 0, obj("", "wall", ["color_0", "wall"], 64 - w, 0, w, 1, 0)))
        others.append(("wall", 0, obj("", "wall", ["color_0", "wall"], 0, 63, w, 1, 0)))
    for o in m["scenery"]:
        col = int([t for t in o["tags"] if t.startswith("color_")][0][6:])
        others.append((o["type"], col, o))
    for mx, my, act in m["markers"]:
        tags = ["color_11", "active", "marker"] if act else ["color_9", "inactive", "marker"]
        others.append(("marker", 11 if act else 9, obj("", "marker", tags, mx, my, 2, 2, 1)))
    if m["blocks"] is not None:
        for k in ("left", "right"):
            bx, by = m["blocks"][k]
            o = obj("block_" + k, "player", ["cyan", "block", "player"], bx, by, 4, 4, 1)
            o["pixels"] = [[10] * 4 for _ in range(4)]
            out.append(o)
        start = 2
    else:
        for px, py in m["players"]:
            others.append(("player", 1, obj("", "player", ["color_1", "armed", "player"], px, py, 4, 4, 1)))
        start = 0
    others.sort(key=lambda t: (t[2]["y"], t[2]["x"]))
    for i, (typ, col, o) in enumerate(others):
        o = dict(o)
        o["name"] = "%s_%d_%d" % (typ, col, start + i)
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    if _mem["canon"] is not None and _canon(state) == _mem["canon"]:
        a, left = _mem["a"], _mem["left"]
    else:
        a = (7 * m["bar"] + 2) // 3 if m["bar"] > 0 else 0
        left = None
    a += 1
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            left = click(m, action["x"], action["y"], left)
    elif action in (1, 2, 3, 4):
        if m["blocks"] is not None:
            step_blocks(m, action)
        else:
            step_marker(m, action)
    out = render(m, a)
    _mem.update(canon=_canon(out), a=a, left=left)
    return out
