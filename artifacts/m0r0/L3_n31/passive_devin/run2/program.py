# Mechanics: mirrored cyan blocks (A1/A2 both y-/+4, A3 apart, A4 together) each blocked by bounds, maze cells and marker cells;
# click a 2x2 marker -> blocks become armed color_1 players and that marker turns active (color_11); A1-A4 then steer the active
# marker by 4 (blocked by maze, other markers, players, bounds); click inactive marker = switch, click player = disarm, else no-op.
# Timer bars color_0 (top-right and bottom-left) have w=floor(3(a+1)/7), a = actions since level start (hidden, continuity-gated).
# Names = type_color_rank by (y,x) (block_left/right reserve 0,1). Hypothesis: maze cells beyond the 3 observed are unknown.
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
STEP = 4
_last = {"out": None, "a": 0}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _bar_w(a):
    return (3 * (a + 1)) // 7


def _fallback_a(w):
    return 0 if w <= 0 else (7 * w - 1) // 3


def parse(state):
    m = {"blocks": None, "players": None, "markers": [], "scenery": [], "bar": 0}
    for o in state:
        t, tags = o.get("type"), o.get("tags", [])
        if t == "player" and "block" in tags:
            m["blocks"] = m["blocks"] or {}
            m["blocks"]["left" if o["name"] == "block_left" else "right"] = [o["x"], o["y"]]
        elif t == "player":
            m["players"] = (m["players"] or []) + [[o["x"], o["y"]]]
        elif t == "marker":
            m["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in tags})
        elif t == "wall" and "color_0" in tags:
            m["bar"] = max(m["bar"], o["w"])
        else:
            m["scenery"].append(o)
    return m


def marker_cell(mk):
    return (mk["x"] - 1, mk["y"] - 1)


def in_bounds(c):
    return 2 <= c[0] <= 58 and 2 <= c[1] <= 58


def block_free(side, c, m):
    lo, hi = (2, 26) if side == "left" else (34, 58)
    if not (lo <= c[0] <= hi and 2 <= c[1] <= 58):
        return False
    return c not in MAZE and all(marker_cell(k) != c for k in m["markers"])


def marker_free(c, m, me):
    if not in_bounds(c) or c in MAZE:
        return False
    if any(marker_cell(k) == c for k in m["markers"] if k is not me):
        return False
    return all(tuple(p) != c for p in (m["players"] or []))


DELTA = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}


def move(m, act):
    if act not in DELTA:
        return
    dx, dy = DELTA[act]
    if m["blocks"] is not None:
        for side, sgn in (("left", 1), ("right", -1)):
            if side not in m["blocks"]:
                continue
            x, y = m["blocks"][side]
            c = (x + dx * sgn, y + dy)
            if block_free(side, c, m):
                m["blocks"][side] = [c[0], c[1]]
    else:
        act_m = next((k for k in m["markers"] if k["active"]), None)
        if act_m is None:
            return
        c = (act_m["x"] - 1 + dx, act_m["y"] - 1 + dy)
        if marker_free(c, m, act_m):
            act_m["x"], act_m["y"] = c[0] + 1, c[1] + 1


def click(m, x, y):
    hit = next((k for k in m["markers"] if k["x"] <= x < k["x"] + 2 and k["y"] <= y < k["y"] + 2), None)
    if hit is not None:
        if hit["active"]:
            return
        for k in m["markers"]:
            k["active"] = k is hit
        if m["blocks"] is not None:
            m["players"] = [list(v) for v in m["blocks"].values()]
            m["blocks"] = None
        return
    if m["players"] is not None:
        for p in m["players"]:
            if p[0] <= x < p[0] + 4 and p[1] <= y < p[1] + 4:
                m["blocks"] = {("left" if q[0] < 32 else "right"): list(q) for q in m["players"]}
                m["players"] = None
                for k in m["markers"]:
                    k["active"] = False
                return


def render(m, a):
    objs = [dict(o) for o in m["scenery"]]
    w = _bar_w(a)
    if w > 0:
        objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True, "w": w, "x": 64 - w, "y": 0, "_c": 0})
        objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True, "w": w, "x": 0, "y": 63, "_c": 0})
    for k in m["markers"]:
        col = 11 if k["active"] else 9
        objs.append({"h": 2, "layer": 1, "tags": ["color_%d" % col, "active" if k["active"] else "inactive", "marker"],
                     "type": "marker", "visible": True, "w": 2, "x": k["x"], "y": k["y"], "_c": col})
    for p in m["players"] or []:
        objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player", "visible": True,
                     "w": 4, "x": p[0], "y": p[1], "_c": 1})
    out = []
    if m["blocks"] is not None:
        for side in ("left", "right"):
            if side in m["blocks"]:
                x, y = m["blocks"][side]
                out.append({"h": 4, "layer": 1, "name": "block_" + side, "pixels": [[10] * 4 for _ in range(4)],
                            "tags": ["cyan", "block", "player"], "type": "player", "visible": True, "w": 4, "x": x, "y": y})
        start = 2
    else:
        start = 0
    objs.sort(key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(objs):
        if "_c" in o:
            c = o.pop("_c")
        else:
            c = int(next(t for t in o["tags"] if t.startswith("color_"))[6:])
        o["name"] = "%s_%d_%d" % (o["type"], c, start + i)
        out.append(o)
    return out


def transition_function(state, action):
    m = parse(state)
    if _last["out"] is not None and _canon(state) == _last["out"]:
        a = _last["a"]
    else:
        a = _fallback_a(m["bar"])
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            click(m, action["x"], action["y"])
    else:
        move(m, action)
    a += 1
    out = render(m, a)
    _last["out"], _last["a"] = _canon(out), a
    return out
