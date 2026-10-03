# Mechanics: mirror-blocks game. Block mode: two cyan 4x4 blocks; A1/A2 move both y-4/+4, A3 apart, A4 together
# (mirrored x); each block moves alone if its target cell stays in its board half and is not a maze/marker/other-block cell.
# Click inactive marker -> armed mode (blocks -> color_1 players, marker active 11); armed A1-A4 move the active marker by 4;
# click another marker switches; click a player -> block mode; other clicks/A5 no-op. Timer bars w=floor(3(a+1)/7), a=actions.
# Hypothesis: maze cells {(6,18),(14,38),(38,14)} inferred from blocked moves; unseen maze cells unknown; renames are re-extraction.
import json

MAZE = {(6, 18), (14, 38), (38, 14)}
_last = {"state": None, "a": 0}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _bar_w(a):
    return (3 * (a + 1)) // 7


def _parse(state):
    m = {"statics": [], "markers": [], "blocks": [], "armed": False, "bar": 0}
    for o in state:
        t = o["type"]
        if t == "player":
            if "armed" in o["tags"]:
                m["armed"] = True
            m["blocks"].append([o["x"], o["y"]])
        elif t == "marker":
            m["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in o["tags"]})
        elif t == "wall" and "color_0" in o["tags"]:
            if o["y"] == 0:
                m["bar"] = o["w"]
        else:
            m["statics"].append(dict(o))
    m["blocks"].sort()
    return m


def _cell_free(cell, m, skip_marker=None, skip_block=None):
    if cell in MAZE:
        return False
    for i, mk in enumerate(m["markers"]):
        if i != skip_marker and (mk["x"] - 1, mk["y"] - 1) == cell:
            return False
    for i, b in enumerate(m["blocks"]):
        if i != skip_block and tuple(b) == cell:
            return False
    return True


def _move_blocks(m, action):
    dy = {1: -4, 2: 4}.get(action, 0)
    for i, b in enumerate(m["blocks"]):
        left = b[0] < 32
        dx = 0
        if action == 3:
            dx = -4 if left else 4
        elif action == 4:
            dx = 4 if left else -4
        nx, ny = b[0] + dx, b[1] + dy
        lo, hi = (2, 26) if left else (34, 58)
        if lo <= nx <= hi and 2 <= ny <= 58 and _cell_free((nx, ny), m, skip_block=i):
            m["blocks"][i] = [nx, ny]


def _move_marker(m, action):
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    for i, mk in enumerate(m["markers"]):
        if mk["active"]:
            cx, cy = mk["x"] - 1 + dx, mk["y"] - 1 + dy
            if 2 <= cx <= 58 and 2 <= cy <= 58 and _cell_free((cx, cy), m, skip_marker=i):
                mk["x"] += dx
                mk["y"] += dy


def _hit(o, x, y, w, h):
    return o[0] <= x < o[0] + w and o[1] <= y < o[1] + h


def _click(m, x, y):
    for i, mk in enumerate(m["markers"]):
        if _hit((mk["x"], mk["y"]), x, y, 2, 2):
            if not mk["active"]:
                for other in m["markers"]:
                    other["active"] = False
                mk["active"] = True
                m["armed"] = True
            return
    if m["armed"]:
        for b in m["blocks"]:
            if _hit(b, x, y, 4, 4):
                m["armed"] = False
                for mk in m["markers"]:
                    mk["active"] = False
                return


def _render(m):
    objs = []
    w = m["bar"]
    if w > 0:
        for bx, by in ((64 - w, 0), (0, 63)):
            objs.append({"type": "wall", "tags": ["color_0", "wall"], "x": bx, "y": by, "w": w, "h": 1,
                         "layer": 0, "visible": True, "_c": "0"})
    for s in m["statics"]:
        s = dict(s)
        s["_c"] = s["tags"][0].split("_")[1]
        objs.append(s)
    for mk in m["markers"]:
        c = "11" if mk["active"] else "9"
        objs.append({"type": "marker", "tags": ["color_" + c, "active" if mk["active"] else "inactive", "marker"],
                     "x": mk["x"], "y": mk["y"], "w": 2, "h": 2, "layer": 1, "visible": True, "_c": c})
    blocks = []
    for bx, by in m["blocks"]:
        if m["armed"]:
            objs.append({"type": "player", "tags": ["color_1", "armed", "player"], "x": bx, "y": by,
                         "w": 4, "h": 4, "layer": 1, "visible": True, "_c": "1"})
        else:
            blocks.append({"name": "block_left" if bx < 32 else "block_right", "type": "player",
                           "tags": ["cyan", "block", "player"], "x": bx, "y": by, "w": 4, "h": 4,
                           "pixels": [[10] * 4 for _ in range(4)], "layer": 1, "visible": True})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    start = len(blocks)
    for k, o in enumerate(objs):
        o["name"] = "%s_%s_%d" % (o["type"], o.pop("_c"), start + k)
    return blocks + objs


def transition_function(state, action):
    m = _parse(state)
    if _last["state"] is not None and _canon(state) == _last["state"]:
        a = _last["a"]
    else:
        a = (7 * m["bar"] - 1) // 3 if m["bar"] > 0 else 0
    if isinstance(action, dict):
        _click(m, action["x"], action["y"])
    elif action in (1, 2, 3, 4):
        if m["armed"]:
            _move_marker(m, action)
        else:
            _move_blocks(m, action)
    a += 1
    m["bar"] = _bar_w(a)
    out = _render(m)
    _last["state"] = _canon(out)
    _last["a"] = a
    return out
