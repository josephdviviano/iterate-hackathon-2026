# Mechanics: blocks (block mode) move together: A1/A2 y-/+4, A3 apart (left x-4, right x+4), A4 together; each moves alone unless its target cell is out of [2,58], a maze cell, a marker cell (marker xy-1) or the other block.
# Click an inactive marker -> blocks become armed players and that marker is active; click a player -> back to blocks; click active marker/block/empty -> no-op. A5 no-op.
# Armed: A1-A4 move the active marker by 4 (cell = xy-1), blocked by bounds, maze, other markers, player cells. Players stay put.
# Timer bars (color_0, top-right x=64-w,y=0 and bottom-left x=0,y=63) have w=floor(3(a+1)/7), a = actions since level start (hidden; continuity-gated, fallback a=ceil(7w/3)).
# Names = type_color_rank by (y,x); in block mode block_left/right take ranks 0,1. Hypothesis: invisible maze {(14,38),(6,18),(38,14),(22,18)} inferred from blocked moves only (step 124 = (22,18)).
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
BLOCK_PIX = [[10] * 4 for _ in range(4)]
_mem = {"last": None, "a": 0, "left_pos": None}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _in_bounds(c):
    return LO <= c[0] <= HI and LO <= c[1] <= HI


def parse(state):
    m = {"armed": False, "blocks": {}, "players": [], "markers": [], "scenery": [], "w": 0}
    for o in state:
        t = o["type"]
        if t == "player":
            if "block" in o["tags"]:
                side = "left" if o["name"] == "block_left" else "right"
                m["blocks"][side] = (o["x"], o["y"])
            else:
                m["armed"] = True
                m["players"].append((o["x"], o["y"]))
        elif t == "marker":
            m["markers"].append([o["x"], o["y"], "active" in o["tags"]])
        elif t == "wall" and "color_0" in o["tags"]:
            m["w"] = max(m["w"], o["w"])
        else:
            m["scenery"].append(dict(o))
    return m


def marker_cell(mk):
    return (mk[0] - 1, mk[1] - 1)


def step_blocks(m, action):
    dx = {3: -4, 4: 4}.get(action, 0)
    dy = {1: -4, 2: 4}.get(action, 0)
    if dx == 0 and dy == 0:
        return
    blocked = MAZE | {marker_cell(mk) for mk in m["markers"]}
    old = dict(m["blocks"])
    for side, (x, y) in old.items():
        ddx = dx if side == "left" else -dx
        tgt = (x + ddx, y + dy)
        others = {p for s, p in old.items() if s != side}
        if _in_bounds(tgt) and tgt not in blocked and tgt not in others:
            m["blocks"][side] = tgt


def step_marker(m, action):
    d = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(action)
    if d is None:
        return
    for mk in m["markers"]:
        if mk[2]:
            tgt = (mk[0] - 1 + d[0], mk[1] - 1 + d[1])
            blocked = MAZE | {marker_cell(o) for o in m["markers"] if o is not mk} | set(m["players"])
            if _in_bounds(tgt) and tgt not in blocked:
                mk[0] += d[0]
                mk[1] += d[1]


def hit(x, y, ox, oy, size):
    return ox <= x < ox + size and oy <= y < oy + size


def click(m, x, y, left_pos):
    for mk in m["markers"]:
        if hit(x, y, mk[0], mk[1], 2):
            if mk[2]:
                return left_pos
            for o in m["markers"]:
                o[2] = o is mk
            if not m["armed"]:
                m["armed"] = True
                left_pos = m["blocks"]["left"]
                m["players"] = [left_pos, m["blocks"]["right"]]
                m["blocks"] = {}
            return left_pos
    if m["armed"]:
        for p in m["players"]:
            if hit(x, y, p[0], p[1], 4):
                ps = sorted(m["players"])
                if left_pos not in ps:
                    left_pos = ps[0]
                l = left_pos
                r = ps[1] if ps[0] == l else ps[0]
                m["blocks"] = {"left": l, "right": r}
                m["players"] = []
                m["armed"] = False
                for o in m["markers"]:
                    o[2] = False
                return None
    return left_pos


def render(m, a):
    objs = []
    for o in m["scenery"]:
        objs.append(dict(o))
    w = 3 * (a + 1) // 7
    if w > 0:
        base = {"type": "wall", "tags": ["color_0", "wall"], "h": 1, "w": w, "layer": 0, "visible": True}
        objs.append(dict(base, x=64 - w, y=0, color=0))
        objs.append(dict(base, x=0, y=63, color=0))
    for mk in m["markers"]:
        c = 11 if mk[2] else 9
        objs.append({"type": "marker", "tags": ["color_%d" % c, "active" if mk[2] else "inactive", "marker"],
                     "x": mk[0], "y": mk[1], "w": 2, "h": 2, "layer": 1, "visible": True, "color": c})
    for p in m["players"]:
        objs.append({"type": "player", "tags": ["color_1", "armed", "player"], "x": p[0], "y": p[1],
                     "w": 4, "h": 4, "layer": 1, "visible": True, "color": 1})
    start = 0 if m["armed"] else 2
    objs.sort(key=lambda o: (o["y"], o["x"]))
    out = []
    for i, o in enumerate(objs):
        if "color" in o:
            c = o.pop("color")
        else:
            c = int(o["tags"][0].split("_")[1])
        o["name"] = "%s_%d_%d" % (o["type"], c, start + i)
        out.append(o)
    for side, (x, y) in m["blocks"].items():
        out.append({"name": "block_" + side, "type": "player", "tags": ["cyan", "block", "player"],
                    "x": x, "y": y, "w": 4, "h": 4, "pixels": [r[:] for r in BLOCK_PIX],
                    "layer": 1, "visible": True})
    return out


def transition_function(state, action):
    m = parse(state)
    if _mem["last"] is not None and _canon(state) == _mem["last"]:
        a, left_pos = _mem["a"], _mem["left_pos"]
    else:
        a = 0 if m["w"] == 0 else (7 * m["w"] + 2) // 3
        left_pos = None
    if isinstance(action, dict):
        left_pos = click(m, action["x"], action["y"], left_pos)
    elif m["armed"]:
        step_marker(m, action)
    else:
        step_blocks(m, action)
    a += 1
    out = render(m, a)
    _mem.update(last=_canon(out), a=a, left_pos=left_pos)
    return out
