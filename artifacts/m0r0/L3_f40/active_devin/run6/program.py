# Mechanics: block mode = two mirrored cyan 4x4 blocks; A1/A2 move both y-4/+4, A3 apart, A4 together;
# each block moves alone if its target cell is in its board half and free of maze/marker/other-block cells.
# Click marker -> armed mode (blocks -> color_1 players, marker active 11); armed A1-A4 move active marker;
# click other marker switches, click active one no-op, click player -> block mode. A5/A7 no-ops.
# Bars wall_0 width floor(3(a+1)/7), a = actions since level start (continuity-gated; fallback = smallest a). Names re-ranked by (y,x).
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
_last = {"canon": None, "a": 0}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def bar_width(a):
    return 3 * (a + 1) // 7


def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def inside(px, py, r):
    x, y, w, h = r
    return x <= px < x + w and y <= py < y + h


def maze_hit(r):
    return any(overlaps(r, (mx, my, 4, 4)) for mx, my in MAZE)


def marker_cell(m):
    return (m["x"] - 1, m["y"] - 1, 4, 4)


def parse(state):
    model = {"mode": "blocks", "pieces": [], "markers": [], "w": 0}
    for o in state:
        t = o["type"]
        if t == "player":
            model["pieces"].append([o["x"], o["y"]])
            if "armed" in o["tags"]:
                model["mode"] = "armed"
        elif t == "marker":
            model["markers"].append({"x": o["x"], "y": o["y"], "active": "active" in o["tags"]})
        elif t == "wall" and "color_0" in o["tags"] and o["y"] == 0:
            model["w"] = o["w"]
    model["pieces"].sort()
    return model


def move_blocks(model, action):
    left, right = model["pieces"]
    dy = {1: -4, 2: 4}.get(action, 0)
    dx = {3: -4, 4: 4}.get(action, 0)
    moves = [(left, dx, (2, 26)), (right, -dx, (34, 58))]
    cur = [tuple(left), tuple(right)]
    for i, (p, ddx, (lo, hi)) in enumerate(moves):
        nx, ny = p[0] + ddx, p[1] + dy
        r = (nx, ny, 4, 4)
        other = cur[1 - i]
        if not (lo <= nx <= hi and 2 <= ny <= 58):
            continue
        if maze_hit(r) or any(overlaps(r, marker_cell(m)) for m in model["markers"]):
            continue
        if overlaps(r, (other[0], other[1], 4, 4)):
            continue
        p[0], p[1] = nx, ny


def move_marker(model, action):
    act = [m for m in model["markers"] if m["active"]]
    if not act:
        return
    m = act[0]
    dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
    r = (m["x"] - 1 + dx, m["y"] - 1 + dy, 4, 4)
    if not (2 <= r[0] <= 58 and 2 <= r[1] <= 58) or maze_hit(r):
        return
    if any(overlaps(r, marker_cell(o)) for o in model["markers"] if o is not m):
        return
    if any(overlaps(r, (p[0], p[1], 4, 4)) for p in model["pieces"]):
        return
    m["x"] += dx
    m["y"] += dy


def click(model, cx, cy):
    hit = [m for m in model["markers"] if inside(cx, cy, (m["x"], m["y"], 2, 2))]
    if hit:
        if hit[0]["active"]:
            return
        for m in model["markers"]:
            m["active"] = m is hit[0]
        model["mode"] = "armed"
        return
    if model["mode"] == "armed" and any(inside(cx, cy, (p[0], p[1], 4, 4)) for p in model["pieces"]):
        model["mode"] = "blocks"
        for m in model["markers"]:
            m["active"] = False


def render(model, a):
    objs = []
    w = bar_width(a)
    if w > 0:
        objs.append(dict(h=1, layer=0, tags=["color_0", "wall"], type="wall", w=w, x=64 - w, y=0, _p="wall_0"))
        objs.append(dict(h=1, layer=0, tags=["color_0", "wall"], type="wall", w=w, x=0, y=63, _p="wall_0"))
    objs.append(dict(h=62, layer=0, tags=["color_15", "wall"], type="wall", w=32, x=0, y=1, _p="wall_15"))
    objs.append(dict(h=62, layer=0, tags=["color_8", "hazard"], type="hazard", w=32, x=32, y=1, _p="hazard_8"))
    for m in model["markers"]:
        if m["active"]:
            tags, p = ["color_11", "active", "marker"], "marker_11"
        else:
            tags, p = ["color_9", "inactive", "marker"], "marker_9"
        objs.append(dict(h=2, layer=1, tags=tags, type="marker", w=2, x=m["x"], y=m["y"], _p=p))
    blocks = []
    for p in model["pieces"]:
        if model["mode"] == "armed":
            objs.append(dict(h=4, layer=1, tags=["color_1", "armed", "player"], type="player",
                             w=4, x=p[0], y=p[1], _p="player_1"))
        else:
            blocks.append(dict(h=4, layer=1, tags=["cyan", "block", "player"], type="player", w=4,
                               x=p[0], y=p[1], pixels=[[10] * 4 for _ in range(4)]))
    out = []
    if blocks:
        blocks.sort(key=lambda b: b["x"])
        blocks[0]["name"], blocks[1]["name"] = "block_left", "block_right"
        out.extend(blocks)
    start = len(blocks)
    objs.sort(key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(objs):
        o["name"] = "%s_%d" % (o.pop("_p"), start + i)
        out.append(o)
    for o in out:
        o["visible"] = True
    return out


def fallback_a(w):
    a = 0
    while bar_width(a) < w:
        a += 1
    return a


def transition_function(state, action):
    model = parse(state)
    c = canon(state)
    a = _last["a"] if c == _last["canon"] else fallback_a(model["w"])
    if isinstance(action, dict):
        click(model, action.get("x", -1), action.get("y", -1))
    elif action in (1, 2, 3, 4):
        if model["mode"] == "blocks":
            move_blocks(model, action)
        else:
            move_marker(model, action)
    a += 1
    out = render(model, a)
    _last["canon"] = canon(out)
    _last["a"] = a
    return out
