# Mechanics: two cyan blocks (left/right half) move together: A1/A2 y-4/+4, A3 apart, A4 together; each is
# blocked by its half bounds, hidden maze cells and marker cells. Clicking an inactive marker arms (blocks ->
# color_1 players, marker active); armed A1-A4 steer the active marker by 4 (blocked by maze/markers/players);
# clicking a player disarms; clicks on the active marker/blocks/empty are no-ops. Timer bars w=floor(3(a+1)/7).
# Hypotheses unconfirmed: maze cells beyond {(14,38),(6,18),(38,14)} are unknown; names = type_color_rank by (y,x).
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
MOVES = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_last = {"canon": None, "a": 0}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _bar_width(a):
    return (3 * (a + 1)) // 7


def _parse(state):
    blocks, players, markers, scenery, bar_w = [], [], [], [], 0
    for o in state:
        tags = o.get("tags", [])
        if o["type"] == "player":
            (blocks if "block" in tags else players).append([o["x"], o["y"]])
        elif o["type"] == "marker":
            markers.append({"x": o["x"], "y": o["y"], "active": "active" in tags})
        elif "color_0" in tags:
            if o["y"] == 0:
                bar_w = o["w"]
        else:
            scenery.append(o)
    return blocks, players, markers, scenery, bar_w


def _cell_free(cell, occupied):
    return cell not in MAZE and cell not in occupied


def _marker_cells(markers, skip=None):
    return {(m["x"] - 1, m["y"] - 1) for i, m in enumerate(markers) if i != skip}


def _move_blocks(blocks, action, markers):
    dx, dy = MOVES[action]
    blocks = sorted(blocks)
    occupied = _marker_cells(markers)
    out = []
    for side, (x, y) in enumerate(blocks):
        bdx = dx if side == 0 else -dx
        if action in (1, 2):
            bdx = 0
        nx, ny = x + bdx, y + dy
        lo, hi = (2, 26) if side == 0 else (34, 58)
        if lo <= nx <= hi and 2 <= ny <= 58 and _cell_free((nx, ny), occupied):
            out.append([nx, ny])
        else:
            out.append([x, y])
    return out


def _move_marker(markers, players, action):
    dx, dy = MOVES[action]
    for i, m in enumerate(markers):
        if not m["active"]:
            continue
        nx, ny = m["x"] + dx, m["y"] + dy
        occupied = _marker_cells(markers, skip=i) | {tuple(p) for p in players}
        cell = (nx - 1, ny - 1)
        if 2 <= cell[0] <= 58 and 2 <= cell[1] <= 58 and _cell_free(cell, occupied):
            m["x"], m["y"] = nx, ny


def _hit(x, y, ox, oy, size):
    return ox <= x < ox + size and oy <= y < oy + size


def _click(blocks, players, markers, cx, cy):
    for m in markers:
        if _hit(cx, cy, m["x"], m["y"], 2):
            if m["active"]:
                return blocks, players
            for k in markers:
                k["active"] = k is m
            if blocks:
                return [], blocks
            return blocks, players
    for p in players:
        if _hit(cx, cy, p[0], p[1], 4):
            for k in markers:
                k["active"] = False
            return players, []
    return blocks, players


def _render(blocks, players, markers, scenery, bar_w):
    others = [dict(o) for o in scenery]
    if bar_w > 0:
        base = {"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall", "visible": True, "w": bar_w}
        others.append(dict(base, x=64 - bar_w, y=0))
        others.append(dict(base, x=0, y=63))
    for m in markers:
        c = "color_11" if m["active"] else "color_9"
        others.append({"h": 2, "layer": 1, "tags": [c, "active" if m["active"] else "inactive", "marker"],
                       "type": "marker", "visible": True, "w": 2, "x": m["x"], "y": m["y"]})
    for p in players:
        others.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                       "visible": True, "w": 4, "x": p[0], "y": p[1]})
    out = []
    start = 0
    if blocks:
        start = 2
        px = [[10] * 4 for _ in range(4)]
        for name, (x, y) in zip(("block_left", "block_right"), sorted(blocks)):
            out.append({"h": 4, "layer": 1, "name": name, "pixels": [r[:] for r in px],
                        "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                        "w": 4, "x": x, "y": y})
    others.sort(key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(others):
        prefix = "hazard" if o["type"] == "hazard" else o["type"]
        o["name"] = "%s_%s_%d" % (prefix, o["tags"][0].split("_", 1)[1], start + i)
        out.append(o)
    return out


def transition_function(state, action):
    blocks, players, markers, scenery, bar_w = _parse(state)
    if _last["canon"] is not None and _canon(state) == _last["canon"]:
        a = _last["a"]
    else:
        a = (7 * bar_w - 1) // 3 if bar_w > 0 else 0
    a += 1
    if isinstance(action, dict):
        blocks, players = _click(blocks, players, markers, action["x"], action["y"])
    elif action in MOVES:
        if blocks:
            blocks = _move_blocks(blocks, action, markers)
        else:
            _move_marker(markers, players, action)
    out = _render(blocks, players, markers, scenery, _bar_width(a))
    _last["canon"], _last["a"] = _canon(out), a
    return out
