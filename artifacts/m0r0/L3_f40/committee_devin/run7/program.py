# Mechanics implemented:
# - Two cyan 4x4 blocks move on a 4px grid: action1/2 = y-4/y+4, 3 = apart, 4 = together.
# - Clicking an inactive marker arms it (color_11) and turns blocks into players; 1-4 then
#   move only the active marker. Clicking a player reverts to blocks; other clicks are no-ops.
# - Two 1px bars grow as w=floor(3*(n+1)/7), n=actions this level (hidden counter, reset on
#   discontinuity, re-inferred from bar width). Maze cells {(6,18),(14,38),(38,14)} + entity cells block moves.
# - Names re-ranked by (y,x): block mode -> blocks take 0,1, rest start at 2; armed mode -> all from 0.

import copy
import json

MAZE = {(6, 18), (14, 38), (38, 14)}
PIX = [[10, 10, 10, 10] for _ in range(4)]
_last = None
_a = 0


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _cls(o):
    tags = o.get("tags", [])
    if o["type"] == "player":
        return "player" if "armed" in tags else "block"
    if o["type"] == "marker":
        return "marker"
    if o["type"] == "wall":
        return "bar" if "color_0" in tags else "wall"
    return o["type"]


def _mk_block(x, y, name):
    return {"h": 4, "layer": 1, "name": name, "pixels": copy.deepcopy(PIX),
            "tags": ["cyan", "block", "player"], "type": "player",
            "visible": True, "w": 4, "x": x, "y": y}


def _mk_player(x, y):
    return {"h": 4, "layer": 1, "name": "player_1_0", "tags": ["color_1", "armed", "player"],
            "type": "player", "visible": True, "w": 4, "x": x, "y": y}


def _mk_marker(x, y, active):
    c = "11" if active else "9"
    st = "active" if active else "inactive"
    return {"h": 2, "layer": 1, "name": "marker_%s_0" % c,
            "tags": ["color_%s" % c, st, "marker"], "type": "marker",
            "visible": True, "w": 2, "x": x, "y": y}


def _mk_wall(x, y):
    return {"h": 62, "layer": 0, "name": "wall_15_0", "tags": ["color_15", "wall"],
            "type": "wall", "visible": True, "w": 32, "x": x, "y": y}


def _mk_hazard(x, y):
    return {"h": 62, "layer": 0, "name": "hazard_8_0", "tags": ["color_8", "hazard"],
            "type": "hazard", "visible": True, "w": 32, "x": x, "y": y}


def _mk_bar(x, y, w):
    return {"h": 1, "layer": 0, "name": "wall_0_0", "tags": ["color_0", "wall"],
            "type": "wall", "visible": True, "w": w, "x": x, "y": y}


def _rank_name(o):
    color = "0"
    for t in o["tags"]:
        if t.startswith("color_"):
            color = t.split("_", 1)[1]
    return "%s_%s" % (o["type"], color)


def transition_function(state, action):
    global _last, _a
    if _last is None or _canon(state) != _last:
        bars = [o for o in state if _cls(o) == "bar"]
        w = bars[0]["w"] if bars else 0
        _a = (7 * w - 1) // 3 if w else 0

    blocks = sorted((o for o in state if _cls(o) == "block"), key=lambda o: (o["x"], o["y"]))
    players = [o for o in state if _cls(o) == "player"]
    markers = [dict(o) for o in state if _cls(o) == "marker"]
    walls = [o for o in state if _cls(o) == "wall"]
    hazards = [o for o in state if _cls(o) == "hazard"]
    armed = bool(players)

    blocks = [dict(o) for o in blocks]
    players = [dict(o) for o in players]
    marker_cells = {(m["x"] - 1, m["y"] - 1) for m in markers}

    def free_cell(cx, cy):
        return 2 <= cx <= 58 and 2 <= cy <= 58 and (cx, cy) not in MAZE

    if isinstance(action, dict):
        cx, cy = action["x"], action["y"]

        def inside(o):
            return o["x"] <= cx < o["x"] + o["w"] and o["y"] <= cy < o["y"] + o["h"]

        hit_marker = next((m for m in markers if inside(m)), None)
        hit_player = next((p for p in players if inside(p)), None)
        hit_block = next((b for b in blocks if inside(b)), None)
        if hit_marker is not None and "active" not in hit_marker["tags"]:
            for m in markers:
                m["active"] = m is hit_marker
            if not armed:
                players = [_mk_player(b["x"], b["y"]) for b in blocks]
                blocks = []
                armed = True
        elif hit_marker is None and hit_player is not None:
            ps = sorted(players, key=lambda p: (p["x"], p["y"]))
            blocks = [_mk_block(ps[0]["x"], ps[0]["y"], "block_left"),
                      _mk_block(ps[1]["x"], ps[1]["y"], "block_right")]
            players = []
            armed = False
            for m in markers:
                m["active"] = False
        # clicks on the active marker, a block, or empty space are no-ops
    elif not armed:
        for i, b in enumerate(blocks):
            if action == 1:
                nx, ny = b["x"], b["y"] - 4
            elif action == 2:
                nx, ny = b["x"], b["y"] + 4
            elif action == 3:
                nx, ny = b["x"] - 4 if i == 0 else b["x"] + 4, b["y"]
            elif action == 4:
                nx, ny = b["x"] + 4 if i == 0 else b["x"] - 4, b["y"]
            else:
                continue
            if free_cell(nx, ny) and (nx, ny) not in marker_cells:
                b["x"], b["y"] = nx, ny
    else:
        active_marker = next((m for m in markers if m.get("active") or "active" in m["tags"]), None)
        d = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(action)
        if active_marker is not None and d is not None:
            ncx = active_marker["x"] - 1 + d[0]
            ncy = active_marker["y"] - 1 + d[1]
            own = (active_marker["x"] - 1, active_marker["y"] - 1)
            obstacles = (marker_cells - {own}) | {(p["x"], p["y"]) for p in players}
            if free_cell(ncx, ncy) and (ncx, ncy) not in obstacles:
                active_marker["x"], active_marker["y"] = ncx + 1, ncy + 1

    _a += 1
    bar_w = (3 * (_a + 1)) // 7

    out = []
    if armed:
        out.extend(players)
    else:
        out.extend(blocks)
    for m in markers:
        act = m.get("active")
        if act is None:
            act = "active" in m["tags"]
        out.append(_mk_marker(m["x"], m["y"], act))
    for wl in walls:
        out.append(_mk_wall(wl["x"], wl["y"]))
    for hz in hazards:
        out.append(_mk_hazard(hz["x"], hz["y"]))
    if bar_w >= 1:
        out.append(_mk_bar(64 - bar_w, 0, bar_w))
        out.append(_mk_bar(0, 63, bar_w))

    if armed:
        ranked = sorted(out, key=lambda o: (o["y"], o["x"]))
        for i, o in enumerate(ranked):
            o["name"] = "%s_%d" % (_rank_name(o), i)
    else:
        rest = sorted((o for o in out if o["type"] != "player"), key=lambda o: (o["y"], o["x"]))
        for i, o in enumerate(rest):
            o["name"] = "%s_%d" % (_rank_name(o), i + 2)

    _last = _canon(out)
    return out
