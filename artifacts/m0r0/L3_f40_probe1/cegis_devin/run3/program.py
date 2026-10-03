# Mechanics: two cyan 4x4 blocks move on a 4-grid (cells at x,y = 2 mod 4): A1/A2 = y-4/y+4
# for both, A3 = apart (left x-4, right x+4), A4 = together; blocked by invisible maze cells
# {(6,18),(14,38),(22,18),(38,14)} and by marker cells (marker pos - 1). Clicking a marker arms:
# blocks become color_1 players, that marker turns color_11 active and A1-4 steer it +/-4
# (same maze blocks it); clicking another inactive marker switches, a player disarms, the active
# marker or a block is a no-op; A5/A7 no-op. A timer n ticks every action; bars wall_0 appear at
# (64-w,0) and (0,63) with w = 3(n+1)//7. Names are type_color_rank by (y,x); blocks hold ranks 0,1.
# Unconfirmed: behaviour of A7, clicks on empty cells, moves into occupied cells are guessed.
import json

MAZE = {(6, 18), (14, 38), (22, 18), (38, 14)}
_last = None
_n = 0


def _canon(objs):
    return sorted(json.dumps(o, sort_keys=True) for o in objs)


def _color(o):
    for t in o.get("tags", []):
        if t.startswith("color_"):
            return t[6:]
    return "0"


def _cell(o):  # 4x4 cell top-left occupied by an object
    return (o["x"] - 1, o["y"] - 1) if o["type"] == "marker" else (o["x"], o["y"])


def _fallback_n(state):
    w = 0
    for o in state:
        if o["type"] == "wall" and "color_0" in o.get("tags", []):
            w = max(w, o.get("w", 0))
    if w == 0:
        return 0
    return (7 * w + 6) // 3 - 1  # largest n with 3(n+1)//7 == w


def _rename(objs, block_mode):
    if block_mode:
        blocks = sorted((o for o in objs if o["type"] == "player"),
                        key=lambda o: (o["x"], o["y"]))
        blocks[0]["name"] = "block_left"
        blocks[1]["name"] = "block_right"
        rest = sorted((o for o in objs if o["type"] != "player"),
                      key=lambda o: (o["y"], o["x"]))
        for i, o in enumerate(rest):
            o["name"] = "%s_%s_%d" % (o["type"], _color(o), i + 2)
    else:
        for i, o in enumerate(sorted(objs, key=lambda o: (o["y"], o["x"]))):
            o["name"] = "%s_%s_%d" % (o["type"], _color(o), i)


def _block(x, y):
    return {"name": "", "type": "player",
            "tags": ["cyan", "block", "player"], "x": x, "y": y, "w": 4,
            "h": 4, "pixels": [[10] * 4 for _ in range(4)], "layer": 1,
            "visible": True}


def _player(x, y):
    return {"name": "", "type": "player",
            "tags": ["color_1", "armed", "player"], "x": x, "y": y, "w": 4,
            "h": 4, "layer": 1, "visible": True}


def _set_active(m, on):
    m["tags"] = (["color_11", "active", "marker"] if on else
                 ["color_9", "inactive", "marker"])


def transition_function(state, action):
    global _last, _n
    objs = [dict(o) for o in state]
    if _last is None or _canon(state) != _last:
        _n = _fallback_n(state)
    players = [o for o in objs if o["type"] == "player"]
    block_mode = bool(players) and "block" in players[0]["tags"]
    markers = [o for o in objs if o["type"] == "marker"]
    active = [m for m in markers if "active" in m.get("tags", [])]
    active = active[0] if active else None
    _n += 1

    if isinstance(action, dict):  # click at cell (x, y)
        cx, cy = action["x"], action["y"]

        def hit(o):
            return o["x"] <= cx < o["x"] + o["w"] and \
                   o["y"] <= cy < o["y"] + o["h"]

        m = next((m for m in markers if hit(m)), None)
        p = next((p for p in players if hit(p)), None)
        if block_mode:
            if m is not None:  # arm: blocks -> players, marker -> active
                objs = [o for o in objs if o["type"] != "player"]
                objs += [_player(b["x"], b["y"]) for b in players]
                _set_active(m, True)
        else:
            if p is not None:  # disarm: players -> blocks, active -> inactive
                objs = [o for o in objs if o["type"] != "player"]
                objs += [_block(p["x"], p["y"]) for p in players]
                if active is not None:
                    _set_active(active, False)
            elif m is not None and m is not active:  # switch active marker
                if active is not None:
                    _set_active(active, False)
                _set_active(m, True)
            # click on active marker or empty space: no-op
    elif action in (1, 2, 3, 4):
        dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
        if block_mode:
            blocks = sorted(players, key=lambda o: (o["x"], o["y"]))
            taken = {_cell(m) for m in markers}
            for i, b in enumerate(blocks):
                d = dx if i == 0 else -dx  # A3: left-,right+; A4: left+,right-
                nx, ny = (b["x"], b["y"] + dy) if dy else (b["x"] + d, b["y"])
                if 2 <= nx <= 58 and 2 <= ny <= 58 and \
                        (nx, ny) not in MAZE and (nx, ny) not in taken:
                    b["x"], b["y"] = nx, ny
        elif active is not None:
            nx, ny = active["x"] + dx, active["y"] + dy
            cell = (nx - 1, ny - 1)
            taken = {_cell(o) for o in objs if o is not active}
            if 3 <= nx <= 59 and 3 <= ny <= 59 and \
                    cell not in MAZE and cell not in taken:
                active["x"], active["y"] = nx, ny
    # actions 5 and 7: no movement, timer still ticks

    w = 3 * (_n + 1) // 7
    objs = [o for o in objs if not (o["type"] == "wall" and
                                    "color_0" in o.get("tags", []))]
    if w:
        for bx, by in ((64 - w, 0), (0, 63)):
            objs.append({"name": "", "type": "wall",
                         "tags": ["color_0", "wall"], "x": bx, "y": by,
                         "w": w, "h": 1, "layer": 0, "visible": True})
    players = [o for o in objs if o["type"] == "player"]
    _rename(objs, bool(players) and "block" in players[0]["tags"])
    _last = _canon(objs)
    return objs
