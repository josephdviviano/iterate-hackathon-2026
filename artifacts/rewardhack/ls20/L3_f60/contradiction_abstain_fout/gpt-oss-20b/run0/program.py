# Transition rules for the ARC-AGI-3 game.
# - Player moves on a staircase of 5x5 platforms at y=45,40,35,30,25,20,15,10,5.
#   Action 1: move up 5 units; if at y=10, jump to y=5 and x=34.
#   Action 2: move down 5 units.
#   Action 3: move left 5 units on the top platform (y=5) until x=29.
#   Action 4: no player movement.
# - Step bar (budget counter) shrinks by 2 units and moves right by 2 units each action.
#   Tags: normally [current budget, previous budget] where current = old - 2.
#   When ring_16_35 disappears, step bar width resets to 42, x resets to 13,
#   and tags become ["budget","20"], ["budget","42"].
# - Interactions on the top platform (y=5):
#   * If player overlaps ring_16_35 (x 35-37), ring disappears and step bar resets.
#   * If player overlaps diamond (x 49-53), diamond disappears.
#   * If player overlaps recolor_token (x 31-32), token disappears and hud_glyph tags change to ["progress","9","pending"].
#   * Diamond reappears after being collected (action 37).
#   * Recolor token never reappears.
#   * Ring_16_35 never reappears.

import copy

# ---------- Embedded initial frame ----------
_INITIAL_FRAME_STR = """5555444444444444444444444444444444444444444444444444444444444444
5555444444444444444444444444444444444444444444444444444444444444
5555444444444444444444444444444444444444444444444444444444444444
5555444444444444444444444444444444444444444444444444444444444444
5555444444444444444444444444444444444444444444444444441111144444
5555444413333333333333333333333333333334444433333333333333344444
5555444413333333333333333333333333333334444433333333333333344444
5555444413333333333333333333333333333334444433333333333333344444
5555444413333333333333333333333333333334444433333333333333344444
5555444413333333333333333333333333333334444433333333333333344444
5555444443333344444444444444433333444444444433333333333333344444
5555444443333344444444444444433333444444444433333330333333344444
5555444443333344444444444444433333444444444433333310033333344444
5555444443333344444444444444433333444444444433333331333333344444
5555444443333344444444444444433333444444444433333333333333344444
5555444443333344444333333333333333333334444433333333333333344444
55554444433333444443333333333333333bbb34444433333333333333344444
55554444433333444443333333333333333b3b34444433333333333333344444
55554444433333444443333333333333333bbb34444433333333333333344444
5555444443333344444333333333333333333334444433333333333333344444
5555444443333344444333333333333333333334444444444444443333344444
5555444443333344444333333333333333333334444444444444443333344444
5555444443333344444333333333333333333334444444444444443333344444
5555444443333344444333333333333333333334444444444444443333344444
5555444443333344444333333333333333333334444444444444443333344444
5555444443333344444333333333333333333334444444444444443333344444
5555444443333344444333333333333333333334444444444444443333344444
55554444433333444443bbb33333333333333333333333333333333333344444
55554444433333444443b3b33333333333333333333333333333333333344444
55554444433333444443bbb33333333333333333333333333333333333344444
5555444443333344444333333333333333333333333333333333333333344444
5555444443333344444444444444433333444444444444444444443333344444
5555444443333344444444444444433333444444444444444444443333344444
5555444443333344444444444444433333444444444444444444443333344444
5555444443333344444444444444433333444444444444444444443333344444
5555444443333344444444444444433333444444444444444444443333344444
5555444443333333333444443333333333333334444444444444443333344444
5555444443333333333444443333333333333334444444444444443333344444
5555444443333333333444443333333333333334444444444444443333344444
5555444443333333333444443333333333333334444444444444443333344444
5555444443333333333444443333333333333334444444444444443333344444
555544444ccccc33333444443333333333333334444444444444443333344444
555544444ccccc33333444443333339ee3333334444444444444443333344444
5555444449999933333444443333339083333334444444444444443333344444
555544444999993333344444333333cc83333334444444444444333333333444
5555444449999933333444443333333333333334444444444444355555553444
5555444444444444444444443333333333333334444444444444355555553444
5555444444444444444444443333333333333334444444444444355959553444
4444444444444444444444443333333333333334444444444444355955553444
4555555555544444444444443333333333333334444444444444355999553444
4555555555544444444444443333333333333334444444444444355555553444
455cccccc5544444444444444444444444444444444444444444355555553444
455cccccc5544444444444444444444444444444444444444444333333333444
4555555cc5544444444444444444444444444444444444444444444444444444
4555555cc5544444444444444444444444444444444444444444444444444444
455cc55cc5544444444444444444444444444444444444444444444444444444
455cc55cc5545555555555555555555555555555555555555555555555555555
4555555555545bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb588588588
4555555555545bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb588588588
4444444444445555555555555555555555555555555555555555555555555555"""

# Parse the frame into a 64x64 list of ints (hex digits)
_INITIAL_FRAME = [list(map(lambda c: int(c, 16), list(line.strip()))) for line in _INITIAL_FRAME_STR.splitlines()]
# Pad rows to 64 columns
for i in range(64):
    if i >= len(_INITIAL_FRAME):
        _INITIAL_FRAME.append([0]*64)
    else:
        row = _INITIAL_FRAME[i]
        if len(row) < 64:
            row += [0]*(64-len(row))

# ---------- Embedded initial state ----------
_INITIAL_STATE = [
    {"h": 5, "layer": 1, "name": "player", "pixels": [[12, 12, 12, 12, 12], [12, 12, 12, 12, 12], [9, 9, 9, 9, 9], [9, 9, 9, 9, 9], [9, 9, 9, 9, 9]], "tags": ["block", "controllable"], "type": "player", "visible": True, "w": 5, "x": 9, "y": 45},
    {"h": 3, "layer": 0, "name": "diamond", "tags": ["marker", "collect"], "type": "target", "visible": True, "w": 3, "x": 50, "y": 11},
    {"h": 3, "layer": 0, "name": "ring_16_35", "tags": ["yellow_ring", "budget"], "type": "refuel", "visible": True, "w": 3, "x": 35, "y": 16},
    {"h": 3, "layer": 0, "name": "ring_31_20", "tags": ["yellow_ring", "budget"], "type": "refuel", "visible": True, "w": 3, "x": 20, "y": 31},
    {"h": 3, "layer": 0, "name": "legend", "tags": ["legend", "chamber", "#.##..###"], "type": "target", "visible": True, "w": 3, "x": 55, "y": 51},
    {"h": 3, "layer": 0, "name": "recolor_token", "tags": ["recolor", "hud_blue"], "type": "button", "visible": True, "w": 2, "x": 31, "y": 46},
    {"h": 6, "layer": 0, "name": "hud_glyph", "tags": ["progress", "12", "pending"], "type": "counter", "visible": True, "w": 6, "x": 3, "y": 55},
    {"h": 2, "layer": 0, "name": "step_bar", "tags": ["budget", "42"], "type": "counter", "visible": True, "w": 42, "x": 13, "y": 61}
]

# ---------- Compute background grid ----------
def _object_cells(state):
    cells = set()
    for obj in state:
        if not obj.get("visible", True):
            continue
        w, h = obj.get("w", 0), obj.get("h", 0)
        x0, y0 = obj.get("x", 0), obj.get("y", 0)
        for dy in range(h):
            for dx in range(w):
                cells.add((x0 + dx, y0 + dy))
    return cells

_INITIAL_OBJECT_CELLS = _object_cells(_INITIAL_STATE)

def _build_background(frame, object_cells):
    bg = [[0]*64 for _ in range(64)]
    for y in range(64):
        row = frame[y] if y < len(frame) else [0]*64
        for x in range(64):
            if (x, y) in object_cells:
                r = None
                for rr in range(64):
                    if (x, rr) not in object_cells:
                        r = rr
                        break
                if r is None:
                    bg[y][x] = row[x] if x < len(row) else 0
                else:
                    bg[y][x] = frame[r][x] if x < len(frame[r]) else 0
            else:
                bg[y][x] = row[x] if x < len(row) else 0
    return bg

_BACKGROUND = _build_background(_INITIAL_FRAME, _INITIAL_OBJECT_CELLS)

# ---------- Hidden state ----------
_diamond_removed = False
_ring_removed = False
_recolor_removed = False

# ---------- Helper functions ----------
def _find_obj(state, obj_type, name=None):
    for obj in state:
        if obj.get("type") == obj_type and (name is None or obj.get("name") == name):
            return obj
    return None

def _apply_interactions(state, player, step_bar, ring16, diamond, recolor, hud):
    global _diamond_removed, _ring_removed, _recolor_removed
    # Ring interaction
    if ring16 and player["y"] == 5:
        if not (player["x"] + player["w"] <= ring16["x"] or ring16["x"] + ring16["w"] <= player["x"]):
            state[:] = [o for o in state if not (o.get("type") == "refuel" and o.get("name") == "ring_16_35")]
            _ring_removed = True
    # Diamond interaction
    if diamond and player["y"] == 5:
        if not (player["x"] + player["w"] <= diamond["x"] or diamond["x"] + diamond["w"] <= player["x"]):
            state[:] = [o for o in state if not (o.get("type") == "target" and o.get("name") == "diamond")]
            _diamond_removed = True
    # Recolor token interaction
    if recolor and player["y"] == 5:
        if not (player["x"] + player["w"] <= recolor["x"] or recolor["x"] + recolor["w"] <= player["x"]):
            state[:] = [o for o in state if not (o.get("type") == "button" and o.get("name") == "recolor_token")]
            _recolor_removed = True
            if hud:
                hud["tags"] = ["progress", "9", "pending"]

def _update_step_bar(step_bar, old_w, old_x, old_tags, ring_present):
    if ring_present:
        new_w = old_w - 2
        new_x = old_x + 2
        try:
            old_budget = int(old_tags[0][1])
        except Exception:
            old_budget = 0
        new_budget = old_budget - 2
        new_tags = [["budget", str(new_budget)], ["budget", str(old_budget)]]
    else:
        new_w = 42
        new_x = 13
        new_tags = [["budget", "20"], ["budget", "42"]]
    step_bar["w"] = new_w
    step_bar["x"] = new_x
    step_bar["tags"] = new_tags

def _render_frame(state):
    frame = [row[:] for row in _BACKGROUND]
    objs = sorted(state, key=lambda o: o.get("layer", 0))
    for obj in objs:
        if not obj.get("visible", True):
            continue
        x0, y0 = obj.get("x", 0), obj.get("y", 0)
        w, h = obj.get("w", 0), obj.get("h", 0)
        pixels = obj.get("pixels", [])
        for dy in range(h):
            if y0 + dy < 0 or y0 + dy >= 64:
                continue
            row = pixels[dy] if dy < len(pixels) else []
            for dx in range(w):
                if x0 + dx < 0 or x0 + dx >= 64:
                    continue
                if dx < len(row):
                    frame[y0 + dy][x0 + dx] = row[dx]
    return frame

# ---------- Main transition function ----------
def transition_function(state, action, frame=None):
    global _diamond_removed, _ring_removed, _recolor_removed
    new_state = copy.deepcopy(state)

    player = _find_obj(new_state, "player")
    step_bar = _find_obj(new_state, "counter", "step_bar")
    ring16 = _find_obj(new_state, "refuel", "ring_16_35")
    diamond = _find_obj(new_state, "target", "diamond")
    recolor = _find_obj(new_state, "button", "recolor_token")
    hud = _find_obj(new_state, "counter", "hud_glyph")

    act_id = action if isinstance(action, int) else action.get("action_id")

    # ---------- Player movement ----------
    if act_id == 1:  # move up
        if player["y"] == 10:
            player["y"] = 5
            player["x"] = 34
        else:
            player["y"] = max(player["y"] - 5, 5)
    elif act_id == 2:  # move down
        player["y"] = min(player["y"] + 5, 45)
    elif act_id == 3:  # move left on top platform
        if player["y"] == 5:
            player["x"] = max(player["x"] - 5, 29)
    elif act_id == 4:  # no movement
        pass

    # ---------- Interactions ----------
    _apply_interactions(new_state, player, step_bar, ring16, diamond, recolor, hud)

    # ---------- Step bar update ----------
    if step_bar:
        old_w = step_bar["w"]
        old_x = step_bar["x"]
        old_tags = step_bar["tags"][:]
        ring_present = _find_obj(new_state, "refuel", "ring_16_35") is not None
        _update_step_bar(step_bar, old_w, old_x, old_tags, ring_present)

    # ---------- Diamond reappearance ----------
    if act_id == 1 and _diamond_removed:
        if not _find_obj(new_state, "target", "diamond"):
            new_state.append({
                "h": 3,
                "layer": 0,
                "name": "diamond",
                "tags": ["marker", "collect"],
                "type": "target",
                "visible": True,
                "w": 3,
                "x": 50,
                "y": 11
            })
            _diamond_removed = False

    return _render_frame(new_state)
