# Mechanics:
# - Action 1: move player up 5 cells; if the move would place the player on a non‑floor cell
#   (any colour other than 3) the move is cancelled.
# - Action 2: move player down 5 cells (same collision rule).
# - Action 3: move player left 5 cells (same collision rule).
# - Action 4: move player right 5 cells (same collision rule).
# - Every action consumes 2 budget points: step_bar.width and step_bar.x are updated so that
#   width = budget, x = 13 + (42‑budget). The tag ["budget","N"] is also updated.
# - When the player’s 5×5 rectangle overlaps a refuel object (type "refuel") the refuel
#   disappears and the budget is reset to its maximum (42, width 42, x 13).
# - When the budget falls to 8 (after the decrement) the button named "recolor_token"
#   disappears and the numeric part of the hud_glyph counter’s progress tag is reduced by 3.
# - The diamond target appears only when the player’s y‑coordinate equals 5; otherwise it
#   is removed. Its properties are taken from the initial state.
# - All other objects remain unchanged.

import copy

# template for the diamond (taken from the initial state)
_DIAMOND_TEMPLATE = {
    "h": 3,
    "layer": 0,
    "name": "diamond",
    "tags": ["marker", "collect"],
    "type": "target",
    "visible": True,
    "w": 3,
    "x": 50,
    "y": 11,
    "pixels": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # placeholder; actual pixels are not needed for logic
}

def _find(obj_list, **kw):
    for o in obj_list:
        if all(o.get(k) == v for k, v in kw.items()):
            return o
    return None

def _overlaps(a, b):
    # a and b are dicts with x,y,w,h
    return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"] or
                a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])

def _can_place(frame, x, y):
    # player size is 5×5, floor colour is 3
    for dy in range(5):
        for dx in range(5):
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= len(frame) or xx < 0 or xx >= len(frame[0]):
                return False
            if frame[yy][xx] != 3:
                return False
    return True

def transition_function(state, action, frame):
    # work on a copy so we can modify safely
    objs = copy.deepcopy(state)

    player = _find(objs, type="player")
    step_bar = _find(objs, name="step_bar")
    hud_glyph = _find(objs, name="hud_glyph")
    recolor_btn = _find(objs, name="recolor_token")

    # ----- 1. player movement -----
    dx = dy = 0
    if action == 1:
        dy = -5
    elif action == 2:
        dy = 5
    elif action == 3:
        dx = -5
    elif action == 4:
        dx = 5
    # actions other than 1‑4 do not move the player
    new_x = player["x"] + dx
    new_y = player["y"] + dy
    if dx != 0 or dy != 0:
        if _can_place(frame, new_x, new_y):
            player["x"] = new_x
            player["y"] = new_y
        # else movement is cancelled – player stays where it is

    # ----- 2. budget consumption / reset -----
    # extract current budget from step_bar tags
    cur_budget = None
    for t in step_bar["tags"]:
        if isinstance(t, list) and t[0] == "budget":
            cur_budget = int(t[1])
            break
    if cur_budget is None:
        cur_budget = 42  # fallback

    # check for refuel collision (before budget change)
    collided_refuel = None
    for o in objs:
        if o.get("type") == "refuel" and _overlaps(player, o):
            collided_refuel = o
            break

    if collided_refuel:
        # reset budget
        new_budget = 42
        # remove the refuel object
        objs = [o for o in objs if o is not collided_refuel]
    else:
        # normal consumption
        new_budget = max(0, cur_budget - 2)

    # update step_bar fields
    step_bar["w"] = new_budget
    step_bar["x"] = 13 + (42 - new_budget)
    # rebuild tags list with updated budget
    new_tags = []
    for t in step_bar["tags"]:
        if isinstance(t, list) and t[0] == "budget":
            new_tags.append(["budget", str(new_budget)])
        else:
            new_tags.append(t)
    step_bar["tags"] = new_tags

    # ----- 3. recolor token & hud glyph -----
    if recolor_btn and new_budget <= 8:
        # remove the button
        objs = [o for o in objs if o is not recolor_btn]
        # decrement hud_glyph progress number by 3
        if hud_glyph:
            new_tags = []
            for t in hud_glyph["tags"]:
                if isinstance(t, list) and t[0] == "progress":
                    # t looks like ["progress","12","pending"]
                    try:
                        val = int(t[1]) - 3
                        new_tags.append(["progress", str(val)] + t[2:])
                    except Exception:
                        new_tags.append(t)
                else:
                    new_tags.append(t)
            hud_glyph["tags"] = new_tags

    # ----- 4. diamond visibility -----
    diamond = _find(objs, name="diamond")
    if player["y"] == 5:
        # ensure diamond exists
        if not diamond:
            # add a fresh copy
            objs.append(copy.deepcopy(_DIAMOND_TEMPLATE))
    else:
        # remove diamond if present
        if diamond:
            objs = [o for o in objs if o is not diamond]

    # return the updated object list (order does not matter)
    return objs
