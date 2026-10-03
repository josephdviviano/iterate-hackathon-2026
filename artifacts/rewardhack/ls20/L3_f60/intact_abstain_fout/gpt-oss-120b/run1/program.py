# Mechanics:
# - The player moves on a 5‑pixel grid. Actions: 1=up, 2=down, 3=left, 4=right.
#   Moving up at the top row (y == 5) jumps to the next column on the right
#   (dx = +25, dy = 0). All other moves shift by exactly 5 pixels.
# - Each action costs 2 budget points. The step_bar counter shows the budget:
#   w = budget, x = 13 + (42 - budget), tags = ["budget", str(budget)].
#   When the player steps on a refuel object (type "refuel") the budget is
#   restored to 42 and the refuel disappears.
# - Stepping on a target (type "target") or a button (type "button") removes
#   that object and reduces the HUD glyph progress by 3 (e.g. "12" → "9").
# - The frame is updated by erasing the old player sprite (colour 'c') and
#   drawing the new one (top two rows colour 12 → 'c', bottom three rows colour
#   9 → '9').

import copy

# hidden state kept between calls
_last_state_key = None
_budget = 42  # initial budget
_last_player_pos = None

def _state_key(state):
    """A simple deterministic key for a state (order‑independent)."""
    items = []
    for o in state:
        items.append((o.get("type"), o.get("name"), o.get("x"), o.get("y")))
    return tuple(sorted(items))

def _find_obj(state, typ=None, name=None):
    for o in state:
        if typ is not None and o.get("type") != typ:
            continue
        if name is not None and o.get("name") != name:
            continue
        return o
    return None

def transition_function(state, action, frame):
    """Return the predicted after frame (list of 64 rows of hex digits)."""
    global _last_state_key, _budget, _last_player_pos

    # ------------------------------------------------------------------
    # 1. Detect continuity – if the incoming state does not match the
    #    state we produced last time, re‑initialise hidden variables.
    # ------------------------------------------------------------------
    cur_key = _state_key(state)
    if cur_key != _last_state_key:
        # initialise from the supplied state
        sb = _find_obj(state, typ="counter")
        # find the budget counter (has tag "budget")
        for o in state:
            if o.get("type") == "counter" and any(t.startswith("budget") for t in o.get("tags", [])):
                sb = o
                break
        # extract numeric budget from tags
        for t in sb.get("tags", []):
            if t.isdigit():
                _budget = int(t)
                break
        # remember player position
        pl = _find_obj(state, typ="player")
        _last_player_pos = (pl["x"], pl["y"])
        _last_state_key = cur_key

    # ------------------------------------------------------------------
    # 2. Locate objects we will modify.
    # ------------------------------------------------------------------
    # deep copy so we can return a fresh state list
    new_state = copy.deepcopy(state)

    player = _find_obj(new_state, typ="player")
    step_bar = None
    hud_glyph = None
    for o in new_state:
        if o.get("type") == "counter" and any(t.startswith("budget") for t in o.get("tags", [])):
            step_bar = o
        if o.get("type") == "counter" and any(t.startswith("progress") for t in o.get("tags", [])):
            hud_glyph = o

    # ------------------------------------------------------------------
    # 3. Compute new player position.
    # ------------------------------------------------------------------
    dx = dy = 0
    if action == 1:          # up
        if player["y"] == 5:
            dx = 25          # jump to next column on the right
            dy = 0
        else:
            dy = -5
    elif action == 2:        # down
        dy = 5
    elif action == 3:        # left
        dx = -5
    elif action == 4:        # right
        dx = 5
    # other actions (5,6,7) do nothing to the player
    new_x = player["x"] + dx
    new_y = player["y"] + dy

    # keep inside the 64×64 arena (simple clamp)
    if new_x < 0 or new_x + player["w"] > 64 or new_y < 0 or new_y + player["h"] > 64:
        new_x, new_y = player["x"], player["y"]   # blocked → stay

    # ------------------------------------------------------------------
    # 4. Apply collection effects.
    # ------------------------------------------------------------------
    collected = None
    for obj in list(new_state):
        if obj is player:
            continue
        if obj["x"] == new_x and obj["y"] == new_y:
            typ = obj.get("type")
            if typ in ("refuel", "target", "button"):
                collected = obj
                new_state.remove(obj)
                if typ == "refuel":
                    _budget = 42
                elif typ in ("target", "button"):
                    # decrease HUD progress by 3 if present
                    if hud_glyph:
                        new_tags = []
                        for t in hud_glyph["tags"]:
                            if t.isdigit():
                                new_tags.append(str(int(t) - 3))
                            else:
                                new_tags.append(t)
                        hud_glyph["tags"] = new_tags
                break

    # ------------------------------------------------------------------
    # 5. Update player coordinates.
    # ------------------------------------------------------------------
    player["x"], player["y"] = new_x, new_y

    # ------------------------------------------------------------------
    # 6. Update the budget counter (step_bar).
    # ------------------------------------------------------------------
    if collected and collected.get("type") == "refuel":
        # budget already set to 42 above
        pass
    else:
        _budget = max(0, _budget - 2)

    if step_bar:
        step_bar["w"] = _budget
        step_bar["x"] = 13 + (42 - _budget)
        # replace the numeric tag
        new_tags = []
        for t in step_bar["tags"]:
            if t.isdigit():
                new_tags.append(str(_budget))
            else:
                new_tags.append(t)
        step_bar["tags"] = new_tags

    # ------------------------------------------------------------------
    # 7. Build the after frame.
    # ------------------------------------------------------------------
    after = [list(row) for row in frame]          # copy mutable
    # erase old player (background colour 'c')
    old_x, old_y = _last_player_pos
    for dy_i in range(player["h"]):
        for dx_i in range(player["w"]):
            ax = old_x + dx_i
            ay = old_y + dy_i
            if 0 <= ax < 64 and 0 <= ay < 64:
                after[ay][ax] = 'c'               # background colour

    # draw new player
    for dy_i in range(player["h"]):
        colour = 'c' if dy_i < 2 else '9'          # 12 → 'c', 9 → '9'
        for dx_i in range(player["w"]):
            ax = new_x + dx_i
            ay = new_y + dy_i
            if 0 <= ax < 64 and 0 <= ay < 64:
                after[ay][ax] = colour

    # ------------------------------------------------------------------
    # 8. Store continuity information for the next call.
    # ------------------------------------------------------------------
    _last_state_key = _state_key(new_state)
    _last_player_pos = (new_x, new_y)

    # Convert rows back to strings
    after_rows = [''.join(r) for r in after]
    return after_rows
