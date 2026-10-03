# program.py
# Mechanics:
# - Actions 1‑4 move the player by 5 cells (up, down, left, right) and cost 2 budget.
# - The step_bar counter shows the remaining budget: its width = budget,
#   its x = 13 + (42‑budget), and its tag "budget" holds the numeric value.
# - Touching a refuel object (type "refuel") resets the budget to the max (42)
#   and permanently removes that refuel.
# - Touching a collectable target (type "target" with tag "collect") toggles its
#   existence: it disappears while the player stands on it and re‑appears when
#   the player leaves the cell.
# - When the budget falls to 10 or less, the recolor button (type "button",
#   tag "recolor") disappears and the progress counter (type "counter",
#   tag "progress") loses 3 points.
# The player sprite is drawn using its stored pixel matrix (values 12 and 9);
# other objects are drawn with a single colour sampled from the initial frame.

import copy

# ----------------------------------------------------------------------
# Global caches – cleared whenever the incoming state does not match the
# previously returned state.
_prev_state_key = None          # a simple hash of the last state we returned
_original_objs = {}             # name -> original object dict (unchanged)
_original_colors = {}           # name -> colour (int) for objects without pixels
_MAX_BUDGET = 42
_FLOOR_COLOUR = 3               # colour of empty floor cells

def _state_key(state):
    """Create a cheap comparable key from a state."""
    parts = []
    for o in sorted(state, key=lambda x: x.get("name", "")):
        parts.append((o.get("name"), o.get("type"), o.get("x"), o.get("y"),
                      o.get("visible", True)))
    return tuple(parts)

def _sample_colour(frame, obj):
    """Sample the colour of the top‑left cell of an object from the frame."""
    y, x = obj["y"], obj["x"]
    if 0 <= y < len(frame) and 0 <= x < len(frame[0]):
        return frame[y][x]
    return _FLOOR_COLOUR

def _rects_intersect(a, b):
    """Check if two axis‑aligned rectangles intersect."""
    ax1, ay1, aw, ah = a["x"], a["y"], a["w"], a["h"]
    bx1, by1, bw, bh = b["x"], b["y"], b["w"], b["h"]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def _clear_rect(frame, obj):
    """Paint a rectangle with the floor colour."""
    for dy in range(obj["h"]):
        for dx in range(obj["w"]):
            y, x = obj["y"] + dy, obj["x"] + dx
            if 0 <= y < len(frame) and 0 <= x < len(frame[0]):
                frame[y][x] = _FLOOR_COLOUR

def _draw_object(frame, obj):
    """Draw an object onto the frame."""
    if "pixels" in obj:
        # use the stored pixel matrix (already numeric colour values)
        for dy, row in enumerate(obj["pixels"]):
            for dx, col in enumerate(row):
                y, x = obj["y"] + dy, obj["x"] + dx
                if 0 <= y < len(frame) and 0 <= x < len(frame[0]):
                    frame[y][x] = col
    else:
        colour = _original_colors.get(obj["name"], _FLOOR_COLOUR)
        for dy in range(obj["h"]):
            for dx in range(obj["w"]):
                y, x = obj["y"] + dy, obj["x"] + dx
                if 0 <= y < len(frame) and 0 <= x < len(frame[0]):
                    frame[y][x] = colour

def transition_function(state, action, frame):
    global _prev_state_key, _original_objs, _original_colors

    # ------------------------------------------------------------------
    # Detect discontinuity – if the incoming state differs from what we
    # produced last time, rebuild caches.
    incoming_key = _state_key(state)
    if incoming_key != _prev_state_key:
        # rebuild original object map and colour map
        _original_objs = {o["name"]: copy.deepcopy(o) for o in state}
        _original_colors = {}
        for o in state:
            if "pixels" not in o:
                _original_colors[o["name"]] = _sample_colour(frame, o)
        _prev_state_key = incoming_key

    # ------------------------------------------------------------------
    # Helper look‑ups
    objs_by_name = {o["name"]: o for o in state}
    player = objs_by_name.get("player")
    step_bar = None
    progress_counter = None
    recolor_button = None
    for o in state:
        if o.get("type") == "counter" and any(t.startswith("budget") for t in o.get("tags", [])):
            step_bar = o
        if o.get("type") == "counter" and any(t.startswith("progress") for t in o.get("tags", [])):
            progress_counter = o
        if o.get("type") == "button" and any(t == "recolor" for t in o.get("tags", [])):
            recolor_button = o

    # ------------------------------------------------------------------
    # Determine movement delta
    dx = dy = 0
    if action == 1:   # up
        dy = -5
    elif action == 2: # down
        dy = 5
    elif action == 3: # left
        dx = -5
    elif action == 4: # right
        dx = 5
    # actions 5,6,7 are ignored (no effect in observed data)

    # ------------------------------------------------------------------
    # Update player position on the frame
    if player:
        old_player = copy.deepcopy(player)
        _clear_rect(frame, old_player)          # erase old sprite

        player["x"] += dx
        player["y"] += dy

        _draw_object(frame, player)             # draw new sprite

    # ------------------------------------------------------------------
    # Budget handling
    if step_bar and action in (1, 2, 3, 4):
        # consume 2 budget
        budget = int(step_bar["tags"][1]) - 2
        if budget < 0:
            budget = 0
        step_bar["tags"][1] = str(budget)
        step_bar["w"] = budget
        step_bar["x"] = 13 + (42 - budget)

    # ------------------------------------------------------------------
    # Refuel handling
    # Find any refuel object intersecting the player after the move.
    if player:
        for o in list(state):
            if o.get("type") == "refuel" and _rects_intersect(player, o):
                # reset budget
                step_bar["tags"][1] = str(_MAX_BUDGET)
                step_bar["w"] = _MAX_BUDGET
                step_bar["x"] = 13
                # remove refuel permanently
                _clear_rect(frame, o)
                state.remove(o)
                break   # only one refuel can be collected at a time

    # ------------------------------------------------------------------
    # Collectable target handling (toggle existence)
    if player:
        for o in list(state):
            if o.get("type") == "target" and "collect" in o.get("tags", []):
                intersect = _rects_intersect(player, o)
                if intersect:
                    # player is on the target – remove it
                    _clear_rect(frame, o)
                    state.remove(o)
                else:
                    # target not intersecting now – ensure it exists (respawn)
                    if o["name"] not in [obj["name"] for obj in state]:
                        # respawn at original location
                        orig = _original_objs.get(o["name"])
                        if orig:
                            new_obj = copy.deepcopy(orig)
                            state.append(new_obj)
                            _draw_object(frame, new_obj)

    # ------------------------------------------------------------------
    # Recolor button & progress handling
    if recolor_button and step_bar:
        budget = int(step_bar["tags"][1])
        if budget <= 10:
            # consume button
            _clear_rect(frame, recolor_button)
            state.remove(recolor_button)
            # reduce progress by 3
            if progress_counter:
                # find numeric progress tag (second element)
                for i, t in enumerate(progress_counter["tags"]):
                    if t.isdigit():
                        new_val = max(0, int(t) - 3)
                        progress_counter["tags"][i] = str(new_val)
                        break

    # ------------------------------------------------------------------
    # Return the updated state (deep‑copy to avoid accidental mutation)
    _prev_state_key = _state_key(state)
    return copy.deepcopy(state)
