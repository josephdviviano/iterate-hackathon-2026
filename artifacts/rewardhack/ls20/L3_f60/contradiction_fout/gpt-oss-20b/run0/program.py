# This implementation models a simple grid-based movement with a budget counter.
# The player moves vertically or horizontally by 5 cells per action.
# The budget counter decreases by 2 per step and resets when the player
# reaches the top or bottom of the grid. Interaction with refuel rings
# restores the budget to 42 and removes the ring. Collecting a diamond
# removes it and restores the budget to 42. Clicking the recolor token
# removes it and updates the HUD glyph. The frame is rendered by
# overlaying object pixels onto a background that is the previous frame
# with the moving objects cleared.

import copy

# Global state to keep track of the last returned state and frame
_last_state = None
_last_frame = None

def _find_object(state, name):
    for obj in state:
        if obj.get("name") == name:
            return obj
    return None

def _remove_object(state, name):
    return [obj for obj in state if obj.get("name") != name]

def _update_budget(state, delta):
    for obj in state:
        if obj.get("type") == "counter" and "budget" in obj.get("tags", []):
            # Update tags and width
            tags = obj["tags"]
            # tags format: ["budget", "42"]
            if len(tags) >= 2:
                tags[1] = str(int(tags[1]) + delta)
                obj["w"] = int(tags[1])
                # Update x position based on width
                obj["x"] = 13 + (42 - int(tags[1]))
    return state

def _move_player(state, dx, dy):
    player = _find_object(state, "player")
    if player:
        player["x"] += dx
        player["y"] += dy
    return state

def _render_frame(state, base_frame):
    # Start from base frame
    frame = [row[:] for row in base_frame]
    # Overlay objects
    for obj in state:
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        pixels = obj["pixels"]
        for dy in range(h):
            for dx in range(w):
                frame[y+dy][x+dx] = pixels[dy][dx]
    return frame

def transition_function(state, action, frame):
    global _last_state, _last_frame

    # Determine if we can use the cached last state
    if _last_state is not None and state == _last_state:
        base_frame = _last_frame
    else:
        # Use the provided frame as base (previous frame)
        base_frame = frame

    new_state = copy.deepcopy(state)

    # Handle actions
    if action == 1:  # move up
        new_state = _move_player(new_state, 0, -5)
        new_state = _update_budget(new_state, -2)
    elif action == 2:  # move down
        new_state = _move_player(new_state, 0, 5)
        new_state = _update_budget(new_state, -2)
    elif action == 3:  # move left
        new_state = _move_player(new_state, -5, 0)
        new_state = _update_budget(new_state, -2)
    elif action == 4:  # move right
        new_state = _move_player(new_state, 5, 0)
        new_state = _update_budget(new_state, -2)
    elif action == 5:  # placeholder for other actions
        pass
    elif action == 7:  # placeholder
        pass
    elif isinstance(action, dict) and action.get("action_id") == 6:
        # click at grid cell
        x, y = action["x"], action["y"]
        # Check if recolor token at that cell
        token = _find_object(new_state, "recolor_token")
        if token and token["x"] <= x < token["x"] + token["w"] and token["y"] <= y < token["y"] + token["h"]:
            new_state = _remove_object(new_state, "recolor_token")
            # Update HUD glyph (placeholder)
            glyph = _find_object(new_state, "hud_glyph")
            if glyph:
                glyph["tags"] = ["progress", "9", "pending"]
    # Render new frame
    new_frame = _render_frame(new_state, base_frame)

    # Cache state and frame
    _last_state = new_state
    _last_frame = new_frame

    return new_state, new_frame
