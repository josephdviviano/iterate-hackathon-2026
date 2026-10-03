# Transition rules for the ARC-AGI-3 corridor game.
# The game is a simple corridor with a player that moves 5 cells per action.
# The step bar budget decreases by 2 per move and its x position increases by 2.
# The diamond toggles when the player moves left from the right corridor or
# moves vertically in the middle corridor.
# The recolor token disappears when the player moves down from y=45 and
# reduces the progress counter by 3.
# The frame is rendered by overlaying objects on a fixed base background.

BASE_FRAME_STR = [
    "5555444444444444444444444444444444444444444444444444444444444444",
    "5555444444444444444444444444444444444444444444444444444444444444",
    "5555444444444444444444444444444444444444444444444444444444444444",
    "5555444444444444444444444444444444444444444444444444444444444444",
    "5555444444444444444444444444444444444444444444444444441111144444",
    "5555444413333333333333333333333333333334444433333333333333344444",
    "5555444413333333333333333333333333333334444433333333333333344444",
    "5555444413333333333333333333333333333334444433333333333333344444",
    "5555444413333333333333333333333333333334444433333333333333344444",
    "5555444413333333333333333333333333333334444433333333333333344444",
    "5555444443333344444444444444433333444444444433333333333333344444",
    "5555444443333344444444444444433333444444444433333330333333344444",
    "5555444443333344444444444444433333444444444433333310033333344444",
    "5555444443333344444444444444433333444444444433333331333333344444",
    "5555444443333344444444444444433333444444444433333333333333344444",
    "5555444443333344444333333333333333333334444433333333333333344444",
    "55554444433333444443333333333333333bbb34444433333333333333344444",
    "55554444433333444443333333333333333b3b34444433333333333333344444",
    "55554444433333444443333333333333333bbb34444433333333333333344444",
    "5555444443333344444333333333333333333334444433333333333333344444",
    "5555444443333344444333333333333333333334444444444444443333344444",
    "5555444443333344444333333333333333333334444444444444443333344444",
    "5555444443333344444333333333333333333334444444444444443333344444",
    "5555444443333344444333333333333333333334444444444444443333344444",
    "5555444443333344444333333333333333333334444444444444443333344444",
    "5555444443333344444333333333333333333334444444444444443333344444",
    "5555444443333344444333333333333333333334444444444444443333344444",
    "5555444443333344444333333333333333333334444444444444443333344444",
    "5555444443333344444333333333333333333334444444444444443333344444",
    "55554444433333444443bbb33333333333333333333333333333333333344444",
    "55554444433333444443b3b33333333333333333333333333333333333344444",
    "55554444433333444443bbb33333333333333333333333333333333333344444",
    "5555444443333344444333333333333333333333333333333333333333344444",
    "5555444443333344444444444444433333444444444444444444443333344444",
    "5555444443333344444444444444433333444444444444444444443333344444",
    "5555444443333344444444444444433333444444444444444444443333344444",
    "5555444443333344444444444444433333444444444444444444443333344444",
    "5555444443333344444444444444433333444444444444444444443333344444",
    "5555444443333333333444443333333333333334444444444444443333344444",
    "5555444443333333333444443333333333333334444444444444443333344444",
    "5555444443333333333444443333333333333334444444444444443333344444",
    "5555444443333333333444443333333333333334444444444444443333344444",
    "5555444443333333333444443333333333333334444444444444443333344444",
    "555544444ccccc33333444443333333333333334444444444444443333344444",
    "555544444ccccc33333444443333339ee3333334444444444444443333344444",
    "5555444449999933333444443333339083333334444444444444443333344444",
    "555544444999993333344444333333cc83333334444444444444333333333444",
    "5555444449999933333444443333333333333334444444444444355555553444",
    "5555444444444444444444443333333333333334444444444444355555553444",
    "5555444444444444444444443333333333333334444444444444355959553444",
    "4444444444444444444444443333333333333334444444444444355955553444",
    "4555555555544444444444443333333333333334444444444444355999553444",
    "4555555555544444444444443333333333333334444444444444355555553444",
    "455cccccc5544444444444444444444444444444444444444444355555553444",
    "455cccccc5544444444444444444444444444444444444444444333333333444",
    "4555555cc5544444444444444444444444444444444444444444444444444444",
    "4555555cc5544444444444444444444444444444444444444444444444444444",
    "455cc55cc5544444444444444444444444444444444444444444444444444444",
    "455cc55cc5545555555555555555555555555555555555555555555555555555",
    "4555555555545bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb588588588",
    "4555555555545bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb588588588",
    "4444444444445555555555555555555555555555555555555555555555555555",
]

# Convert base frame to integers
BASE_FRAME = [[int(ch) for ch in row] for row in BASE_FRAME_STR]

def transition_function(state, action, frame=None):
    # Helper to find object by type
    def find(type_name):
        for o in state:
            if o.get("type") == type_name:
                return o
        return None

    # Copy state to avoid mutating input
    new_state = [o.copy() for o in state]

    # Find objects
    player = find("player")
    step_bar = find("counter")
    recolor_token = find("button")
    diamond = find("target")
    hud_glyph = find("counter")  # the progress counter

    # Update step bar
    if step_bar and "budget" in step_bar.get("tags", []):
        old_budget = int(step_bar["tags"][1])
        new_budget = max(old_budget - 2, 0)
        step_bar["tags"] = ["budget", str(new_budget)]
        step_bar["w"] = new_budget
        step_bar["x"] = step_bar["x"] + 2

    # Update player position
    if player:
        old_x, old_y = player["x"], player["y"]
        dx, dy = 0, 0
        if action == 1:  # up
            dx, dy = 0, -5
        elif action == 2:  # down
            dx, dy = 0, 5
        elif action == 3:  # left
            dx, dy = -5, 0
        elif action == 4:  # right
            dx, dy = 5, 0
        # Teleport rules
        if action == 1 and old_y == 5 and old_x == 9:
            new_x, new_y = 34, 5
        elif action == 3 and old_y == 5 and old_x == 49:
            new_x, new_y = 34, 5
        elif action == 4 and old_y == 5 and old_x == 34:
            new_x, new_y = 49, 45
        else:
            new_x, new_y = old_x + dx, old_y + dy
        # Clamp within corridor bounds
        new_x = max(9, min(53, new_x))
        new_y = max(5, min(45, new_y))
        player["x"], player["y"] = new_x, new_y

    # Diamond toggle
    diamond_present = diamond is not None
    toggle = False
    if action == 3 and old_y == 5 and old_x == 54:  # left from right corridor
        toggle = True
    if action in (1, 2) and old_x == 34:  # vertical in middle corridor
        toggle = True
    if toggle:
        if diamond_present:
            new_state = [o for o in new_state if not (o.get("type") == "target" and "marker" in o.get("tags", []) and "collect" in o.get("tags", []))]
        else:
            new_state.append({
                "h": 3, "layer": 0, "name": "diamond",
                "tags": ["marker", "collect"], "type": "target",
                "visible": True, "w": 3, "x": 50, "y": 11
            })

    # Recolor token removal and HUD update
    if recolor_token and action == 2 and old_y == 45:
        new_state = [o for o in new_state if not (o.get("type") == "button" and "recolor" in o.get("tags", []))]
        if hud_glyph:
            old_progress = int(hud_glyph["tags"][1])
            new_progress = max(old_progress - 3, 0)
            hud_glyph["tags"] = ["progress", str(new_progress), "pending"]

    # Render frame from updated state
    def render(state_objs):
        # Start from the base frame
        frame = [row[:] for row in BASE_FRAME]

        # Overlay objects by layer
        for obj in sorted(state_objs, key=lambda o: o.get("layer", 0)):
            if not obj.get("visible", True):
                continue
            if "pixels" not in obj:
                continue
            w, h = obj["w"], obj["h"]
            x0, y0 = obj["x"], obj["y"]
            pixels = obj["pixels"]
            for dy in range(h):
                for dx in range(w):
                    val = pixels[dy][dx]
                    if val is None:
                        continue
                    fx, fy = x0 + dx, y0 + dy
                    if 0 <= fx < 64 and 0 <= fy < 64:
                        frame[fy][fx] = val
        return frame

    return render(new_state)
