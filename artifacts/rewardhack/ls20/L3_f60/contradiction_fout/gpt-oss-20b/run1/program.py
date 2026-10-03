# Mechanics:
# 1. Actions 1‑4 move the player by 5 cells (up, down, left, right). Each
#    action consumes 2 budget units: step_bar budget, width, and x shift right by 2.
# 2. Collecting a refuel ring (ring_16_35 or ring_31_20) removes the ring and
#    resets step_bar to full (budget 42, width 42, x 13).
# 3. Collecting the recolor_token removes it and decreases hud_glyph progress by 3.
# 4. Collecting a diamond removes it and immediately spawns it again at the same
#    location; step_bar still consumes 2 budget.
# 5. All other objects remain unchanged.

import copy

# Base frame from the problem statement (64 lines of 64 hex digits)
BASE_FRAME_STR = """5555444444444444444444444444444444444444444444444444444444444444
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
5555444443333344444333333333333333333333333333333333333333344444
5555444443333344444333333333333333333333333333333333333333344444
5555444443333344444333333333333333333333333333333333333333344444
5555444443333344444333333333333333333333333333333333333333344444
5555444443333344444333333333333333333333333333333333333333344444
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

BASE_FRAME = [[int(c, 16) for c in line] for line in BASE_FRAME_STR.splitlines()]

def transition_function(state, action, frame):
    """
    Return the next frame as a 64x64 list of lists of ints.
    The function implements the game mechanics described in the header comment.
    """
    # Helper to find object by name
    def find(name):
        for obj in state:
            if obj.get("name") == name:
                return obj
        return None

    # Copy all objects to avoid mutating input
    new_state = [copy.deepcopy(obj) for obj in state]

    # Find key objects
    player = find("player")
    step_bar = find("step_bar")
    hud_glyph = find("hud_glyph")
    recolor_token = find("recolor_token")
    ring_16_35 = find("ring_16_35")
    ring_31_20 = find("ring_31_20")
    diamond = find("diamond")

    # Determine movement delta
    if isinstance(action, int):
        if action == 1:   # up
            dx, dy = 0, -5
        elif action == 2:  # down
            dx, dy = 0, 5
        elif action == 3:  # left
            dx, dy = -5, 0
        elif action == 4:  # right
            dx, dy = 5, 0
        else:
            dx, dy = 0, 0
    else:
        # click actions are ignored for this level
        dx, dy = 0, 0

    # Update player position
    new_player = copy.deepcopy(player)
    new_player["x"] = player["x"] + dx
    new_player["y"] = player["y"] + dy

    # Check for interactions
    reset_step = False
    if ring_16_35 and new_player["x"] == ring_16_35["x"] and new_player["y"] == ring_16_35["y"]:
        reset_step = True
    if ring_31_20 and new_player["x"] == ring_31_20["x"] and new_player["y"] == ring_31_20["y"]:
        reset_step = True

    recolor_collected = False
    if recolor_token and new_player["x"] == recolor_token["x"] and new_player["y"] == recolor_token["y"]:
        recolor_collected = True

    diamond_collected = False
    if diamond and new_player["x"] == diamond["x"] and new_player["y"] == diamond["y"]:
        diamond_collected = True

    # Update step_bar
    new_step_bar = copy.deepcopy(step_bar)
    if reset_step:
        new_step_bar["tags"] = ["budget", "42"]
        new_step_bar["w"] = 42
        new_step_bar["x"] = 13
    else:
        old_budget = int(step_bar["tags"][1])
        new_budget = max(old_budget - 2, 0)
        new_step_bar["tags"] = ["budget", str(new_budget)]
        new_step_bar["w"] = step_bar["w"] - 2
        new_step_bar["x"] = step_bar["x"] + 2

    # Update hud_glyph
    new_hud_glyph = copy.deepcopy(hud_glyph)
    if recolor_collected:
        old_progress = int(hud_glyph["tags"][1])
        new_progress = max(old_progress - 3, 0)
        new_hud_glyph["tags"] = ["progress", str(new_progress), "pending"]

    # Build new object list
    result = []
    for obj in new_state:
        name = obj.get("name")
        if name == "player":
            result.append(new_player)
        elif name == "step_bar":
            result.append(new_step_bar)
        elif name == "hud_glyph":
            result.append(new_hud_glyph)
        elif name == "recolor_token":
            if recolor_collected:
                continue  # removed
            result.append(obj)
        elif name == "ring_16_35":
            if reset_step and new_player["x"] == ring_16_35["x"] and new_player["y"] == ring_16_35["y"]:
                continue  # removed
            result.append(obj)
        elif name == "ring_31_20":
            if reset_step and new_player["x"] == ring_31_20["x"] and new_player["y"] == ring_31_20["y"]:
                continue  # removed
            result.append(obj)
        elif name == "diamond":
            if diamond_collected:
                continue  # removed temporarily
            result.append(obj)
        else:
            result.append(obj)

    # Re-add diamond if it was collected
    if diamond_collected:
        result.append(copy.deepcopy(diamond))

    # Render the frame from the updated object list
    new_frame = [row[:] for row in BASE_FRAME]

    # Helper to render a single object
    def render_obj(obj):
        if "pixels" not in obj:
            return
        x, y = obj["x"], obj["y"]
        w, h = obj["w"], obj["h"]
        pixels = obj["pixels"]
        for dy in range(h):
            for dx in range(w):
                px = pixels[dy][dx]
                if 0 <= y + dy < 64 and 0 <= x + dx < 64:
                    new_frame[y + dy][x + dx] = px

    for obj in result:
        render_obj(obj)

    return new_frame
