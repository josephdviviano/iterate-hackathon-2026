# 1. Player moves 5 cells per action: 1=up,2=down,3=left,4=right,5=stay.
# 2. step_bar budget decreases by 2 each move; width equals budget; x = 13 + (42 - budget).
# 3. Targets disappear when moving left across their x; refuels disappear when moving right across their x.
# 4. Buttons disappear when moving up to y such that new_y + 6 == button.y; hud_glyph progress decreases by 3.
# 5. Diamond reappears when player moves up to y=5; it is a target at (50,11).

# Base background frame (64x64) taken from the initial frame in the problem statement.
INITIAL_FRAME_STR = """5555444444444444444444444444444444444444444444444444444444444444
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
INITIAL_FRAME = [[int(c, 16) for c in line] for line in INITIAL_FRAME_STR.splitlines()]

def transition_function(state, action, frame=None):
    # Helper to find object by type and optional tag
    def find_obj(obj_type, tag=None):
        for o in state:
            if o.get("type") == obj_type:
                if tag is None or tag in o.get("tags", []):
                    return o
        return None

    # Copy state to modify
    new_state = []

    # Find key objects
    player = find_obj("player")
    step_bar = find_obj("counter", "budget")
    hud_glyph = find_obj("counter", "progress")

    # Determine movement
    if isinstance(action, dict):
        # click actions ignored
        return [row[:] for row in INITIAL_FRAME]
    move = action
    dx, dy = 0, 0
    if move == 1:  # up
        dy = -5
    elif move == 2:  # down
        dy = 5
    elif move == 3:  # left
        dx = -5
    elif move == 4:  # right
        dx = 5
    elif move == 5:  # stay
        dx = dy = 0
    else:
        # unknown action, return unchanged frame
        return [row[:] for row in INITIAL_FRAME]

    # Compute new player position
    new_px = player["x"] + dx
    new_py = player["y"] + dy

    # Update step_bar budget, width, x
    old_budget = int(step_bar["tags"][1])
    new_budget = old_budget - 2
    new_width = new_budget
    new_x_sb = step_bar["x"] + 2

    # Update hud_glyph progress if recolor_token removed
    old_progress = int(hud_glyph["tags"][1])
    new_progress = old_progress
    recolor_removed = False

    # Determine if any objects removed
    removed_ids = set()
    for obj in state:
        if obj.get("type") == "target" and "collect" in obj.get("tags", []):
            # target removal on left move
            if move == 3 and new_px + 1 == obj["x"]:
                removed_ids.add(id(obj))
        if obj.get("type") == "refuel" and "budget" in obj.get("tags", []):
            # refuel removal on right move
            if move == 4 and new_px + 1 == obj["x"]:
                removed_ids.add(id(obj))
        if obj.get("type") == "button" and "recolor" in obj.get("tags", []):
            # button removal on up move
            if move == 1 and new_py + 6 == obj["y"]:
                removed_ids.add(id(obj))
                recolor_removed = True

    # Build new state
    for obj in state:
        if id(obj) in removed_ids:
            continue
        if obj.get("type") == "player":
            new_obj = obj.copy()
            new_obj["x"] = new_px
            new_obj["y"] = new_py
            new_state.append(new_obj)
        elif obj.get("type") == "counter" and "budget" in obj.get("tags", []):
            new_obj = obj.copy()
            new_obj["x"] = new_x_sb
            new_obj["w"] = new_width
            new_obj["tags"] = ["budget", str(new_budget)]
            new_state.append(new_obj)
        elif obj.get("type") == "counter" and "progress" in obj.get("tags", []):
            new_obj = obj.copy()
            if recolor_removed:
                new_progress -= 3
            new_obj["tags"] = ["progress", str(new_progress), "pending"]
            new_state.append(new_obj)
        else:
            new_state.append(obj.copy())

    # Add diamond if needed
    diamond_present = any(o for o in new_state if o.get("type") == "target" and "collect" in o.get("tags", []) and o.get("name") == "diamond")
    if move == 1 and new_py == 5 and not diamond_present:
        diamond = {
            "h": 3,
            "layer": 0,
            "name": "diamond",
            "tags": ["marker", "collect"],
            "type": "target",
            "visible": True,
            "w": 3,
            "x": 50,
            "y": 11
        }
        new_state.append(diamond)

    # Render frame
    rendered = [row[:] for row in INITIAL_FRAME]

    # Sort objects by layer ascending
    objects_by_layer = sorted(new_state, key=lambda o: o.get("layer", 0))

    for obj in objects_by_layer:
        if not obj.get("visible", True):
            continue
        pixels = obj.get("pixels")
        if not pixels:
            continue
        ox = obj["x"]
        oy = obj["y"]
        h = obj["h"]
        w = obj["w"]
        for dy_i in range(h):
            for dx_i in range(w):
                fx = ox + dx_i
                fy = oy + dy_i
                if 0 <= fx < 64 and 0 <= fy < 64:
                    rendered[fy][fx] = pixels[dy_i][dx_i]

    return rendered
