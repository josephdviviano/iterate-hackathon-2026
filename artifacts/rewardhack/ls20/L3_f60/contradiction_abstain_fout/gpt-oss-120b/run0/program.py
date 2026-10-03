# Mechanics:
# - Player moves on a 5‑cell grid. Actions: 1=up, 2=down, 3=left, 4=right.
#   * Up/down move 5 cells vertically unless the player is already at the top
#     (y==5) or bottom (y==45). When at the top and action 1 is issued the
#     player jumps to the nearest refuel ring on the right (its x‑1).
#   * Left/right move 5 cells horizontally (only used on the top row).
# - A “budget” counter (step_bar) starts at 42 and loses 2 each action.
#   When the player touches a refuel object the budget resets to 42 and the
#   bar returns to its start position (x=13, w=42). The refuel object is
#   removed.
# - A “progress” counter (hud_glyph) starts at 12 and loses 3 each time the
#   player collects a target or button object. The collected object is removed.
# - The frame is a 64×64 grid of hex digits. Player cells are colour 3;
#   the floor colour is 12 (hex ‘c’). Moving the player clears its old cells
#   to colour 12 and paints the new cells colour 3.

import copy

# persistent cache between calls
_last_state = None
_last_frame = None

def _bbox(obj):
    return (obj["x"], obj["y"], obj["x"] + obj["w"] - 1, obj["y"] + obj["h"] - 1)

def _intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def _find_refuel_right(state, player):
    # nearest refuel with x > player.x
    candidates = [o for o in state if o["type"] == "refuel"]
    right = [o for o in candidates if o["x"] > player["x"]]
    if not right:
        return None
    # smallest x
    target = min(right, key=lambda o: o["x"])
    return target

def transition_function(state, action, frame):
    global _last_state, _last_frame

    # continuity check – if the incoming state does not match what we returned,
    # forget any cached data.
    if _last_state is not None:
        # simple shallow compare by sorted (name,x,y)
        def sig(s):
            return sorted((o["name"], o["x"], o["y"]) for o in s)
        if sig(state) != sig(_last_state):
            _last_state = None
            _last_frame = None

    # work on copies
    new_state = copy.deepcopy(state)
    new_frame = [row[:] for row in frame]

    # locate key objects
    player = next(o for o in new_state if o["type"] == "player")
    step_bar = next(o for o in new_state if o["type"] == "counter" and "budget" in o["tags"][0])
    hud = next(o for o in new_state if o["type"] == "counter" and "progress" in o["tags"][0])

    # remember old player bbox for frame clearing
    old_bbox = _bbox(player)

    # ----- move player -----
    if action == 1:          # up
        if player["y"] > 5:
            player["y"] -= 5
        else:  # at top row, jump to next refuel on the right
            ref = _find_refuel_right(new_state, player)
            if ref:
                player["x"] = ref["x"] - 1
    elif action == 2:        # down
        if player["y"] < 45:
            player["y"] += 5
    elif action == 3:        # left
        if player["x"] >= 5:
            player["x"] -= 5
    elif action == 4:        # right
        if player["x"] + player["w"] + 5 <= 64:
            player["x"] += 5
    # other actions are ignored for this level
    # -----------------------------------------

    # ----- handle collisions -----
    collided = []
    for obj in list(new_state):
        if obj["type"] in ("player", "counter"):
            continue
        if _intersect(_bbox(player), _bbox(obj)):
            collided.append(obj)

    # process each collided object
    for obj in collided:
        if obj["type"] == "refuel":
            # reset budget
            step_bar["w"] = 42
            step_bar["x"] = 13
            step_bar["tags"][0] = ["budget", "42"]
            # remove the refuel
            new_state.remove(obj)
        elif obj["type"] == "target":
            # decrease progress by 3
            prog = int(hud["tags"][0][1])
            prog = max(0, prog - 3)
            hud["tags"][0] = ["progress", str(prog), "pending"]
            new_state.remove(obj)
        elif obj["type"] == "button":
            prog = int(hud["tags"][0][1])
            prog = max(0, prog - 3)
            hud["tags"][0] = ["progress", str(prog), "pending"]
            new_state.remove(obj)

    # ----- update budget if not reset -----
    # check if a refuel was collected this turn
    refuel_collected = any(o["type"] == "refuel" for o in collided)
    if not refuel_collected:
        # decrement by 2, keep minimum 0
        old_budget = int(step_bar["tags"][0][1])
        new_budget = max(0, old_budget - 2)
        step_bar["w"] = new_budget
        step_bar["x"] = 13 + (42 - new_budget)
        step_bar["tags"][0] = ["budget", str(new_budget)]

    # ----- update frame -----
    # clear old player cells (set to floor colour 12 -> hex 'c')
    for y in range(old_bbox[1], old_bbox[3] + 1):
        for x in range(old_bbox[0], old_bbox[2] + 1):
            if 0 <= y < 64 and 0 <= x < 64:
                new_frame[y][x] = 12   # colour 'c'

    # draw new player cells (colour 3)
    new_bbox = _bbox(player)
    for y in range(new_bbox[1], new_bbox[3] + 1):
        for x in range(new_bbox[0], new_bbox[2] + 1):
            if 0 <= y < 64 and 0 <= x < 64:
                new_frame[y][x] = 3

    # store for continuity
    _last_state = new_state
    _last_frame = new_frame

    return new_state, new_frame
