# transition_function implements a simple 2‑D grid game.
# Mechanics inferred from the observations:
#  - Action 1: move player up 5 cells.
#  - Action 2: move player down 5 cells.
#  - Action 3: move player left 5 cells.
#  - Action 4: move player right 5 cells.
#  - Each action consumes 2 units of “budget”. The budget is shown as a
#    horizontal bar (colour 0xb) on rows 61‑62.  When the budget is spent the
#    bar shrinks by 2 cells and shifts right by 2 cells.
#  - Touching a visible object of type “refuel” or “target” resets the budget
#    to its maximum (42) and moves the bar back to its start position
#    (x = 13, width = 42).  The object disappears but its visual representation
#    is not part of the frame, so we only need to reset the bar.
#  - The player is a 5×5 block drawn with colour 0xc (decimal 12).  The floor
#    colour is 0x3 (decimal 3).  When the player moves, the old cells are
#    restored to the floor colour.
#  - The function is stateless: it recomputes the next frame solely from the
#    supplied state, action and before‑frame.

from copy import deepcopy

PLAYER_COLOR = 12   # hex 'c'
FLOOR_COLOR = 3     # hex '3'
BAR_COLOR = 11      # hex 'b'
BAR_ROWS = (61, 62)  # rows that contain the budget bar
PLAYER_SIZE = 5
MAX_BUDGET = 42
BAR_START_X = 13


def _rect_overlap(ax, ay, aw, ah, bx, by, bw, bh):
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def transition_function(state, action, frame):
    # ----------------------------------------------------------------------
    # locate relevant objects
    player = next(o for o in state if o.get("type") == "player")
    bar = next(
        o
        for o in state
        if o.get("type") == "counter"
        and any(isinstance(t, str) and t.startswith("budget") for t in o.get("tags", []))
    )
    # ----------------------------------------------------------------------
    # current player rectangle
    px, py = player["x"], player["y"]
    pw, ph = player["w"], player["h"]

    # ----------------------------------------------------------------------
    # compute new player position according to the action
    nx, ny = px, py
    if action == 1:          # up
        ny = py - PLAYER_SIZE
    elif action == 2:        # down
        ny = py + PLAYER_SIZE
    elif action == 3:        # left
        nx = px - PLAYER_SIZE
    elif action == 4:        # right
        nx = px + PLAYER_SIZE
    # (actions 5,6,… are ignored – they only affect the budget)

    # ----------------------------------------------------------------------
    # budget handling
    old_budget = int(next(t for t in bar["tags"] if isinstance(t, str) and t.startswith("budget")).split(",")[0].split()[-1])
    # default consumption
    new_budget = max(0, old_budget - 2)
    new_bar_x = bar["x"] + 2
    new_bar_w = new_budget

    # check for collision with a refuel or target object
    reset = False
    for obj in state:
        if obj.get("type") in ("refuel", "target") and obj.get("visible"):
            if _rect_overlap(px, py, pw, ph,
                             obj["x"], obj["y"], obj["w"], obj["h"]):
                reset = True
                break
    if reset:
        new_budget = MAX_BUDGET
        new_bar_x = BAR_START_X
        new_bar_w = MAX_BUDGET

    # ----------------------------------------------------------------------
    # start from a copy of the before frame
    new_frame = [row[:] for row in frame]

    # erase old player
    for dy in range(ph):
        for dx in range(pw):
            yy = py + dy
            xx = px + dx
            if 0 <= yy < 64 and 0 <= xx < 64:
                new_frame[yy][xx] = FLOOR_COLOR

    # draw new player
    for dy in range(ph):
        for dx in range(pw):
            yy = ny + dy
            xx = nx + dx
            if 0 <= yy < 64 and 0 <= xx < 64:
                new_frame[yy][xx] = PLAYER_COLOR

    # erase old budget bar
    old_bar_x, old_bar_w = bar["x"], bar["w"]
    for col in range(old_bar_x, old_bar_x + old_bar_w):
        for row in BAR_ROWS:
            if 0 <= row < 64 and 0 <= col < 64:
                new_frame[row][col] = FLOOR_COLOR

    # draw new budget bar
    for col in range(new_bar_x, new_bar_x + new_bar_w):
        for row in BAR_ROWS:
            if 0 <= row < 64 and 0 <= col < 64:
                new_frame[row][col] = BAR_COLOR

    return new_frame
