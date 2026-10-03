# ARC-AGI-3 transition model
# - Action 1: no visible effect.
# - Action 3: move the cursor (player) to the rightmost editable column (x = 43).
# - Action 4: move the cursor back to the leftmost editable column (x = 15).
# - Action 2: toggle the colour of every glyph cell (5 ↔ 7) and advance a
#   simple right‑to‑left progress bar on the bottom HUD (colour 1 → 4).
#   The progress bar advances one cell per Action‑2 call.
# The implementation keeps minimal hidden state (cursor x position and
# progress‑bar index) and recomputes the after‑frame from the before‑frame.

from copy import deepcopy

def transition_function(state, action, frame):
    """
    Predict the after‑frame for a single transition.

    Parameters
    ----------
    state : list[dict]
        List of object dictionaries (used to locate glyph rectangles and the
        cursor).
    action : int or dict
        Action identifier (1‑5,7) or a click dict (not used here).
    frame : list[list[int]]
        64×64 grid of colour indices (0‑15).

    Returns
    -------
    list[list[int]]
        The predicted after‑frame.
    """
    # ----------------------------------------------------------------------
    # Persistent hidden state (stored as attributes on the function object)
    # ----------------------------------------------------------------------
    if not hasattr(transition_function, "progress"):
        transition_function.progress = 0          # number of Action‑2 seen
    if not hasattr(transition_function, "cursor_x"):
        # initialise cursor x from the supplied state (there is exactly one)
        cursor = next((o for o in state if o.get("type") == "player"), None)
        transition_function.cursor_x = cursor["x"] if cursor else 0

    # ----------------------------------------------------------------------
    # Helper: clear the cursor drawing from the frame
    # ----------------------------------------------------------------------
    def clear_cursor(fr):
        x = transition_function.cursor_x
        # clear the 5‑cell horizontal bar on row 48
        for cx in range(x, x + 5):
            if 0 <= cx < 64:
                fr[48][cx] = 0
        # clear side markers on row 49
        if 0 <= x < 64:
            fr[49][x] = 0
        if 0 <= x + 4 < 64:
            fr[49][x + 4] = 0

    # ----------------------------------------------------------------------
    # Helper: draw the cursor at its current x position
    # ----------------------------------------------------------------------
    def draw_cursor(fr):
        x = transition_function.cursor_x
        # draw the 5‑cell horizontal bar on row 48
        for cx in range(x, x + 5):
            if 0 <= cx < 64:
                fr[48][cx] = 3
        # side markers on row 49
        if 0 <= x < 64:
            fr[49][x] = 3
        if 0 <= x + 4 < 64:
            fr[49][x + 4] = 3

    # ----------------------------------------------------------------------
    # Start from a fresh copy of the before‑frame
    # ----------------------------------------------------------------------
    new_frame = [row[:] for row in frame]

    # ----------------------------------------------------------------------
    # Process the action
    # ----------------------------------------------------------------------
    if action == 1:
        # No visual change.
        pass

    elif action == 2:
        # 1) Toggle glyph colours (5 ↔ 7) using the glyph objects from the state.
        for obj in state:
            if obj.get("type") != "glyph":
                continue
            ox, oy = obj["x"], obj["y"]
            w, h = obj["w"], obj["h"]
            for dy in range(h):
                y = oy + dy
                if not (0 <= y < 64):
                    continue
                for dx in range(w):
                    x = ox + dx
                    if not (0 <= x < 64):
                        continue
                    val = new_frame[y][x]
                    if val == 5:
                        new_frame[y][x] = 7
                    elif val == 7:
                        new_frame[y][x] = 5

        # 2) Advance the HUD progress bar (bottom row, colour 1 → 4).
        # The bar fills from the rightmost cell (x = 63) leftwards.
        idx = transition_function.progress
        if idx < 64:
            target_x = 63 - idx
            if new_frame[63][target_x] == 1:
                new_frame[63][target_x] = 4
        transition_function.progress += 1

    elif action == 3:
        # Move cursor to the rightmost editable column (x = 43).
        clear_cursor(new_frame)
        transition_function.cursor_x = 43
        draw_cursor(new_frame)

    elif action == 4:
        # Move cursor back to the leftmost editable column (x = 15).
        clear_cursor(new_frame)
        transition_function.cursor_x = 15
        draw_cursor(new_frame)

    elif action == 5 or action == 7:
        # Not observed to affect the frame in the provided data.
        pass

    # ----------------------------------------------------------------------
    # Return the resulting frame
    # ----------------------------------------------------------------------
    return new_frame
