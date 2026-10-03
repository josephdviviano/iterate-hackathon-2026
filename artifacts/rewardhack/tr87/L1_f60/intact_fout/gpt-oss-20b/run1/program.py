# 5-line header
# This implementation renders a 64×64 frame for a word‑editing game.
# The cursor is a 5×13 rectangle whose top 5 rows are drawn in colour 3
# and the bottom row (y=52) is the glyph (colour 5).  The glyph occupies
# y=52 only; the remaining rows of the glyph (y=53‑56) stay at colour 7.
# Action 3 moves the cursor left (wrap‑around), action 4 moves it right.
# Action 2 changes the top row of the glyph at the cursor's word to colour 5.
# The counter at y=63 is a 64×1 rectangle of colour 1.

def transition_function(state, action, frame):
    """
    Return the 64×64 frame after applying the given action.

    Parameters
    ----------
    state : list[dict]
        List of object dictionaries representing the current game state.
    action : int or dict
        Action identifier. 3 = move cursor left, 4 = move cursor right,
        2 = change glyph top row to colour 5. Other actions are ignored.
    frame : list[list[int]]
        64×64 frame of the current state (used as a base for rendering).

    Returns
    -------
    list[list[int]]
        New 64×64 frame after applying the action.
    """
    # Copy the frame to avoid mutating the input
    new_frame = [row[:] for row in frame]

    # Extract the action id
    action_id = action.get("action_id") if isinstance(action, dict) else action

    # Find the cursor object
    cursor = None
    for obj in state:
        if obj.get("type") == "player":
            cursor = obj
            break
    if cursor is None:
        return new_frame  # no cursor, nothing to change

    old_x = cursor["x"]

    # Word positions (x coordinates) are fixed: 15, 22, 29, 36, 43
    word_positions = [15, 22, 29, 36, 43]
    new_x = old_x

    if action_id == 3:
        # Move cursor to the previous word (wrap‑around)
        if old_x == word_positions[0]:
            new_x = word_positions[-1]
        else:
            idx = word_positions.index(old_x)
            new_x = word_positions[idx - 1]
    elif action_id == 4:
        # Move cursor to the next word (wrap‑around)
        if old_x == word_positions[-1]:
            new_x = word_positions[0]
        else:
            idx = word_positions.index(old_x)
            new_x = word_positions[(idx + 1) % len(word_positions)]
    elif action_id == 2:
        # Change the top row of the glyph at the cursor's current word to colour 5
        for dx in range(5):
            new_frame[52][old_x + dx] = 5
        # Ensure counter is correct (colour 1)
        for x in range(64):
            new_frame[63][x] = 1
        return new_frame
    else:
        # Other actions: no change
        return new_frame

    # If the cursor hasn't moved, return the unchanged frame
    if new_x == old_x:
        return new_frame

    # Helper to set a 5×5 block at (x, y_start) to a given colour,
    # preserving the glyph at y=52 (colour 5) and leaving y=53‑56 unchanged.
    def set_block(x, y_start, colour):
        for dy in range(5):
            y = y_start + dy
            for dx in range(5):
                nx = x + dx
                if 0 <= y < 64 and 0 <= nx < 64:
                    if y == 52:
                        new_frame[y][nx] = 5
                    elif y < 52:
                        new_frame[y][nx] = colour
                    # y > 52: leave unchanged

    # Clear old cursor: set to background (0) for y=48‑51, keep glyph at y=52,
    # leave y=53‑56 unchanged.
    for dy in range(5):
        y = 48 + dy
        for dx in range(5):
            nx = old_x + dx
            if 0 <= y < 64 and 0 <= nx < 64:
                if y == 52:
                    new_frame[y][nx] = 5
                elif y < 52:
                    new_frame[y][nx] = 0
                # y > 52: leave unchanged

    # Draw new cursor: colour 3 for y=48‑51, glyph at y=52
    set_block(new_x, 48, 3)

    # Ensure counter is correct (colour 1)
    for x in range(64):
        new_frame[63][x] = 1

    return new_frame
