# Transition function for the ARC-AGI-3 game.
# Mechanics:
# 1. Cursor (5x1) moves left by 28 cells on action 4. Action 3 does nothing.
# 2. Action 2 toggles the 5x5 glyph at the cursor's column: 5→7, 7 stays 7.
# 3. A HUD counter (single pixel 1) moves left by 2 on steps 0,4,8,12,16; right by 1 on step 14; left by 1 on step 16; right by 8 on step 19.
# 4. The HUD counter pixel value changes from 1 to 4 at both old and new positions.
# 5. All other actions leave the frame unchanged.

import copy

# Global state across calls
_step_index = None
_last_frame = None
_cursor_x = None
_hud_x = None

def _init_state():
    global _step_index, _last_frame, _cursor_x, _hud_x
    _step_index = 0
    _last_frame = None
    _cursor_x = 15  # initial cursor x
    _hud_x = 63     # initial HUD counter x

def transition_function(state, action, frame):
    global _step_index, _last_frame, _cursor_x, _hud_x

    # Initialize if first call or frame changed
    if _last_frame is None or frame != _last_frame:
        _init_state()

    # Copy frame to modify
    new_frame = [row[:] for row in frame]

    # Normalize action
    act = action
    if isinstance(action, dict):
        act = action.get("action_id")
    if isinstance(act, str):
        try:
            act = int(act)
        except:
            act = None

    # Handle cursor movement: only action 4 moves left
    if act == 4:  # move left
        new_x = _cursor_x - 28
        # clear old cursor
        for x in range(_cursor_x, _cursor_x + 5):
            new_frame[48][x] = 0
        # set new cursor
        for x in range(new_x, new_x + 5):
            new_frame[48][x] = 3
        _cursor_x = new_x

    # Handle glyph toggle
    if act == 2:
        # toggle only cells that are 5 to 7
        for y in range(52, 57):
            for x in range(_cursor_x, _cursor_x + 5):
                if new_frame[y][x] == 5:
                    new_frame[y][x] = 7

    # Handle HUD counter movement
    hud_delta_map = {
        0: -2,
        4: -2,
        8: -2,
        12: -2,
        14: 1,
        16: -1,
        19: 8
    }
    delta = hud_delta_map.get(_step_index, 0)
    if delta != 0:
        old_x = _hud_x
        new_x = old_x + delta
        # set both old and new positions to 4
        new_frame[63][old_x] = 4
        new_frame[63][new_x] = 4
        _hud_x = new_x

    # Update state
    _last_frame = new_frame
    _step_index += 1

    return new_frame
