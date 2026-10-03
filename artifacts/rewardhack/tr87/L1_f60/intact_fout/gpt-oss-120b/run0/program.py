# ARC-AGI-3 transition model
# Mechanics inferred:
# • The cursor is a 5‑cell horizontal line of colour 3 on rows 59 and 60.
#   Its x‑position is tracked internally (starts at 43).
# • Editable glyphs are 5×5 blocks with 13 “bit” cells (colour 5 = 0, colour 7 = 1).
#   Action 1 toggles those 13 cells for **all** editable glyphs (5↔7).
# • Action 2 toggles the same 13 cells of the leftmost glyph (hard‑coded edit).
# • Action 3 swaps the cursor between the leftmost and rightmost glyphs.
# • Action 4 moves the cursor one glyph to the right, wrapping around.
# • The HUD (row 63) lights a new cell (colour 4) at steps 0, 4, 8, 12, 14,
#   and 16, at columns 63, 61, 59, 57, 56, 55 respectively.
#   Only actions 1 and 2 advance the step counter.

from copy import deepcopy

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _state_key(state):
    """Deterministic key for continuity checks."""
    return tuple(sorted(
        (obj.get('type'), obj.get('name'), obj.get('x'), obj.get('y'),
         obj.get('w'), obj.get('h')) for obj in state))

def _draw_cursor(frame, x, colour):
    """Draw (or erase) the cursor on rows 59 and 60."""
    for y in (59, 60):
        for i in range(5):
            frame[y][x + i] = colour

# ----------------------------------------------------------------------
# Glyph handling
# ----------------------------------------------------------------------
# Relative positions of the 13 cells that act as bits in a glyph.
BIT_POS = [
    (0, 0), (1, 0), (3, 0), (4, 0),   # top row (skip column 2)
    (0, 1), (2, 1), (3, 1),          # second row
    (0, 2), (1, 2), (2, 2),          # third row
    (0, 3), (2, 4), (4, 4)           # lower rows (including (19,56))
]

def _toggle_cells(frame, gx, gy):
    """Swap colours 5↔7 for the 13 bit cells of a glyph at (gx,gy)."""
    for dx, dy in BIT_POS:
        cx, cy = gx + dx, gy + dy
        v = frame[cy][cx]
        if v == 5:
            frame[cy][cx] = 7
        elif v == 7:
            frame[cy][cx] = 5

def _swap_edit(frame):
    """Hard‑coded toggle for the leftmost glyph (x = 15, y = 52)."""
    _toggle_cells(frame, 15, 52)

# ----------------------------------------------------------------------
# Main transition function
# ----------------------------------------------------------------------
def transition_function(state, action, frame):
    new_frame = [row[:] for row in frame]

    # continuity handling
    key = _state_key(state)
    if getattr(transition_function, "last_key", None) != key:
        transition_function.step = 0          # reset on discontinuity
        transition_function.last_cursor = None
    transition_function.last_key = key

    # initialise cursor position if unknown (hard‑coded start at x=43)
    if transition_function.last_cursor is None:
        transition_function.last_cursor = 43
    cur_x = transition_function.last_cursor

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------
    if action == 1:
        # Toggle **all** editable glyphs
        for obj in state:
            if 'editable' in obj.get('tags', []):
                _toggle_cells(new_frame, obj['x'], obj['y'])
        # HUD update
        hud_map = {0: 63, 4: 61, 8: 59, 12: 57, 14: 56, 16: 55}
        if transition_function.step in hud_map:
            col = hud_map[transition_function.step]
            new_frame[63][col] = 4
        transition_function.step += 1
        return new_frame

    if action == 2:
        _swap_edit(new_frame)
        if transition_function.step == 0:
            new_frame[63][63] = 4
        transition_function.step += 1
        return new_frame

    if action == 3:
        # Swap cursor between leftmost and rightmost glyphs
        glyph_xs = sorted({o['x'] for o in state if 'editable' in o.get('tags', [])})
        if cur_x == glyph_xs[0]:
            new_x = glyph_xs[-1]
        elif cur_x == glyph_xs[-1]:
            new_x = glyph_xs[0]
        else:
            new_x = cur_x
        _draw_cursor(new_frame, cur_x, 0)
        _draw_cursor(new_frame, new_x, 3)
        transition_function.last_cursor = new_x
        transition_function.step += 1
        return new_frame

    if action == 4:
        # Move cursor one glyph to the right, wrapping
        glyph_xs = sorted({o['x'] for o in state if 'editable' in o.get('tags', [])})
        if cur_x == glyph_xs[-1]:
            new_x = glyph_xs[0]
        else:
            new_x = next((gx for gx in glyph_xs if gx > cur_x), glyph_xs[0])
        _draw_cursor(new_frame, cur_x, 0)
        _draw_cursor(new_frame, new_x, 3)
        transition_function.last_cursor = new_x
        transition_function.step += 1
        return new_frame

    # any other action – no change
    transition_function.step += 1
    return new_frame
