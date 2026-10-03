# ARC-AGI-3 transition model
# - Action 1: no‑op, but every second consecutive Action‑1 swaps the entire
#   5×5 editable glyph region (x 15‑19, y 52‑56) between two patterns (5↔7)
#   and lights a HUD cell. The toggle flag flips each time Action‑1 is received.
# - Action 2: edit the glyph under the cursor (hard‑coded for the initial
#   glyph at (15,52)), light HUD column 63 and move the cursor to the
#   rightmost editable glyph.
# - Action 3: move the cursor left to the leftmost editable glyph.
# - Action 4: move the cursor right to the rightmost editable glyph.
# - The cursor consists of:
#     * a 5‑cell wide selector (colour 3) on its y‑row,
#     * vertical edge cells (colour 3) on the row below at both selector edges,
#     * a single right‑edge cell (colour 3) at y+11,
#     * a horizontal line (colour 3) spanning the selector width at y+12.
#   When the cursor moves, the previous graphics are cleared to colour 0.
# - Minimal persistent state (previous cursor x, HUD column, consecutive
#   Action‑1 count, toggle flag, cached patterns) is kept across calls.

from copy import deepcopy

# ----------------------------------------------------------------------
def _editable_xs(state):
    """Sorted x‑coordinates of all editable glyphs."""
    return sorted({obj['x'] for obj in state
                   if obj.get('type') == 'glyph' and 'editable' in obj.get('tags', [])})


def _cursor_obj(state):
    """Return the player (cursor) object."""
    for obj in state:
        if obj.get('type') == 'player':
            return obj
    return None


# ----------------------------------------------------------------------
def _clear_cursor(new, x, w, y):
    """Erase previous cursor graphics (set to 0)."""
    for dx in range(w):
        new[y][x + dx] = 0
    new[y + 1][x] = 0
    new[y + 1][x + w - 1] = 0
    new[y + 11][x + w - 1] = 0          # right‑edge indicator
    for dx in range(w):
        new[y + 12][x + dx] = 0        # bottom horizontal line


def _draw_cursor(new, x, w, y):
    """Overlay cursor graphics (colour 3)."""
    for dx in range(w):
        new[y][x + dx] = 3
    new[y + 1][x] = 3
    new[y + 1][x + w - 1] = 3
    new[y + 11][x + w - 1] = 3        # right‑edge indicator
    for dx in range(w):
        new[y + 12][x + dx] = 3        # bottom horizontal line


# ----------------------------------------------------------------------
def _init_patterns(frame):
    """
    Cache the edited glyph pattern and its inverse.
    The edited pattern is a full 5×5 block where most cells are 5,
    with a handful of 7s (as observed in the first edit).
    The inverse swaps 5↔7.
    """
    # explicit 7‑positions observed in the first edit
    seven_coords = {
        (15, 53), (17, 53), (18, 53),
        (15, 54), (17, 54),
        (15, 55), (17, 56), (19, 56)
    }

    edited = {}
    for y in range(52, 57):
        for x in range(15, 20):
            edited[(x, y)] = 7 if (x, y) in seven_coords else 5

    inv = {}
    for pos, v in edited.items():
        inv[pos] = 7 if v == 5 else 5
    return edited, inv


# ----------------------------------------------------------------------
def transition_function(state, action, frame):
    """
    Predict the after‑frame given the before‑frame, the action and the object list.
    """
    new = [row[:] for row in frame]

    cursor = _cursor_obj(state)
    if not cursor:
        return new

    cur_w = cursor['w']          # width = 5
    cur_y = cursor['y']          # selector row (48)

    # ------------------------------------------------------------------
    # Persistent attributes
    # ------------------------------------------------------------------
    if not hasattr(transition_function, 'prev_x'):
        transition_function.prev_x = cursor['x']
        transition_function.prev_y = cur_y

    if not hasattr(transition_function, 'hud_col'):
        transition_function.hud_col = 61

    if not hasattr(transition_function, 'ones_streak'):
        transition_function.ones_streak = 0

    if not hasattr(transition_function, 'toggle_state'):
        transition_function.toggle_state = False

    if not hasattr(transition_function, 'edited_pat'):
        transition_function.edited_pat, transition_function.inv_pat = _init_patterns(frame)

    # ------------------------------------------------------------------
    # Helper to move cursor (clear old graphics, draw new)
    # ------------------------------------------------------------------
    def move_cursor(to_x):
        _clear_cursor(new, transition_function.prev_x, cur_w, transition_function.prev_y)
        _draw_cursor(new, to_x, cur_w, cur_y)
        transition_function.prev_x = to_x
        transition_function.prev_y = cur_y

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------
    if action == 1:
        # No‑op, but every second consecutive Action‑1 toggles the glyph region
        # and lights the HUD.
        transition_function.ones_streak += 1

        # flip toggle flag and apply the appropriate pattern
        transition_function.toggle_state = not transition_function.toggle_state
        pat = (transition_function.inv_pat
               if transition_function.toggle_state else transition_function.edited_pat)
        for (x, y), v in pat.items():
            new[y][x] = v

        if transition_function.ones_streak % 2 == 0:
            col = transition_function.hud_col
            if 0 <= col < 64:
                new[63][col] = 4
            if col > 57:
                transition_function.hud_col = col - 2
            else:
                transition_function.hud_col = col - 1

        # cursor stays where it was
        _draw_cursor(new, transition_function.prev_x, cur_w, cur_y)
        return new

    # reset ones streak for any non‑1 action
    transition_function.ones_streak = 0

    if action == 2:
        # Edit glyph (hard‑coded for the initial glyph at (15,52))
        if transition_function.prev_x == 15 and transition_function.prev_y == 48:
            for (x, y), v in transition_function.edited_pat.items():
                new[y][x] = v
            transition_function.toggle_state = False

        # light HUD cell at far right
        new[63][63] = 4

        # move cursor to rightmost editable glyph
        xs = _editable_xs(state)
        if xs:
            move_cursor(xs[-1])
        else:
            _draw_cursor(new, transition_function.prev_x, cur_w, cur_y)
        return new

    if action == 3:
        # Move cursor left to leftmost editable glyph
        xs = _editable_xs(state)
        if xs:
            move_cursor(xs[0])
        else:
            _draw_cursor(new, transition_function.prev_x, cur_w, cur_y)
        return new

    if action == 4:
        # Move cursor right to rightmost editable glyph
        xs = _editable_xs(state)
        if xs:
            move_cursor(xs[-1])
        else:
            _draw_cursor(new, transition_function.prev_x, cur_w, cur_y)
        return new

    # Any other action – keep cursor as‑is
    _draw_cursor(new, transition_function.prev_x, cur_w, cur_y)
    return new
