# program.py
# Mechanics (deduced):
# - The cursor is a horizontal line of colour 3 on row y=48, plus edge cells on rows
#   y=49 and y=59 (leftmost and rightmost positions). It starts at the leftmost
#   possible x (0) and moves between the editable glyph blocks (type "glyph").
#   Action 3 moves it right (wrap), action 4 moves it left (wrap). Action 2 clears
#   the cursor and forgets its position.
# - Editable glyphs are 5×5 blocks at y≈52. A fixed mask of offsets toggles colours
#   5 ↔ 7 inside a glyph. When the cursor is not over a glyph, actions 1 and 2
#   toggle the first glyph.
# - Action 2 also sets the three HUD cells at (61‑63, 63) to colour 4.
# - No other HUD changes are required.
# The implementation keeps hidden cursor state and resets it if the supplied
# state does not match the last returned state.

from copy import deepcopy

# Offsets to toggle inside a glyph (excluding the two cells three rows above)
_TOGGLE_MASK = [
    (0, 0), (1, 0), (3, 0), (4, 0),
    (0, 1), (2, 1), (3, 1),
    (0, 2), (1, 2), (2, 2),
    (0, 3),
    (2, 4), (4, 4),            # bottom‑right cell
]

_last_state_key = None   # hash of the last observed state
_last_frame = None       # frame we returned last time
_cursor_x = None         # current cursor x (None means cursor hidden)
_prev_cursor_x = None    # previous cursor x (for erasing)


def _state_key(state):
    """Deterministic hash of a state (ignores order of objects)."""
    return tuple(sorted(
        (obj.get('name'), obj.get('type'), tuple(sorted(obj.get('tags', []))),
         obj.get('x'), obj.get('y')) for obj in state))


def _find_objects(state, typ=None, tag=None):
    """Yield objects matching a type and/or a tag."""
    for obj in state:
        if typ is not None and obj.get('type') != typ:
            continue
        if tag is not None and tag not in obj.get('tags', []):
            continue
        yield obj


def _clear_cursor(frame, x):
    """Erase the cursor line (row 48) and its edge cells on rows 49 and 59."""
    for cx in range(x, x + 5):
        frame[48][cx] = 0
    frame[49][x] = 0
    frame[49][x + 4] = 0
    frame[59][x] = 0
    frame[59][x + 4] = 0


def _draw_cursor(frame, x):
    """Draw the cursor line (row 48) and its edge cells on rows 49 and 59."""
    for cx in range(x, x + 5):
        frame[48][cx] = 3
    frame[49][x] = 3
    frame[49][x + 4] = 3
    frame[59][x] = 3
    frame[59][x + 4] = 3


def _toggle_glyph(frame, top_x, top_y):
    """Toggle colours 5↔7 at the fixed mask positions within a glyph."""
    for dx, dy in _TOGGLE_MASK:
        x = top_x + dx
        y = top_y + dy
        c = frame[y][x]
        if c == 5:
            frame[y][x] = 7
        elif c == 7:
            frame[y][x] = 5


def transition_function(state, action, frame):
    """
    Predict the after‑frame given the before‑frame, the state and an action.
    Returns a 64×64 list of lists of colour indices.
    """
    global _last_state_key, _last_frame, _cursor_x, _prev_cursor_x

    # Reset hidden state if the incoming state differs from what we produced last time
    cur_key = _state_key(state)
    if _last_state_key != cur_key:
        _last_state_key = cur_key
        _last_frame = deepcopy(frame)
        # Initialise cursor: assume it starts visible at the leftmost position (x=0)
        # unless the frame already shows a cursor elsewhere (detected via edge cells).
        # Simple detection: look for the distinctive edge cells.
        found = None
        for x in range(0, 60):
            if (frame[49][x] == 3 and frame[49][x + 4] == 3 and
                frame[59][x] == 3 and frame[59][x + 4] == 3):
                found = x
                break
        _cursor_x = found if found is not None else 0
        _prev_cursor_x = None

    # Work on a fresh copy of the last frame
    new_frame = deepcopy(_last_frame)

    # Glyph objects sorted by x (editable blocks)
    glyphs = sorted(list(_find_objects(state, typ='glyph')), key=lambda o: o['x'])

    # Determine which glyph (if any) the cursor currently overlaps
    current_glyph = None
    if _cursor_x is not None:
        for g in glyphs:
            if _cursor_x >= g['x'] and _cursor_x < g['x'] + 5:
                current_glyph = g
                break

    def _target_glyph():
        return current_glyph if current_glyph is not None else (glyphs[0] if glyphs else None)

    # ------------------- action handling -------------------
    if action == 1:
        tg = _target_glyph()
        if tg:
            _toggle_glyph(new_frame, tg['x'], tg['y'])

    elif action == 2:
        # Clear cursor if it exists
        if _cursor_x is not None:
            _clear_cursor(new_frame, _cursor_x)
        _cursor_x = None
        _prev_cursor_x = None

        tg = _target_glyph()
        if tg:
            _toggle_glyph(new_frame, tg['x'], tg['y'])

        # HUD: set the three right‑most cells of the bottom row to colour 4
        for x in range(61, 64):
            new_frame[63][x] = 4

    elif action == 3:  # move cursor right (wrap)
        if glyphs:
            if _cursor_x is None:
                new_x = glyphs[0]['x']
            else:
                # find index of glyph currently under cursor
                idx = next((i for i, g in enumerate(glyphs)
                            if _cursor_x >= g['x'] and _cursor_x < g['x'] + 5), 0)
                new_idx = (idx + 1) % len(glyphs)
                new_x = glyphs[new_idx]['x']
                _clear_cursor(new_frame, _cursor_x)
                _prev_cursor_x = _cursor_x
            _draw_cursor(new_frame, new_x)
            _cursor_x = new_x

    elif action == 4:  # move cursor left (wrap)
        if glyphs:
            if _cursor_x is None:
                new_x = glyphs[-1]['x']
            else:
                idx = next((i for i, g in enumerate(glyphs)
                            if _cursor_x >= g['x'] and _cursor_x < g['x'] + 5), 0)
                new_idx = (idx - 1) % len(glyphs)
                new_x = glyphs[new_idx]['x']
                _clear_cursor(new_frame, _cursor_x)
                _prev_cursor_x = _cursor_x
            _draw_cursor(new_frame, new_x)
            _cursor_x = new_x

    # For actions that do not move the cursor, ensure it stays drawn
    if action not in (3, 4) and _cursor_x is not None:
        if _prev_cursor_x is not None and _prev_cursor_x != _cursor_x:
            _clear_cursor(new_frame, _prev_cursor_x)
        _draw_cursor(new_frame, _cursor_x)
        _prev_cursor_x = _cursor_x

    # Store frame for the next call
    _last_frame = deepcopy(new_frame)

    return new_frame
