# This implementation renders the game frame by:
# 1. Moving the cursor (action 3: right 28, action 4: left 28).
# 2. Toggling the glyph pattern under the cursor (action 2: swap 5↔7) and
#    erasing the cursor from the frame.
# 3. Incrementing a single‑pixel counter at (61,63) on every action 1.
# 4. Drawing the cursor as a 5×5 block of color 3 only on action 1.
# 5. Drawing each glyph as a 5×5 block of its current pattern.
# Hidden state is stored in module globals and reset if the incoming
# state differs from the last returned state.

# Hidden state
_last_state = None
_cursor_pos = None
_glyph_patterns = {}   # key: glyph x, value: 5×5 pattern list of lists
_counter = 0

def _copy_frame(frame):
    return [row[:] for row in frame]

def _extract_pattern(frame, x, y):
    return [row[x:x+5] for row in frame[y:y+5]]

def _apply_pattern(frame, pattern, x, y):
    for dy, row in enumerate(pattern):
        frame[y+dy][x:x+5] = row

def _draw_cursor(frame, x, y):
    for dy in range(5):
        frame[y+dy][x:x+5] = [3]*5

def _clear_cursor(frame, x, y):
    for dy in range(5):
        frame[y+dy][x:x+5] = [0]*5

def _draw_counter(frame, value):
    # counter is a single pixel at (61,63)
    frame[63][61] = value

def transition_function(state, action, frame):
    global _last_state, _cursor_pos, _glyph_patterns, _counter

    # Reset hidden state if continuity broken
    if _last_state is None or _last_state != state:
        _cursor_pos = None
        _glyph_patterns = {}
        _counter = 0
        # initialize cursor position
        for obj in state:
            if obj.get("type") == "player":
                _cursor_pos = (obj["x"], obj["y"])
                break
        # initialize glyph patterns from the provided frame
        for obj in state:
            if obj.get("type") == "glyph":
                x, y = obj["x"], obj["y"]
                _glyph_patterns[x] = _extract_pattern(frame, x, y)

    # Update cursor position
    if action == 3:   # move right 28
        _cursor_pos = (_cursor_pos[0] + 28, _cursor_pos[1])
    elif action == 4: # move left 28
        _cursor_pos = (_cursor_pos[0] - 28, _cursor_pos[1])

    # Toggle glyph under cursor
    if action == 2:
        cx, cy = _cursor_pos
        if cx in _glyph_patterns:
            pat = _glyph_patterns[cx]
            # swap 5 and 7
            new_pat = [[7 if v == 5 else 5 if v == 7 else v for v in row] for row in pat]
            _glyph_patterns[cx] = new_pat

    # Increment counter on action 1
    if action == 1:
        _counter += 1

    # Render new frame
    new_frame = _copy_frame(frame)

    # Draw glyphs (glyph y is fixed at 52)
    for x, pat in _glyph_patterns.items():
        _apply_pattern(new_frame, pat, x, 52)

    # Handle cursor: erase on action 2, draw only on action 1
    if action == 2:
        _clear_cursor(new_frame, _cursor_pos[0], _cursor_pos[1])
    elif action == 1:
        _draw_cursor(new_frame, _cursor_pos[0], _cursor_pos[1])

    # Draw counter
    _draw_counter(new_frame, _counter)

    _last_state = state
    return new_frame
