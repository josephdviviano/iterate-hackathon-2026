# program.py
# Mechanics inferred:
# - Action 1: tick – hide the cursor.
# - Action 2: toggle the glyph under the cursor (invert its stored colours) and hide the cursor.
# - Action 3: move cursor to the right‑most editable glyph and show it.
# - Action 4: move cursor to the left‑most editable glyph and show it.
# - Action 6 (click) is unused.
# - Each editable glyph stores a pixel matrix in its “pixels” field. The stored matrix
#   represents the visible colour (5 or 7). When the cursor is hidden the glyph is shown
#   as stored. When the cursor is visible the glyph is hidden (background 0) by clearing
#   any cells it overlaps. Toggling a glyph swaps its stored colours (5↔7).
# - The cursor (colour 3) is drawn on background cells only; any glyph cells it covers
#   are cleared to background (0).

import copy

_last_state_key = None          # hash of the state we returned last time
_step_counter = 0               # total number of calls
_prev_cursor_rect = None        # (x, y, w, h) of cursor in previous frame
_cursor_visible = False         # whether the cursor should be drawn this step
_glyph_state = {}               # name -> mutable pixel matrix (list of lists)

def _state_key(state):
    """Hashable representation of a state (order‑independent)."""
    return tuple(sorted(
        (obj.get('type'), obj.get('name'), obj.get('x'), obj.get('y'))
        for obj in state
    ))

def _find_cursor(state):
    for obj in state:
        if obj.get('type') == 'player' and 'cursor' in obj.get('tags', []):
            return obj
    return None

def _editable_x_positions(state):
    """Sorted list of x‑coordinates of all editable glyphs."""
    xs = [obj['x'] for obj in state
          if obj.get('type') == 'glyph' and 'editable' in obj.get('tags', [])]
    return sorted(xs)

def _glyph_cells(state):
    """Set of (x, y) coordinates occupied by any editable glyph."""
    cells = set()
    for obj in state:
        if obj.get('type') == 'glyph' and 'editable' in obj.get('tags', []):
            x0, y0 = obj['x'], obj['y']
            for dy in range(obj['h']):
                for dx in range(obj['w']):
                    cells.add((x0 + dx, y0 + dy))
    return cells

def _clear_rect(frame, rect):
    """Set every cell inside rect to background (0)."""
    if rect is None:
        return
    x0, y0, w, h = rect
    for dy in range(h):
        y = y0 + dy
        if 0 <= y < len(frame):
            row = frame[y]
            for dx in range(w):
                x = x0 + dx
                if 0 <= x < len(row):
                    row[x] = 0

def _render_glyphs(frame, state):
    """Render all editable glyphs using their stored colours."""
    for obj in state:
        if obj.get('type') != 'glyph' or 'editable' not in obj.get('tags', []):
            continue
        name = obj['name']
        mat = _glyph_state.get(name)
        if not mat:
            continue
        x0, y0 = obj['x'], obj['y']
        for dy, row_pixels in enumerate(mat):
            y = y0 + dy
            if not (0 <= y < len(frame)):
                continue
            frame_row = frame[y]
            for dx, col in enumerate(row_pixels):
                x = x0 + dx
                if not (0 <= x < len(frame_row)):
                    continue
                frame_row[x] = col

def _invert_stored_glyph(name):
    """Swap stored colours 5↔7 for the given glyph."""
    mat = _glyph_state.get(name)
    if not mat:
        return
    for dy in range(len(mat)):
        for dx in range(len(mat[dy])):
            col = mat[dy][dx]
            if col == 5:
                mat[dy][dx] = 7
            elif col == 7:
                mat[dy][dx] = 5

def _clear_glyphs_under_cursor(frame, cursor_rect, glyph_cells):
    """Set to background (0) any glyph cells that lie inside cursor_rect."""
    if cursor_rect is None:
        return
    x0, y0, w, h = cursor_rect
    for dy in range(h):
        y = y0 + dy
        if not (0 <= y < len(frame)):
            continue
        row = frame[y]
        for dx in range(w):
            x = x0 + dx
            if not (0 <= x < len(row)):
                continue
            if (x, y) in glyph_cells:
                row[x] = 0

def transition_function(state, action, frame):
    global _last_state_key, _step_counter, _prev_cursor_rect, _cursor_visible, _glyph_state

    # Reset if continuity is broken.
    cur_key = _state_key(state)
    if _last_state_key is None or cur_key != _last_state_key:
        _step_counter = 0
        _prev_cursor_rect = None
        _cursor_visible = False
        _glyph_state = {}

    _step_counter += 1

    # Work on copies.
    new_state = copy.deepcopy(state)
    new_frame = [list(row) for row in frame]

    cursor = _find_cursor(new_state)
    if cursor is None:
        cursor = {'type': 'player', 'x': 0, 'y': 0, 'w': 5, 'h': 13,
                  'tags': ['cursor', 'selector']}
        new_state.append(cursor)

    # Initialise stored glyph matrices on first use.
    if not _glyph_state:
        for obj in new_state:
            if obj.get('type') == 'glyph' and 'editable' in obj.get('tags', []):
                name = obj['name']
                pixels = obj.get('pixels')
                if pixels:
                    _glyph_state[name] = [list(row) for row in pixels]

    # ----- handle action ---------------------------------------------------
    editable_xs = _editable_x_positions(new_state)

    if action == 1:
        _cursor_visible = False

    elif action == 2:
        # Toggle *all* editable glyphs (only one is under the cursor in practice)
        for name in list(_glyph_state.keys()):
            _invert_stored_glyph(name)
        _cursor_visible = False

    elif action == 3:
        # Move cursor to the right‑most editable glyph and show it.
        if editable_xs:
            cursor['x'] = max(editable_xs)
        _cursor_visible = True

    elif action == 4:
        # Move cursor to the left‑most editable glyph and show it.
        if editable_xs:
            cursor['x'] = min(editable_xs)
        _cursor_visible = True

    elif isinstance(action, dict) and action.get('action_id') == 6:
        # Click – not observed.
        pass

    # ----- render new frame ------------------------------------------------
    # 1. Clear previous cursor rectangle (restore background and glyphs).
    _clear_rect(new_frame, _prev_cursor_rect)

    # 2. Render all glyphs (using stored colours).
    _render_glyphs(new_frame, new_state)

    # 3. Render cursor (colour 3) on background cells only.
    if _cursor_visible:
        glyph_cells = _glyph_cells(new_state)
        for dy in range(cursor['h']):
            y = cursor['y'] + dy
            if 0 <= y < len(new_frame):
                row = new_frame[y]
                for dx in range(cursor['w']):
                    x = cursor['x'] + dx
                    if 0 <= x < len(row):
                        if (x, y) in glyph_cells:
                            continue          # will be cleared later
                        row[x] = 3
        # Hide any glyph cells that the cursor overlaps.
        _clear_glyphs_under_cursor(new_frame,
                                   (cursor['x'], cursor['y'],
                                    cursor['w'], cursor['h']),
                                   glyph_cells)

    # Remember current cursor rectangle for the next step.
    _prev_cursor_rect = (cursor['x'], cursor['y'],
                         cursor['w'], cursor['h']) if _cursor_visible else None

    # Store state key for continuity checking.
    _last_state_key = _state_key(new_state)

    return new_frame
