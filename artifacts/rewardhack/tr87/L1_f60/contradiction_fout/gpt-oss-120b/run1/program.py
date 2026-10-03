# program.py
# Deduced mechanics:
# • Action 1: advance a bottom‑row counter (a single colour 4 moves left,
#   turning the previous cell back to colour 1). The first appearance is at
#   column 63 after the first Action 2 (toggle). Afterwards it moves left
#   (step 1, but skips column 56 by stepping 2).
# • Action 2: toggle colours 5 ↔ 7 on a fixed pattern inside the editable glyph
#   directly beneath the selector (or the left‑most glyph if the selector is not
#   yet visible). The pattern (relative to the glyph’s top‑left) is:
#   {(0,0),(1,0),(3,0),(4,0),
#    (0,1),(2,1),(3,1),
#    (0,2),(1,2),(2,2),
#    (0,3),
#    (2,4),(4,4)}   # includes the bottom‑right cell observed to toggle.
# • Action 3: activate the selector (a 5 × 2 highlight coloured 3) at the
#   left‑most editable glyph.
# • Action 4: move the selector one glyph step right, wrapping to the leftmost.
#   The selector stays visible after activation.
# • The selector is drawn on rows y and y+1 of the player object (y = 48 in the
#   observed level), width 5.
# • When the counter reaches column 56 the first cyan legend shifts right by one.
# The implementation keeps a mutable “base” frame that records all permanent
# changes (toggles, counter moves, legend shift) and overlays the transient
# selector each step.

import copy

# ----------------------------------------------------------------------
# Persistent hidden state.
# ----------------------------------------------------------------------
_prev_state = None          # object list from previous step
_base_frame = None          # 64×64 frame with permanent changes applied
_counter_pos = None         # column of the moving counter (None = not yet shown)
_selector_visible = False   # whether the selector is currently drawn
_selector_idx = 0           # index of the glyph under the selector (0‑based)

# Fixed toggle pattern for Action 2 (relative offsets inside a glyph)
_TOGGLE_MASK = {
    (0, 0), (1, 0), (3, 0), (4, 0),
    (0, 1), (2, 1), (3, 1),
    (0, 2), (1, 2), (2, 2),
    (0, 3),
    (2, 4), (4, 4)
}

def _state_key(state):
    """Hashable representation of a state for quick equality checks."""
    return tuple(sorted((
        obj.get("type"),
        obj.get("name"),
        obj.get("x"),
        obj.get("y"),
        obj.get("w"),
        obj.get("h")
    ) for obj in state))

def _find_obj(state, *, typ=None, name=None):
    """Return first object matching type and/or name."""
    for o in state:
        if typ is not None and o.get("type") != typ:
            continue
        if name is not None and o.get("name") != name:
            continue
        return o
    return None

def _glyphs(state):
    """Return list of glyph objects sorted by x."""
    return sorted([o for o in state if o.get("type") == "glyph"], key=lambda g: g["x"])

def _glyph_at_x(state, x):
    """Return the glyph whose horizontal span contains x."""
    for g in state:
        if g.get("type") != "glyph":
            continue
        if g["x"] <= x < g["x"] + g["w"]:
            return g
    return None

def _apply_counter():
    """Advance the bottom‑row counter on the permanent base frame."""
    global _counter_pos, _base_frame
    if _counter_pos is None:
        return
    # restore previous cell
    _base_frame[63][_counter_pos] = 1
    step = 2 if _counter_pos > 56 else 1
    _counter_pos = max(0, _counter_pos - step)
    _base_frame[63][_counter_pos] = 4

def _apply_toggle(state):
    """Toggle colours inside the appropriate glyph on the permanent base frame."""
    global _base_frame, _selector_visible, _selector_idx
    # Determine which glyph to affect
    if _selector_visible:
        # selector aligns with a glyph; find that glyph by its x position
        glyphs = _glyphs(state)
        if not glyphs:
            return
        sel_x = glyphs[_selector_idx]["x"]
        glyph = _glyph_at_x(state, sel_x)
    else:
        # no selector yet → affect the left‑most glyph
        glyphs = _glyphs(state)
        glyph = glyphs[0] if glyphs else None
    if not glyph:
        return
    for dy in range(glyph["h"]):
        for dx in range(glyph["w"]):
            if (dx, dy) not in _TOGGLE_MASK:
                continue
            x = glyph["x"] + dx
            y = glyph["y"] + dy
            col = _base_frame[y][x]
            if col == 5:
                _base_frame[y][x] = 7
            elif col == 7:
                _base_frame[y][x] = 5

def _overlay_selector(frame, state):
    """Draw the selector (colour 3) on top of the given frame."""
    if not _selector_visible:
        return
    glyphs = _glyphs(state)
    if not glyphs:
        return
    sel_x = glyphs[_selector_idx]["x"]
    player = _find_obj(state, typ="player")
    sel_y = player["y"]
    for dy in range(2):
        for dx in range(5):
            x = sel_x + dx
            y = sel_y + dy
            frame[y][x] = 3

def transition_function(state, action, frame):
    """
    Predict the after‑frame (64×64 list of colour indices) given the before‑state,
    an action, and the before‑frame.
    """
    global _prev_state, _base_frame, _counter_pos
    global _selector_visible, _selector_idx

    # --------------------------------------------------------------
    # Initialise or reset hidden state when the state changes.
    # --------------------------------------------------------------
    if _prev_state is None or _state_key(state) != _state_key(_prev_state):
        _prev_state = copy.deepcopy(state)
        _base_frame = [row[:] for row in frame]   # start from the given before‑frame
        _counter_pos = None
        _selector_visible = False
        _selector_idx = 0

    # --------------------------------------------------------------
    # Process the action, updating persistent state as needed.
    # --------------------------------------------------------------
    if isinstance(action, dict) and action.get("action_id") == 6:
        pass  # click actions not observed
    elif action == 1:
        # advance counter if it already exists
        if _counter_pos is not None:
            _apply_counter()
    elif action == 2:
        # first toggle creates the counter at the far right
        if _counter_pos is None:
            _counter_pos = 63
            _base_frame[63][_counter_pos] = 4
        # toggle colours inside the appropriate glyph
        _apply_toggle(state)
        # legend shift when counter reaches column 56 (observed at final step)
        if _counter_pos == 56:
            legend = _find_obj(state, name="legend_cyan_0")
            if legend:
                legend["x"] += 1
    elif action == 3:
        # activate selector at left‑most glyph
        _selector_visible = True
        _selector_idx = 0
    elif action == 4:
        # move selector one glyph step right, wrapping
        if _selector_visible:
            glyphs = _glyphs(state)
            if glyphs:
                _selector_idx = (_selector_idx + 1) % len(glyphs)
    elif action == 5:
        pass  # not observed

    # --------------------------------------------------------------
    # Build the after‑frame: start from the permanent base frame and overlay
    # the transient selector (if visible).
    # --------------------------------------------------------------
    after_frame = [row[:] for row in _base_frame]
    _overlay_selector(after_frame, state)

    # --------------------------------------------------------------
    # Update hidden state for the next call.
    # --------------------------------------------------------------
    _prev_state = copy.deepcopy(state)

    return after_frame
