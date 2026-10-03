# ARC-AGI-3 transition model
# Mechanics inferred:
# - Actions 1 and 2 trigger a glyph animation step.
#   Each glyph (type "glyph") is a 5×5 bitmap where colour 5 = 0 and colour 7 = 1.
#   The bitmap is interpreted as a little‑endian binary number with reverse
#   row‑major ordering (bottom‑right cell = least‑significant bit, then left,
#   then up). On an animation step the number is decremented by one (mod 2^(w·h))
#   and the resulting bits are written back as colours 5/7. The visual glyph is
#   drawn one row **above** its stored y coordinate (i.e. at y‑1 … y+3).
# - Action 2 also shifts a cyan legend (type "legend", tag "cyan") one cell to
#   the right if the cursor overlaps it.
# - Action 3 moves the cursor to the previous glyph (wrap around);
#   Action 4 moves the cursor to the next glyph (wrap around).
#   The cursor rectangle (colour 3) is drawn four rows above the top of the
#   target glyph. After a move the cursor is drawn **full** (all 13 rows = 3).
#   After the first non‑move action (action 2) the top two rows become
#   background 0; after any later non‑move action (action 1) the top three rows
#   become background 0. The cursor never overwrites glyph cells (5/7).
# - A move‑counter lives on the bottom row (y = 63). Each cursor move (actions 3
#   or 4) shifts the counter digit left by two cells and paints it colour 4;
#   the previous digit is cleared to colour 1.
# - Action 1 otherwise does nothing; Action 2 does not move the cursor.

import copy

# hidden state kept between calls
_hidden = {
    "last_after_state": None,   # for continuity checking
    "cursor_idx": 0,            # index of the glyph the cursor currently points at
    "counter_x": 61,            # x‑position of the next counter digit
    "clear_rows": 0,            # how many top rows of the cursor should be cleared (0 = full)
}

def _find_objects(state, typ=None, tag=None):
    """Return objects matching a type and/or a tag."""
    res = []
    for obj in state:
        if typ is not None and obj.get("type") != typ:
            continue
        if tag is not None and tag not in obj.get("tags", []):
            continue
        res.append(obj)
    return res

def _overlaps(a, b):
    """True if rectangles a and b overlap (inclusive)."""
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def _draw_rect(frame, rect, colour):
    """Fill a rectangle (dict with x,y,w,h) with the given colour."""
    for dy in range(rect["h"]):
        y = rect["y"] + dy
        for dx in range(rect["w"]):
            x = rect["x"] + dx
            frame[y][x] = colour

def _decrement_glyph(frame, glyph):
    """Treat the glyph rectangle as a little‑endian binary number (5→0, 7→1)
    with reverse row‑major ordering and decrement it by one.
    The visual glyph is drawn starting at y‑1 (one row above the stored y)."""
    w, h = glyph["w"], glyph["h"]
    # read bits from visual area (y‑1 … y+h‑2)
    bits = []
    for ry in range(h - 1, -1, -1):
        y = glyph["y"] - 1 + ry
        for rx in range(w - 1, -1, -1):
            x = glyph["x"] + rx
            bits.append(1 if frame[y][x] == 7 else 0)
    # decrement
    val = 0
    for i, b in enumerate(bits):
        val |= (b << i)
    val = (val - 1) % (1 << (w * h))
    # write back
    idx = 0
    for ry in range(h - 1, -1, -1):
        y = glyph["y"] - 1 + ry
        for rx in range(w - 1, -1, -1):
            x = glyph["x"] + rx
            bit = (val >> idx) & 1
            frame[y][x] = 7 if bit else 5
            idx += 1

def _draw_cursor(frame, cursor, clear_rows):
    """Draw the cursor rectangle.
    The top `clear_rows` rows become background 0.
    Remaining rows become 3 **unless** the cell already contains a glyph colour
    (5 or 7), in which case it is left unchanged."""
    # clear top part
    for dy in range(clear_rows):
        y = cursor["y"] + dy
        for dx in range(cursor["w"]):
            x = cursor["x"] + dx
            frame[y][x] = 0
    # draw the rest, preserving glyph cells
    for dy in range(clear_rows, cursor["h"]):
        y = cursor["y"] + dy
        for dx in range(cursor["w"]):
            x = cursor["x"] + dx
            if frame[y][x] in (0, 3):
                frame[y][x] = 3

def _clear_cursor(frame, cursor):
    """Erase the whole cursor rectangle (set to background 0)."""
    _draw_rect(frame, cursor, 0)

def _update_counter(frame, hidden):
    """Shift the counter digit left by two cells and set it to colour 4."""
    y = 63
    old_x = hidden["counter_x"]
    if 0 <= old_x < len(frame[0]):
        frame[y][old_x] = 1          # clear previous digit
    new_x = old_x - 2
    if new_x < 0:
        new_x = 0
    hidden["counter_x"] = new_x
    if 0 <= new_x < len(frame[0]):
        frame[y][new_x] = 4

def transition_function(state, action, frame):
    """
    Given before‑state, an action, and the before‑frame, return the after‑frame.
    """
    global _hidden

    # continuity check – reset hidden state if the incoming state differs
    if _hidden["last_after_state"] is not None:
        def norm(s):
            return sorted((obj.get("type"), obj.get("name",""), obj["x"], obj["y"]) for obj in s)
        if norm(state) != norm(_hidden["last_after_state"]):
            _hidden = {
                "last_after_state": None,
                "cursor_idx": 0,
                "counter_x": 61,
                "clear_rows": 0,
            }

    cursor = next((o for o in state if o.get("type") == "player"), None)
    glyphs = _find_objects(state, typ="glyph")
    glyphs.sort(key=lambda o: o["x"])

    # keep cursor_idx in sync with the glyph currently overlapped
    if cursor:
        for i, g in enumerate(glyphs):
            if _overlaps(cursor, g):
                _hidden["cursor_idx"] = i
                break

    new_frame = [row[:] for row in frame]

    if action == 1:
        for g in glyphs:
            _decrement_glyph(new_frame, g)
        _hidden["clear_rows"] = 3          # top three rows cleared after action 1
        if cursor:
            _draw_cursor(new_frame, cursor, _hidden["clear_rows"])

    elif action == 2:
        for g in glyphs:
            _decrement_glyph(new_frame, g)
        # shift overlapped cyan legend right by one cell
        if cursor:
            legends = _find_objects(state, typ="legend", tag="cyan")
            for leg in legends:
                if _overlaps(cursor, leg):
                    leg["x"] += 1
                    break
        _hidden["clear_rows"] = 2          # top two rows cleared after the first action 2
        if cursor:
            _draw_cursor(new_frame, cursor, _hidden["clear_rows"])

    elif action == 3:
        if glyphs:
            if cursor:
                _clear_cursor(new_frame, cursor)
            _hidden["cursor_idx"] = (_hidden["cursor_idx"] - 1) % len(glyphs)
            target = glyphs[_hidden["cursor_idx"]]
            cursor["x"] = target["x"]
            cursor["y"] = target["y"] - 4
            _hidden["clear_rows"] = 0        # full cursor after a move
            _draw_cursor(new_frame, cursor, 0)
            _update_counter(new_frame, _hidden)

    elif action == 4:
        if glyphs:
            if cursor:
                _clear_cursor(new_frame, cursor)
            _hidden["cursor_idx"] = (_hidden["cursor_idx"] + 1) % len(glyphs)
            target = glyphs[_hidden["cursor_idx"]]
            cursor["x"] = target["x"]
            cursor["y"] = target["y"] - 4
            _hidden["clear_rows"] = 0
            _draw_cursor(new_frame, cursor, 0)
            _update_counter(new_frame, _hidden)

    # store after‑state for continuity checking
    _hidden["last_after_state"] = copy.deepcopy(state)

    return new_frame
