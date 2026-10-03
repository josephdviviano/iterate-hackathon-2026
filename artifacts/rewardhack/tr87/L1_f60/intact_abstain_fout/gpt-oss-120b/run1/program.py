# program.py
# Inferred mechanics:
# 1. Editable glyphs (type "glyph") are 5×5 blocks using colour 5 (off) and 7 (on).
#    The glyph under the cursor (matching x) is incremented as a binary counter.
#    The least‑significant bit is the top‑left cell; bits are ordered column‑major
#    (iterate x, then y). Actions 1 and 2 both perform this increment.
# 2. The cursor (type "player") is a selector consisting of:
#      • a 5×2 block at (x, y) coloured 3,
#      • a horizontal line 11 rows below (y+11) also coloured 3.
#    It is invisible initially. Action 3 makes it appear at the left glyph (x = 15);
#    action 4 moves it to the right glyph (x = 43). All other actions leave the
#    cursor unchanged and invisible.
# 3. The row immediately above each glyph (y = glyph.y − 1) is a permanent
#    highlight coloured 7, independent of the cursor.
# 4. The counter object is left untouched.

from copy import deepcopy

def _find_objects(state, obj_type):
    return [o for o in state if o.get("type") == obj_type]

def _read_block(frame, x0, y0, w, h):
    return [[frame[y][x] for x in range(x0, x0 + w)] for y in range(y0, y0 + h)]

def _write_block(frame, x0, y0, block, on_color=7, off_color=5):
    for dy, row in enumerate(block):
        for dx, bit in enumerate(row):
            frame[y0 + dy][x0 + dx] = on_color if bit else off_color

def _glyph_bits(frame, obj):
    raw = _read_block(frame, obj["x"], obj["y"], obj["w"], obj["h"])
    return [[cell == 7 for cell in row] for row in raw]

def _increment_bits_col_major(bits):
    """Increment a 5×5 boolean matrix treating bits column‑major (LSB at (0,0))."""
    w, h = 5, 5
    # flatten column‑major
    flat = [bits[y][x] for x in range(w) for y in range(h)]
    carry = 1
    for i in range(len(flat)):
        if carry == 0:
            break
        new = flat[i] + carry
        flat[i] = new % 2
        carry = new // 2
    # reshape back to 5×5 matrix (row‑major for writing)
    it = iter(flat)
    return [[next(it) for _ in range(w)] for _ in range(h)]

def _clear_cursor_region(frame, cursor):
    # top 5×2 block
    for dy in range(2):
        for dx in range(cursor["w"]):
            frame[cursor["y"] + dy][cursor["x"] + dx] = 0
    # bottom line at y+11
    y_line = cursor["y"] + 11
    for dx in range(cursor["w"]):
        frame[y_line][cursor["x"] + dx] = 0

def _draw_cursor(frame, cursor):
    for dy in range(2):
        for dx in range(cursor["w"]):
            frame[cursor["y"] + dy][cursor["x"] + dx] = 3
    y_line = cursor["y"] + 11
    for dx in range(cursor["w"]):
        frame[y_line][cursor["x"] + dx] = 3

def _set_highlights(frame, glyphs):
    for g in glyphs:
        y_high = g["y"] - 1
        if y_high < 0:
            continue
        for dx in range(g["w"]):
            frame[y_high][g["x"] + dx] = 7

def transition_function(state, action, frame):
    new_frame = deepcopy(frame)

    # ----------------------------------------------------------------------
    # Cursor handling (visible only on actions 3 and 4)
    # ----------------------------------------------------------------------
    cursor_objs = _find_objects(state, "player")
    if cursor_objs:
        cursor = cursor_objs[0]

        if action == 3:
            new_x = 15
        elif action == 4:
            new_x = 43
        else:
            new_x = cursor["x"]

        if new_x != cursor["x"]:
            _clear_cursor_region(new_frame, cursor)
            cursor["x"] = new_x

        if action in (3, 4):
            _draw_cursor(new_frame, cursor)

    # ----------------------------------------------------------------------
    # Glyph handling
    # ----------------------------------------------------------------------
    glyph_objs = _find_objects(state, "glyph")
    if glyph_objs:
        selected = None
        for g in glyph_objs:
            if cursor and g["x"] == cursor["x"]:
                selected = g
                break

        if action in (1, 2) and selected:
            bits = _glyph_bits(frame, selected)
            bits = _increment_bits_col_major(bits)
            _write_block(new_frame, selected["x"], selected["y"], bits)

        _set_highlights(new_frame, glyph_objs)

    # Counter and other objects are left unchanged.
    return new_frame
