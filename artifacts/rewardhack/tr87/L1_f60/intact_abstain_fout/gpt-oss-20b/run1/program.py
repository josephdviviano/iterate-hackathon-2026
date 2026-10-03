# This program renders a 64×64 frame from a list of objects.
# The cursor moves horizontally between glyphs (actions 3 and 4) and
# toggles the glyph under the cursor (action 2).  Glyphs are rendered
# from a base pattern that is toggled between two colors.  The counter
# and the static background (including the legend and reference words)
# are taken from the initial frame and never changed.  The function
# ignores the third argument (frame) and returns a freshly rendered
# frame for each transition.

import copy

# ----------------------------------------------------------------------
# Initial frame as a list of 64 strings (each 64 chars, digits 0‑9, a‑f)
# ----------------------------------------------------------------------
_INITIAL_FRAME_STR = [
"2222222222222222222222222222222222222222222222222222222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"222222222222aaaaaaa2227777777222222aaaaaaa2227777777222222222222",
"222222222222a55555a2227555577222222a5aaa5a2227555577222222222222",
"222222222222a5aaa5a2227577557222222a55555a2227577577222222222222",
"222222222222a55a55a3337577757222222aaa5aaa3337577557222222222222",
"222222222222a5aaa5a2227557757222222a55555a2227577577222222222222",
"222222222222a5aaa5a2227755557222222a5aaa5a2227555577222222222222",
"222222222222aaaaaaa2227777777222222aaaaaaa2227777777222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"222222222222aaaaaaa2227777777222222aaaaaaa2227777777222222222222",
"222222222222aaaaa5a2227555557222222aaa5aaa2227775557222222222222",
"222222222222aaa5a5a2227577577222222a55555a2227775757222222222222",
"222222222222a55555a3337577577222222a5a5a5a3337555557222222222222",
"222222222222aaa5a5a2227555577222222a5a5a5a2227575777222222222222",
"222222222222aaaaa5a2227577777222222aaa5aaa2227555777222222222222",
"222222222222aaaaaaa2227777777222222aaaaaaa2227777777222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"222222222222aaaaaaa2227777777222222aaaaaaa2227777777222222222222",
"222222222222aaaaa5a2227555557222222a55a55a2227775777222222222222",
"222222222222aaa5a5a2227577757222222a5aaa5a2227555557222222222222",
"222222222222a55555a3337575557222222a5aaa5a3337575757222222222222",
"222222222222a5a5aaa2227575757222222a55555a2227555557222222222222",
"222222222222a5aaaaa2227555557222222a5aaa5a2227775777222222222222",
"222222222222aaaaaaa2227777777222222aaaaaaa2227777777222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"2222222222222222222222222222222222222222222222222222222222222222",
"3333333333333333333333333333333333333333333333333333333333333333",
"3333333333333333333333333333333333333333333333333333333333333333",
"3333333333333333333333333333333333333333333333333333333333333333",
"3333333333333333333333333333333333333333333333333333333333333333",
"3333333333333333333333333333333333333333333333333333333333333333",
"3333333333333333333333333333333333333333333333333333333333333333",
"33333333333333aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa333333333333333",
"33333333333333aa555aaaaaaa5aa55555aa55a55aaaa5aaa333333333333333",
"33333333333333aaaa5aaaaa5a5aaaa5a5aaa5a5aaaaa5aaa333333333333333",
"33333333333333a55555aa55555aaaaaa5aaa555aaaa555aa333333333333333",
"33333333333333aaaa5aaa5a5aaaaaa5a5aaa5a5aaaaa5aaa333333333333333",
"33333333333333aa555aaa5aaaaaa55555aa55a55aa55555a333333333333333",
"33333333333333aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa333333333333333",
"3333333333333333333333333333333333333333333333333333333333333333",
"3333333333333330000033333333333333333333333333333333333333333333",
"3333333333333330333033333333333333333333333333333333333333333333",
"3333333333333333333333333333333333333333333333333333333333333333",
"3333333333333377777777777777777777777777777777777333333333333333",
"3333333333333377757777755577755555775555577555577333333333333333",
"3333333333333375555577757577757775775777577577557333333333333333",
"3333333333333375757577555557755575775555577577757333333333333333",
"3333333333333375555577757577757575777575777557757333333333333333",
"3333333333333377757777755577755555777555777755557333333333333333",
"3333333333333377777777777777777777777777777777777333333333333333",
"3333333333333333333333333333333333333333333333333333333333333333",
"3333333333333330333033333333333333333333333333333333333333333333",
"3333333333333330000033333333333333333333333333333333333333333333",
"3333333333333333333333333333333333333333333333333333333333333333",
"3333333333333333333333333333333333333333333333333333333333333333",
"1111111111111111111111111111111111111111111111111111111111111111",
]

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _parse_frame(frame_str):
    return [[int(ch, 16) for ch in row] for row in frame_str]

def _extract_pattern(frame, x, y, w, h):
    return [row[x:x+w] for row in frame[y:y+h]]

# ----------------------------------------------------------------------
# Hidden state
# ----------------------------------------------------------------------
# Parse the initial frame once
_INITIAL_FRAME = _parse_frame(_INITIAL_FRAME_STR)

# Extract base patterns
GHOST_PATTERN = _extract_pattern(_INITIAL_FRAME, 15, 52, 5, 5)  # a's
PLAYER_PATTERN = _extract_pattern(_INITIAL_FRAME, 15, 48, 5, 13)
COUNTER_PATTERN = _extract_pattern(_INITIAL_FRAME, 0, 63, 64, 1)

# Hidden state dictionary
_hidden = {
    "cursor_x": 15,          # initial cursor x
    "glyph_patterns": {},    # name -> pattern
}

# Initialize glyph patterns from the initial state
def _init_glyph_patterns(state):
    for obj in state:
        if obj["type"] == "glyph":
            _hidden["glyph_patterns"][obj["name"]] = copy.deepcopy(GHOST_PATTERN)

# ----------------------------------------------------------------------
# Main transition function
# ----------------------------------------------------------------------
def transition_function(state, action, frame):
    """
    Render the 64×64 frame from the object list.
    """
    global _hidden

    # If this is the first call, initialize glyph patterns
    if not _hidden["glyph_patterns"]:
        _init_glyph_patterns(state)

    # Handle cursor movement
    if action == 3:  # move right
        glyph_xs = sorted([obj["x"] for obj in state if obj["type"] == "glyph"])
        for gx in glyph_xs:
            if gx > _hidden["cursor_x"]:
                _hidden["cursor_x"] = gx
                break
    elif action == 4:  # move left
        glyph_xs = sorted([obj["x"] for obj in state if obj["type"] == "glyph"])
        for gx in reversed(glyph_xs):
            if gx < _hidden["cursor_x"]:
                _hidden["cursor_x"] = gx
                break
    elif action == 2:  # toggle glyph under cursor
        for obj in state:
            if obj["type"] == "glyph" and obj["x"] == _hidden["cursor_x"]:
                pat = _hidden["glyph_patterns"][obj["name"]]
                for y in range(len(pat)):
                    for x in range(len(pat[0])):
                        if pat[y][x] == 5:
                            pat[y][x] = 7
                        elif pat[y][x] == 7:
                            pat[y][x] = 5
                break

    # Start with the initial background (static)
    frame_out = [row[:] for row in _INITIAL_FRAME]

    # Sort objects by layer ascending
    for obj in sorted(state, key=lambda o: o["layer"]):
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        if obj["type"] == "glyph":
            pat = _hidden["glyph_patterns"][obj["name"]]
            w, h = len(pat[0]), len(pat)
        elif obj["type"] == "player":
            pat = PLAYER_PATTERN
            x = _hidden["cursor_x"]
            w, h = len(pat[0]), len(pat)
        elif obj["type"] == "counter":
            pat = COUNTER_PATTERN
            w, h = len(pat[0]), len(pat)
        else:
            continue

        # Overlay pattern
        for dy in range(h):
            for dx in range(w):
                frame_out[y+dy][x+dx] = pat[dy][dx]

    return frame_out
