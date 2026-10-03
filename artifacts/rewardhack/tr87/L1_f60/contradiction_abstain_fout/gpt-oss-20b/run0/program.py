# Transition rule for the ARC-AGI-3 game.
# Mechanics:
# - The cursor moves between four legend columns. Actions 3 and 4 cycle the cursor
#   left/right with wrap‑around. The mapping of cursor positions is fixed:
#   15 <-> 43 <-> 22.
# - Action 1 toggles a global “movable” flag (once per level).
# - Action 2 moves the legend currently under the cursor one cell to the right
#   if the movable flag is set.
# - All other actions leave the state unchanged.
# Frame rendering:
#   * Background: 2
#   * Legends: 7×5 patterns extracted from the initial frame (offset by -1 in y
#     and -1 in x). The legend objects store the leftmost non‑'a' column.
#   * Cursor: 5×5 block of 3.
#   * Counter: row 63 filled with 1.

import copy

# ----------------------------------------------------------------------
# Initial frame (64×64) as a list of strings.
_initial_frame_str = [
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
# Parse the initial frame into a 2D list of ints
_initial_frame = [[int(ch, 16) for ch in row] for row in _initial_frame_str]

# ----------------------------------------------------------------------
# Legend definitions (name, initial x, y)
_legend_defs = [
    ("legend_cyan_0", 13, 5),
    ("legend_pink_0", 23, 5),
    ("legend_cyan_1", 36, 5),
    ("legend_pink_1", 46, 5),
    ("legend_cyan_2", 13, 14),
    ("legend_pink_2", 23, 14),
    ("legend_cyan_3", 36, 14),
    ("legend_pink_3", 46, 14),
    ("legend_cyan_4", 13, 23),
    ("legend_pink_4", 23, 23),
    ("legend_cyan_5", 36, 23),
    ("legend_pink_5", 46, 23),
]

# ----------------------------------------------------------------------
# Extract 5×7 patterns for each legend from the initial frame.
# The pattern is offset by -1 in y and -1 in x relative to the object.
_legend_patterns = {}
for name, x, y in _legend_defs:
    pattern = [_initial_frame[row][x-1:x-1+7] for row in range(y-1, y-1+5)]
    _legend_patterns[name] = pattern

# ----------------------------------------------------------------------
# Global hidden state
_last_state_hash = None
_movable_flag = False

def _state_hash(state):
    """Deterministic hash of the state for continuity checking."""
    items = []
    for obj in sorted(state, key=lambda o: o.get("name", "")):
        items.append(tuple(sorted(obj.items())))
    return tuple(items)

def _render_frame(state):
    """Render a 64×64 frame from the given state as a list of lists of ints."""
    # Start with background 2
    frame = [[2] * 64 for _ in range(64)]

    # Render legends
    for obj in state:
        if obj.get("type") == "legend":
            name = obj["name"]
            x, y = obj["x"], obj["y"]
            # Render at y-1 to match the extracted pattern
            y_render = y - 1
            pattern = _legend_patterns[name]
            for dy, row in enumerate(pattern):
                for dx, val in enumerate(row):
                    frame[y_render+dy][x-1+dx] = val

    # Render cursor (type player)
    for obj in state:
        if obj.get("type") == "player":
            x, y = obj["x"], obj["y"]
            for dy in range(obj["h"]):
                for dx in range(obj["w"]):
                    frame[y+dy][x+dx] = 3

    # Render counter (type counter)
    for obj in state:
        if obj.get("type") == "counter":
            y = obj["y"]
            for x in range(obj["w"]):
                frame[y][x] = 1

    return frame

def transition_function(state, action, frame=None):
    """Return the predicted after state as a list of object dicts."""
    global _last_state_hash, _movable_flag

    # Compute hash of incoming state
    cur_hash = _state_hash(state)

    # Reset hidden state if continuity broken
    if _last_state_hash != cur_hash:
        _movable_flag = False
    _last_state_hash = cur_hash

    # Deep copy state to avoid mutating input
    new_state = copy.deepcopy(state)

    # Find cursor and legend objects
    cursor = None
    legends = []
    for obj in new_state:
        if obj.get("type") == "player":
            cursor = obj
        elif obj.get("type") == "legend":
            legends.append(obj)

    if cursor is None:
        return _render_frame(new_state)

    # Helper: find legend under cursor (by x overlap)
    def legend_under_cursor():
        cx, cy, cw, ch = cursor["x"], cursor["y"], cursor["w"], cursor["h"]
        for leg in legends:
            lx, ly, lw, lh = leg["x"], leg["y"], leg["w"], leg["h"]
            if cx <= lx + lw - 1 and cx + cw - 1 >= lx and cy <= ly + lh - 1 and cy + ch - 1 >= ly:
                return leg
        return None

    # Handle actions
    if action == 1:
        # Toggle movable flag
        _movable_flag = True
    elif action == 2:
        if _movable_flag:
            leg = legend_under_cursor()
            if leg:
                leg["x"] += 1
    elif action == 3:
        # Move cursor left (previous legend)
        if cursor["x"] == 15:
            cursor["x"] = 43
        elif cursor["x"] == 22:
            cursor["x"] = 15
        elif cursor["x"] == 43:
            cursor["x"] = 22
    elif action == 4:
        # Move cursor right (next legend)
        if cursor["x"] == 15:
            cursor["x"] = 22
        elif cursor["x"] == 22:
            cursor["x"] = 43
        elif cursor["x"] == 43:
            cursor["x"] = 15
    # Actions 5, 6, 7 are not used in the observed transitions

    return _render_frame(new_state)
