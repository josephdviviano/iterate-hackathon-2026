"""Mechanics: the cursor (player) moves between editable word slots.
Action 3 moves it one slot left, action 4 one slot right; both wrap around.
Actions 1, 2, 5, 7 and clicks change no observed object.
Unconfirmed: action 1 probably edits the glyph under the cursor (pixels not
extracted); the legend_cyan_0 shift on one action 2 is not explained.
"""


def _slot_xs(state):
    return sorted(o["x"] for o in state if o["type"] == "glyph")


def _move_cursor(cursor, slots, delta):
    if cursor["x"] not in slots:
        return
    i = (slots.index(cursor["x"]) + delta) % len(slots)
    cursor["x"] = slots[i]


def transition_function(state, action):
    out = [dict(o) for o in state]
    slots = _slot_xs(out)
    for o in out:
        if o["type"] == "player" and slots:
            if action == 3:
                _move_cursor(o, slots, -1)
            elif action == 4:
                _move_cursor(o, slots, 1)
    return out
