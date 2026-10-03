"""Mechanics: legends form rows (same y); in a row, legends with a gap <= 2 form one word.
The cursor sits on the first legend of a word, offset vertically by its row offset.
Action 4 moves to the next word (wraps to next row, last row wraps to first row).
Action 3 moves to the previous word (wrap is assumed symmetric; not observed).
Hypothesis not confirmed: actions 1, 2, 5, 6, 7 change nothing (no effect observed).
"""

WORD_GAP = 2


def _word_starts(legends):
    rows = {}
    for o in legends:
        rows.setdefault(o["y"], []).append(o)
    slots = []
    for y in sorted(rows):
        prev_end = None
        for o in sorted(rows[y], key=lambda o: o["x"]):
            if prev_end is None or o["x"] - prev_end > WORD_GAP:
                slots.append((o["x"], y))
            prev_end = o["x"] + o["w"]
    return slots


def _current_slot(cursor, slots):
    for i, (x, y) in enumerate(slots):
        if x == cursor["x"] and cursor["y"] <= y < cursor["y"] + cursor["h"]:
            return i
    return None


def transition_function(state, action):
    if action not in (3, 4):
        return state
    cursor = next((o for o in state if o["type"] == "player"), None)
    legends = [o for o in state if o["type"] == "legend"]
    if cursor is None or not legends:
        return state
    slots = _word_starts(legends)
    i = _current_slot(cursor, slots)
    if i is None:
        return state
    offset = slots[i][1] - cursor["y"]
    j = (i + (1 if action == 4 else -1)) % len(slots)
    cursor["x"] = slots[j][0]
    cursor["y"] = slots[j][1] - offset
    return state
