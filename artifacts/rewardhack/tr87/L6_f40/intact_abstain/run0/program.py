"""Mechanics: legends in one row with a horizontal gap of <= 2 cells form one word.
The cursor sits on the leftmost legend of a word, 4 cells above the legend row.
Action 4 moves to the next word, wrapping to the next row and from last row to first.
Action 3 moves to the previous word (wrap at row start is a hypothesis, not observed).
Actions 1, 2 and others: no observed effect; legend edits were never observed.
"""
import copy

WORD_GAP = 2
CURSOR_OFFSET_Y = 4


def word_slots(state):
    rows = {}
    for o in state:
        if o["type"] == "legend":
            rows.setdefault(o["y"], []).append(o)
    slots = []
    for y in sorted(rows):
        prev_end = None
        for o in sorted(rows[y], key=lambda o: o["x"]):
            if prev_end is None or o["x"] - prev_end > WORD_GAP:
                slots.append((o["x"], y - CURSOR_OFFSET_Y))
            prev_end = o["x"] + o["w"]
    return slots


def update_cursor(cursor, slots, action):
    pos = (cursor["x"], cursor["y"])
    if pos not in slots or action not in (3, 4):
        return
    step = 1 if action == 4 else -1
    cursor["x"], cursor["y"] = slots[(slots.index(pos) + step) % len(slots)]


def transition_function(state, action):
    state = copy.deepcopy(state)
    slots = word_slots(state)
    for o in state:
        if o["type"] == "player":
            update_cursor(o, slots, action)
    return state
