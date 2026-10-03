"""Mechanics: the cursor selects a word slot in a grid of legend rows. Legends in one row
whose gap is <= 2 cells form one word; the slots are the word starts. Action 4 moves the
cursor to the next slot, wraps to the first slot of the next row, last row wraps to row 0.
Action 3 moves to the previous slot (wrap at row start is assumed symmetric, unconfirmed).
Actions 1, 2 change nothing (their edit effect on legends is not observed). Cursor y = row y - 4.
"""
import copy

WORD_GAP = 2
CURSOR_DY = 4


def word_rows(state):
    rows = {}
    for o in state:
        if o["type"] == "legend":
            rows.setdefault(o["y"], []).append(o)
    slots = []
    for y in sorted(rows):
        starts, end = [], None
        for o in sorted(rows[y], key=lambda o: o["x"]):
            if end is None or o["x"] - end > WORD_GAP:
                starts.append(o["x"])
            end = o["x"] + o["w"]
        slots.extend((x, y) for x in starts)
    return slots


def move_cursor(cursor, slots, step):
    pos = (cursor["x"], cursor["y"] + CURSOR_DY)
    if pos not in slots:
        return
    x, y = slots[(slots.index(pos) + step) % len(slots)]
    cursor["x"], cursor["y"] = x, y - CURSOR_DY


def transition_function(state, action):
    state = copy.deepcopy(state)
    step = {4: 1, 3: -1}.get(action)
    if step is None:
        return state
    slots = word_rows(state)
    for o in state:
        if o["type"] == "player":
            move_cursor(o, slots, step)
    return state
