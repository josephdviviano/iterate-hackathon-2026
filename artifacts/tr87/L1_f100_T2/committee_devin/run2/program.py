# Mechanics: word-edit game. Pairwise rule player(cursor)<->glyph(editable word letter):
#   ACTION3/ACTION4 move the cursor to the previous/next glyph column (sorted by x), wrapping.
#   ACTION1/ACTION2 edit the glyph under the cursor (cycle letter), but the schema has no
#   pixels, so the edit is invisible -> no observable change. Legends, targets, counter static.
# Unconfirmed: whether ACTION1/2 can ever change glyph geometry; counter never moves here.
import copy

MOVE = {3: -1, 4: +1}


def glyph_columns(state):
    return sorted({o["x"] for o in state if o.get("type") == "glyph"})


def cursor_step(cursor, cols, delta):
    """Guard 'select': cursor snaps to the neighbouring glyph column, wrapping around."""
    if not cols:
        return
    if cursor["x"] in cols:
        i = cols.index(cursor["x"])
    else:
        i = min(range(len(cols)), key=lambda k: abs(cols[k] - cursor["x"]))
    cursor["x"] = cols[(i + delta) % len(cols)]


def transition_function(state, action):
    new = copy.deepcopy(state)
    aid = action.get("action_id") if isinstance(action, dict) else action
    if aid in MOVE:
        cols = glyph_columns(new)
        for o in new:
            if o.get("type") == "player":
                cursor_step(o, cols, MOVE[aid])
    return new
