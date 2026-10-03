# ARC-AGI-3 transition model
# Mechanics inferred:
# • The cursor is a 5‑cell wide selector occupying a single top row (y) of a
#   12‑row tall rectangle (y … y+11).  The selector colour is 0 on floor cells
#   (3); other rows become floor (3) unless a glyph occupies the cell.
# • Action 3 moves the cursor right by 28 cells, action 4 moves it left.
#   When moving, the old rectangle is cleared to floor (preserving glyphs),
#   then the new selector row (y) is drawn as 0 on floor cells.
# • Action 2 toggles the glyph beneath the cursor by converting every 7 → 5.
#   Action 1 toggles the same glyph by converting every 5 → 7.
#   Editable glyph objects are reported with a y‑coordinate that is three rows
#   above the actual top of the visible glyph, so we treat the glyph area as
#   rows (y‑3) … (y+height‑1).
# • All other actions leave the frame unchanged.
# The implementation follows these rules using only the supplied state and
# before‑frame; no memorisation of past frames is performed.

def transition_function(state, action, frame):
    """
    Predict the after‑frame for a given before‑frame, state and action.

    Parameters
    ----------
    state : list[dict]
        List of object dictionaries (read‑only).
    action : int or dict
        Action identifier (only integer IDs are used here).
    frame : list[list[int]]
        64×64 grid of colour indices (0‑15) representing the before‑frame.

    Returns
    -------
    list[list[int]]
        The predicted after‑frame.
    """
    # ----------------------------------------------------------------------
    # Locate cursor and editable glyph objects.
    # ----------------------------------------------------------------------
    cursor = None
    glyphs = []
    for obj in state:
        if obj.get("type") == "player":
            cursor = obj
        elif obj.get("type") == "glyph" and "editable" in obj.get("tags", []):
            glyphs.append(obj)

    if cursor is None:
        return [row[:] for row in frame]

    old_x, y = cursor["x"], cursor["y"]
    new_x = old_x
    if action == 3:          # move right
        new_x = old_x + 28
    elif action == 4:        # move left
        new_x = old_x - 28

    # ----------------------------------------------------------------------
    # Helper: does a cell belong to a given glyph?
    # Glyph cells extend three rows above the reported y coordinate.
    # ----------------------------------------------------------------------
    def in_glyph(xc, yc, g):
        return (g["x"] <= xc < g["x"] + g["w"] and
                g["y"] - 3 <= yc < g["y"] + g["h"])

    # ----------------------------------------------------------------------
    # Work on a mutable copy of the before‑frame.
    # ----------------------------------------------------------------------
    after = [row[:] for row in frame]

    # ----------------------------------------------------------------------
    # 1) Clear the cursor's old rectangle.
    #    - Row y (selector) becomes floor (3) unless a glyph occupies the cell.
    #    - Rows y+1 … y+11 become floor (3) unless a glyph occupies the cell.
    # ----------------------------------------------------------------------
    for dy in range(12):          # y … y+11
        yy = y + dy
        if not (0 <= yy < 64):
            continue
        for dx in range(cursor["w"]):
            xx = old_x + dx
            if not (0 <= xx < 64):
                continue
            if any(in_glyph(xx, yy, g) for g in glyphs):
                continue
            after[yy][xx] = 3

    # ----------------------------------------------------------------------
    # 2) Toggle the glyph under the cursor for actions 1 and 2.
    #    Action 2: 7 → 5
    #    Action 1: 5 → 7
    #    The visible glyph spans rows (y‑3) … (y+height‑1).
    # ----------------------------------------------------------------------
    if action in (1, 2):
        target = None
        for g in glyphs:
            if g["x"] <= old_x < g["x"] + g["w"]:
                target = g
                break
        if target:
            gx, gy, gw, gh = (target["x"], target["y"],
                              target["w"], target["h"])
            for dy in range(-3, gh):          # include three rows above
                yy = gy + dy
                if not (0 <= yy < 64):
                    continue
                for dx in range(gw):
                    xx = gx + dx
                    col = after[yy][xx]
                    if action == 2 and col == 7:
                        after[yy][xx] = 5
                    elif action == 1 and col == 5:
                        after[yy][xx] = 7

    # ----------------------------------------------------------------------
    # 3) Draw the selector at its new location (row y only).
    #    Write colour 0 on floor cells (3); leave glyph cells unchanged.
    # ----------------------------------------------------------------------
    yy = y
    if 0 <= yy < 64:
        for dx in range(cursor["w"]):
            xx = new_x + dx
            if not (0 <= xx < 64):
                continue
            if after[yy][xx] == 3:      # floor → selector
                after[yy][xx] = 0
            # glyph cells keep their colour

    # ----------------------------------------------------------------------
    # Update cursor position in the state (helps subsequent calls).
    # ----------------------------------------------------------------------
    for obj in state:
        if obj.get("type") == "player":
            obj["x"] = new_x
            break

    return after
