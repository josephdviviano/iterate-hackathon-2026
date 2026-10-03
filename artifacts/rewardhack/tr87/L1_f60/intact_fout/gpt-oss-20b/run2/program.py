# Transition function for ARC-AGI-3 game.
# Mechanics:
# - The only dynamic object is the cursor (player). It moves left/right among
#   five glyph columns (x = 15, 22, 29, 36, 43). Action 3 moves left,
#   action 4 moves right, all other actions leave it in place.
# - All other objects (legend, glyph, counter) are static and unchanged.
# - The function accepts an optional frame argument (ignored) to satisfy the
#   checker signature.

def transition_function(state, action, frame=None):
    # Find the cursor object
    cursor = next(o for o in state if o.get('type') == 'player')
    glyph_positions = [15, 22, 29, 36, 43]
    idx = glyph_positions.index(cursor['x'])
    if action == 3:          # move left
        new_idx = (idx - 1) % len(glyph_positions)
    elif action == 4:        # move right
        new_idx = (idx + 1) % len(glyph_positions)
    else:
        new_idx = idx
    new_x = glyph_positions[new_idx]

    # Build new state list
    new_state = []
    for obj in state:
        if obj.get('type') == 'player':
            new_obj = obj.copy()
            new_obj['x'] = new_x
            new_state.append(new_obj)
        else:
            new_state.append(obj.copy())
    return new_state
