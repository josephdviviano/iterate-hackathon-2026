# Mechanics: The cursor moves left and right between the editable words. The player can click to select a word, but the cursor movement is the primary action.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if cursor is None:
        return state

    if action == 2:
        # Move cursor left
        cursor['x'] = max(15, cursor['x'] - 7)
    elif action == 3:
        # Move cursor right
        cursor['x'] = min(43, cursor['x'] + 7)
    elif action == 4:
        # Swap cursor position with the next word
        if cursor['x'] == 15:
            cursor['x'] = 22
        elif cursor['x'] == 22:
            cursor['x'] = 29
        elif cursor['x'] == 29:
            cursor['x'] = 36
        elif cursor['x'] == 36:
            cursor['x'] = 43
        elif cursor['x'] == 43:
            cursor['x'] = 15

    return state
