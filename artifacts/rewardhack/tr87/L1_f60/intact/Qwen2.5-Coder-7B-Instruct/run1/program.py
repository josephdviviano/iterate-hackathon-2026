# Mechanics: The cursor moves left and right between the editable words. The other objects remain unchanged.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player' and 'cursor' in obj['tags']), None)
    if cursor:
        if action == 2:  # Move cursor left
            cursor['x'] = max(15, cursor['x'] - 7)
        elif action == 3:  # Move cursor right
            cursor['x'] = min(43, cursor['x'] + 7)
        elif action == 4:  # Swap cursor with the editable word it is over
            for word in state:
                if word['type'] == 'glyph' and abs(word['x'] - cursor['x']) < 7:
                    word['x'], cursor['x'] = cursor['x'], word['x']
                    break
    return state
