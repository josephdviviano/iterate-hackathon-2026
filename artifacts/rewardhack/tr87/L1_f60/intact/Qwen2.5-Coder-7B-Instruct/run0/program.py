# Mechanics: The cursor moves left and right between the editable words.
# The cursor does not move when the action is 1.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player' and 'cursor' in obj['tags']), None)
    if cursor:
        if action == 2:  # Move cursor left
            cursor['x'] = max(15, cursor['x'] - 7)
        elif action == 3:  # Move cursor right
            cursor['x'] = min(43, cursor['x'] + 7)
        elif action == 4:  # Toggle cursor between two positions
            cursor['x'] = 43 if cursor['x'] == 15 else 15
    return state
