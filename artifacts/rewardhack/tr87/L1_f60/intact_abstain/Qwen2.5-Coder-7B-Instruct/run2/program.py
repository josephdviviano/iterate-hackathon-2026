# Mechanics: The game has a cursor that can move horizontally between editable words. The cursor's x-coordinate changes based on the action taken. The rest of the objects remain unchanged.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player' and 'cursor' in obj['tags']), None)
    if cursor:
        if action == 2:
            cursor['x'] = max(0, cursor['x'] - 7)
        elif action == 3:
            cursor['x'] = min(50, cursor['x'] + 7)
        elif action == 4:
            cursor['x'] = 22
    return state
