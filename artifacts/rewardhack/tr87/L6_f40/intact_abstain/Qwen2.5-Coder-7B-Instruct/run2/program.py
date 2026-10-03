# Mechanics: The game involves a cursor that can move horizontally and vertically on a grid. The cursor's position is updated based on the action taken. The cursor can move left, right, up, and down. The cursor's movement is constrained within the grid boundaries.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if cursor is None:
        return state

    x, y = cursor['x'], cursor['y']
    if action == 1:  # Move left
        x = max(0, x - 1)
    elif action == 2:  # Move right
        x = min(50, x + 1)
    elif action == 3:  # Move up
        y = max(0, y - 1)
    elif action == 4:  # Move down
        y = min(64, y + 1)

    cursor['x'], cursor['y'] = x, y
    return state
