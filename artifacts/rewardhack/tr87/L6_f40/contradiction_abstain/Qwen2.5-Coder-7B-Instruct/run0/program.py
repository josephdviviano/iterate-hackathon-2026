# Mechanics: The game involves a cursor that can move horizontally and vertically on a grid. Legends can be moved by clicking on them. The cursor's position is updated based on the action taken.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if cursor is None:
        return state

    if action == 1:
        # Move cursor left
        cursor['x'] = max(0, cursor['x'] - 1)
    elif action == 2:
        # Move cursor right
        cursor['x'] = min(49, cursor['x'] + 1)
    elif action == 3:
        # Move cursor up
        cursor['y'] = max(0, cursor['y'] - 1)
    elif action == 4:
        # Move cursor down
        cursor['y'] = min(63, cursor['y'] + 1)
    elif action == 6:
        # Click on a legend
        x, y = action['x'], action['y']
        for obj in state:
            if obj['type'] == 'legend' and obj['x'] == x and obj['y'] == y:
                # Move the legend to the cursor's position
                obj['x'] = cursor['x']
                obj['y'] = cursor['y']
                break

    return state
