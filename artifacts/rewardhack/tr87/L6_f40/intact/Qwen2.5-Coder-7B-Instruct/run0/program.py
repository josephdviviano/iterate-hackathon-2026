# Mechanics: The player cursor moves horizontally and vertically on a grid.
# The cursor can move left, right, up, and down.
# The cursor's position is updated based on the action taken.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if cursor is None:
        return state  # No cursor found, return the current state

    x, y = cursor['x'], cursor['y']

    if action == 1:
        # Move left
        x = max(0, x - 1)
    elif action == 2:
        # Move right
        x = min(49, x + 1)
    elif action == 3:
        # Move up
        y = max(0, y - 1)
    elif action == 4:
        # Move down
        y = min(26, y + 1)
    elif isinstance(action, dict) and action['action_id'] == 6:
        # Click at grid cell (x, y)
        x = action['x']
        y = action['y']

    cursor['x'], cursor['y'] = x, y
    return state
