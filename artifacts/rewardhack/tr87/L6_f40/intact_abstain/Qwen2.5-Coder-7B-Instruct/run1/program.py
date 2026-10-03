# Mechanics: The game involves a cursor that can move horizontally and vertically on a grid. The cursor's position is updated based on the action taken. The cursor can move left, right, up, and down. The cursor's position is represented by its x and y coordinates.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if cursor is None:
        return state

    new_x = cursor['x']
    new_y = cursor['y']

    if action == 1:  # Move left
        new_x -= 10
    elif action == 2:  # Move right
        new_x += 10
    elif action == 3:  # Move up
        new_y -= 12
    elif action == 4:  # Move down
        new_y += 12
    elif isinstance(action, dict) and action['action_id'] == 6:
        new_x = action['x']
        new_y = action['y']

    # Ensure the cursor stays within the grid boundaries
    grid_width = 50
    grid_height = 60
    new_x = max(0, min(new_x, grid_width - 5))
    new_y = max(0, min(new_y, grid_height - 13))

    new_cursor = {**cursor, 'x': new_x, 'y': new_y}
    return [obj if obj != cursor else new_cursor for obj in state]
