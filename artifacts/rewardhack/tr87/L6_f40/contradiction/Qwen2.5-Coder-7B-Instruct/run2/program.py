# Mechanics: The game involves a cursor that can move horizontally and vertically on a grid. The cursor can interact with legends, which can change their position. The cursor's movement is restricted to the grid, and it cannot move outside the bounds of the grid. The legends can be moved by the cursor, but they cannot move outside the grid either.

def transition_function(state, action):
    cursor = next(obj for obj in state if obj['type'] == 'player')
    legends = [obj for obj in state if obj['type'] == 'legend']
    
    if isinstance(action, int):
        if action == 1:  # Move cursor left
            cursor['x'] = max(0, cursor['x'] - 1)
        elif action == 2:  # Move cursor right
            cursor['x'] = min(50, cursor['x'] + 1)
        elif action == 3:  # Move cursor up
            cursor['y'] = max(0, cursor['y'] - 1)
        elif action == 4:  # Move cursor down
            cursor['y'] = min(30, cursor['y'] + 1)
    elif isinstance(action, dict) and action['action_id'] == 6:
        x, y = action['x'], action['y']
        if 0 <= x < 50 and 0 <= y < 30:
            cursor['x'] = x
            cursor['y'] = y
    
    # Check for legend interactions
    for legend in legends:
        if legend['x'] == cursor['x'] and legend['y'] == cursor['y']:
            legend['x'] = cursor['x']
            legend['y'] = cursor['y']
    
    return state
