# Mechanics: The game involves a cursor that can move horizontally and vertically on a grid. The cursor can interact with objects of type "legend" and "target". The cursor's position is updated based on the action taken. The "legend" objects can be moved horizontally, while the "target" objects remain stationary.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if not cursor:
        return state

    if action == 1:
        # Move cursor left
        cursor['x'] -= 1
    elif action == 2:
        # Move cursor right
        cursor['x'] += 1
    elif action == 3:
        # Move cursor up
        cursor['y'] -= 1
    elif action == 4:
        # Move cursor down
        cursor['y'] += 1
    elif action == 6:
        # Click at grid cell (x, y)
        x, y = action['x'], action['y']
        # Check if there is a legend at the clicked position
        legend = next((obj for obj in state if obj['type'] == 'legend' and obj['x'] == x and obj['y'] == y), None)
        if legend:
            # Move the legend to the cursor's position
            legend['x'] = cursor['x']
            legend['y'] = cursor['y']

    return state
