# Mechanics: The game involves a cursor that can move horizontally and vertically on a grid. The cursor can interact with objects of type "legend" and "target". The cursor's movement is controlled by actions 1, 2, and 3. Action 1 moves the cursor right, action 2 moves the cursor down, and action 3 moves the cursor left. The cursor's position is updated based on the action taken.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if cursor is None:
        return state

    x, y = cursor['x'], cursor['y']

    if action == 1:  # Move right
        x = (x + 5) % 50
    elif action == 2:  # Move down
        y = (y + 13) % 64
    elif action == 3:  # Move left
        x = (x - 5) % 50

    cursor['x'] = x
    cursor['y'] = y

    return state
