# Mechanics: The game involves a cursor that can move horizontally and vertically on a grid. Legends can be moved by clicking on them. The cursor can also move to a reference word by clicking on it. The game state is updated based on the cursor's position and the objects it interacts with.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if not cursor:
        return state

    if action == 1:
        cursor['x'] = (cursor['x'] + 1) % 50
    elif action == 2:
        cursor['y'] = (cursor['y'] + 1) % 50
    elif action == 3:
        cursor['x'] = (cursor['x'] - 1) % 50
    elif action == 4:
        cursor['y'] = (cursor['y'] - 1) % 50
    elif action == 6:
        x, y = action['x'], action['y']
        for obj in state:
            if obj['x'] == x and obj['y'] == y:
                if obj['type'] == 'legend':
                    obj['x'] = cursor['x']
                    obj['y'] = cursor['y']
                elif obj['type'] == 'target':
                    cursor['x'] = obj['x']
                    cursor['y'] = obj['y']
                break

    return state
