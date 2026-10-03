# Mechanics: The game involves a player-controlled cursor that can move horizontally and vertically on a grid. Legends can be moved by clicking on them. The cursor can also move to a reference word by clicking on it. The game state is updated based on the player's actions.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if cursor is None:
        return state

    if isinstance(action, int):
        if action == 1:
            cursor['x'] = (cursor['x'] + 1) % 50
        elif action == 2:
            cursor['y'] = (cursor['y'] + 1) % 50
        elif action == 3:
            cursor['x'] = (cursor['x'] - 1) % 50
        elif action == 4:
            cursor['y'] = (cursor['y'] - 1) % 50
        elif action == 7:
            pass  # No action
    elif isinstance(action, dict) and action['action_id'] == 6:
        x, y = action['x'], action['y']
        for obj in state:
            if obj['type'] == 'legend' and obj['x'] == x and obj['y'] == y:
                obj['x'] = (obj['x'] + 1) % 50
                break
            elif obj['type'] == 'target' and obj['x'] == x and obj['y'] == y:
                cursor['x'] = x
                cursor['y'] = y
                break

    return state
