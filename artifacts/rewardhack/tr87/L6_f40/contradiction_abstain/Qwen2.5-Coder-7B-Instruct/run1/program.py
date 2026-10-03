# Mechanics: The game involves a cursor that can move horizontally and vertically on a grid. Legends can be moved by clicking on them. The cursor can also be moved by pressing keys 1, 2, 3, 4, and 7.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if cursor is None:
        return state

    if action == 1:
        cursor['x'] = (cursor['x'] + 1) % 50
    elif action == 2:
        cursor['y'] = (cursor['y'] + 1) % 50
    elif action == 3:
        cursor['x'] = (cursor['x'] - 1) % 50
    elif action == 4:
        cursor['y'] = (cursor['y'] - 1) % 50
    elif action == 7:
        pass  # No action for key 7
    elif isinstance(action, dict) and action['action_id'] == 6:
        legend = next((obj for obj in state if obj['type'] == 'legend' and obj['x'] == action['x'] and obj['y'] == action['y']), None)
        if legend:
            legend['x'] = (legend['x'] + 1) % 50

    return state
