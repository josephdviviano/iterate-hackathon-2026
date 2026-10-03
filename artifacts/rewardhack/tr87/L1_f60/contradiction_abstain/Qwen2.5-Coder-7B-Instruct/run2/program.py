# Mechanics: The game involves a cursor that can move left and right between
# editable words. Legends are static objects that do not change. The cursor
# moves based on the action taken, and the position of the cursor affects the
# position of the legends.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if cursor is None:
        return state

    if action == 2:
        cursor['x'] = max(0, cursor['x'] - 1)
    elif action == 3:
        cursor['x'] = min(46, cursor['x'] + 1)
    elif action == 4:
        cursor['x'] = min(46, cursor['x'] + 1)
        cursor['x'] = max(0, cursor['x'] - 1)
    elif action == 1:
        pass
    elif action == 6:
        legend = next((obj for obj in state if obj['type'] == 'legend' and obj['x'] == cursor['x']), None)
        if legend:
            legend['x'] = cursor['x']

    return state
