# Mechanics: The game involves a cursor that can move between editable words and legends. The cursor moves left and right, and legends can be moved up and down. The cursor can also select legends, which causes them to move up or down.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    if cursor is None:
        return state

    if action == 1:
        # Move cursor left
        cursor['x'] -= 7
        if cursor['x'] < 13:
            cursor['x'] = 46
    elif action == 2:
        # Move cursor right
        cursor['x'] += 7
        if cursor['x'] > 46:
            cursor['x'] = 13
    elif action == 3:
        # Move cursor up
        cursor['y'] -= 9
        if cursor['y'] < 5:
            cursor['y'] = 52
    elif action == 4:
        # Move cursor down
        cursor['y'] += 9
        if cursor['y'] > 52:
            cursor['y'] = 5

    legends = [obj for obj in state if obj['type'] == 'legend']
    for legend in legends:
        if abs(legend['x'] - cursor['x']) <= 6:
            if action == 5:
                # Select legend and move it up
                legend['y'] -= 9
                if legend['y'] < 5:
                    legend['y'] = 52
            elif action == 6:
                # Select legend and move it down
                legend['y'] += 9
                if legend['y'] > 52:
                    legend['y'] = 5

    return state
