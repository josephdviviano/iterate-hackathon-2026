# Mechanics: The game involves a cursor that can move left and right between
# editable words. Legends are static objects that can be moved by the cursor.
# The cursor moves by clicking on the left or right arrow keys (actions 2 and 3),
# and legends can be moved by the cursor (action 4). The cursor can also skip
# over legends (action 1).

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    legends = [obj for obj in state if obj['type'] == 'legend']
    
    if cursor and legends:
        if action == 2:  # Move cursor left
            cursor['x'] = max(0, cursor['x'] - 1)
        elif action == 3:  # Move cursor right
            cursor['x'] = min(max(legends, key=lambda x: x['x'])['x'] + 5, cursor['x'] + 1)
        elif action == 4:  # Move legend under cursor
            for legend in legends:
                if legend['x'] <= cursor['x'] < legend['x'] + legend['w']:
                    legend['x'] = cursor['x']
                    break
    
    return state
