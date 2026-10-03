# Mechanics: The game involves a cursor that can move horizontally between
# editable words. Legends are static objects that can be moved horizontally
# by the cursor. The cursor can move left or right, and legends can be moved
# by the cursor's movement. No other objects are affected.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    legends = [obj for obj in state if obj['type'] == 'legend']
    
    if cursor is None:
        return state
    
    if action == 1:
        # Move cursor left
        cursor['x'] = max(0, cursor['x'] - 1)
    elif action == 2:
        # Move cursor right
        cursor['x'] = min(max(obj['x'] + obj['w'] for obj in state), cursor['x'] + 1)
    elif action == 3:
        # Move cursor to the leftmost legend
        cursor['x'] = min(legend['x'] for legend in legends)
    elif action == 4:
        # Move cursor to the rightmost legend
        cursor['x'] = max(legend['x'] + legend['w'] for legend in legends)
    elif action == 5:
        # Move cursor to the center of the screen
        cursor['x'] = 32
    
    # Move legends if they are within the cursor's range
    for legend in legends:
        if legend['x'] < cursor['x'] < legend['x'] + legend['w']:
            legend['x'] = cursor['x']
    
    return state
