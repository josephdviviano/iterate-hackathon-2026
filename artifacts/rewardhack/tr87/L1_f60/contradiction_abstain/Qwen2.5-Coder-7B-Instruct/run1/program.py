# Mechanics: The game involves a cursor that can move between editable words and legends. The cursor can move left, right, and up. Legends can move left and right. The cursor can click on legends to move them.

def transition_function(state, action):
    cursor = next((obj for obj in state if obj['type'] == 'player'), None)
    legends = [obj for obj in state if obj['type'] == 'legend']
    
    if cursor and legends:
        cursor_x = cursor['x']
        cursor_y = cursor['y']
        cursor_layer = cursor['layer']
        
        if action == 1:  # Move cursor left
            if cursor_x > 0:
                cursor['x'] -= 1
        elif action == 2:  # Move cursor right
            if cursor_x < 46:
                cursor['x'] += 1
        elif action == 3:  # Move cursor up
            if cursor_y > 5:
                cursor['y'] -= 1
        elif action == 4:  # Move cursor down
            if cursor_y < 41:
                cursor['y'] += 1
        elif action == 5:  # Move legend left
            for legend in legends:
                if legend['x'] > 0:
                    legend['x'] -= 1
        elif action == 6:  # Move legend right
            for legend in legends:
                if legend['x'] < 46:
                    legend['x'] += 1
        elif action == 7:  # Click on legend
            for legend in legends:
                if legend['x'] == cursor_x and legend['y'] == cursor_y:
                    if legend['tags'][1] == 'cyan':
                        legend['tags'][1] = 'pink'
                    else:
                        legend['tags'][1] = 'cyan'
                    break
    
    return state
