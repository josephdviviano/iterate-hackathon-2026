# Mechanics: 
# - The cursor moves horizontally between the editable words.
# - Legends move horizontally between their positions.
# - No other objects change.

def transition_function(state, action):
    new_state = state.copy()
    
    if action == 1:
        # Move cursor to the next editable word
        cursor = next(obj for obj in new_state if obj['type'] == 'player')
        edit_words = sorted([obj for obj in new_state if obj['type'] == 'glyph'], key=lambda x: x['x'])
        current_index = edit_words.index(next(obj for obj in edit_words if obj['x'] == cursor['x']))
        next_index = (current_index + 1) % len(edit_words)
        cursor['x'] = edit_words[next_index]['x']
    
    elif action == 2:
        # Move legend_cyan_0 to the right
        legend_cyan_0 = next(obj for obj in new_state if obj['name'] == 'legend_cyan_0')
        legend_cyan_0['x'] += 1
    
    elif action == 3:
        # Move legend_pink_0 to the right
        legend_pink_0 = next(obj for obj in new_state if obj['name'] == 'legend_pink_0')
        legend_pink_0['x'] += 1
    
    elif action == 4:
        # Move cursor to the previous editable word
        cursor = next(obj for obj in new_state if obj['type'] == 'player')
        edit_words = sorted([obj for obj in new_state if obj['type'] == 'glyph'], key=lambda x: x['x'])
        current_index = edit_words.index(next(obj for obj in edit_words if obj['x'] == cursor['x']))
        previous_index = (current_index - 1) % len(edit_words)
        cursor['x'] = edit_words[previous_index]['x']
    
    elif action == 5:
        # Move legend_cyan_0 to the left
        legend_cyan_0 = next(obj for obj in new_state if obj['name'] == 'legend_cyan_0')
        legend_cyan_0['x'] -= 1
    
    elif action == 6:
        # Move legend_pink_0 to the left
        legend_pink_0 = next(obj for obj in new_state if obj['name'] == 'legend_pink_0')
        legend_pink_0['x'] -= 1
    
    return new_state
