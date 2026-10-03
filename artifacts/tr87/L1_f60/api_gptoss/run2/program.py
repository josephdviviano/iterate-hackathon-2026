"""
Mechanics implemented:
1. Cursor movement: Player cursor moves left/right between word positions
2. Word editing: When cursor moves to a new position, the editable glyph 
   at that position changes to match the reference word at that position
3. Action mapping:
   - Action 1: Move cursor right
   - Action 2: Move cursor left  
   - Action 3: Move cursor to rightmost position (x=43)
   - Action 4: Move cursor to leftmost position (x=15)
   - Action 5: No movement

The cursor can move between positions 15, 22, 29, 36, 43 (corresponding to 
the x positions of reference words). The cursor maintains its y-position (48),
but when it moves, the editable glyph should match the reference word at that position.

The editable glyphs are at y=52, reference words at y=41.
"""

import copy

# Global state to track previous state for continuity check
_last_state_hash = None
_cursor_positions = [15, 22, 29, 36, 43]

def _hash_state(state):
    """Create a hashable representation of the state for continuity checking."""
    return tuple(sorted((obj['name'], obj['x'], obj['y']) for obj in state))

def _find_object_by_name(state, name):
    """Find an object by name in the state."""
    for obj in state:
        if obj['name'] == name:
            return obj
    return None

def _find_target_word_at_position(state, x_pos):
    """Find the target word at a specific x position."""
    for obj in state:
        if (obj['type'] == 'target' and obj['x'] == x_pos and 
            'word' in obj['tags']):
            return obj
    return None

def _find_editable_glyph_at_position(state, x_pos):
    """Find the editable glyph at a specific x position."""
    for obj in state:
        if (obj['type'] == 'glyph' and obj['x'] == x_pos and 
            'word' in obj['tags'] and 'editable' in obj['tags']):
            return obj
    return None

def transition_function(state, action):
    global _last_state_hash
    
    # Create a copy of the state to avoid modifying the original
    result_state = copy.deepcopy(state)
    
    # Check if this is a continuation of the previous state
    current_hash = _hash_state(state)
    if _last_state_hash != current_hash:
        # Reset if state doesn't match previous
        _last_state_hash = current_hash
    
    # Find the cursor
    cursor = _find_object_by_name(result_state, 'cursor')
    if not cursor:
        return result_state
    
    current_x = cursor['x']
    
    # Determine next position based on action
    next_x = current_x
    
    if action == 1:  # Move right
        # Find next position to the right
        positions = sorted(_cursor_positions)
        try:
            idx = positions.index(current_x)
            if idx < len(positions) - 1:
                next_x = positions[idx + 1]
        except ValueError:
            pass  # Stay in place if not found
            
    elif action == 2:  # Move left
        # Find next position to the left
        positions = sorted(_cursor_positions)
        try:
            idx = positions.index(current_x)
            if idx > 0:
                next_x = positions[idx - 1]
        except ValueError:
            pass  # Stay in place if not found
            
    elif action == 3:  # Move to rightmost
        next_x = max(_cursor_positions)
        
    elif action == 4:  # Move to leftmost
        next_x = min(_cursor_positions)
        
    elif action == 5:  # No movement
        pass  # Keep current position
        
    # Update cursor position if changed
    if next_x != current_x:
        cursor['x'] = next_x
        
        # Find the target word at the new position
        target_word = _find_target_word_at_position(result_state, next_x)
        if target_word:
            # Update the editable glyph to match the target word at new position
            # The editable glyph should be at the same x position as target word
            # but keep its y-coordinate (which is 52)
            editable_glyph = _find_editable_glyph_at_position(result_state, next_x)
            if editable_glyph:
                # Just change the x position to match the target word
                # The y position should remain the same (52)
                editable_glyph['x'] = target_word['x']
                
    # Update the global state tracker
    _last_state_hash = current_hash
    
    return result_state
