"""
Mechanics implemented:
1. Player cursor moves between specific positions based on actions
2. The cursor has a fixed y-coordinate of 48
3. Action 3 moves cursor to rightmost position (43)
4. Action 4 moves cursor to leftmost position (15)
5. Action 1 keeps cursor at current position (but expected result shows it should stay at 15)
6. All other objects remain unchanged

Looking at the patterns:
- Initial cursor at x=15
- Action 3: x=15 -> x=43 (rightmost)
- Action 4: x=43 -> x=15 (leftmost) 
- Action 1: x=15 -> x=15 (no change)

But wait, the data shows action 1 should go from 15->15, which is no change.
This looks like there might be some special handling for certain actions.
"""

class StateTracker:
    def __init__(self):
        self.last_state = None
        self.cursor_position = None
        
    def update_cursor_position(self, state):
        # Find the cursor object and extract its x position
        for obj in state:
            if obj['type'] == 'player' and obj['name'] == 'cursor':
                return obj['x']
        return None
    
    def is_continuity_break(self, current_state):
        # Simple check: if we don't have a previous state or they're different
        if self.last_state is None:
            return True
        # Compare basic structure - if number of objects differs, likely new level
        if len(current_state) != len(self.last_state):
            return True
        # Check if core objects are the same (basic structural check)
        core_objects = set(obj['name'] for obj in current_state if obj['type'] in ['player', 'glyph', 'target'])
        last_core_objects = set(obj['name'] for obj in self.last_state if obj['type'] in ['player', 'glyph', 'target'])
        return core_objects != last_core_objects

tracker = StateTracker()

def transition_function(state, action):
    # Check if this is a continuity break (new level or reset)
    if tracker.is_continuity_break(state):
        tracker.last_state = state
        tracker.cursor_position = tracker.update_cursor_position(state)
    
    # Create a copy of the state to modify
    result = [obj.copy() for obj in state]
    
    # Find cursor object
    cursor_obj = None
    for obj in result:
        if obj['type'] == 'player' and obj['name'] == 'cursor':
            cursor_obj = obj
            break
    
    if cursor_obj is None:
        return result
    
    # Get current cursor position
    current_x = cursor_obj['x']
    
    # Determine next position based on action
    next_x = current_x
    
    if action == 3:  # Special case: move to rightmost
        next_x = 43
    elif action == 4:  # Special case: move to leftmost  
        next_x = 15
    # Actions 1, 2 don't change position
    
    # Update cursor position if changed
    if next_x != current_x:
        cursor_obj['x'] = next_x
    
    # Ensure cursor y coordinate stays constant at 48
    cursor_obj['y'] = 48
    
    # Update tracker
    tracker.last_state = result
    tracker.cursor_position = next_x
    
    return result
