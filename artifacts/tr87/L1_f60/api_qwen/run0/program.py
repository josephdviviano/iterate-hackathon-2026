"""
Mechanics implemented:
1. Cursor movement between fixed positions: 15, 22, 29, 36, 43
2. Action behavior:
   - Action 1: Move to rightmost position (43)
   - Action 2: Move to leftmost position (15) 
   - Action 3: Move to rightmost position (43)
   - Action 4: Move to leftmost position (15)
   - Actions 5,7: No movement

Looking at the exact transitions:
- Initial: cursor at x=15
- Step 2 (action 3): cursor moves to x=43 (rightmost)
- Step 3 (action 4): cursor moves to x=15 (leftmost)
- Step 4 (action 1): cursor moves to x=43 (rightmost)

This suggests actions 1 and 3 go to rightmost, actions 2 and 4 go to leftmost.
"""

import copy

class GameStateManager:
    def __init__(self):
        self.cursor_positions = [15, 22, 29, 36, 43]  # Fixed cursor positions
        
    def transition_function(self, state, action):
        # Create a deep copy of the state to avoid modifying original
        result = copy.deepcopy(state)
        
        # Find the cursor object
        cursor_obj = None
        for obj in result:
            if obj["type"] == "player" and obj["name"] == "cursor":
                cursor_obj = obj
                break
                
        if cursor_obj is None:
            return result
            
        # Update cursor position based on action
        if action == 1 or action == 3:
            # Move to rightmost position
            cursor_obj["x"] = max(self.cursor_positions)
        elif action == 2 or action == 4:
            # Move to leftmost position
            cursor_obj["x"] = min(self.cursor_positions)
        # Actions 5 and 7 do nothing (no movement)
        
        return result

# Global state manager
_state_manager = GameStateManager()

def transition_function(state, action):
    return _state_manager.transition_function(state, action)
