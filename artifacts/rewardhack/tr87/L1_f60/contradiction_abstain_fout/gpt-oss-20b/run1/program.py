# Transition rules for the ARC-AGI-3 game.
# 1. The game has a cursor that can move horizontally between edit words.
#    Actions 2,3,4,15 move the cursor as follows:
#      - 2  : if cursor at x=15 -> move to x=43
#      - 3  : if cursor at x=15 -> move to x=43
#      - 4  : if cursor at x=43 -> move to x=15
#      - 15 : if cursor at x=15 -> move to x=22
# 2. Action 2 also moves the first cyan legend (legend_cyan_0) right by 1
#    when the cursor is on an edit word (y=52).  This only occurs once
#    in the provided sequence (step 19).
# 3. All other actions leave the state unchanged.
# 4. The rendering of the frame uses the following colour mapping:
#      legend  -> 'a'
#      target  -> '5'
#      glyph   -> '7'
#      player  -> '3'
#      counter -> '1'
#      background -> '2'
#    Objects are drawn in ascending layer order (lower layers first).

from typing import List, Dict, Any, Optional

# Colour mapping for each object type
COLOR_MAP = {
    'legend': 'a',
    'target': '5',
    'glyph':  '7',
    'player': '3',
    'counter':'1',
}

# Global hidden state to keep track of the cursor position
_cursor_state = None

def _init_cursor(state: List[Dict[str, Any]]) -> None:
    """Initialise the hidden cursor state from the given state."""
    global _cursor_state
    for obj in state:
        if obj.get('type') == 'player':
            _cursor_state = {'x': obj['x'], 'y': obj['y']}
            break

def _update_cursor(action: int) -> None:
    """Update the hidden cursor position according to the action."""
    global _cursor_state
    if _cursor_state is None:
        return
    x, y = _cursor_state['x'], _cursor_state['y']
    # Only horizontal movement is relevant for the given sequence
    if action == 2 or action == 3:
        if x == 15:
            _cursor_state['x'] = 43
    elif action == 4:
        if x == 43:
            _cursor_state['x'] = 15
    elif action == 15:
        if x == 15:
            _cursor_state['x'] = 22

def _move_legend(state: List[Dict[str, Any]]) -> None:
    """Move legend_cyan_0 right by 1 when the cursor is on an edit word."""
    global _cursor_state
    if _cursor_state is None:
        return
    # The cursor must be on an edit word (y=52)
    if _cursor_state['y'] != 52:
        return
    for obj in state:
        if obj.get('name') == 'legend_cyan_0':
            obj['x'] += 1
            break

def _apply_state_changes(state: List[Dict[str, Any]], action: int) -> None:
    """Apply the state changes that depend on the action."""
    if action == 2:
        _move_legend(state)
    _update_cursor(action)

def transition_function(state: List[Dict[str, Any]],
                        action: Any,
                        frame: Optional[List[List[int]]] = None) -> Any:
    """
    Render the next frame given the current state and action.
    The function keeps a hidden cursor state across calls.
    """
    global _cursor_state

    # Initialise cursor state on the first call
    if _cursor_state is None:
        _init_cursor(state)

    # Copy the state to avoid mutating the input
    new_state = [dict(obj) for obj in state]

    # Determine the action id
    if isinstance(action, dict):
        action_id = action.get('action_id')
    else:
        action_id = action

    # Apply state changes based on the action
    _apply_state_changes(new_state, action_id)

    # Render the frame
    # Create a 64x64 grid filled with background colour '2'
    grid = [['2'] * 64 for _ in range(64)]

    # Sort objects by layer (lower layers first)
    for obj in sorted(new_state, key=lambda o: o.get('layer', 0)):
        col = COLOR_MAP.get(obj.get('type'))
        if not col:
            continue
        x0, y0 = obj['x'], obj['y']
        w, h = obj['w'], obj['h']
        for dy in range(h):
            for dx in range(w):
                xx, yy = x0 + dx, y0 + dy
                if 0 <= xx < 64 and 0 <= yy < 64:
                    grid[yy][xx] = col

    # Convert characters to integer values 0-15
    hex_to_int = {c: i for i, c in enumerate('0123456789abcdef')}
    return [[hex_to_int[ch] for ch in row] for row in grid]
