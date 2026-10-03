# Transition rule for the ARC-AGI-3 game.
# Mechanics:
# - Player moves 5 cells per action: 1=up, 2=down, 3=left, 4=right.
# - Each action costs 2 budget; step_bar shows remaining budget and moves right 2 cells.
# - Action 6 is a click: player teleports to the clicked cell, budget resets to 42, step_bar resets.
# - Clicking on a diamond removes it; diamond reappears after 15 moves from removal.
# - Stepping on recolor_token removes it and decreases hud_glyph progress by 3.
# The function returns the 64x64 frame after the action.

# Hidden state for diamond reappearance timing
_diamond_counter = None

def transition_function(state, action, frame):
    global _diamond_counter

    # Helper to find object by type and name
    def find_obj(t, n=None):
        for o in state:
            if o.get('type') == t and (n is None or o.get('name') == n):
                return o
        return None

    # Find key objects
    player = find_obj('player')
    step_bar = find_obj('counter', 'step_bar')
    hud_glyph = find_obj('counter', 'hud_glyph')
    recolor_token = find_obj('button', 'recolor_token')
    diamond = find_obj('target', 'diamond')

    # Determine action type
    if isinstance(action, int):
        act = action
    else:
        act = action.get('action_id')
        click_x = action.get('x')
        click_y = action.get('y')

    # Update player position
    new_player = player.copy()
    if act in {1, 2, 3, 4}:
        dx, dy = 0, 0
        if act == 1:
            dy = -1
        if act == 2:
            dy = 1
        if act == 3:
            dx = -1
        if act == 4:
            dx = 1
        # Move 5 cells
        new_x = player['x'] + dx * 5
        new_y = player['y'] + dy * 5
        # If movement blocked (player unchanged), keep same
        if new_x == player['x'] and new_y == player['y']:
            new_player['x'] = player['x']
            new_player['y'] = player['y']
        else:
            new_player['x'] = new_x
            new_player['y'] = new_y
    elif act == 6:
        # Teleport to clicked cell
        new_player['x'] = click_x
        new_player['y'] = click_y
    else:
        # Unknown action: keep unchanged
        new_player = player.copy()

    # Update step_bar
    new_step_bar = step_bar.copy()
    if act == 6:
        new_budget = 42
        new_x = 13
    else:
        # Parse current budget from tags
        current_budget = int(step_bar['tags'][1])
        new_budget = current_budget - 2
        new_x = step_bar['x'] + 2
    new_step_bar['w'] = new_budget
    new_step_bar['x'] = new_x
    new_step_bar['tags'] = ['budget', str(new_budget)]

    # Update hud_glyph if recolor_token removed
    recolor_removed = False
    if recolor_token:
        # Check if new player position overlaps recolor_token
        if new_player['x'] == recolor_token['x'] and new_player['y'] == recolor_token['y']:
            recolor_removed = True
            # Decrease progress by 3
            tags = hud_glyph['tags']
            if tags[0] == 'progress':
                prog = int(tags[1])
                hud_glyph['tags'] = ['progress', str(prog - 3), tags[2]]

    # Update diamond removal on click
    new_diamond = diamond
    if act == 6 and diamond:
        # If click on diamond, remove it
        if click_x == diamond['x'] and click_y == diamond['y']:
            new_diamond = None
            _diamond_counter = 0  # start counter for reappearance

    # Handle diamond reappearance after 15 moves from removal
    if _diamond_counter is not None:
        if act in {1, 2, 3, 4}:
            _diamond_counter += 1
            if _diamond_counter >= 15:
                # Reappear diamond
                new_diamond = {
                    "h": 3,
                    "layer": 0,
                    "name": "diamond",
                    "tags": ["marker", "collect"],
                    "type": "target",
                    "visible": True,
                    "w": 3,
                    "x": 50,
                    "y": 11
                }
                _diamond_counter = None

    # Build new state (not returned, only used for frame update)
    new_state = []
    for o in state:
        if o.get('type') == 'player':
            new_state.append(new_player)
        elif o.get('type') == 'counter' and o.get('name') == 'step_bar':
            new_state.append(new_step_bar)
        elif o.get('type') == 'counter' and o.get('name') == 'hud_glyph':
            new_state.append(hud_glyph)
        elif o.get('type') == 'button' and o.get('name') == 'recolor_token':
            if recolor_removed:
                continue  # removed
            new_state.append(o)
        elif o.get('type') == 'target' and o.get('name') == 'diamond':
            if new_diamond is None:
                continue  # removed
            new_state.append(new_diamond)
        else:
            new_state.append(o)

    # Ensure diamond is added if missing and 15 moves passed
    if act in {1, 2, 3, 4} and new_diamond is None and _diamond_counter is None:
        # No diamond present and counter finished
        new_state.append({
            "h": 3,
            "layer": 0,
            "name": "diamond",
            "tags": ["marker", "collect"],
            "type": "target",
            "visible": True,
            "w": 3,
            "x": 50,
            "y": 11
        })

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------
    # Background color is 5 (from initial frame)
    BG = 5
    # Player color is 12
    PLAYER_COLOR = 12
    # Diamond pattern
    DIAMOND_PATTERN = [
        [3, 3, 3],
        [1, 0, 0],
        [3, 1, 3]
    ]
    # Recolor_token color is 3 (same as diamond, but removed when used)
    RECOLOR_COLOR = 3
    # Step_bar color is 2
    STEP_COLOR = 2

    # Copy before frame
    new_frame = [row[:] for row in frame]

    # Helper to clear a rectangle
    def clear_rect(x, y, w, h):
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                new_frame[yy][xx] = BG

    # Helper to draw a rectangle
    def draw_rect(x, y, w, h, color):
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                new_frame[yy][xx] = color

    # Clear old player
    clear_rect(player['x'], player['y'], player['w'], player['h'])
    # Draw new player
    draw_rect(new_player['x'], new_player['y'], new_player['w'], new_player['h'], PLAYER_COLOR)

    # Clear old diamond if present
    if diamond:
        clear_rect(diamond['x'], diamond['y'], diamond['w'], diamond['h'])
    # Draw new diamond if present
    if new_diamond:
        x0, y0 = new_diamond['x'], new_diamond['y']
        for dy in range(3):
            for dx in range(3):
                new_frame[y0 + dy][x0 + dx] = DIAMOND_PATTERN[dy][dx]

    # Clear old recolor_token if present
    if recolor_token:
        clear_rect(recolor_token['x'], recolor_token['y'], recolor_token['w'], recolor_token['h'])
    # Recolor_token is never added back, so no need to draw

    # Clear old step_bar
    clear_rect(step_bar['x'], step_bar['y'], step_bar['w'], step_bar['h'])
    # Draw new step_bar
    draw_rect(new_step_bar['x'], new_step_bar['y'], new_step_bar['w'], new_step_bar['h'], STEP_COLOR)

    return new_frame
