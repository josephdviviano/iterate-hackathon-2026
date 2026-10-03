# This program models a simple grid‑based game where the player moves in
# 5‑cell steps up (action 1) or left (action 3).  Each action consumes 2
# budget units, shrinking a horizontal step‑bar counter.  Stepping on a
# refuel ring resets the budget to 42 and the bar to its initial
# position.  Stepping on a diamond removes it and schedules a respawn
# after the next action.  Stepping on a recolor token removes it and
# reduces the HUD glyph progress by 3.  The code keeps hidden state
# across calls to handle diamond respawns and resets when the input
# state changes.  It renders the resulting frame by overlaying object
# pixels on the supplied before‑frame, clearing the old player, step‑bar
# and ring areas to the background colour (3) before drawing the new
# ones.

import copy

# Hidden state across calls
_last_state_repr = None
_diamond_respawn = False
_diamond_pos = None
_diamond_pixels = None

def _state_repr(state):
    """Canonical representation of a state for comparison."""
    def normalize_tags(tags):
        return tuple(
            tuple(t) if isinstance(t, list) else t for t in tags
        )
    def normalize_pixels(pixels):
        return tuple(tuple(row) for row in pixels)
    return tuple(sorted(
        (
            obj.get('name'),
            obj.get('type'),
            obj.get('x'),
            obj.get('y'),
            obj.get('w'),
            obj.get('h'),
            obj.get('layer'),
            obj.get('visible'),
            normalize_tags(obj.get('tags', [])),
            normalize_pixels(obj.get('pixels', []))
        )
        for obj in state
    ))

def _find_obj(state, name=None, type_=None, tags=None):
    for obj in state:
        if name is not None and obj.get('name') != name:
            continue
        if type_ is not None and obj.get('type') != type_:
            continue
        if tags is not None and not tags(obj.get('tags', [])):
            continue
        return obj
    return None

def _budget_from_tags(tags):
    for t in tags:
        if isinstance(t, list) and 'budget' in t:
            try:
                return int(t[1])
            except (IndexError, ValueError):
                continue
    if isinstance(tags, list) and len(tags) >= 2 and tags[0] == 'budget':
        try:
            return int(tags[1])
        except ValueError:
            pass
    return 0

def _set_budget_tags(tags, budget):
    for i, t in enumerate(tags):
        if isinstance(t, list) and t and t[0] == 'budget':
            tags[i] = ['budget', str(budget)]
            return
    tags.append(['budget', str(budget)])

def transition_function(state, action, frame):
    global _last_state_repr, _diamond_respawn, _diamond_pos, _diamond_pixels

    # Reset hidden state if the incoming state differs from the last one
    if _last_state_repr is None or _state_repr(state) != _last_state_repr:
        _diamond_respawn = False
        _diamond_pos = None
        _diamond_pixels = None

    new_state = copy.deepcopy(state)

    player = _find_obj(new_state, type_='player')
    step_bar = _find_obj(new_state, type_='counter', tags=lambda t: 'budget' in t)
    hud_glyph = _find_obj(new_state, type_='counter', tags=lambda t: 'progress' in t)

    def update_step_bar(budget, x_offset):
        _set_budget_tags(step_bar['tags'], budget)
        step_bar['w'] = budget
        step_bar['x'] = x_offset
        # generate pixels: 2 rows of width w, color 11 ('b')
        step_bar['pixels'] = [[11] * budget for _ in range(2)]

    # Capture old positions for clearing
    old_player_x, old_player_y = player['x'], player['y']
    old_step_x, old_step_w = step_bar['x'], step_bar['w']

    # Compute new player position
    new_x, new_y = old_player_x, old_player_y
    if action == 1:          # up
        new_y -= 5
    elif action == 3:        # left
        new_x -= 5
    elif action == 2:        # down
        new_y += 5
    # actions 4,5,6,7: no movement

    player['x'], player['y'] = new_x, new_y

    # Consume budget
    current_budget = _budget_from_tags(step_bar['tags'])
    new_budget = current_budget - 2
    new_x_offset = step_bar['x'] + 2

    # Check for refuel ring
    refuel = _find_obj(new_state, type_='refuel')
    ring_removed = False
    if refuel and (new_x >= refuel['x'] and new_x < refuel['x'] + refuel['w'] and
                   new_y >= refuel['y'] and new_y < refuel['y'] + refuel['h']):
        ring_removed = True
        ring_area = (refuel['x'], refuel['y'], refuel['w'], refuel['h'])
        new_state.remove(refuel)
        new_budget = 42
        new_x_offset = 13
    else:
        if new_budget < 0:
            new_budget = 0

    update_step_bar(new_budget, new_x_offset)

    # Diamond interaction
    diamond = _find_obj(new_state, type_='target', tags=lambda t: 'marker' in t)
    if diamond and (new_x >= diamond['x'] and new_x < diamond['x'] + diamond['w'] and
                    new_y >= diamond['y'] and new_y < diamond['y'] + diamond['h']):
        _diamond_pixels = diamond.get('pixels', None)
        new_state.remove(diamond)
        _diamond_respawn = True
        _diamond_pos = (diamond['x'], diamond['y'], diamond['w'], diamond['h'])
    else:
        if _diamond_respawn:
            new_state.append({
                "h": 3,
                "layer": 0,
                "name": "diamond",
                "tags": ["marker", "collect"],
                "type": "target",
                "visible": True,
                "w": 3,
                "x": _diamond_pos[0],
                "y": _diamond_pos[1],
                "pixels": _diamond_pixels
            })
            _diamond_respawn = False

    # Recolor token interaction
    recolor = _find_obj(new_state, type_='button', tags=lambda t: 'recolor' in t)
    if recolor and (new_x >= recolor['x'] and new_x < recolor['x'] + recolor['w'] and
                    new_y >= recolor['y'] and new_y < recolor['y'] + recolor['h']):
        new_state.remove(recolor)
        if hud_glyph:
            progress = _budget_from_tags(hud_glyph['tags'])
            new_progress = max(progress - 3, 0)
            _set_budget_tags(hud_glyph['tags'], new_progress)

    # Render frame
    out_frame = [row[:] for row in frame]  # deep copy

    # Clear old player area to background color 3
    for dy in range(5):
        y = old_player_y + dy
        if 0 <= y < 64:
            for dx in range(5):
                x = old_player_x + dx
                if 0 <= x < 64:
                    out_frame[y][x] = 3

    # Clear old step bar area to background color 3
    for dy in range(2):
        y = 61 + dy
        if 0 <= y < 64:
            for dx in range(old_step_w):
                x = old_step_x + dx
                if 0 <= x < 64:
                    out_frame[y][x] = 3

    # Clear ring area if a ring was removed
    if ring_removed:
        rx, ry, rw, rh = ring_area
        for dy in range(rh):
            y = ry + dy
            if 0 <= y < 64:
                for dx in range(rw):
                    x = rx + dx
                    if 0 <= x < 64:
                        out_frame[y][x] = 3

    # Sort objects by layer ascending (lower layers first)
    for obj in sorted(new_state, key=lambda o: o.get('layer', 0)):
        if 'pixels' not in obj:
            continue
        w, h = obj['w'], obj['h']
        px = obj['pixels']
        ox, oy = obj['x'], obj['y']
        for dy in range(h):
            y = oy + dy
            if y < 0 or y >= 64:
                continue
            row = out_frame[y]
            for dx in range(w):
                x = ox + dx
                if x < 0 or x >= 64:
                    continue
                row[x] = px[dy][dx]
    _last_state_repr = _state_repr(new_state)
    return out_frame
