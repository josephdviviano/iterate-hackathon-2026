# Mechanics:
#  - Player moves 5 cells per action: 1=up, 2=down, 3=left, 4=right.
#  - Movement is allowed only within predefined corridor rectangles.
#  - Step_bar (budget counter) moves right 2 cells and shrinks width by 2 each action.
#  - Collecting ring_16_35 resets budget to 42 and step_bar to leftmost.
#  - Collecting diamond reduces budget by 2 and respawns after one action.
#  - Collecting recolor_token reduces budget by 2 and hud_glyph progress by 3.
#  - Recolor_token and ring_16_35 disappear permanently after collection.
#  - Teleport: action 4 from (49,5) moves player to (54,45).

import copy

# Hidden state across calls
_last_state = None
_last_action = None
_diamond_respawn = 0
_object_patterns = {}   # name -> pixel pattern (list of lists of ints)
_diamond_initial_pattern = None
_ring16_initial_pattern = None

# Corridor definitions: (x_min, x_max, y_min, y_max)
CORRIDORS = [
    (9, 13, 45, 49),   # bottom left
    (19, 23, 31, 35),  # middle left
    (29, 33, 5, 9),    # top left
    (34, 38, 5, 9),    # top middle
    (49, 53, 5, 9),    # top right
]

def _rect_intersect(ax, ay, aw, ah, bx, by, bw, bh):
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

def _find_obj(state, name=None, type_=None):
    for o in state:
        if name is not None and o.get('name') != name:
            continue
        if type_ is not None and o.get('type') != type_:
            continue
        return o
    return None

def _update_step_bar(step_bar, reset=False):
    if reset:
        step_bar['x'] = 13
        step_bar['w'] = 42
    else:
        step_bar['x'] += 2
        step_bar['w'] -= 2
    step_bar['tags'] = ['budget', str(step_bar['w'])]

def transition_function(state, action, frame):
    global _last_state, _last_action, _diamond_respawn
    global _object_patterns, _diamond_initial_pattern, _ring16_initial_pattern

    # Reset hidden state if state changed
    if _last_state is None or state != _last_state:
        _diamond_respawn = 0
        _object_patterns = {}
        _last_state = copy.deepcopy(state)
        _last_action = action

        # Extract initial patterns from the first frame
        for o in state:
            name = o.get('name')
            if name:
                x, y, w, h = o['x'], o['y'], o['w'], o['h']
                pattern = [row[x:x+w] for row in frame[y:y+h]]
                _object_patterns[name] = pattern
                if name == 'diamond':
                    _diamond_initial_pattern = pattern
                if name == 'ring_16_35':
                    _ring16_initial_pattern = pattern

    # Work on a new blank frame (background 4, with 5 at top-left 4x4)
    new_frame = [[4]*64 for _ in range(64)]
    for yy in range(4):
        for xx in range(4):
            new_frame[yy][xx] = 5

    # Find objects
    player = _find_obj(state, type_='player')
    step_bar = _find_obj(state, type_='counter')
    hud_glyph = _find_obj(state, name='hud_glyph')
    recolor_token = _find_obj(state, name='recolor_token')
    ring16 = _find_obj(state, name='ring_16_35')
    ring31 = _find_obj(state, name='ring_31_20')
    diamond = _find_obj(state, name='diamond')

    # Movement deltas
    deltas = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
    dx, dy = deltas.get(action, (0, 0))
    new_x = player['x'] + dx
    new_y = player['y'] + dy

    # Check corridor containment
    def in_corridor(x, y):
        for (xmin, xmax, ymin, ymax) in CORRIDORS:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False

    moved = False
    if in_corridor(new_x, new_y):
        player['x'] = new_x
        player['y'] = new_y
        moved = True
    else:
        # Special teleport for action 4 from corridor 49-53
        if action == 4 and player['x'] == 49 and player['y'] == 5:
            player['x'] = 54
            player['y'] = 45
            moved = True

    # Interactions after movement
    reset_step_bar = False
    # Ring 16-35
    if ring16 and _rect_intersect(player['x'], player['y'], player['w'], player['h'],
                                  ring16['x'], ring16['y'], ring16['w'], ring16['h']):
        state = [o for o in state if o.get('name') != 'ring_16_35']
        reset_step_bar = True
    # Ring 31-20
    if ring31 and _rect_intersect(player['x'], player['y'], player['w'], player['h'],
                                  ring31['x'], ring31['y'], ring31['w'], ring31['h']):
        state = [o for o in state if o.get('name') != 'ring_31_20']
        reset_step_bar = True
    # Diamond
    if diamond and _rect_intersect(player['x'], player['y'], player['w'], player['h'],
                                   diamond['x'], diamond['y'], diamond['w'], diamond['h']):
        state = [o for o in state if o.get('name') != 'diamond']
        _diamond_respawn = 1
    # Recolor token
    if recolor_token and _rect_intersect(player['x'], player['y'], player['w'], player['h'],
                                         recolor_token['x'], recolor_token['y'],
                                         recolor_token['w'], recolor_token['h']):
        state = [o for o in state if o.get('name') != 'recolor_token']
        if hud_glyph and len(hud_glyph['tags']) >= 2:
            try:
                val = int(hud_glyph['tags'][1]) - 3
                hud_glyph['tags'][1] = str(val)
            except:
                pass

    # Update step_bar
    _update_step_bar(step_bar, reset=reset_step_bar)

    # Handle diamond respawn
    if _diamond_respawn > 0:
        _diamond_respawn -= 1
        if _diamond_respawn == 0:
            if _diamond_initial_pattern:
                new_diamond = {
                    "h": 3,
                    "layer": 0,
                    "name": "diamond",
                    "tags": ["marker", "collect"],
                    "type": "target",
                    "visible": True,
                    "w": 3,
                    "x": 50,
                    "y": 11,
                    "pixels": _diamond_initial_pattern
                }
                state.append(new_diamond)
                _object_patterns['diamond'] = _diamond_initial_pattern

    # Render new frame
    # 1. Draw step_bar
    sb = _find_obj(state, type_='counter')
    for x in range(sb['x'], sb['x'] + sb['w']):
        if 0 <= sb['y'] < 64 and 0 <= x < 64:
            new_frame[sb['y']][x] = 3

    # 2. Draw other objects
    for name in ['ring_16_35', 'ring_31_20', 'diamond', 'recolor_token']:
        o = _find_obj(state, name=name)
        if o:
            pattern = _object_patterns.get(name)
            if pattern is None:
                pattern = [row[o['x']:o['x']+o['w']] for row in frame[o['y']:o['y']+o['h']]]
                _object_patterns[name] = pattern
            for yy in range(o['h']):
                for xx in range(o['w']):
                    val = pattern[yy][xx]
                    if 0 <= o['y'] + yy < 64 and 0 <= o['x'] + xx < 64:
                        new_frame[o['y'] + yy][o['x'] + xx] = val

    # 3. Draw player
    px, py, pw, ph = player['x'], player['y'], player['w'], player['h']
    player_pixels = player.get('pixels')
    if player_pixels:
        pattern = [[12 if v == 12 else 9 for v in row] for row in player_pixels]
    else:
        pattern = [[12]*pw for _ in range(ph)]
    for yy in range(ph):
        for xx in range(pw):
            val = pattern[yy][xx]
            if 0 <= py + yy < 64 and 0 <= px + xx < 64:
                new_frame[py + yy][px + xx] = val

    # Update hidden state
    _last_state = copy.deepcopy(state)
    _last_action = action

    return new_frame
