# Unarmed blocks move by four cells, with mirrored horizontal directions.
# Clicking an inactive marker arms the blocks and selects that marker.
# Armed mode moves only the selected marker; clicking an armed block exits.
# Destinations must be floor colour 5; HUD width rounds 64*actions/150.
# HUD phase uses frame continuity; action 7 as a counted no-op is unconfirmed.

FLOOR = 5
BLOCK = 10
ARMED_BLOCK = 1
INACTIVE_MARKER = 9
ACTIVE_MARKER = 11
DIRECTIONS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
_last_frame = None
_action_count = 0


def contains(obj, x, y):
    return (obj['x'] <= x < obj['x'] + obj['w']
            and obj['y'] <= y < obj['y'] + obj['h'])


def marker_cell(marker):
    return {'x': marker['x'] - 1, 'y': marker['y'] - 1,
            'w': marker['w'] + 2, 'h': marker['h'] + 2}


def destination_is_floor(frame, footprint, dx, dy):
    for y in range(footprint['y'] + dy,
                   footprint['y'] + dy + footprint['h']):
        for x in range(footprint['x'] + dx,
                       footprint['x'] + dx + footprint['w']):
            if not (0 <= x < 64 and 1 <= y < 63):
                return False
            if not contains(footprint, x, y) and frame[y][x] != FLOOR:
                return False
    return True


def clicked_inactive_marker(markers, click, frame):
    x, y = click.get('x', -1), click.get('y', -1)
    if not (0 <= x < 64 and 0 <= y < 64):
        return None
    if frame[y][x] != INACTIVE_MARKER:
        return None
    return next((m for m in markers if contains(m, x, y)), None)


def clicked_armed_block(players, click, frame):
    x, y = click.get('x', -1), click.get('y', -1)
    return (0 <= x < 64 and 0 <= y < 64
            and frame[y][x] == ARMED_BLOCK
            and any(contains(p, x, y) for p in players))


def update_player(player, action_id, frame, armed, is_right):
    if armed or action_id not in DIRECTIONS:
        return
    dx, dy = DIRECTIONS[action_id]
    if is_right:
        dx = -dx
    if destination_is_floor(frame, player, dx, dy):
        player['x'] += dx
        player['y'] += dy


def update_marker(marker, action_id, frame, selected):
    if marker is not selected or action_id not in DIRECTIONS:
        return
    dx, dy = DIRECTIONS[action_id]
    if destination_is_floor(frame, marker_cell(marker), dx, dy):
        marker['x'] += dx
        marker['y'] += dy


def paint(frame, obj, colour):
    for y in range(obj['y'], obj['y'] + obj['h']):
        for x in range(obj['x'], obj['x'] + obj['w']):
            frame[y][x] = int(colour)


def update_hud(frame, before):
    global _last_frame, _action_count
    width = sum(c == 0 for c in before[0])
    if _last_frame != before:
        _action_count = (150 * width + 74) // 64 if width else 0
    _action_count += 1
    width = min(64, (64 * _action_count + 75) // 150)
    frame[0] = [FLOOR] * (64 - width) + [0] * width
    frame[63] = [0] * width + [FLOOR] * (64 - width)
    _last_frame = [row[:] for row in frame]


def transition_function(state, action, frame):
    output = [list(map(int, row)) for row in frame]
    players = sorted((dict(o) for o in state
                      if o['type'] == 'player' and o.get('visible', True)),
                     key=lambda o: (o['x'], o['y']))
    markers = [dict(o) for o in state
               if o['type'] == 'marker' and o.get('visible', True)]
    selected = next((m for m in markers
                     if frame[m['y']][m['x']] == ACTIVE_MARKER), None)
    armed = selected is not None
    action_id = action.get('action_id') if isinstance(action, dict) else action

    for player in players:
        paint(output, player, FLOOR)
    for marker in markers:
        paint(output, marker, FLOOR)

    if action_id == 6 and isinstance(action, dict):
        clicked = clicked_inactive_marker(markers, action, frame)
        if clicked is not None:
            selected, armed = clicked, True
        elif clicked_armed_block(players, action, frame):
            selected, armed = None, False
    else:
        for i, player in enumerate(players):
            update_player(player, action_id, frame, armed, i > 0)
        for marker in markers:
            update_marker(marker, action_id, frame, selected)

    for player in players:
        paint(output, player, ARMED_BLOCK if armed else BLOCK)
    for marker in markers:
        paint(output, marker,
              ACTIVE_MARKER if marker is selected else INACTIVE_MARKER)
    update_hud(output, frame)
    return output
