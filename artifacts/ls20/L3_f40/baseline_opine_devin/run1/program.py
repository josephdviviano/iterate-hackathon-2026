# Arrows move a player-sized cell; colour 4 walls block movement.
# Colour 1 strips beside a destination convey it away until a wall.
# Every action spends two budget pixels; covered rings refill the track.
# Covering the persistent diamond rotates the HUD glyph clockwise.
# Hidden sprites require continuity; other actions and chamber entry are untested.

_last_frame = None
_hidden_targets = []


def cells(obj):
    return [(x, y) for y in range(obj['y'], obj['y'] + obj['h'])
            for x in range(obj['x'], obj['x'] + obj['w'])]


def covers(rect, obj):
    x, y, w, h = rect
    return (x <= obj['x'] and y <= obj['y'] and
            obj['x'] + obj['w'] <= x + w and
            obj['y'] + obj['h'] <= y + h)


def wall_blocks(ground, x, y, w, h):
    if x < 0 or y < 0 or x + w > 64 or y + h > 64:
        return True
    colours = [ground[yy][xx] for yy in range(y, y + h)
               for xx in range(x, x + w)]
    return 4 in colours or 3 not in colours


def conveyor_direction(ground, x, y, w, h):
    edges = [((1, 0), [(x - 1, yy) for yy in range(y, y + h)]),
             ((-1, 0), [(x + w, yy) for yy in range(y, y + h)]),
             ((0, 1), [(xx, y - 1) for xx in range(x, x + w)]),
             ((0, -1), [(xx, y + h) for xx in range(x, x + w)])]
    for direction, edge in edges:
        if all(0 <= xx < 64 and 0 <= yy < 64 and ground[yy][xx] == 1
               for xx, yy in edge):
            return direction
    return None


def update_player(player, action, ground):
    x, y, w, h = (player[k] for k in ('x', 'y', 'w', 'h'))
    dx, dy = {1: (0, -h), 2: (0, h), 3: (-w, 0),
              4: (w, 0)}.get(action, (0, 0))
    nx, ny = x + dx, y + dy
    if wall_blocks(ground, nx, ny, w, h):
        return x, y, w, h
    if dx or dy:
        direction = conveyor_direction(ground, nx, ny, w, h)
        if direction:
            sx, sy = direction[0] * w, direction[1] * h
            while not wall_blocks(ground, nx + sx, ny + sy, w, h):
                nx, ny = nx + sx, ny + sy
    return nx, ny, w, h


def update_refuel(objects, rect, ground):
    refilled = False
    for obj in objects:
        if obj['type'] == 'refuel' and covers(rect, obj):
            refilled = True
            for x, y in cells(obj):
                ground[y][x] = 3
    return refilled


def update_target(targets, old_rect, new_rect, ground, glyph):
    for obj, sprite in targets:
        if covers(new_rect, obj) and not covers(old_rect, obj) and glyph:
            x, y, w, h = (glyph[k] for k in ('x', 'y', 'w', 'h'))
            if w == h:
                patch = [row[x:x + w] for row in ground[y:y + h]]
                rotated = [list(row) for row in zip(*patch[::-1])]
                for j in range(h):
                    ground[y + j][x:x + w] = rotated[j]


def update_budget(bar, refilled, ground):
    if not bar:
        return
    right = bar['x'] + bar['w']
    left = bar['x']
    while left > 0 and ground[bar['y']][left - 1] != 5:
        left -= 1
    width = right - left if refilled else max(0, bar['w'] - 2)
    for y in range(bar['y'], bar['y'] + bar['h']):
        for x in range(left, right):
            ground[y][x] = 11 if x >= right - width else 3


def transition_function(state, action, frame):
    global _last_frame, _hidden_targets
    aid = action.get('action_id') if isinstance(action, dict) else action
    ground = [row[:] for row in frame]
    player = next((o for o in state if o['type'] == 'player'), None)
    if player is None:
        _last_frame = ground
        _hidden_targets = []
        return ground
    old_rect = tuple(player[k] for k in ('x', 'y', 'w', 'h'))
    continuing = frame == _last_frame
    targets = [(o, s) for o, s in _hidden_targets
               if continuing and covers(old_rect, o)]
    for obj in state:
        if obj['type'] == 'target' and 'collect' in obj.get('tags', []):
            sprite = [(x, y, frame[y][x]) for x, y in cells(obj)]
            targets = [(o, s) for o, s in targets if o['name'] != obj['name']]
            targets.append((dict(obj), sprite))
    for x, y in cells(player):
        ground[y][x] = 3
    for obj, sprite in targets:
        for x, y, colour in sprite:
            ground[y][x] = colour
    new_rect = update_player(player, aid, ground)
    refilled = update_refuel(state, new_rect, ground)
    glyph = next((o for o in state if o['type'] == 'counter' and
                  'progress' in o.get('tags', [])), None)
    update_target(targets, old_rect, new_rect, ground, glyph)
    bar = next((o for o in state if o['type'] == 'counter' and
                'budget' in o.get('tags', [])), None)
    update_budget(bar, refilled, ground)
    nx, ny, w, h = new_rect
    sprite = player.get('pixels', [row[player['x']:player['x'] + w]
                                 for row in frame[player['y']:player['y'] + h]])
    for y in range(h):
        for x in range(w):
            ground[ny + y][nx + x] = int(sprite[y][x])
    _hidden_targets = [(o, s) for o, s in targets if covers(new_rect, o)]
    _last_frame = [row[:] for row in ground]
    return ground
