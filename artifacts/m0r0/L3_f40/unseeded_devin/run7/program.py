# Mechanics: two cyan 4x4 block players slide on a 4x4-cell grid (cells at 2 mod
# 4). A1/A2 move both blocks up/down one cell; A3 pushes them apart, A4 pulls
# them together; each block's move is cancelled by invisible maze cells, marker
# cells, or leaving its half of the board. Clicking a marker arms the level:
# blocks become colour-1 players, that marker turns active (colour 11) and is
# then steered one cell by A1-4; clicking an inactive marker moves the active
# flag, clicking a player disarms back to blocks, clicks on the active marker,
# a block, or empty space are no-ops. Two 1px wall_0 timer bars (top-right and
# bottom-left) grow to w = floor(3(n+1)/7) after n actions since level start.
# Objects are renamed f"{type}_{color}_{rank}" sorted by (y,x); in block mode
# the two blocks hold ranks 0 and 1 so everything else starts at 2.
# Unconfirmed: click hit-testing outside object top-lefts (all observed clicks
# land exactly on an object origin); marker-vs-marker/player blocking.

import json

MAZE = {(14, 38), (6, 18), (38, 14)}
LO, HI = 2, 58                    # legal cell coords (multiple of 4 offset 2)
BLOCK_PIX = [[10, 10, 10, 10]] * 4

_last = None                      # canonical form of the last state we returned
_n = 0                            # actions taken since level start


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _color(obj):
    for t in obj.get('tags', []):
        if t.startswith('color_'):
            return t.split('_', 1)[1]
    return '0'


def _is_left(block):
    return block['x'] < 32


def _mcell(marker):
    return (marker['x'] - 1, marker['y'] - 1)


def _free_block_cell(block, nx, ny, blocks, markers):
    if not (LO <= ny <= HI):
        return False
    if _is_left(block):
        if not (LO <= nx <= 26):
            return False
    elif not (34 <= nx <= HI):
        return False
    if (nx, ny) in MAZE:
        return False
    if (nx, ny) in {_mcell(m) for m in markers}:
        return False
    return not any(o is not block and o['x'] == nx and o['y'] == ny
                   for o in blocks)


def _free_marker_cell(marker, nx, ny, markers, players):
    cell = (nx - 1, ny - 1)
    if cell in MAZE or not (LO <= cell[0] <= HI and LO <= cell[1] <= HI):
        return False
    occupied = {_mcell(m) for m in markers if m is not marker}
    occupied |= {(p['x'], p['y']) for p in players}
    return cell not in occupied


def _hit(state, x, y):
    for o in state:
        if o.get('type') == 'marker' and \
                o['x'] <= x < o['x'] + o['w'] and o['y'] <= y < o['y'] + o['h']:
            return o
    for o in state:
        if o.get('type') == 'player' and \
                o['x'] <= x < o['x'] + o['w'] and o['y'] <= y < o['y'] + o['h']:
            return o
    return None


def _mark(marker, active):
    marker['tags'] = (['color_11', 'active', 'marker'] if active
                      else ['color_9', 'inactive', 'marker'])


def _bar_width(state):
    w = 0
    for o in state:
        if o.get('type') == 'wall' and 'color_0' in o.get('tags', []):
            w = max(w, o.get('w', 0))
    return w


def _rename(state, armed):
    def rank_key(o):
        return (o['y'], o['x'])
    if armed:
        for rank, o in enumerate(sorted(state, key=rank_key)):
            o['name'] = '{}_{}_{}'.format(o['type'], _color(o), rank)
    else:
        rest = []
        for o in state:
            if 'block' in o.get('tags', []):
                o['name'] = 'block_left' if _is_left(o) else 'block_right'
            else:
                rest.append(o)
        for rank, o in enumerate(sorted(rest, key=rank_key)):
            o['name'] = '{}_{}_{}'.format(o['type'], _color(o), rank + 2)


def transition_function(state, action):
    global _last, _n
    if _last is None or _canon(state) != _last:
        w = _bar_width(state)
        _n = (7 * w - 1) // 3 if w else 0
    _n += 1

    objs = [dict(o) for o in state]
    markers = [o for o in objs if o.get('type') == 'marker']
    blocks = [o for o in objs if 'block' in o.get('tags', [])]
    players = [o for o in objs
               if o.get('type') == 'player' and 'armed' in o.get('tags', [])]
    armed = bool(players)

    if isinstance(action, dict) and action.get('action_id') == 6:
        hit = _hit(objs, action['x'], action['y'])
        if hit is not None and hit['type'] == 'marker':
            if not (armed and 'active' in hit['tags']):
                for m in markers:
                    _mark(m, m is hit)
                for b in blocks:            # arm: blocks become players
                    b['tags'] = ['color_1', 'armed', 'player']
                    b.pop('pixels', None)
        elif hit is not None and 'armed' in hit.get('tags', []):
            for p in players:               # disarm: players become blocks
                p['tags'] = ['cyan', 'block', 'player']
                p['pixels'] = [row[:] for row in BLOCK_PIX]
            for m in markers:
                _mark(m, False)
        # clicks on blocks, the active marker or empty space do nothing
    elif isinstance(action, int) and action in (1, 2, 3, 4):
        if armed:
            am = next((m for m in markers if 'active' in m['tags']), None)
            if am is not None:
                dx, dy = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}[action]
                nx, ny = am['x'] + dx, am['y'] + dy
                if _free_marker_cell(am, nx, ny, markers, players):
                    am['x'], am['y'] = nx, ny
        else:
            for b in blocks:
                if action in (1, 2):
                    dx, dy = 0, (-4 if action == 1 else 4)
                elif _is_left(b):
                    dx, dy = (-4 if action == 3 else 4), 0
                else:
                    dx, dy = (4 if action == 3 else -4), 0
                nx, ny = b['x'] + dx, b['y'] + dy
                if _free_block_cell(b, nx, ny, blocks, markers):
                    b['x'], b['y'] = nx, ny
    # actions 5 and 7 only advance the timer

    objs = [o for o in objs
            if not (o.get('type') == 'wall' and 'color_0' in o.get('tags', []))]
    w = (3 * (_n + 1)) // 7
    if w:
        objs.append({'name': '', 'type': 'wall', 'tags': ['color_0', 'wall'],
                     'x': 64 - w, 'y': 0, 'w': w, 'h': 1,
                     'layer': 0, 'visible': True})
        objs.append({'name': '', 'type': 'wall', 'tags': ['color_0', 'wall'],
                     'x': 0, 'y': 63, 'w': w, 'h': 1,
                     'layer': 0, 'visible': True})

    armed = any('armed' in o.get('tags', []) for o in objs)
    _rename(objs, armed)
    _last = _canon(objs)
    return objs
