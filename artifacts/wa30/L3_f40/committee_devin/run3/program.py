# Mechanics: player (A1-4 = up/down/left/right by 4) turns to the action direction (cap_*), moves unless blocked by
# wall/box/chaser/bounds; A5 toggles grab on the box it faces; a grabbed box moves rigidly with the player (boxes ignore
# walls). Chaser (autonomous, every step): if carrying a seated box -> release; if carrying -> drag it (box at chaser-4)
# toward the nearest empty slot; else go to the right side of the nearest free box and grab it; moves x-first, y if blocked. Box tags box+carried/grabbed/faced/idle(+seated), border colour by state, names box_i by (y,x); wall dither
# masked by objects above. Unconfirmed: hypothesis 'block A3 [player,wall]' = facing (faced iff player cap points at it); pulling, detours, cap_down.
import copy

DIRS = {1: (0, -4, 'cap_up'), 2: (0, 4, 'cap_down'), 3: (-4, 0, 'cap_left'), 4: (4, 0, 'cap_right')}
BORDER = {'idle': 4, 'carried': 5, 'faced': 3, 'grabbed': 0}
GRID = 64


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o, x=None, y=None):
    return (o['x'] if x is None else x, o['y'] if y is None else y, o['w'], o['h'])


def in_bounds(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= GRID and r[1] + r[3] <= GRID


def box_state(b):
    for t in ('carried', 'grabbed', 'faced'):
        if t in b['tags']:
            return t
    return 'idle'


def blocked(r, solids):
    return not in_bounds(r) or any(overlap(r, s) for s in solids)


def player_cap(p):
    for t in p['tags']:
        if t.startswith('cap_'):
            return t
    return 'cap_up'


def cap_dir(cap):
    for a, (dx, dy, c) in DIRS.items():
        if c == cap:
            return dx, dy
    return 0, -4


def player_pixels(cap, w, h):
    px = [[14] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if (cap == 'cap_up' and y == 0) or (cap == 'cap_down' and y == h - 1) or \
               (cap == 'cap_left' and x == 0) or (cap == 'cap_right' and x == w - 1):
                px[y][x] = 0
    return px


def box_pixels(state, w, h):
    c = BORDER[state]
    return [[9 if 0 < x < w - 1 and 0 < y < h - 1 else c for x in range(w)] for y in range(h)]


def step_player(player, boxes, walls, chasers, action):
    if action not in DIRS:
        if action == 5:
            for b in boxes:
                s = box_state(b)
                if s == 'faced':
                    b['_state'] = 'grabbed'
                elif s == 'grabbed':
                    b['_state'] = 'faced'
        return
    dx, dy, cap = DIRS[action]
    grabbed = [b for b in boxes if b['_state'] == 'grabbed']
    if not grabbed:
        player['_cap'] = cap
    nr = rect(player, player['x'] + dx, player['y'] + dy)
    others = [rect(b) for b in boxes if b not in grabbed] + [rect(c) for c in chasers]
    if blocked(nr, others + [rect(w) for w in walls]):
        return
    for b in grabbed:
        br = rect(b, b['x'] + dx, b['y'] + dy)
        if blocked(br, others + [nr]):
            return
    player['x'] += dx
    player['y'] += dy
    for b in grabbed:
        b['x'] += dx
        b['y'] += dy


def slot_at(b, slots):
    for s in slots:
        if s['x'] == b['x'] and s['y'] == b['y']:
            return s
    return None


def dist(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def chaser_move(c, tx, ty, solids, carried):
    def try_move(dx, dy):
        cr = rect(c, c['x'] + dx, c['y'] + dy)
        if blocked(cr, solids):
            return False
        if carried is not None:
            br = rect(carried, carried['x'] + dx, carried['y'] + dy)
            if not in_bounds(br) or any(overlap(br, s) for s in solids if s != rect(c)):
                return False
        c['x'] += dx
        c['y'] += dy
        if carried is not None:
            carried['x'] += dx
            carried['y'] += dy
        return True
    sx = (tx > c['x']) - (tx < c['x'])
    sy = (ty > c['y']) - (ty < c['y'])
    if sx and try_move(4 * sx, 0):
        return
    if sy:
        try_move(0, 4 * sy)


def step_chaser(c, player, boxes, walls, slots):
    carried = [b for b in boxes if b['_state'] == 'carried']
    others = lambda ex: [rect(b) for b in boxes if b is not ex] + [rect(w) for w in walls] + [rect(player)]
    if carried:
        b = carried[0]
        if slot_at(b, slots) is not None:
            b['_state'] = 'idle'
            return
        empty = [s for s in slots if 'empty' in s['tags'] and not any(x['x'] == s['x'] and x['y'] == s['y'] for x in boxes)]
        if not empty:
            return
        s = min(empty, key=lambda s: dist(b['x'], b['y'], s['x'], s['y']))
        chaser_move(c, s['x'] + c['w'], s['y'], others(b), b)
        return
    free = [b for b in boxes if b['_state'] in ('idle', 'faced') and slot_at(b, slots) is None]
    if not free:
        return
    b = min(free, key=lambda b: dist(c['x'], c['y'], b['x'] + b['w'], b['y']))
    ax, ay = b['x'] + b['w'], b['y']
    if (c['x'], c['y']) == (ax, ay):
        b['_state'] = 'carried'
        return
    chaser_move(c, ax, ay, others(None), None)


def render_wall(wall, above):
    px = wall['pixels']
    period = {}
    for r, row in enumerate(px):
        if any(v != -1 for v in row):
            period.setdefault((wall['y'] + r) % 4, row)
    out = []
    for r in range(wall['h']):
        y = wall['y'] + r
        base = period.get(y % 4, [-1] * wall['w'])
        row = []
        for i in range(wall['w']):
            x = wall['x'] + i
            covered = any(o['x'] <= x < o['x'] + o['w'] and o['y'] <= y < o['y'] + o['h'] for o in above)
            row.append(-1 if covered else base[i])
        out.append(row)
    return out


def transition_function(state, action):
    st = copy.deepcopy(state)
    aid = action.get('action_id') if isinstance(action, dict) else action
    players = [o for o in st if o['type'] == 'player']
    boxes = [o for o in st if o['type'] == 'block']
    walls = [o for o in st if o['type'] == 'wall']
    chasers = [o for o in st if o['type'] == 'chaser']
    slots = [o for o in st if o['type'] == 'target']
    for b in boxes:
        b['_state'] = box_state(b)
    for p in players:
        p['_cap'] = player_cap(p)
        step_player(p, boxes, walls, chasers, aid)
    for c in chasers:
        if players:
            step_chaser(c, players[0], boxes, walls, slots)
    for p in players:
        p['tags'] = ['player', p['_cap']]
        p['pixels'] = player_pixels(p['_cap'], p['w'], p['h'])
        dx, dy = cap_dir(p['_cap'])
        for b in boxes:
            if b['_state'] in ('idle', 'faced'):
                b['_state'] = 'faced' if (b['x'], b['y']) == (p['x'] + dx, p['y'] + dy) else 'idle'
        del p['_cap']
    for b in boxes:
        s = b.pop('_state')
        b['tags'] = ['box', s] + (['seated'] if slot_at(b, slots) is not None else [])
        b['pixels'] = box_pixels(s, b['w'], b['h'])
    for i, b in enumerate(sorted(boxes, key=lambda b: (b['y'], b['x']))):
        b['name'] = 'box_%d' % i
    for s in slots:
        filled = any(b['x'] == s['x'] and b['y'] == s['y'] for b in boxes)
        s['tags'] = [t for t in s['tags'] if t not in ('empty', 'filled')] + ['filled' if filled else 'empty']
    above = [o for o in st if o['layer'] > 0 and o.get('visible', True)]
    for w in walls:
        w['pixels'] = render_wall(w, above)
    return st
