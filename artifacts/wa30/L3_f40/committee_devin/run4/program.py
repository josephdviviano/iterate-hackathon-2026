# Mechanics: player (A1-4 move 4 cells, sets cap_*; bumping a box marks it 'faced' without moving; A5 toggles
# faced<->grabbed; a grabbed box moves with the player; moving away un-faces). Wall blocks player/chaser, not boxes.
# Chaser (autonomous, acts every step after the player): release a carried+seated box > pick up an idle unseated
# box whose right side it touches > step 4 toward nearest pickup point (box.x+4,box.y) or empty slot+(4,0) when
# carrying, x first, y if x reached/blocked. Unconfirmed: chaser vs wall crossing, pulling a grabbed box, tie-breaks.
import copy

DIRS = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}
CAPS = {1: 'cap_up', 2: 'cap_down', 3: 'cap_left', 4: 'cap_right'}
BOX_BORDER = {'idle': 4, 'carried': 5, 'faced': 3, 'grabbed': 0}
GRID = 64


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o, dx=0, dy=0):
    return (o['x'] + dx, o['y'] + dy, o['w'], o['h'])


def in_bounds(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= GRID and r[1] + r[3] <= GRID


def box_state(b):
    for t in ('carried', 'grabbed', 'faced', 'idle'):
        if t in b['tags']:
            return t
    return 'idle'


def set_box_state(b, st):
    seated = 'seated' in b['tags']
    b['tags'] = ['box', st] + (['seated'] if seated else [])


def box_pixels(st):
    c = BOX_BORDER[st]
    return [[c, c, c, c], [c, 9, 9, c], [c, 9, 9, c], [c, c, c, c]]


def player_pixels(cap):
    p = [[14] * 4 for _ in range(4)]
    for i in range(4):
        if cap == 'cap_up':
            p[0][i] = 0
        elif cap == 'cap_down':
            p[3][i] = 0
        elif cap == 'cap_left':
            p[i][0] = 0
        else:
            p[i][3] = 0
    return p


def hits_wall(r, walls):
    return any(overlap(r, rect(w)) for w in walls)


def step_player(player, boxes, walls, chasers, action):
    if action == 5:
        for b in boxes:
            st = box_state(b)
            if st == 'faced':
                set_box_state(b, 'grabbed')
            elif st == 'grabbed':
                set_box_state(b, 'faced')
        return
    if action not in DIRS:
        return
    dx, dy = DIRS[action]
    cap = CAPS[action]
    player['tags'] = ['player', cap]
    nr = rect(player, dx, dy)
    grabbed = [b for b in boxes if box_state(b) == 'grabbed']
    others = [b for b in boxes if b not in grabbed]
    if not in_bounds(nr) or hits_wall(nr, walls) or any(overlap(nr, rect(c)) for c in chasers):
        return
    if grabbed:
        g = grabbed[0]
        gr = rect(g, dx, dy)
        if any(overlap(nr, rect(b)) for b in others) or not in_bounds(gr) \
                or any(overlap(gr, rect(b)) for b in others) or any(overlap(gr, rect(c)) for c in chasers):
            return
        player['x'], player['y'] = nr[0], nr[1]
        g['x'], g['y'] = gr[0], gr[1]
        return
    bumped = [b for b in boxes if overlap(nr, rect(b))]
    if bumped:
        for b in boxes:
            if box_state(b) == 'faced':
                set_box_state(b, 'idle')
        if box_state(bumped[0]) == 'idle':
            set_box_state(bumped[0], 'faced')
        return
    player['x'], player['y'] = nr[0], nr[1]
    for b in boxes:
        if box_state(b) == 'faced':
            set_box_state(b, 'idle')


def seat_if_on_slot(box, slots):
    for s in slots:
        if s['x'] == box['x'] and s['y'] == box['y'] and 'empty' in s['tags']:
            box['tags'] = box['tags'] + ['seated'] if 'seated' not in box['tags'] else box['tags']
            s['tags'] = [t if t != 'empty' else 'filled' for t in s['tags']]
            return


def chaser_blocked(r, carried, boxes, walls, player):
    if not in_bounds(r) or hits_wall(r, walls) or overlap(r, rect(player)):
        return True
    return any(overlap(r, rect(b)) for b in boxes if b is not carried)


def step_chaser(ch, boxes, slots, walls, player):
    carried = next((b for b in boxes if box_state(b) == 'carried'), None)
    if carried is not None and 'seated' in carried['tags']:
        set_box_state(carried, 'idle')
        return
    if carried is None:
        for b in boxes:
            if box_state(b) == 'idle' and 'seated' not in b['tags'] \
                    and b['x'] + b['w'] == ch['x'] and b['y'] == ch['y']:
                set_box_state(b, 'carried')
                return
        goals = [(b['x'] + b['w'], b['y']) for b in boxes
                 if box_state(b) == 'idle' and 'seated' not in b['tags']]
    else:
        goals = [(s['x'] + s['w'], s['y']) for s in slots if 'empty' in s['tags']]
    if not goals:
        return
    cx, cy = ch['x'], ch['y']
    tx, ty = min(goals, key=lambda g: abs(g[0] - cx) + abs(g[1] - cy))
    moves = []
    if tx != cx:
        moves.append((4 if tx > cx else -4, 0))
    if ty != cy:
        moves.append((0, 4 if ty > cy else -4))
    for dx, dy in moves:
        if chaser_blocked(rect(ch, dx, dy), carried, boxes, walls, player):
            continue
        if carried is not None:
            cr = rect(carried, dx, dy)
            if not in_bounds(cr) or any(overlap(cr, rect(b)) for b in boxes if b is not carried) \
                    or overlap(cr, rect(player)):
                continue
        ch['x'] += dx
        ch['y'] += dy
        if carried is not None:
            carried['x'] += dx
            carried['y'] += dy
            seat_if_on_slot(carried, slots)
        return


def render_wall(w, occluders):
    old = w['pixels']
    H, W = len(old), len(old[0]) if old else 0
    occ = lambda x, y: any(o['x'] <= x < o['x'] + o['w'] and o['y'] <= y < o['y'] + o['h'] for o in occluders)
    base = [[None] * W for _ in range(H)]
    for j in range(H):
        for i in range(W):
            if old[j][i] != -1 or not occ(w['x'] + i, w['y'] + j):
                base[j][i] = old[j][i]
    period = 4
    for j in range(H):
        for i in range(W):
            if base[j][i] is None:
                for k in range(j % period, H, period):
                    if base[k][i] is not None:
                        base[j][i] = base[k][i]
                        break
    return base


def transition_function(state, action):
    st = copy.deepcopy(state)
    if isinstance(action, dict):
        action = action.get('action_id')
    player = next((o for o in st if o['type'] == 'player'), None)
    boxes = [o for o in st if o['type'] == 'block']
    walls = [o for o in st if o['type'] == 'wall']
    chasers = [o for o in st if o['type'] == 'chaser']
    slots = [o for o in st if o['type'] == 'target']
    before_occ = [o for o in st if o['type'] in ('block', 'player', 'chaser')]
    wall_base = {id(w): render_wall(w, before_occ) for w in walls}
    if player is not None:
        step_player(player, boxes, walls, chasers, action)
    for ch in chasers:
        step_chaser(ch, boxes, slots, walls, player)
    for b in boxes:
        b['pixels'] = box_pixels(box_state(b))
    for i, b in enumerate(sorted(boxes, key=lambda b: (b['y'], b['x']))):
        b['name'] = 'box_%d' % i
    if player is not None:
        player['pixels'] = player_pixels(player['tags'][1])
    occ = [o for o in st if o['type'] in ('block', 'player', 'chaser')]
    for w in walls:
        base = wall_base[id(w)]
        w['pixels'] = [[-1 if any(o['x'] <= w['x'] + i < o['x'] + o['w'] and o['y'] <= w['y'] + j < o['y'] + o['h']
                                  for o in occ) or v is None else v
                        for i, v in enumerate(row)] for j, row in enumerate(base)]
    return st
