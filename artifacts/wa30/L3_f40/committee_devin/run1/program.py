# Player: A1-4 turn cap and move 4 unless blocked (wall/box/chaser/bounds); box adjacent in cap dir = 'faced'; A5 toggles faced<->grabbed; grabbed box moves with player.
# Chaser acts every step: releases a seated carried box; else picks up an idle box whose right side it touches; else steps 4 (x first, y if blocked)
# toward nearest pickup point (box.x+4, box.y) or, when carrying, nearest empty slot (slot.x+4, slot.y). Mixed x/y under A5 = x-first fallback/arrival.
# Carried box rides at chaser.x-4 and seats on exact empty-slot hit (slot filled). Boxes renamed box_i by (y,x); wall pixels occluded by layers>0.
# Unconfirmed: pulling/turning while grabbed, chaser-player collisions, chaser through wall, behaviour once all slots are filled.
import copy

STEP = 4
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
CAPS = {1: 'cap_up', 2: 'cap_down', 3: 'cap_left', 4: 'cap_right'}
CAP_DIR = {v: DIRS[k] for k, v in CAPS.items()}
BOX_COLOR = {'idle': 4, 'carried': 5, 'faced': 3, 'grabbed': 0}
BOARD = 64


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o, dx=0, dy=0):
    return (o['x'] + dx, o['y'] + dy, o['w'], o['h'])


def in_board(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= BOARD and r[1] + r[3] <= BOARD


def box_state(b):
    for t in ('grabbed', 'faced', 'carried', 'idle'):
        if t in b['tags']:
            return t
    return 'idle'


def set_box(b, st, seated=None):
    if seated is None:
        seated = 'seated' in b['tags']
    b['tags'] = ['box', st] + (['seated'] if seated else [])
    c = BOX_COLOR[st]
    b['pixels'] = [[c if i in (0, b['w'] - 1) or j in (0, b['h'] - 1) else 9
                    for i in range(b['w'])] for j in range(b['h'])]


def player_pixels(p, cap):
    w, h = p['w'], p['h']
    def px(i, j):
        edge = {'cap_up': j == 0, 'cap_down': j == h - 1, 'cap_left': i == 0, 'cap_right': i == w - 1}[cap]
        return 0 if edge else 14
    return [[px(i, j) for i in range(w)] for j in range(h)]


# ---------------- player ----------------
def player_step(player, boxes, walls, chaser, action):
    cap = next((t for t in player['tags'] if t.startswith('cap_')), 'cap_up')
    grabbed = [b for b in boxes if box_state(b) == 'grabbed']
    if action == 5:
        faced = [b for b in boxes if box_state(b) == 'faced']
        if faced:
            for b in faced:
                set_box(b, 'grabbed')
        else:
            for b in grabbed:
                set_box(b, 'faced')
        return
    if action not in DIRS:
        return
    dx, dy = DIRS[action]
    if not grabbed:
        cap = CAPS[action]
    new = rect(player, dx, dy)
    ok = in_board(new) and not any(overlap(new, w) for w in walls)
    ok = ok and not any(overlap(new, rect(b)) for b in boxes if b not in grabbed)
    ok = ok and not (chaser and overlap(new, rect(chaser)))
    for g in grabbed:
        gr = rect(g, dx, dy)
        ok = ok and in_board(gr) and not any(overlap(gr, rect(b)) for b in boxes if b is not g)
    if ok:
        player['x'] += dx
        player['y'] += dy
        for g in grabbed:
            g['x'] += dx
            g['y'] += dy
    player['tags'] = ['player', cap]
    player['pixels'] = player_pixels(player, cap)
    update_faced(player, cap, boxes)


def update_faced(player, cap, boxes):
    fx, fy = CAP_DIR[cap]
    front = rect(player, fx, fy)
    for b in boxes:
        st = box_state(b)
        if st in ('idle', 'faced') and 'seated' not in b['tags']:
            hit = (b['x'], b['y']) == (front[0], front[1])
            if hit and st == 'idle':
                set_box(b, 'faced')
            elif not hit and st == 'faced':
                set_box(b, 'idle')


# ---------------- chaser ----------------
def chaser_move(ch, target, obstacles, carried):
    tx, ty = target
    opts = []
    if tx != ch['x']:
        opts.append((STEP if tx > ch['x'] else -STEP, 0))
    if ty != ch['y']:
        opts.append((0, STEP if ty > ch['y'] else -STEP))
    for dx, dy in opts:
        new = rect(ch, dx, dy)
        if not in_board(new) or any(overlap(new, o) for o in obstacles):
            continue
        ch['x'] += dx
        ch['y'] += dy
        if carried:
            carried['x'] = ch['x'] - carried['w']
            carried['y'] = ch['y']
        return


def chaser_step(ch, boxes, walls, slots, player):
    carried = next((b for b in boxes if box_state(b) == 'carried'), None)
    others = [rect(b) for b in boxes if b is not carried] + list(walls) + [rect(player)]
    dist = lambda t: abs(t[0] - ch['x']) + abs(t[1] - ch['y'])
    if carried:
        if 'seated' in carried['tags']:
            set_box(carried, 'idle', True)
            return
        empty = [s for s in slots if 'empty' in s['tags']]
        if not empty:
            return
        s = min(empty, key=lambda s: (dist((s['x'] + carried['w'], s['y'])), s['y'], s['x']))
        chaser_move(ch, (s['x'] + carried['w'], s['y']), others, carried)
        for s in empty:
            if (s['x'], s['y']) == (carried['x'], carried['y']):
                set_box(carried, 'carried', True)
                s['tags'] = [t if t != 'empty' else 'filled' for t in s['tags']]
                break
        return
    idle = [b for b in boxes if box_state(b) == 'idle' and 'seated' not in b['tags']]
    for b in idle:
        if (b['x'] + b['w'], b['y']) == (ch['x'], ch['y']):
            set_box(b, 'carried', False)
            return
    if idle:
        b = min(idle, key=lambda b: (dist((b['x'] + b['w'], b['y'])), b['y'], b['x']))
        chaser_move(ch, (b['x'] + b['w'], b['y']), others, None)


# ---------------- wall rendering ----------------
def render_wall(wall, objs):
    pix = wall.get('pixels')
    if not pix:
        return
    pattern = {}
    for j, row in enumerate(pix):
        if any(v != -1 for v in row):
            pattern.setdefault((wall['y'] + j) % 4, row)
    covers = [rect(o) for o in objs if o is not wall and o.get('visible', True)
              and o.get('layer', 0) > wall.get('layer', 0) and o.get('pixels')]
    out = []
    for j in range(wall['h']):
        y = wall['y'] + j
        base = pattern.get(y % 4, pix[j])
        out.append([-1 if any(overlap((wall['x'] + i, y, 1, 1), c) for c in covers) else base[i]
                    for i in range(wall['w'])])
    wall['pixels'] = out


def transition_function(state, action):
    objs = copy.deepcopy(state)
    act = action.get('action_id') if isinstance(action, dict) else action
    player = next((o for o in objs if o['type'] == 'player'), None)
    chaser = next((o for o in objs if o['type'] == 'chaser'), None)
    boxes = [o for o in objs if o['type'] == 'block']
    wall_objs = [o for o in objs if o['type'] == 'wall']
    walls = [rect(w) for w in wall_objs]
    slots = [o for o in objs if o['type'] == 'target']
    if player:
        player_step(player, boxes, walls, chaser, act)
    if chaser:
        chaser_step(chaser, boxes, walls, slots, player)
    for i, b in enumerate(sorted(boxes, key=lambda b: (b['y'], b['x']))):
        b['name'] = 'box_%d' % i
    for w in wall_objs:
        render_wall(w, objs)
    return objs
