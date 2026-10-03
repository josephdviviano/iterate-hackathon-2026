# Mechanics: player (A1-4 up/down/left/right by 4) turns to face the move; an occupied cell (box/chaser/wall/bounds)
# blocks it; A5 toggles the box in front between faced(3) and grabbed(0); a grabbed box moves with the player.
# Idle box in front of the player = faced. Chaser: seeks nearest reachable idle unseated box from its right side,
# grabs it (carried, 5), drags it (x first, then y; greedy, fallback other axis) to nearest empty slot, then releases.
# Wall = hatched column; boxes may enter it and hide its pixels. Hypothesis (unconfirmed): pulling/perpendicular grabbed moves.
import copy

STEP = 4
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
CAP = {(0, -1): 'cap_up', (0, 1): 'cap_down', (-1, 0): 'cap_left', (1, 0): 'cap_right'}
BOX_COLOR = {'idle': 4, 'faced': 3, 'grabbed': 0, 'carried': 5}
BOARD = 64


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o['x'], o['y'], o['w'], o['h'])


def box_state(o):
    for t in ('grabbed', 'faced', 'carried', 'idle'):
        if t in o['tags']:
            return t
    return 'idle'


def player_pixels(cap):
    px = [[14] * 4 for _ in range(4)]
    for i in range(4):
        if cap == 'cap_up': px[0][i] = 0
        if cap == 'cap_down': px[3][i] = 0
        if cap == 'cap_left': px[i][0] = 0
        if cap == 'cap_right': px[i][3] = 0
    return px


def box_pixels(state):
    c = BOX_COLOR[state]
    return [[c] * 4, [c, 9, 9, c], [c, 9, 9, c], [c] * 4]


class World:
    def __init__(self, state):
        self.objs = copy.deepcopy(state)
        self.player = next(o for o in self.objs if o['type'] == 'player')
        self.chaser = next((o for o in self.objs if o['type'] == 'chaser'), None)
        self.boxes = [o for o in self.objs if o['type'] == 'block']
        self.slots = [o for o in self.objs if o['type'] == 'target']
        self.walls = [o for o in self.objs if o['type'] == 'wall']
        self.state = {id(b): box_state(b) for b in self.boxes}
        self.wall_info = [self.parse_wall(w) for w in self.walls]

    def parse_wall(self, w):
        pattern, gaps = {}, set()
        for r, row in enumerate(w['pixels']):
            if any(v != -1 for v in row):
                pattern[(w['y'] + r) % 4] = row
        for r, row in enumerate(w['pixels']):
            if all(v == -1 for v in row):
                cell = (w['x'], w['y'] + r, w['w'], 1)
                if not any(overlap(cell, rect(b)) for b in self.boxes):
                    gaps.add(r)
        return pattern, gaps

    def wall_solid(self, rc):
        for w, (_, gaps) in zip(self.walls, self.wall_info):
            for r in range(w['h']):
                if r not in gaps and overlap(rc, (w['x'], w['y'] + r, w['w'], 1)):
                    return True
        return False

    def in_bounds(self, rc):
        return rc[0] >= 0 and rc[1] >= 0 and rc[0] + rc[2] <= BOARD and rc[1] + rc[3] <= BOARD

    def occupied(self, rc, ignore=()):
        movers = self.boxes + [self.player] + ([self.chaser] if self.chaser else [])
        return any(overlap(rc, rect(o)) for o in movers if not any(o is i for i in ignore))

    def facing(self):
        for t in self.player['tags']:
            for d, c in CAP.items():
                if t == c:
                    return d
        return (0, -1)

    def box_at(self, x, y):
        return next((b for b in self.boxes if b['x'] == x and b['y'] == y), None)

    # ---- player ----
    def player_step(self, action):
        p = self.player
        grabbed = next((b for b in self.boxes if self.state[id(b)] == 'grabbed'), None)
        if action == 5:
            d = self.facing()
            b = self.box_at(p['x'] + d[0] * STEP, p['y'] + d[1] * STEP)
            if b is not None and self.state[id(b)] == 'grabbed':
                self.state[id(b)] = 'faced'
            elif b is not None and self.state[id(b)] in ('faced', 'idle'):
                self.state[id(b)] = 'grabbed'
            return
        if action not in DIRS:
            return
        d = DIRS[action]
        np_ = (p['x'] + d[0] * STEP, p['y'] + d[1] * STEP, p['w'], p['h'])
        if grabbed is not None:
            nb = (grabbed['x'] + d[0] * STEP, grabbed['y'] + d[1] * STEP, grabbed['w'], grabbed['h'])
            ok = (self.in_bounds(np_) and self.in_bounds(nb) and not self.wall_solid(np_)
                  and not self.occupied(np_, (p, grabbed)) and not self.occupied(nb, (p, grabbed)))
            if ok:
                p['x'], p['y'] = np_[0], np_[1]
                grabbed['x'], grabbed['y'] = nb[0], nb[1]
            return
        self.set_cap(CAP[d])
        if self.in_bounds(np_) and not self.wall_solid(np_) and not self.occupied(np_, (p,)):
            p['x'], p['y'] = np_[0], np_[1]

    def set_cap(self, cap):
        p = self.player
        p['tags'] = [t for t in p['tags'] if not t.startswith('cap_')] + [cap]
        p['pixels'] = player_pixels(cap)

    # ---- chaser ----
    def seated(self, b):
        return any(s['x'] == b['x'] and s['y'] == b['y'] for s in self.slots)

    def chaser_free(self, x, y, ignore):
        rc = (x, y, self.chaser['w'], self.chaser['h'])
        return self.in_bounds(rc) and not self.wall_solid(rc) and not self.occupied(rc, ignore)

    def reachable(self):
        c = self.chaser
        start = (c['x'], c['y'])
        dist, frontier = {start: 0}, [start]
        while frontier:
            nxt = []
            for (x, y) in frontier:
                for dx, dy in DIRS.values():
                    q = (x + dx * STEP, y + dy * STEP)
                    if q not in dist and self.chaser_free(q[0], q[1], (c,)):
                        dist[q] = dist[(x, y)] + 1
                        nxt.append(q)
            frontier = nxt
        return dist

    def greedy_move(self, tx, ty, carried):
        c = self.chaser
        ign = (c, carried) if carried is not None else (c,)
        moves = []
        if tx != c['x']:
            moves.append((STEP if tx > c['x'] else -STEP, 0))
        if ty != c['y']:
            moves.append((0, STEP if ty > c['y'] else -STEP))
        for dx, dy in moves:
            if not self.chaser_free(c['x'] + dx, c['y'] + dy, ign):
                continue
            if carried is not None:
                nb = (carried['x'] + dx, carried['y'] + dy, carried['w'], carried['h'])
                if not self.in_bounds(nb) or self.occupied(nb, ign):
                    continue
                carried['x'] += dx
                carried['y'] += dy
            c['x'] += dx
            c['y'] += dy
            return

    def chaser_step(self):
        c = self.chaser
        if c is None:
            return
        carried = next((b for b in self.boxes if self.state[id(b)] == 'carried'), None)
        if carried is not None:
            if self.seated(carried):
                self.state[id(carried)] = 'idle'
                return
            filled = {(b['x'], b['y']) for b in self.boxes}
            empty = [s for s in self.slots if (s['x'], s['y']) not in filled]
            if not empty:
                return
            s = min(empty, key=lambda s: (abs(s['x'] - carried['x']) + abs(s['y'] - carried['y']), s['y'], s['x']))
            self.greedy_move(s['x'] + (c['x'] - carried['x']), s['y'] + (c['y'] - carried['y']), carried)
            return
        dist = self.reachable()
        cands = []
        for b in self.boxes:
            if self.state[id(b)] != 'idle' or self.seated(b):
                continue
            a = (b['x'] + b['w'], b['y'])
            if a in dist:
                cands.append((dist[a], b['y'], b['x'], a, b))
        if not cands:
            return
        cands.sort(key=lambda t: t[:3])
        _, _, _, a, b = cands[0]
        if (c['x'], c['y']) == a:
            self.state[id(b)] = 'carried'
        else:
            self.greedy_move(a[0], a[1], None)

    # ---- derived fields ----
    def finish(self):
        d = self.facing()
        p = self.player
        front = self.box_at(p['x'] + d[0] * STEP, p['y'] + d[1] * STEP)
        for b in self.boxes:
            st = self.state[id(b)]
            if st in ('idle', 'faced'):
                st = 'faced' if b is front else 'idle'
            b['tags'] = ['box', st] + (['seated'] if self.seated(b) else [])
            b['pixels'] = box_pixels(st)
        for i, b in enumerate(sorted(self.boxes, key=lambda b: (b['y'], b['x']))):
            b['name'] = 'box_%d' % i
        filled = {(b['x'], b['y']) for b in self.boxes}
        for s in self.slots:
            s['tags'] = [t for t in s['tags'] if t not in ('empty', 'filled')] + \
                ['filled' if (s['x'], s['y']) in filled else 'empty']
        for w, (pattern, gaps) in zip(self.walls, self.wall_info):
            px = []
            for r in range(w['h']):
                base = pattern.get((w['y'] + r) % 4, [-1] * w['w'])
                row = list(base) if r not in gaps else [-1] * w['w']
                for ci in range(w['w']):
                    cell = (w['x'] + ci, w['y'] + r, 1, 1)
                    if any(overlap(cell, rect(b)) for b in self.boxes):
                        row[ci] = -1
                px.append(row)
            w['pixels'] = px
        return self.objs


def transition_function(state, action):
    aid = action.get('action_id') if isinstance(action, dict) else action
    w = World(state)
    w.player_step(aid)
    w.chaser_step()
    return w.finish()
