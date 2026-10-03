# Mechanics: player A1-4 sets cap_dir and moves 4 (blocked by wall/box/chaser/bounds); a grabbed box moves rigidly with it (boxes ignore the wall).
# Box state: faced iff idle box sits at player+cap (positional), A5 toggles faced<->grabbed; border colour idle4 carried5 faced3 grabbed0, centre 9.
# Chaser acts every step after the player: release a seated carried box > pick up a free box touching its left side > step 4 toward the nearest
# pickup point (box.x+4,box.y) or, carrying, nearest empty slot+4; x-axis first, y if x is blocked. Seated/filled are positional; box names by (y,x).
# Wall = period-4 dither masked by layer>0 objects. Unconfirmed: cap_down sprite, turning while grabbed, player/chaser collisions, tie-breaks.
import copy

DIRS = {1: (0, -4, 'cap_up'), 2: (0, 4, 'cap_down'), 3: (-4, 0, 'cap_left'), 4: (4, 0, 'cap_right')}
BORDER = {'idle': 4, 'carried': 5, 'faced': 3, 'grabbed': 0}
DITHER = [[2, -1, 2, 2], [-1, 2, 2, 2], [2, 2, 2, -1], [2, 2, -1, 2]]
STATES = ('idle', 'carried', 'faced', 'grabbed')
N = 64


def cap_pixels(cap):
    p = [[14] * 4 for _ in range(4)]
    for i in range(4):
        if cap == 'cap_up': p[0][i] = 0
        elif cap == 'cap_down': p[3][i] = 0
        elif cap == 'cap_left': p[i][0] = 0
        else: p[i][3] = 0
    return p


def box_pixels(state):
    b = BORDER[state]
    return [[b] * 4, [b, 9, 9, b], [b, 9, 9, b], [b] * 4]


def overlap(ax, ay, aw, ah, bx, by, bw, bh):
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def in_bounds(x, y):
    return 0 <= x <= N - 4 and 0 <= y <= N - 4


class World:
    def __init__(self, state):
        self.objs = state
        self.player = next(o for o in state if o['type'] == 'player')
        self.chaser = next((o for o in state if o['type'] == 'chaser'), None)
        self.boxes = [o for o in state if o['type'] == 'block']
        self.wall = next((o for o in state if o['type'] == 'wall'), None)
        self.slots = [o for o in state if o['type'] == 'target']
        self.wx0 = self.wall['x'] if self.wall else None
        for b in self.boxes:
            b['state'] = next(t for t in b['tags'] if t in STATES)

    def hits_wall(self, x, y):
        w = self.wall
        return w is not None and overlap(x, y, 4, 4, w['x'], 0, w['w'], N - 1)

    def box_at(self, x, y, exclude=()):
        return any(overlap(x, y, 4, 4, b['x'], b['y'], 4, 4) for b in self.boxes if b not in exclude)

    def seated(self, b):
        return any(s['x'] == b['x'] and s['y'] == b['y'] for s in self.slots)

    def cap(self):
        return next(t for t in self.player['tags'] if t.startswith('cap_'))

    # --- player ---
    def step_player(self, action):
        p = self.player
        grabbed = next((b for b in self.boxes if b['state'] == 'grabbed'), None)
        if action in DIRS:
            dx, dy, cap = DIRS[action]
            p['tags'] = [t if not t.startswith('cap_') else cap for t in p['tags']]
            nx, ny = p['x'] + dx, p['y'] + dy
            ex = (grabbed,) if grabbed else ()
            ok = in_bounds(nx, ny) and not self.hits_wall(nx, ny) and not self.box_at(nx, ny, ex) \
                and not self.chaser_at(nx, ny)
            if ok and grabbed:
                bx, by = grabbed['x'] + dx, grabbed['y'] + dy
                ok = in_bounds(bx, by) and not self.box_at(bx, by, ex) and not self.chaser_at(bx, by)
                if ok:
                    grabbed['x'], grabbed['y'] = bx, by
            if ok:
                p['x'], p['y'] = nx, ny
        elif action == 5:
            faced = next((b for b in self.boxes if b['state'] == 'faced'), None)
            if faced:
                faced['state'] = 'grabbed'
            elif grabbed:
                grabbed['state'] = 'faced'
        self.update_faced()

    def chaser_at(self, x, y):
        c = self.chaser
        return c is not None and overlap(x, y, 4, 4, c['x'], c['y'], 4, 4)

    def update_faced(self):
        p = self.player
        dx, dy = next((d[0], d[1]) for d in DIRS.values() if d[2] == self.cap())
        fx, fy = p['x'] + dx, p['y'] + dy
        for b in self.boxes:
            if b['state'] in ('idle', 'faced'):
                b['state'] = 'faced' if (b['x'], b['y']) == (fx, fy) else 'idle'

    # --- chaser ---
    def step_chaser(self):
        c = self.chaser
        if c is None:
            return
        carried = next((b for b in self.boxes if b['state'] == 'carried'), None)
        if carried:
            if self.seated(carried):
                carried['state'] = 'idle'
                self.update_faced()
                return
            filled = {(b['x'], b['y']) for b in self.boxes if b is not carried}
            goals = [(s['x'] + c['x'] - carried['x'], s['y'] + c['y'] - carried['y'])
                     for s in self.slots if (s['x'], s['y']) not in filled]
        else:
            free = [b for b in self.boxes if b['state'] in ('idle', 'faced') and not self.seated(b)]
            for b in free:
                if b['state'] == 'idle' and (b['x'] + 4, b['y']) == (c['x'], c['y']):
                    b['state'] = 'carried'
                    return
            goals = [(b['x'] + 4, b['y']) for b in free]
        if not goals:
            return
        gx, gy = min(goals, key=lambda g: abs(g[0] - c['x']) + abs(g[1] - c['y']))
        moves = []
        if gx != c['x']:
            moves.append((4 if gx > c['x'] else -4, 0))
        if gy != c['y']:
            moves.append((0, 4 if gy > c['y'] else -4))
        for dx, dy in moves:
            if self.chaser_can(dx, dy, carried):
                c['x'] += dx; c['y'] += dy
                if carried:
                    carried['x'] += dx; carried['y'] += dy
                return

    def chaser_can(self, dx, dy, carried):
        c, p = self.chaser, self.player
        nx, ny = c['x'] + dx, c['y'] + dy
        ex = (carried,) if carried else ()
        if not in_bounds(nx, ny) or self.hits_wall(nx, ny) or self.box_at(nx, ny, ex):
            return False
        if overlap(nx, ny, 4, 4, p['x'], p['y'], 4, 4):
            return False
        if carried:
            bx, by = carried['x'] + dx, carried['y'] + dy
            if not in_bounds(bx, by) or self.box_at(bx, by, ex) or overlap(bx, by, 4, 4, p['x'], p['y'], 4, 4):
                return False
        return True

    # --- render ---
    def render(self):
        p = self.player
        p['pixels'] = cap_pixels(self.cap())
        for b in self.boxes:
            st = b.pop('state')
            b['tags'] = ['box', st] + (['seated'] if self.seated(b) else [])
            b['pixels'] = box_pixels(st)
        for i, b in enumerate(sorted(self.boxes, key=lambda b: (b['y'], b['x']))):
            b['name'] = 'box_%d' % i
        occ = {(b['x'], b['y']) for b in self.boxes}
        for s in self.slots:
            s['tags'] = [t for t in s['tags'] if t not in ('empty', 'filled')] + \
                ['filled' if (s['x'], s['y']) in occ else 'empty']
        if self.wall:
            self.render_wall()
        return self.objs

    def render_wall(self):
        w = self.wall
        x0, wid, hgt = self.wx0, 4, N - 1
        covers = [o for o in self.objs if o['layer'] > 0 and o.get('visible', True)]
        grid = []
        for r in range(hgt):
            row = []
            for cx in range(wid):
                v = DITHER[r % 4][cx]
                if any(overlap(x0 + cx, r, 1, 1, o['x'], o['y'], o['w'], o['h']) for o in covers):
                    v = -1
                row.append(v)
            grid.append(row)
        rows = [r for r in range(hgt) if any(v != -1 for v in grid[r])]
        cols = [c for c in range(wid) if any(grid[r][c] != -1 for r in range(hgt))]
        if not rows:
            self.objs.remove(w); return
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
        w['x'], w['y'], w['w'], w['h'] = x0 + c0, r0, c1 - c0 + 1, r1 - r0 + 1
        w['pixels'] = [grid[r][c0:c1 + 1] for r in range(r0, r1 + 1)]


def transition_function(state, action):
    state = copy.deepcopy(state)
    if isinstance(action, dict):
        action = action.get('action_id')
    w = World(state)
    w.step_player(action)
    w.step_chaser()
    return w.render()
