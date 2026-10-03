# Mechanics: ring (player) + beam (arm) slide on a rail by 6; on-beam blocks (fully inside beam cols, overlapping
# beam rows, any tag) are carried; moved rects push other blocks (chain). A1/A2 cancel if any block/ring leaves the
# colour-4 field; A4/A3 grow/shrink beam by 6 (w in [1, field]); blocks that would leave the field -> beam alone.
# Tags: hooked next (x == tip-5) with all other on-beam blocks collected -> collected; collected off beam -> next.
# HUD row: 3s from right = (n-1)//3, n = non-click actions (memo by returned frame; fallback 3*threes+1). Unconfirmed: A5/A7.
STEP = 6
_memo = {}


def key(frame):
    return tuple(tuple(r) for r in frame)


class World:
    def __init__(self, frame):
        self.hud = next(y for y in range(64) if all(c not in (4, 5) for c in frame[y]))
        cells = [(x, y) for y in range(self.hud) for x in range(64) if frame[y][x] == 4]
        self.fx0 = min(x for x, _ in cells)
        self.fx1 = max(x for x, _ in cells)
        self.fy0 = min(y for _, y in cells)
        self.fy1 = max(y for _, y in cells)

    def inside(self, r):
        x, y, w, h = r
        return x >= self.fx0 and x + w - 1 <= self.fx1 and y >= self.fy0 and y + h - 1 <= self.fy1

    def bg(self, x, y, ring):
        if self.fx0 <= x <= self.fx1 and self.fy0 <= y <= self.fy1:
            return 4
        if x in (ring['x'] + 2, ring['x'] + 3) and self.fy0 + 2 <= y <= self.fy1 - 2:
            return 2 if (y - self.fy0 - 2) % 6 < 2 else 3
        return 5


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o['x'], o['y'], o['w'], o['h'])


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def on_beam(b, beam):
    tip = beam['x'] + beam['w']
    rows = (beam['y'], beam['y'] + beam['h'])
    return (b['x'] >= beam['x'] and b['x'] + b['w'] <= tip
            and b['y'] < rows[1] and rows[0] < b['y'] + b['h'])


def push_chain(blocks, movers, pushers, dx, dy):
    """movers: set of indices; pushers: extra rects already at their new place. Returns closed mover set."""
    movers = set(movers)
    changed = True
    while changed:
        changed = False
        rects = list(pushers) + [shift(rect(blocks[i]), dx, dy) for i in movers]
        for i, b in enumerate(blocks):
            if i not in movers and any(overlap(r, rect(b)) for r in rects):
                movers.add(i)
                changed = True
    return movers


def colour(b):
    for t in b.get('tags', []):
        if t.isdigit():
            return int(t)
    return b['pixels'][0][0]


def status(b):
    for t in ('collected', 'next', 'todo'):
        if t in b.get('tags', []):
            return t
    return 'todo'


def vertical(world, ring, beam, blocks, dy):
    nr = shift(rect(ring), 0, dy)
    if nr[1] < world.fy0 or nr[1] + nr[3] - 1 > world.fy1:
        return
    carried = {i for i, b in enumerate(blocks) if on_beam(b, beam)}
    nb = shift(rect(beam), 0, dy)
    movers = push_chain(blocks, carried, [nb, nr], 0, dy)
    if not all(world.inside(shift(rect(blocks[i]), 0, dy)) for i in movers):
        return
    for i in movers:
        blocks[i]['y'] += dy
    ring['y'] += dy
    beam['y'] += dy


def horizontal(world, beam, blocks, sign):
    nw = beam['w'] + sign * STEP
    tip = beam['x'] + beam['w']
    if nw < 1 or beam['x'] + nw - 1 > world.fx1:
        return
    dx = sign * STEP
    movers = {i for i, b in enumerate(blocks) if on_beam(b, beam)}
    pushers = []
    if sign > 0:
        pushers.append((tip, beam['y'], STEP, beam['h']))
    movers = push_chain(blocks, movers, pushers, dx, 0)
    if all(world.inside(shift(rect(blocks[i]), dx, 0)) for i in movers):
        for i in movers:
            blocks[i]['x'] += dx
    beam['w'] = nw


def update_tags(beam, blocks, order):
    tip = beam['x'] + beam['w']
    st = [status(b) for b in blocks]
    onb = [on_beam(b, beam) for b in blocks]
    hooked = [onb[i] and b['x'] == tip - 5 for i, b in enumerate(blocks)]
    for i in range(len(blocks)):
        if st[i] == 'next' and hooked[i] and all(st[j] == 'collected' for j in range(len(blocks)) if j != i and onb[j]):
            st[i] = 'collected'
            todo = [j for j in order if st[j] == 'todo']
            if todo:
                st[todo[0]] = 'next'
            return st
    for i in range(len(blocks)):
        if st[i] == 'collected' and not onb[i]:
            for j in range(len(blocks)):
                if st[j] == 'next':
                    st[j] = 'todo'
            st[i] = 'next'
            return st
    return st


def legend_icons(frame, world):
    icons = {}
    for y in range(world.hud + 1, 64):
        for x in range(64):
            c = frame[y][x]
            if c not in icons:
                icons[c] = (x, y)
    return icons


def hud_count(frame, world, action):
    k = key(frame)
    row = frame[world.hud]
    if k in _memo:
        n = _memo[k]
    else:
        threes = sum(1 for c in row if c == 3)
        n = 3 * threes + 1 if threes else 0
    if not isinstance(action, dict):
        n += 1
    return n


def transition_function(state, action, frame):
    world = World(frame)
    objs = [dict(o) for o in state]
    ring = next(o for o in objs if o['type'] == 'player')
    beam = next(o for o in objs if o['type'] == 'arm')
    blocks = [o for o in objs if o['type'] == 'block']
    old = [rect(o) for o in [ring, beam] + blocks]
    old_ring = dict(ring)
    st_before = [status(b) for b in blocks]

    aid = action['action_id'] if isinstance(action, dict) else action
    if aid in (1, 2):
        vertical(world, ring, beam, blocks, -STEP if aid == 1 else STEP)
    elif aid in (3, 4):
        horizontal(world, beam, blocks, 1 if aid == 4 else -1)

    icons = legend_icons(frame, world)
    order = sorted(range(len(blocks)), key=lambda i: icons.get(colour(blocks[i]), (99, 99)))
    st = update_tags(beam, blocks, order) if aid in (1, 2, 3, 4) else st_before

    out = [list(r) for r in frame]
    for (x, y, w, h) in old:
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                if 0 <= xx < 64 and 0 <= yy < world.hud:
                    out[yy][xx] = world.bg(xx, yy, old_ring)
    for i in range(ring['h']):
        for j in range(ring['w']):
            out[ring['y'] + i][ring['x'] + j] = ring['pixels'][i][j]
    for i in range(beam['h']):
        for j in range(beam['w']):
            out[beam['y'] + i][beam['x'] + j] = 2 if (i + j) % 3 == 1 else 1
    for b in blocks:
        c = colour(b)
        for i in range(b['h']):
            for j in range(b['w']):
                out[b['y'] + i][b['x'] + j] = c

    for i, b in enumerate(blocks):
        c = colour(b)
        if c in icons:
            x0, y0 = icons[c]
            for yy in (y0 + 1, y0 + 2):
                for xx in (x0 + 1, x0 + 2):
                    out[yy][xx] = 0 if st[i] == 'collected' else c

    n = hud_count(frame, world, action)
    filled = max(0, (n - 1) // 3)
    for x in range(64):
        if out[world.hud][x] in (2, 3):
            out[world.hud][x] = 3 if x >= 64 - filled else 2
    _memo[key(out)] = n
    return out
