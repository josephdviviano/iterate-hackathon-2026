# Crane/skewer game. Ring (player) moves y by 6 on A1/A2 carrying blocks on the beam; beam (arm) w +-6 on A4/A3.
# Hook: any block on the beam rows whose x == tip-4 (tag-free; step-87 todo block is dragged), then all blocks fully
# on the beam follow the tip; unhooked extend pushes blocks in the swept strip. Blocked chains: vertical cancels all,
# horizontal leaves the beam sliding alone. Tags: hooked next alone on beam -> collected; collected off the beam reverts.
# HUD row: 3s = (n-1)//3, n = non-click actions; n kept in a memo of returned frames (fallback 3*bars+1 or 0).

MEMO = {}


def key_of(frame):
    return tuple(tuple(r) for r in frame)


class World:
    def __init__(self, frame):
        self.f = frame
        self.hud = next(y for y in range(64) if all(v not in (4, 5) for v in frame[y]))
        cells = [(x, y) for y in range(self.hud) for x in range(64) if frame[y][x] == 4]
        self.left = min(c[0] for c in cells)
        self.right = max(c[0] for c in cells)
        self.top = min(c[1] for c in cells)
        self.bot = max(c[1] for c in cells)

    def inside(self, r):
        x, y, w, h = r
        return x >= self.left and x + w - 1 <= self.right and y >= self.top and y + h - 1 <= self.bot

    def bg(self, x, y, ring):
        if self.left <= x <= self.right and self.top <= y <= self.bot:
            return 4
        if x in (ring['x'] + 2, ring['x'] + 3) and self.top + 2 <= y <= self.bot - 2:
            return 2 if (y - self.top - 2) % 6 < 2 else 3
        return 5


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o['x'], o['y'], o['w'], o['h'])


def shift(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def push_chain(blocks, movers, solids, dx, dy, world):
    """movers: names of blocks moving; solids: other moved rects (ring/beam/swept). Returns set or None."""
    moving = set(movers)
    frontier = [shift(rect(blocks[n]), dx, dy) for n in moving] + list(solids)
    while frontier:
        r = frontier.pop()
        for n, b in blocks.items():
            if n not in moving and overlap(r, rect(b)):
                moving.add(n)
                frontier.append(shift(rect(b), dx, dy))
    for n in moving:
        if not world.inside(shift(rect(blocks[n]), dx, dy)):
            return None
    return moving


def beam_rect(beam):
    return (beam['x'], beam['y'], beam['w'], beam['h'])


def on_beam(b, beam):
    return overlap(rect(b), beam_rect(beam))


def fully_on_beam(b, beam):
    return on_beam(b, beam) and b['x'] >= beam['x'] and b['x'] + b['w'] <= beam['x'] + beam['w']


def hooked(blocks, beam):
    tip = beam['x'] + beam['w'] - 1
    return [n for n, b in blocks.items() if on_beam(b, beam) and b['x'] == tip - 4]


def step_vertical(ring, beam, blocks, dy, world):
    nr = shift(rect(ring), 0, dy)
    if nr[1] < world.top or nr[1] + nr[3] - 1 > world.bot:
        return False
    carried = [n for n, b in blocks.items() if on_beam(b, beam)]
    moving = push_chain(blocks, carried, [nr, shift(beam_rect(beam), 0, dy)], 0, dy, world)
    if moving is None:
        return False
    ring['y'] += dy
    beam['y'] += dy
    for n in moving:
        blocks[n]['y'] += dy
    return True


def step_horizontal(beam, blocks, d, world):
    tip = beam['x'] + beam['w'] - 1
    if d > 0 and tip + d > world.right:
        return False
    if d < 0 and beam['w'] + d < 1:
        return False
    hook = hooked(blocks, beam)
    moving = None
    if hook:
        group = [n for n, b in blocks.items() if fully_on_beam(b, beam)]
        moving = push_chain(blocks, group, [], d, 0, world)
    elif d > 0:
        strip = (tip + 1, beam['y'], d, beam['h'])
        moving = push_chain(blocks, [], [strip], d, 0, world)
    beam['w'] += d
    for n in moving or ():
        blocks[n]['x'] += d
    return True


def legend_order(world, blocks):
    def lx(n):
        c = int(blocks[n]['tags'][1])
        xs = [x for y in range(world.hud + 1, 64) for x in range(64) if world.f[y][x] == c]
        return min(xs) if xs else 99
    return sorted(blocks, key=lx)


def set_tag(b, t):
    b['tags'] = b['tags'][:-1] + [t]


def update_tags(world, beam, blocks):
    hook = set(hooked(blocks, beam))
    order = legend_order(world, blocks)
    for n in order:
        b = blocks[n]
        if b['tags'][-1] == 'collected' and not on_beam(b, beam):
            for m in order:
                if blocks[m]['tags'][-1] == 'next':
                    set_tag(blocks[m], 'todo')
            set_tag(b, 'next')
            return
    onb = [n for n, b in blocks.items() if on_beam(b, beam)]
    for n in hook:
        if blocks[n]['tags'][-1] == 'next' and onb == [n]:
            set_tag(blocks[n], 'collected')
            for m in order:
                if blocks[m]['tags'][-1] == 'todo':
                    set_tag(blocks[m], 'next')
                    break
            return


def beam_pixels(w, h):
    return [[2 if (i + j) % 3 == 1 else 1 for i in range(w)] for j in range(h)]


def paint(out, o, pix):
    for j, row in enumerate(pix):
        for i, v in enumerate(row):
            x, y = o['x'] + i, o['y'] + j
            if 0 <= x < 64 and 0 <= y < 64 and v >= 0:
                out[y][x] = v


def render(world, frame, before, ring, beam, blocks, n):
    out = [list(r) for r in frame]
    for o in before:
        for j in range(o['h']):
            for i in range(o['w']):
                x, y = o['x'] + i, o['y'] + j
                if 0 <= x < 64 and 0 <= y < world.hud:
                    out[y][x] = world.bg(x, y, ring)
    paint(out, ring, ring['pixels'])
    paint(out, beam, beam_pixels(beam['w'], beam['h']))
    for b in blocks.values():
        paint(out, b, [[int(b['tags'][1])] * b['w'] for _ in range(b['h'])])
    for b in blocks.values():
        c = int(b['tags'][1])
        pts = [(x, y) for y in range(world.hud + 1, 64) for x in range(64) if frame[y][x] == c]
        if not pts:
            continue
        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
        v = 0 if b['tags'][-1] == 'collected' else c
        for dy in (1, 2):
            for dx in (1, 2):
                out[y0 + dy][x0 + dx] = v
    k = (n - 1) // 3 if n > 0 else 0
    for x in range(64):
        out[world.hud][x] = 3 if x >= 64 - k else 2
    return out


def count_bar(world, frame):
    return sum(1 for v in frame[world.hud] if v == 3)


def transition_function(state, action, frame):
    world = World(frame)
    key = key_of(frame)
    if key in MEMO:
        n = MEMO[key]
    else:
        b = count_bar(world, frame)
        n = 3 * b + 1 if b else 0
    if isinstance(action, dict):
        MEMO[key] = n
        return [list(r) for r in frame]
    objs = [dict(o) for o in state]
    ring = next(o for o in objs if o['type'] == 'player')
    beam = next(o for o in objs if o['type'] == 'arm')
    blocks = {o['name']: o for o in objs if o['type'] == 'block'}
    for b in blocks.values():
        b['tags'] = list(b['tags'])
    n += 1
    if action in (1, 2):
        step_vertical(ring, beam, blocks, -6 if action == 1 else 6, world)
    elif action in (3, 4):
        step_horizontal(beam, blocks, 6 if action == 4 else -6, world)
    update_tags(world, beam, blocks)
    out = render(world, frame, state, ring, beam, blocks, n)
    MEMO[key_of(out)] = n
    return out
