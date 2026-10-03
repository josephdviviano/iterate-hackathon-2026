# Crane/skewer game. ring (player) + beam (arm, h=2, from ring.x+5) + 4x4 blocks tagged todo/next/collected.
# A1/A2: ring+beam y -/+6, carrying blocks on the beam rows and pushing blocks hit (chain); cancel if any block y outside [2,58].
# A4/A3: beam w +/-6 in [1,43]. Hooked (next/collected block at x == beam.x+w-5) drags all on-beam blocks; unhooked extend pushes
# blocks in swept cols; a blocked group (x>48 or into the ring) leaves blocks put and the beam changes alone. Clicks/A5/A7 no-op.
# Tags: hooked next with every other on-beam block collected -> collected (rightmost todo -> next); collected off beam -> next. Order of collection beyond rightmost-todo unconfirmed.
import copy

STEP = 6
MAX_W = 43


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o['x'], o['y'], o['w'], o['h'])


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def on_beam(block, beam):
    return overlap(rect(block), rect(beam))


def is_hooked(block, beam):
    return block['tags'][-1] in ('next', 'collected') and on_beam(block, beam) and block['x'] == beam['x'] + beam['w'] - 5


def push_chain(blocks, movers, solids, dx, dy):
    """Move `movers` by (dx,dy); blocks overlapping moved solids/movers get pushed too. Returns moved set (ids)."""
    moved = set(id(b) for b in movers)
    for b in movers:
        b['x'] += dx
        b['y'] += dy
    frontier = list(solids) + [rect(b) for b in movers]
    while frontier:
        r = frontier.pop()
        for b in blocks:
            if id(b) not in moved and overlap(r, rect(b)):
                moved.add(id(b))
                b['x'] += dx
                b['y'] += dy
                frontier.append(rect(b))
    return moved


def block_ok(b, ring):
    return 2 <= b['y'] <= 58 and b['x'] <= 48 and b['x'] >= ring['x'] + ring['w']


def update_tags(blocks, beam):
    def tag(b):
        return b['tags'][-1]
    for b in blocks:
        if tag(b) == 'next' and is_hooked(b, beam):
            others = [o for o in blocks if o is not b and on_beam(o, beam)]
            if all(tag(o) == 'collected' for o in others):
                b['tags'][-1] = 'collected'
                todos = [o for o in blocks if tag(o) == 'todo']
                if todos:
                    max(todos, key=lambda o: (o['x'], -o['y']))['tags'][-1] = 'next'
                return
    off = [b for b in blocks if tag(b) == 'collected' and not on_beam(b, beam)]
    if off:
        for b in blocks:
            if tag(b) == 'next':
                b['tags'][-1] = 'todo'
        max(off, key=lambda o: o['x'])['tags'][-1] = 'next'


def vertical(ring, beam, blocks, dy):
    if not (0 <= ring['y'] + dy <= 64 - ring['h']):
        return False
    carried = [b for b in blocks if on_beam(b, beam)]
    ring['y'] += dy
    beam['y'] += dy
    push_chain(blocks, carried, [rect(ring), rect(beam)], 0, dy)
    return all(block_ok(b, ring) for b in blocks)


def horizontal(ring, beam, blocks, dw):
    nw = beam['w'] + dw
    if nw < 1 or nw > MAX_W:
        return False
    hooked = any(is_hooked(b, beam) for b in blocks)
    trial = copy.deepcopy(blocks)
    if hooked:
        push_chain(trial, [b for b in trial if on_beam(b, beam)], [], dw, 0)
    elif dw > 0:
        swept = (beam['x'] + beam['w'], beam['y'], dw, beam['h'])
        push_chain(trial, [], [swept], dw, 0)
    if all(block_ok(b, ring) for b in trial):
        for b, t in zip(blocks, trial):
            b['x'] = t['x']
    beam['w'] = nw
    beam['pixels'] = beam_pixels(nw)
    return True


def transition_function(state, action):
    new = copy.deepcopy(state)
    ring = next((o for o in new if o['type'] == 'player'), None)
    beam = next((o for o in new if o['type'] == 'arm'), None)
    blocks = [o for o in new if o['type'] == 'block']
    if ring is None or beam is None or not isinstance(action, int):
        return new
    if action in (1, 2):
        ok = vertical(ring, beam, blocks, -STEP if action == 1 else STEP)
        if not ok:
            return copy.deepcopy(state)
    elif action in (3, 4):
        if not horizontal(ring, beam, blocks, STEP if action == 4 else -STEP):
            return new
    else:
        return new
    update_tags(blocks, beam)
    return new
