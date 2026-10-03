# Crane/skewer: ring (player) moves y-+6 on A1/A2 with the beam (arm); A4/A3 grow/shrink beam w by 6.
# Blocks on the beam are carried vertically; blocks hit by moved ring/beam/blocks are pushed (chains).
# Beam end hooks ANY block at x == tip-5 (tag-agnostic) and drags all on-beam blocks; else extension
# pushes any block in the swept strip. Out-of-field group -> beam moves alone (vertical: cancel all).
# HUD row: 3s from right = (n-1)//3, n = non-click actions (continuity via own outputs); legend centre 0 = collected.
import copy

HIST = {}


def world(frame):
    hud = next(y for y in range(64) if not any(v in (4, 5) for v in frame[y]))
    cells = [(x, y) for y in range(hud) for x in range(64) if frame[y][x] == 4]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return {'hud': hud, 'L': min(xs), 'R': max(xs), 'T': min(ys), 'B': max(ys)}


def rail_cols(frame, W, covered):
    cols = set()
    for y in range(W['hud']):
        for x in range(64):
            if (x, y) in covered or W['L'] <= x <= W['R']:
                continue
            if frame[y][x] in (2, 3):
                cols.add(x)
    return cols


def bg(W, rails, x, y):
    if W['L'] <= x <= W['R'] and W['T'] <= y <= W['B']:
        return 4
    if x in rails and W['T'] + 2 <= y <= W['B'] - 2:
        return 2 if (y - W['T'] - 2) % 6 < 2 else 3
    return 5


def rect(o):
    return (o['x'], o['y'], o['w'], o['h'])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def in_field(W, o):
    return W['L'] <= o['x'] and o['x'] + o['w'] - 1 <= W['R'] and W['T'] <= o['y'] and o['y'] + o['h'] - 1 <= W['B']


def on_rows(beam, b):
    return b['y'] < beam['y'] + beam['h'] and beam['y'] < b['y'] + b['h']


def fully_on(beam, b):
    return on_rows(beam, b) and b['x'] >= beam['x'] and b['x'] + b['w'] <= beam['x'] + beam['w']


def hooked(beam, blocks):
    tip = beam['x'] + beam['w'] - 5
    return [b for b in blocks if b['x'] == tip and on_rows(beam, b)]


def push_chain(blocks, movers, extra, dx, dy):
    """Move movers by (dx,dy); blocks overlapping moved rects or extra rects join. Returns moved set."""
    moved = set()
    frontier = list(movers)
    solids = list(extra)
    while True:
        for i in frontier:
            if i in moved:
                continue
            moved.add(i)
            blocks[i]['x'] += dx
            blocks[i]['y'] += dy
            solids.append(rect(blocks[i]))
        frontier = [i for i, b in enumerate(blocks) if i not in moved and any(overlap(rect(b), s) for s in solids)]
        if not frontier:
            return moved


def tag(b):
    return b['tags'][-1]


def set_tag(b, t):
    b['tags'] = b['tags'][:-1] + [t]


def update_tags(beam, blocks, order):
    for b in blocks:
        if tag(b) == 'next' and b in hooked(beam, blocks):
            others = [o for o in blocks if o is not b and fully_on(beam, o)]
            if all(tag(o) == 'collected' for o in others):
                set_tag(b, 'collected')
                for c in order:
                    bb = blocks_by_colour(blocks, c)
                    if bb is not None and tag(bb) == 'todo':
                        set_tag(bb, 'next')
                        break
                return
    off = [c for c in order if blocks_by_colour(blocks, c) is not None
           and tag(blocks_by_colour(blocks, c)) == 'collected'
           and not fully_on(beam, blocks_by_colour(blocks, c))]
    if off:
        for b in blocks:
            if tag(b) == 'next':
                set_tag(b, 'todo')
        set_tag(blocks_by_colour(blocks, off[-1]), 'next')


def blocks_by_colour(blocks, c):
    for b in blocks:
        if b['pixels'][0][0] == c:
            return b
    return None


def legend(frame, W, colours):
    pos = {}
    for c in colours:
        cs = [(x, y) for y in range(W['hud'] + 1, 64) for x in range(64) if frame[y][x] == c]
        if cs:
            pos[c] = (min(p[0] for p in cs), min(p[1] for p in cs))
    return pos


def step(state, action, W):
    S = copy.deepcopy(state)
    ring = next(o for o in S if o['type'] == 'player')
    beam = next(o for o in S if o['type'] == 'arm')
    blocks = [o for o in S if o['type'] == 'block']
    if action in (1, 2):
        d = -6 if action == 1 else 6
        if not (W['T'] <= ring['y'] + d and ring['y'] + d + ring['h'] - 1 <= W['B']):
            return state, False
        trial = copy.deepcopy(blocks)
        carried = [i for i, b in enumerate(trial) if overlap(rect(beam), rect(b))]
        nb = dict(beam, y=beam['y'] + d)
        nr = dict(ring, y=ring['y'] + d)
        moved = push_chain(trial, carried, [rect(nb), rect(nr)], 0, d)
        if any(not in_field(W, trial[i]) for i in moved):
            return state, False
        ring['y'] += d
        beam['y'] += d
        for b, t in zip(blocks, trial):
            b['y'] = t['y']
            b['x'] = t['x']
        return S, True
    if action in (3, 4):
        dw = 6 if action == 4 else -6
        nw = beam['w'] + dw
        if nw < 1 or beam['x'] + nw - 1 > W['R']:
            return state, False
        trial = copy.deepcopy(blocks)
        hk = hooked(beam, trial)
        if hk:
            movers = [i for i, b in enumerate(trial) if fully_on(beam, b)]
            moved = push_chain(trial, movers, [], dw, 0)
        elif dw > 0:
            strip = (beam['x'] + beam['w'], beam['y'], dw, beam['h'])
            movers = [i for i, b in enumerate(trial) if overlap(strip, rect(b))]
            moved = push_chain(trial, movers, [], dw, 0) if movers else set()
        else:
            moved = set()
        ok = all(in_field(W, trial[i]) for i in moved)
        beam['w'] = nw
        if ok:
            for b, t in zip(blocks, trial):
                b['x'] = t['x']
        return S, True
    return state, False


def beam_pixels(w, h):
    return [[2 if (i + j) % 3 == 1 else 1 for i in range(w)] for j in range(h)]


def transition_function(state, action, frame):
    W = world(frame)
    out = [list(map(int, r)) for r in frame]
    covered = set()
    for o in state:
        for j in range(o['h']):
            for i in range(o['w']):
                covered.add((o['x'] + i, o['y'] + j))
    rails = rail_cols(frame, W, covered)
    blocks0 = [o for o in state if o['type'] == 'block']
    colours = [b['pixels'][0][0] for b in blocks0]
    leg = legend(frame, W, colours)
    order = sorted(leg, key=lambda c: leg[c][0])
    click = isinstance(action, dict)
    new = state
    if not click:
        new, changed = step(state, action, W)
        new = copy.deepcopy(new)
        nb = next(o for o in new if o['type'] == 'arm')
        update_tags(nb, [o for o in new if o['type'] == 'block'], order)
    # erase old objects, draw new by layer
    for (x, y) in covered:
        if 0 <= x < 64 and 0 <= y < W['hud']:
            out[y][x] = bg(W, rails, x, y)
    for o in sorted(new, key=lambda o: o['layer']):
        px = beam_pixels(o['w'], o['h']) if o['type'] == 'arm' else o['pixels']
        for j in range(o['h']):
            for i in range(o['w']):
                x, y = o['x'] + i, o['y'] + j
                v = px[j][i]
                if v >= 0 and 0 <= x < 64 and 0 <= y < W['hud']:
                    out[y][x] = v
    # legend centres
    for b in new:
        if b['type'] != 'block':
            continue
        c = b['pixels'][0][0]
        if c in leg:
            lx, ly = leg[c]
            v = 0 if tag(b) == 'collected' else c
            for j in (1, 2):
                for i in (1, 2):
                    out[ly + j][lx + i] = v
    # HUD action bar
    hy = W['hud']
    key = tuple(tuple(r) for r in frame)
    if key in HIST:
        n = HIST[key]
    else:
        threes = sum(1 for v in frame[hy] if v == 3)
        n = 3 * threes + 1 if threes else 0
    if not click:
        n += 1
    k = max(0, (n - 1) // 3)
    for x in range(64):
        out[hy][x] = 3 if x >= 64 - k else 2
    HIST[tuple(tuple(r) for r in out)] = n
    if len(HIST) > 16:
        HIST.pop(next(iter(HIST)))
    return out

