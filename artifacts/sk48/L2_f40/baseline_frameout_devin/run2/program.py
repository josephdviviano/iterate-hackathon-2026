# Mechanics: crane ring on a rail moves y-/+6 (A1/A2) with its beam (h=2, x=ring.x+5); A4/A3 grow/shrink the beam by 6.
# Blocks overlapping the beam ride vertically; blocks hit by new ring/beam rects are pushed (chain); any block onto a
# colour-5 (off-field) cell or the ring cancels the move. Hooked (next/collected block at tip-4) drags on-beam blocks.
# Tags: hooked next alone on beam -> collected (HUD icon centre 0), rightmost todo -> next; unhooked collected reverts.
# HUD row 53 gains one 3 from the right every 3 non-click actions; phase hidden (continuity-gated; unconfirmed: A5/A7, full bar).
import copy

BG_OUT = 5
MEM = {}


def rect(o):
    return (o['x'], o['y'], o['w'], o['h'])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shifted(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def covered(state):
    s = set()
    for o in state:
        for y in range(o['y'], o['y'] + o['h']):
            for x in range(o['x'], o['x'] + o['w']):
                s.add((x, y))
    return s


def field_rows(frame):
    rows = [y for y in range(64) if any(v != BG_OUT for v in frame[y])]
    hud = next(y for y in rows if all(v != BG_OUT for v in frame[y]))
    play = [y for y in rows if y < hud]
    return hud, min(play), max(play)


def background(frame, state):
    """Frame with objects removed. The board is mirror-symmetric top/bottom about the field centre, and the
    rail/field textures repeat every 6 rows: a covered cell copies its mirror cell, else the nearest uncovered
    cell 6 rows away (above unless that is off-field)."""
    cov = covered(state)
    hud, top, bot = field_rows(frame)
    bg = [row[:] for row in frame]
    for (x, y) in cov:
        if not (0 <= x < 64 and 0 <= y < 64):
            continue
        m = top + bot - y
        if y < hud and 0 <= m < hud and (x, m) not in cov:
            bg[y][x] = frame[m][x]
            continue
        cands = []
        for d in (-6, 6):
            yy = y + d
            while 0 <= yy < 64 and (x, yy) in cov:
                yy += d
            cands.append(frame[yy][x] if 0 <= yy < hud else None)
        up, dn = cands
        bg[y][x] = dn if up is None or (up == BG_OUT and dn is not None) else up
    return bg


def off_field(bg, r):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            if not (0 <= x < 64 and 0 <= y < 64) or bg[y][x] == BG_OUT:
                return True
    return False


def push_chain(blocks, movers, solids, dx, dy):
    """Move movers by (dx,dy), pulling in every block hit by a solid or a moved block. Returns moved set or None."""
    moving = set(movers)
    changed = True
    while changed:
        changed = False
        hitters = list(solids) + [shifted(rect(blocks[i]), dx, dy) for i in moving]
        for i, b in enumerate(blocks):
            if i not in moving and any(overlap(rect(b), h) for h in hitters):
                moving.add(i)
                changed = True
    return moving


def chain_ok(bg, blocks, moving, dx, dy, ring_r):
    for i in moving:
        r = shifted(rect(blocks[i]), dx, dy)
        if off_field(bg, r) or overlap(r, ring_r):
            return False
    return True


def beam_rect(beam):
    return (beam['x'], beam['y'], beam['w'], beam['h'])


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def status(b):
    for t in ('todo', 'next', 'collected'):
        if t in b['tags']:
            return t
    return None


def set_status(b, s):
    b['tags'] = [s if t in ('todo', 'next', 'collected') else t for t in b['tags']]


def on_beam(blocks, beam):
    br = beam_rect(beam)
    return [i for i, b in enumerate(blocks) if overlap(rect(b), br)]


def hooked(blocks, beam):
    tip = beam['x'] + beam['w'] - 1
    for i in on_beam(blocks, beam):
        b = blocks[i]
        if status(b) in ('next', 'collected') and b['x'] == tip - 4:
            return i
    return None


def move_vertical(bg, ring, beam, blocks, dy):
    nring = shifted(rect(ring), 0, dy)
    nbeam = shifted(beam_rect(beam), 0, dy)
    if off_field(bg, (nbeam[0] + 1, nbeam[1], 1, nbeam[3])):
        return
    carried = on_beam(blocks, beam)
    moving = push_chain(blocks, carried, [nring, nbeam], 0, dy)
    if not chain_ok(bg, blocks, moving, 0, dy, (-99, -99, 0, 0)):
        return
    ring['y'] += dy
    beam['y'] += dy
    for i in moving:
        blocks[i]['y'] += dy


def move_horizontal(bg, ring, beam, blocks, dw):
    nw = beam['w'] + dw
    if nw < 1:
        return
    if dw > 0 and off_field(bg, (beam['x'] + beam['w'], beam['y'], dw, beam['h'])):
        return
    h = hooked(blocks, beam)
    if h is not None:
        moving = set(on_beam(blocks, beam))
    elif dw > 0:
        swept = (beam['x'] + beam['w'], beam['y'], dw, beam['h'])
        moving = push_chain(blocks, [], [swept], dw, 0)
    else:
        moving = set()
    if moving and chain_ok(bg, blocks, moving, dw, 0, rect(ring)):
        for i in moving:
            blocks[i]['x'] += dw
    beam['w'] = nw
    beam['pixels'] = beam_pixels(nw)


def update_tags(blocks, beam):
    h = hooked(blocks, beam)
    ob = on_beam(blocks, beam)
    for i, b in enumerate(blocks):
        if status(b) == 'collected' and i != h:
            nxt = [j for j, c in enumerate(blocks) if status(c) == 'next']
            for j in nxt:
                set_status(blocks[j], 'todo')
            set_status(b, 'next')
    if h is not None and status(blocks[h]) == 'next' and \
            all(j == h or status(blocks[j]) == 'collected' for j in ob):
        set_status(blocks[h], 'collected')
        todo = [j for j, c in enumerate(blocks) if status(c) == 'todo']
        if todo:
            j = max(todo, key=lambda k: blocks[k]['x'])
            set_status(blocks[j], 'next')


def step_objects(state, action, bg):
    st = copy.deepcopy(state)
    ring = next(o for o in st if o['type'] == 'player')
    beam = next(o for o in st if o['type'] == 'arm')
    blocks = [o for o in st if o['type'] == 'block']
    if action == 1:
        move_vertical(bg, ring, beam, blocks, -6)
    elif action == 2:
        move_vertical(bg, ring, beam, blocks, 6)
    elif action == 3:
        move_horizontal(bg, ring, beam, blocks, -6)
    elif action == 4:
        move_horizontal(bg, ring, beam, blocks, 6)
    update_tags(blocks, beam)
    return st


def draw(frame, objs):
    for o in sorted(objs, key=lambda o: o['layer']):
        if o['type'] == 'arm':
            o = dict(o, pixels=beam_pixels(o['w']))
        for j, row in enumerate(o['pixels']):
            for i, v in enumerate(row):
                x, y = o['x'] + i, o['y'] + j
                if 0 <= x < 64 and 0 <= y < 64 and v >= 0:
                    frame[y][x] = v


def block_colour(b):
    return b['pixels'][0][0]


def draw_hud_icons(frame, blocks, hud):
    """Each block has a HUD icon of its colour below the HUD row; a collected block's icon gets a 0 centre."""
    for b in blocks:
        c = block_colour(b)
        cells = [(x, y) for y in range(hud + 1, 64) for x in range(64) if frame[y][x] == c]
        if not cells:
            continue
        x0, x1 = min(p[0] for p in cells), max(p[0] for p in cells)
        y0, y1 = min(p[1] for p in cells), max(p[1] for p in cells)
        fill = 0 if status(b) == 'collected' else c
        for y in range(y0 + 1, y1):
            for x in range(x0 + 1, x1):
                frame[y][x] = fill


def hud_count(frame, hud):
    return sum(1 for v in frame[hud] if v == 3)


def draw_hud_bar(frame, n, hud):
    k = max(0, (n - 1) // 3)
    for i in range(64):
        frame[hud][63 - i] = 3 if i < k else 2


def transition_function(state, action, frame):
    hud = field_rows(frame)[0]
    if MEM.get('frame') == frame:
        n = MEM['n']
    else:
        z = hud_count(frame, hud)
        n = 0 if z == 0 else 3 * z + 1
    is_click = isinstance(action, dict)
    if not is_click:
        n += 1
    bg = background(frame, state)
    after = step_objects(state, action, bg)
    out = [row[:] for row in bg]
    draw(out, after)
    draw_hud_icons(out, [o for o in after if o['type'] == 'block'], hud)
    if not is_click:
        draw_hud_bar(out, n, hud)
    MEM['frame'] = [row[:] for row in out]
    MEM['n'] = n
    return out
