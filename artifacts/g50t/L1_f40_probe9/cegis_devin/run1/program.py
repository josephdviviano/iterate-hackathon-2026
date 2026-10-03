# Mechanics: player 5x5 moves 6px (A1-4); blocked by wall px in the dest box (except the plate), y<8, x<14, board edge.
# First input of a fresh level (counter w=64) is a no-op. Counter loses 1px on odd calls (parity hidden; fallback n odd).
# Plate covered by player/ghost -> wall px left of the line shift +6 toward it; uncovered -> canonical wall restored.
# A5 (player off spawn): toggle ghost mode (red hud at (1,1), other huds x+4), player -> spawn, trail=[start, spawn].
# Ghost = player trail 2 moves behind, hidden under the player (step-43 fix). Unconfirmed: x>=14 bound, START/SPAWN defaults.
import copy

STEP, TOP, LEFT, HI = 6, 8, 14, 57
DEFAULT_START, DEFAULT_SPAWN = (20, 8), (14, 8)
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
RED = 2
_mem = {}


def canon(s):
    return sorted(repr(sorted(o.items())) for o in s)


def idx(o):
    try:
        return int(o['name'].rsplit('_', 1)[1])
    except Exception:
        return 999


def wall_px(o):
    return {(o['x'] + i, o['y'] + j) for j, row in enumerate(o['pixels'])
            for i, v in enumerate(row) if v >= 0}


def line_x(px):
    cols = {}
    for x, _ in px:
        cols[x] = cols.get(x, 0) + 1
    return max(cols, key=lambda c: (cols[c], c))


def find_plate(px, bodies):
    for (x, y) in sorted(px, key=lambda q: (q[1], q[0])):
        if all((x + a, y + b) in px for a in (-1, 0, 1) for b in (-1, 0, 1)):
            return (x, y)
    lx = line_x(px)
    top = min(y for x, y in px if x == lx)
    for bx, by in bodies:
        if bx <= lx < bx + 5 and by + 5 == top:
            return (bx + 2, by + 2)
    return None


def press(px, lx):
    return {(x, y) for x, y in px if x >= lx} | \
           {(x + STEP, y) for x, y in px if x < lx and x + STEP < lx}


def unpress(px, lx):
    out = {(x, y) for x, y in px if x >= lx} | {(x - STEP, y) for x, y in px if x < lx}
    for y in {y for x, y in px if x == lx - 1}:
        out |= {(x, y) for x in range(lx - STEP, lx)}
    return out


def restore(px, plate, bodies):
    px = set(px)
    if plate:
        cx, cy = plate
        px |= {(cx + a, cy + b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
        lx = line_x(px)
        top = min([y for x, y in px if x == lx and y > cy + 1] or [cy + 2])
        px |= {(lx, y) for y in range(cy + 2, top)}
    return px


def covers(body, pt):
    return body[0] <= pt[0] < body[0] + 5 and body[1] <= pt[1] < body[1] + 5


def stateless(state):
    objs = sorted(state, key=idx)
    player = next(o for o in objs if o['type'] == 'player')
    ghost = next((o for o in objs if o['type'] == 'ghost'), None)
    wall = next(o for o in objs if o['type'] == 'wall')
    counter = next(o for o in objs if o['type'] == 'counter')
    red = [o for o in objs if o['type'] == 'hud' and o['pixels'] and o['pixels'][0][0] == RED]
    p = (player['x'], player['y'])
    bodies = [p] + ([(ghost['x'], ghost['y'])] if ghost else [])
    px = wall_px(wall)
    plate = find_plate(px, bodies)
    px = restore(px, plate, bodies)
    if plate and any(covers(b, plate) for b in bodies):
        px = unpress(px, line_x(px))
    w = counter['w']
    n = 0 if w >= 64 else 2 * (64 - w) - 1
    spawn = DEFAULT_SPAWN
    trail = [DEFAULT_START, spawn] if p == spawn else [p, p]
    if ghost:
        g = (ghost['x'], ghost['y'])
        mid = ((g[0] + p[0]) // 2, (g[1] + p[1]) // 2)
        trail = [g, mid, p]
    return {'n': n, 'start': DEFAULT_START, 'spawn': spawn, 'trail': trail,
            'ghost_mode': bool(red), 'wall': px, 'plate': plate, 'p': p}


def blocked(m, dest):
    x, y = dest
    if x < LEFT or y < TOP or x > HI or y > HI:
        return True
    cur = cur_wall(m)
    for (a, b) in cur:
        if x <= a < x + 5 and y <= b < y + 5:
            if m['plate'] and covers(dest, m['plate']):
                return False
            return True
    return False


def ghost_pos(m):
    if m['ghost_mode'] and len(m['trail']) >= 3:
        return m['trail'][-3]
    return None


def cur_wall(m):
    bodies = [m['p']] + ([ghost_pos(m)] if ghost_pos(m) else [])
    if m['plate'] and any(covers(b, m['plate']) for b in bodies):
        return press(m['wall'], line_x(m['wall']))
    return m['wall']


def step(m, action):
    first = m['n'] == 0
    m['n'] += 1
    aid = action['action_id'] if isinstance(action, dict) else action
    if first:
        return
    if aid in DIRS:
        dx, dy = DIRS[aid]
        dest = (m['p'][0] + dx * STEP, m['p'][1] + dy * STEP)
        if not blocked(m, dest):
            m['p'] = dest
            m['trail'].append(dest)
            if m['spawn'] is None:
                m['spawn'] = dest
    elif aid == 5:
        spawn = m['spawn'] or DEFAULT_SPAWN
        if m['p'] != spawn:
            m['ghost_mode'] = not m['ghost_mode']
            m['p'] = spawn
            m['trail'] = [m['start'], spawn]


def render(state, m):
    base = [o for o in sorted(state, key=idx)
            if o['type'] not in ('ghost', 'wall')
            and not (o['type'] == 'hud' and o['pixels'] and o['pixels'][0][0] == RED)]
    tmpl = {t: next((o for o in state if o['type'] == t), None) for t in ('ghost', 'wall', 'player', 'hud')}
    out = []
    for o in base:
        o = copy.deepcopy(o)
        if o['type'] == 'hud':
            o['x'] = 5 if m['ghost_mode'] else 1
        if o['type'] == 'player':
            o['x'], o['y'] = m['p']
        if o['type'] == 'counter':
            o['w'] = max(0, 64 - (m['n'] + 1) // 2)
            o['pixels'] = [[o['pixels'][0][0]] * o['w']] if o['w'] else []
        out.append(o)
    if m['ghost_mode']:
        ring = copy.deepcopy(tmpl['hud'])
        ring.update(x=1, y=1, w=3, h=3, pixels=[[RED] * 3, [RED, -1, RED], [RED] * 3])
        out.append(ring)
    g = ghost_pos(m)
    bodies = [m['p']]
    if g and g != m['p']:
        gh = copy.deepcopy(tmpl['ghost']) if tmpl['ghost'] else copy.deepcopy(tmpl['player'])
        pix = [[RED] * 5 for _ in range(5)]
        pix[2][2] = -1
        gh.update(type='ghost', tags=['ghost', 'snake_tail', 'follower'], x=g[0], y=g[1],
                  w=5, h=5, layer=1, visible=True, pixels=pix)
        out.append(gh)
        bodies.append(g)
    px = {q for q in cur_wall(m) if not any(covers(b, q) for b in bodies)}
    if px:
        wall = copy.deepcopy(tmpl['wall'])
        x0, y0 = min(a for a, _ in px), min(b for _, b in px)
        x1, y1 = max(a for a, _ in px), max(b for _, b in px)
        col = next(v for row in tmpl['wall']['pixels'] for v in row if v >= 0)
        wall.update(x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1,
                    pixels=[[col if (x, y) in px else -1 for x in range(x0, x1 + 1)]
                            for y in range(y0, y1 + 1)])
        out.append(wall)
    for i, o in enumerate(out):
        o['name'] = '%s_%03d' % (o['type'], i)
    return out


def transition_function(state, action):
    global _mem
    if _mem.get('last') is not None and canon(state) == _mem['last']:
        m = _mem['model']
    else:
        m = stateless(state)
    step(m, action)
    out = render(state, m)
    _mem = {'last': canon(out), 'model': m}
    return out
