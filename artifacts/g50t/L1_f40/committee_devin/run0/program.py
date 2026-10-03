# Mechanics: player (5x5) moves 6 cells per A1-A4 unless the box leaves the field (y<8), hits a wall pixel or an
# invisible blocked cell; the wall's 3x3 handle cell is walkable and a body on it slides the wall's key end +6 right.
# A5 (only after a move since the last reset): no ghost -> player history becomes a ghost replay, HUD gains a colour-2
# slot, player -> spawn; ghost present -> clear ghost/slot, player -> spawn. Ghost steps along the recorded path once
# per successful player move. Counter w = 64-(n+1)//2. Unconfirmed: (20,14) invisible block (t0), ghost collisions.
import copy, json

STEP, BOX, TOP, SPAWN, SLOT_DX = 6, 5, 8, (14, 8), 4
INVISIBLE_BLOCKED = {(20, 14)}
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
_mem = {}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells(o):
    return {(o['x'] + i, o['y'] + j) for j, r in enumerate(o['pixels']) for i, c in enumerate(r) if c >= 0}


def box(p):
    return {(p[0] + i, p[1] + j) for i in range(BOX) for j in range(BOX)}


def is_slot(o):
    return o['type'] == 'hud' and any(c == 2 for r in o['pixels'] for c in r)


# ---- wall model: base (unpressed) pixel set + column x, bar row, key x, handle cell ----
def wall_geometry(base):
    xs, ys = {}, {}
    for x, y in base:
        xs[x] = xs.get(x, 0) + 1
        ys[y] = ys.get(y, 0) + 1
    colx = max(xs, key=lambda k: (xs[k], k))
    bary = max(ys, key=lambda k: (ys[k], k))
    keyx = min(x for x, y in base if y == bary)
    htop = min(y for x, y in base if x == colx)
    return colx, bary, keyx, (colx - 2, htop - 1)


def wall_cells(base, pressed):
    if not pressed:
        return set(base)
    colx, bary, keyx, _ = wall_geometry(base)
    return ({(x + STEP, y) for x, y in base if x < keyx + BOX} |
            {(x, y) for x, y in base if x >= keyx + STEP})


def parse_wall(wall, bodies):
    vis = cells(wall)
    colx, bary, keyx, _ = wall_geometry(vis)
    top = min(y for x, y in vis if x == colx)
    for b in bodies:
        if b[0] <= colx < b[0] + BOX and b[1] < top <= b[1] + BOX:   # body standing on the handle
            cx, cy = b[0] + 2, b[1] + 2
            hidden = {(cx + i, cy + j) for i in (-1, 0, 1) for j in (-1, 0, 1)}
            hidden |= {(colx, y) for y in range(cy + 2, top)}
            vis |= hidden
            base = {(x - STEP, y) for x, y in vis if x < keyx + BOX} | {(x, y) for x, y in vis if x >= keyx}
            base |= {(x, bary) for x in range(keyx - STEP, colx + 1)}
            return base
    return vis


def pressed_by(base, bodies):
    return any(tuple(b) == wall_geometry(base)[3] for b in bodies)


# ---- player rule ----
def try_move(pos, action, wall, handle):
    dx, dy = MOVES[action]
    np = (pos[0] + dx, pos[1] + dy)
    if np[1] < TOP or np[0] < 0 or np[0] + BOX > 64 or np[1] + BOX > 63 or np in INVISIBLE_BLOCKED:
        return pos
    if np != handle and box(np) & wall:
        return pos
    return np


def parse(state):
    get = lambda t: [o for o in state if o['type'] == t]
    player = get('player')[0]
    ghosts = get('ghost')
    m = {'player': (player['x'], player['y']), 'mode': any(is_slot(o) for o in state)}
    m['ghost'] = (ghosts[0]['x'], ghosts[0]['y']) if ghosts else (SPAWN if m['mode'] else None)
    m['path'] = [m['ghost']] if m['ghost'] else []
    m['k'] = 1 if ghosts else 0
    bodies = [m['player']] + ([m['ghost']] if ghosts else [])
    m['base'] = parse_wall(get('wall')[0], bodies)
    w = get('counter')[0]['w']
    m['n'] = 2 * (64 - w)
    m['record'] = [m['player']] if m['player'] == SPAWN else [SPAWN, m['player']]
    m['moved'] = m['player'] != SPAWN or m['k'] > 0
    return m


def step(m, action):
    m['n'] += 1
    aid = action['action_id'] if isinstance(action, dict) else action
    bodies = [m['player']] + ([m['ghost']] if m['ghost'] else [])
    wall = wall_cells(m['base'], pressed_by(m['base'], bodies))
    handle = wall_geometry(m['base'])[3]
    if aid in MOVES:
        np = try_move(m['player'], aid, wall, handle)
        if np != m['player']:
            m['player'] = np
            m['moved'] = True
            if not m['mode']:
                m['record'].append(np)
            elif m['k'] + 1 < len(m['path']):
                m['k'] += 1
                m['ghost'] = m['path'][m['k']]
    elif aid == 5 and m['moved']:
        if not m['mode']:
            m['mode'], m['path'], m['k'] = True, list(m['record']), 0
            m['ghost'] = m['path'][0]
        else:
            m['mode'], m['ghost'], m['path'], m['k'] = False, None, [], 0
        m['player'], m['moved'], m['record'] = SPAWN, False, [SPAWN]
    return m


def crop(o, solid, covers):
    vis = {p: c for p, c in solid.items() if p not in covers}
    if not vis:
        return None
    x0, y0 = min(p[0] for p in vis), min(p[1] for p in vis)
    x1, y1 = max(p[0] for p in vis), max(p[1] for p in vis)
    o.update(x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1,
             pixels=[[vis.get((x, y), -1) for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)])
    return o


def render(state, m):
    order = sorted(state, key=lambda o: int(o['name'].rsplit('_', 1)[1]))
    base = [copy.deepcopy(o) for o in order if o['type'] not in ('wall', 'ghost') and not is_slot(o)]
    player = next(o for o in state if o['type'] == 'player')
    wall_t = next(o for o in state if o['type'] == 'wall')
    pbox = box(m['player'])
    out = []
    for o in base:
        if o['type'] == 'hud':
            o['x'] = 1 + (SLOT_DX if m['mode'] else 0) + (o['x'] - 1) % SLOT_DX
        elif o['type'] == 'player':
            o['x'], o['y'] = m['player']
        elif o['type'] == 'counter':
            o['w'] = 64 - (m['n'] + 1) // 2
            o['pixels'] = [[9] * o['w']]
        out.append(o)
    if m['mode']:
        hud = next(o for o in state if o['type'] == 'hud')
        out.append(dict(copy.deepcopy(hud), x=1, y=1, w=3, h=3, pixels=[[2, 2, 2], [2, -1, 2], [2, 2, 2]]))
    covers = set(pbox)
    if m['ghost']:
        gpix = {(m['ghost'][0] + i, m['ghost'][1] + j): (-1 if i == j == 2 else 2) for i in range(BOX) for j in range(BOX)}
        gpix = {p: c for p, c in gpix.items() if c >= 0}
        g = dict(copy.deepcopy(player), type='ghost', tags=['ghost', 'snake_tail', 'follower'])
        g = crop(g, gpix, pbox)
        if g:
            out.append(g)
        covers |= box(m['ghost'])
    bodies = [m['player']] + ([m['ghost']] if m['ghost'] else [])
    wcells = wall_cells(m['base'], pressed_by(m['base'], bodies))
    w = crop(copy.deepcopy(wall_t), {p: 8 for p in wcells}, covers)
    if w:
        out.append(w)
    for i, o in enumerate(out):
        o['name'] = '%s_%03d' % (o['type'], i)
    return out


def transition_function(state, action):
    global _mem
    if _mem.get('last') == canon(state):
        m = copy.deepcopy(_mem['model'])
    else:
        m = parse(state)
    m = step(m, action)
    out = render(state, m)
    _mem = {'last': canon(out), 'model': copy.deepcopy(m)}
    return out
