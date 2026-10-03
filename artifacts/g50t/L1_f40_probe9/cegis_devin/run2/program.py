# Mechanics: player 5x5 moves 6px (A1-4); blocked by HUD band (y<8), x<14, screen edge, or any wall px in the
# destination box unless the box holds the plate centre. First input of a fresh level (counter w=64) is a no-op.
# Plate (3x3 top of the wall rope) covered by player/ghost -> key+bar left of the rope column shift +6 (door opens).
# A5 off spawn toggles ghost mode: record life trail -> red HUD ring, white huds x+4, ghost replays trail[k-1] after
# k successful moves; 2nd A5 clears. Ghost landing on the player's pre-move cell is hidden (step 43); counter -1 on odd calls.
import json

SPAWN = (14, 8)
FULL = 64
_mem = {}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def by_type(state, t):
    return [o for o in state if o['type'] == t]


def obj_pixels(o):
    pts = set()
    for j, row in enumerate(o['pixels']):
        for i, v in enumerate(row):
            if v >= 0:
                pts.add((o['x'] + i, o['y'] + j))
    return pts


def wall_geometry(w0):
    cols = {}
    for x, y in w0:
        cols[x] = cols.get(x, 0) + 1
    colx = max(cols, key=lambda c: (cols[c], c))
    top = min(y for _, y in w0)
    rows = {}
    for x, y in w0:
        rows.setdefault(y, set()).add(x)
    pb = top
    while pb + 1 in rows and len(rows[pb + 1]) > 1:
        pb += 1
    plate = [(x, y) for x, y in w0 if y <= pb]
    pc = (sorted(x for x, _ in plate)[len(plate) // 2], sorted(y for _, y in plate)[len(plate) // 2])
    return colx, pb, pc


def render_wall(w0, pressed):
    colx, pb, _ = wall_geometry(w0)
    if not pressed:
        return set(w0)
    keep = {(x, y) for x, y in w0 if not (x < colx and y > pb)}
    moved = {(x + 6, y) for x, y in w0 if x < colx and y > pb and x + 6 < colx}
    return keep | moved


def in_box(p, pos, n=5):
    return pos[0] <= p[0] < pos[0] + n and pos[1] <= p[1] < pos[1] + n


def parse_wall(state):
    pts = set()
    for o in by_type(state, 'wall'):
        pts |= obj_pixels(o)
    return pts


def stateless(state):
    counter = by_type(state, 'counter')[0]
    w = counter['w']
    calls = 0 if w >= FULL else 2 * (FULL - w) - 1
    player = by_type(state, 'player')[0]
    pos = (player['x'], player['y'])
    ghosts = by_type(state, 'ghost')
    red = [o for o in by_type(state, 'hud') if o['pixels'][0][0] == 2]
    ghost_mode = bool(red)
    gpos = (ghosts[0]['x'], ghosts[0]['y']) if ghosts else None
    return {
        'calls': calls,
        'pos': pos,
        'prev': None,
        'moved': pos != SPAWN,
        'trail': [pos],
        'ghost_mode': ghost_mode,
        'ghost_trail': [gpos] if gpos else [SPAWN],
        'k': 1 if gpos else 0,
        'w0': parse_wall(state),
        'hud_tpl': [o for o in by_type(state, 'hud') if o['pixels'][0][0] != 2],
        'red_tpl': red[0] if red else None,
        'player_tpl': player,
        'ghost_tpl': ghosts[0] if ghosts else None,
        'statics': [o for o in state if o['type'] in ('gate', 'target')],
        'counter_tpl': counter,
    }


def ghost_pos(m):
    if not m['ghost_mode'] or m['k'] == 0:
        return None
    gt = m['ghost_trail']
    return gt[min(m['k'] - 1, len(gt) - 1)]


def pressed(m):
    _, _, pc = wall_geometry(m['w0'])
    bodies = [m['pos']]
    g = ghost_pos(m)
    if g:
        bodies.append(g)
    return any(in_box(pc, b) for b in bodies)


def can_move(m, dest):
    x, y = dest
    if x < 14 or y < 8 or x + 5 > 64 or y + 5 > 63:
        return False
    _, _, pc = wall_geometry(m['w0'])
    if in_box(pc, dest):
        return True
    wall = render_wall(m['w0'], pressed(m))
    return not any(in_box(p, dest) for p in wall)


def step(m, action):
    m = dict(m)
    fresh = m['calls'] == 0
    if isinstance(action, int) and action in (1, 2, 3, 4) and not fresh:
        dx, dy = {1: (0, -6), 2: (0, 6), 3: (-6, 0), 4: (6, 0)}[action]
        dest = (m['pos'][0] + dx, m['pos'][1] + dy)
        if can_move(m, dest):
            m['prev'] = m['pos']
            m['pos'] = dest
            m['moved'] = True
            m['trail'] = m['trail'] + [dest]
            if m['ghost_mode']:
                m['k'] += 1
    elif action == 5 and m['moved']:
        if m['ghost_mode']:
            m['ghost_mode'] = False
        else:
            m['ghost_mode'] = True
            m['ghost_trail'] = list(m['trail'])
        m['k'] = 0
        m['pos'] = SPAWN
        m['prev'] = None
        m['moved'] = False
        m['trail'] = [SPAWN]
    m['calls'] += 1
    return m


def mk(tpl, name, **kw):
    o = json.loads(json.dumps(tpl))
    o.update(kw)
    o['name'] = name
    return o


def body(color, tags, typ, pos):
    px = [[color] * 5 for _ in range(5)]
    px[2][2] = -1
    return {'type': typ, 'tags': tags, 'x': pos[0], 'y': pos[1], 'w': 5, 'h': 5,
            'layer': 1, 'visible': True, 'pixels': px}


def render(m):
    out = []
    shift = 4 if m['ghost_mode'] else 0
    huds = sorted(m['hud_tpl'], key=lambda o: o['h'] == 1)
    for o in huds:
        o = dict(o)
        o['x'] = 1 + shift
        out.append(o)
    p = dict(m['player_tpl'])
    p['x'], p['y'] = m['pos']
    out.append(p)
    out += [dict(o) for o in sorted(m['statics'], key=lambda o: o['type'] != 'gate')]
    w = FULL - (m['calls'] + 1) // 2
    c = dict(m['counter_tpl'])
    c['w'] = w
    c['pixels'] = [[9] * w]
    out.append(c)
    bodies = [m['pos']]
    if m['ghost_mode']:
        r = m['red_tpl'] or {'type': 'hud', 'tags': ['hud'], 'layer': 0, 'visible': True, 'w': 3, 'h': 3,
                             'pixels': [[2, 2, 2], [2, -1, 2], [2, 2, 2]]}
        r = dict(r)
        r['x'], r['y'] = 1, 1
        out.append(r)
        g = ghost_pos(m)
        if g and g != m['prev']:
            out.append(body(2, ['ghost', 'snake_tail', 'follower'], 'ghost', g))
            bodies.append(g)
    wall = {q for q in render_wall(m['w0'], pressed(m)) if not any(in_box(q, b) for b in bodies)}
    if wall:
        x0 = min(x for x, _ in wall); y0 = min(y for _, y in wall)
        x1 = max(x for x, _ in wall); y1 = max(y for _, y in wall)
        px = [[8 if (x, y) in wall else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]
        out.append({'type': 'wall', 'tags': ['wall'], 'x': x0, 'y': y0, 'w': x1 - x0 + 1, 'h': y1 - y0 + 1,
                    'layer': 0, 'visible': True, 'pixels': px})
    for i, o in enumerate(out):
        o['name'] = '%s_%03d' % (o['type'], i)
    return out


def transition_function(state, action):
    global _mem
    if _mem.get('last') is not None and canon(state) == _mem['last']:
        m = _mem['m']
    else:
        m = stateless(state)
    walls = by_type(state, 'wall')
    if walls:
        m['wall_tags'] = walls[0]['tags']
    m = step(m, action)
    out = render(m)
    for o in out:
        if o['type'] == 'wall' and 'wall_tags' in m:
            o['tags'] = m['wall_tags']
    _mem = {'last': canon(out), 'm': m}
    return out
