# Mechanics: player 5x5 moves 6px; blocked by y<8 HUD band or wall px in dest box (unless box holds plate centre); first input (n==0) no-op.
# Wall = 3x3 plate atop line column + arm row + key; body on plate shifts key/arm rows left of the line +6; layer-1 boxes crop wall px.
# A5 off-spawn toggles ghost mode: record life trail [spawn, start, moves...] (+red hud, huds x+4) or clear; player respawns.
# Ghost shows trail[m], m = moves since record, hidden while on the player (step-43: trail[1]=player pos). Counter -1px on even n.
# Unconfirmed: spawn constant (14,8) with life 1 starting off-spawn (unseen step 0); counter parity hidden (fallback from width).
import json

STEP = 6
DEFAULT_SPAWN = (14, 8)
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
GHOST_PX = [[2] * 5, [2] * 5, [2, 2, -1, 2, 2], [2] * 5, [2] * 5]
GHUD_PX = [[2, 2, 2], [2, -1, 2], [2, 2, 2]]
_mem = {}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def px_set(o):
    return {(o['x'] + i, o['y'] + j) for j, row in enumerate(o['pixels'])
            for i, v in enumerate(row) if v != -1}


def in_box(p, b):
    return b[0] <= p[0] < b[0] + 5 and b[1] <= p[1] < b[1] + 5


# ---------- wall model ----------
def parse_wall(W, bodies):
    cols, rows = {}, {}
    for x, y in W:
        cols[x] = cols.get(x, 0) + 1
        rows[y] = rows.get(y, 0) + 1
    xc = max(cols, key=lambda c: (cols[c], -c))
    arm = max(rows, key=lambda r: (rows[r], r))
    full = set(W)
    side = [p for p in W if p[0] in (xc - 1, xc + 1) and p[1] < arm - 2]
    if side:
        plate = (xc, min(p[1] for p in side) + 1)
    else:
        ytop = min(y for x, y in W if x == xc)
        body = next((b for b in bodies if in_box((xc, ytop - 1), b)), None)
        py = body[1] + 2 if body else ytop - 2
        plate = (xc, py)
        full |= {(xc + dx, py + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        full |= {(xc, y) for y in range(py + 2, ytop)}
    return {'xc': xc, 'arm': arm, 'plate': plate}, full


def pressed(model, bodies):
    return any(in_box(model['plate'], b) for b in bodies)


def shift(model, P):
    xc, a = model['xc'], model['arm']
    door = lambda x, y: x < xc and abs(y - a) <= 2
    return {(x + STEP, y) if door(x, y) else (x, y) for x, y in P if not door(x, y) or x + STEP < xc}


def unshift(model, P):
    xc, a = model['xc'], model['arm']
    out = {(x - STEP, y) if x < xc and abs(y - a) <= 2 else (x, y) for x, y in P}
    return out | {(x, model['arm']) for x in range(xc - STEP, xc)}


def wall_obj(P, template):
    o = dict(template)
    if not P:
        return None
    x0, y0 = min(p[0] for p in P), min(p[1] for p in P)
    x1, y1 = max(p[0] for p in P), max(p[1] for p in P)
    color = 8
    if template.get('pixels'):
        color = next(v for r in template['pixels'] for v in r if v != -1)
    o.update(x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1,
             pixels=[[color if (x, y) in P else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)])
    return o


# ---------- counter ----------
def counter_n(c):
    w = c['w']
    return 0 if w >= 64 else 2 * (64 - w) - 1


def counter_after(c, n):
    o = dict(c)
    if n % 2 == 0 and o['w'] > 0:
        o['w'] -= 1
        o['pixels'] = [[o['pixels'][0][0]] * o['w']] if o['w'] else []
    return o


# ---------- main ----------
def transition_function(state, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    player = next(o for o in state if o['type'] == 'player')
    counter = next((o for o in state if o['type'] == 'counter'), None)
    walls = [o for o in state if o['type'] == 'wall']
    ghosts = [o for o in state if o['type'] == 'ghost']
    huds = [o for o in state if o['type'] == 'hud']
    ghud = [h for h in huds if any(v == 2 for r in h['pixels'] for v in r)]
    phuds = [h for h in huds if h not in ghud]
    pos = (player['x'], player['y'])
    bodies = [pos] + [(g['x'], g['y']) for g in ghosts]

    W = set()
    for w in walls:
        W |= px_set(w)
    model, full = parse_wall(W, bodies) if W else (None, set())
    base = unshift(model, full) if model and pressed(model, bodies) else full

    if _mem.get('canon') == canon(state):
        h = _mem['h']
        spawn, n, life, gtrail, m = h['spawn'], h['n'], list(h['life']), h['gtrail'], h['m']
    else:
        spawn = DEFAULT_SPAWN
        n = counter_n(counter) if counter else 1
        life = [spawn] + ([pos] if pos != spawn else [])
        gtrail = [(ghosts[0]['x'], ghosts[0]['y'])] if ghosts else ([spawn] if ghud else None)
        m = 0
    ghost_mode = gtrail is not None

    newpos = pos
    if aid in MOVES and n > 0:
        dx, dy = MOVES[aid]
        cand = (pos[0] + dx, pos[1] + dy)
        box_px = [(cand[0] + i, cand[1] + j) for i in range(5) for j in range(5)]
        ok = cand[1] >= 8 and cand[0] >= 0 and cand[0] + 5 <= 64 and cand[1] + 5 <= 63
        if ok and model and any(p in base for p in box_px) and not in_box(model['plate'], cand):
            ok = False
        if ok:
            newpos = cand
            life.append(cand)
            if ghost_mode:
                m += 1
    elif aid == 5 and pos != spawn:
        if ghost_mode:
            gtrail = None
        else:
            gtrail, m = life, 0
        life = [spawn]
        m = 0
        newpos = spawn
    ghost_mode = gtrail is not None

    gpos = gtrail[min(m, len(gtrail) - 1)] if ghost_mode else None
    gvis = ghost_mode and gpos != newpos
    after_bodies = [newpos] + ([gpos] if gvis else [])
    P = shift(model, base) if model and pressed(model, after_bodies) else set(base)
    P = {p for p in P if not any(in_box(p, b) for b in after_bodies)}

    off = 4 if ghost_mode else 0
    out = []
    for o in state:
        if o['type'] in ('wall', 'ghost') or o in ghud:
            continue
        o = dict(o)
        if o['type'] == 'player':
            o['x'], o['y'] = newpos
        elif o['type'] == 'counter':
            o = counter_after(o, n)
        elif o in phuds or (o['type'] == 'hud' and o['name'] in [h['name'] for h in phuds]):
            o['x'] = o['x'] - (4 if ghud else 0) + off
        out.append(o)
    if ghost_mode:
        out.append({'name': 'hud', 'type': 'hud', 'tags': ['hud'], 'x': 1, 'y': 1, 'w': 3, 'h': 3,
                    'pixels': [r[:] for r in GHUD_PX], 'layer': 0, 'visible': True})
    if gvis:
        out.append({'name': 'ghost', 'type': 'ghost', 'tags': ['ghost', 'snake_tail', 'follower'],
                    'x': gpos[0], 'y': gpos[1], 'w': 5, 'h': 5, 'pixels': [r[:] for r in GHOST_PX],
                    'layer': 1, 'visible': True})
    if walls:
        wo = wall_obj(P, walls[0])
        if wo:
            out.append(wo)
    for i, o in enumerate(out):
        o['name'] = '%s_%03d' % (o['type'], i)
    _mem['canon'] = canon(out)
    _mem['h'] = {'spawn': spawn, 'n': n + 1, 'life': life, 'gtrail': gtrail, 'm': m}
    return out
