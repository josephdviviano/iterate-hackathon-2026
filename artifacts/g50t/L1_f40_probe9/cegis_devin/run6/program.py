# Mechanics: A1-4 move player 6px (blocked by y<8, board edge, or wall px in dest box unless it holds the plate centre);
# first call of a level (counter w=64) is a no-op; counter loses 1px on every odd call (hidden parity).
# Plate (3x3 top of the rope line) covered by player/ghost -> key/arm rows left of the line slide +6 (clipped at line).
# A5: on spawn -> no-op; ghost mode on -> clear red hud/ghost, huds x-4, respawn; else record: red hud (1,1), huds x+4, respawn.
# Ghost = player trail lag 2, trail reset to [level start, spawn] on record; hidden on the player. Unconfirmed: plate by ghost.
import json

DIRS = {1: (0, -6), 2: (0, 6), 3: (-6, 0), 4: (6, 0)}
DEF_START, DEF_SPAWN = (20, 8), (14, 8)
MEM = {}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def box_has(b, p):
    return b[0] <= p[0] < b[0] + 5 and b[1] <= p[1] < b[1] + 5


def wall_px(o):
    return {(o['x'] + i, o['y'] + j): v for j, r in enumerate(o['pixels']) for i, v in enumerate(r) if v != -1}


def parse_rope(px):
    cols, rows = {}, {}
    for (x, y) in px:
        cols[x] = cols.get(x, 0) + 1
        rows[y] = rows.get(y, 0) + 1
    xl = max(cols, key=lambda k: (cols[k], k))
    ya = max(rows, key=lambda k: (rows[k], k))
    return xl, ya


def press(base, xl, ya):
    out = {}
    for (x, y), v in base.items():
        if x < xl and abs(y - ya) <= 2:
            if x + 6 < xl:
                out[(x + 6, y)] = v
        else:
            out[(x, y)] = v
    for (x, y), v in base.items():
        if x == xl:
            out[(x, y)] = v
    return out


def unpress(px, xl, ya):
    out = {}
    for (x, y), v in px.items():
        if x < xl and abs(y - ya) <= 2:
            out[(x - 6, y)] = v
        else:
            out[(x, y)] = v
    arm = [x for (x, y) in out if y == ya]
    col = px.get((xl, ya), 8)
    for x in range(min(arm), xl):
        out.setdefault((x, ya), col)
    return out


def find_plate(px, xl):
    top = min(y for (x, y) in px if x == xl)
    if all((xl + dx, top + dy) in px for dx in (-1, 0, 1) for dy in (0, 1, 2)):
        return (xl, top + 1)
    return None


def stateless(state):
    objs = {o['type']: o for o in state}
    counter = objs['counter']
    w = counter['w']
    n = 0 if w == 64 else 2 * (64 - w) - 1
    wall = objs['wall']
    px = wall_px(wall)
    xl, ya = parse_rope(px)
    bodies = [(o['x'], o['y']) for o in state if o['type'] in ('player', 'ghost')]
    plate = find_plate(px, xl)
    if plate is None:
        top = min(y for (x, y) in px if x == xl)
        occ = [b for b in bodies if box_has(b, (xl, top - 1))]
        plate = (xl, occ[0][1] + 2) if occ else (xl, top - 2)
    pressed = any(box_has(b, plate) for b in bodies)
    base = unpress(px, xl, ya) if pressed else dict(px)
    col = px.get((xl, ya), 8)
    ys = [y for (x, y) in base if x == xl]
    for y in range(plate[1] - 1, max(ys) + 1):
        base.setdefault((xl, y), col)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            base.setdefault((plate[0] + dx, plate[1] + dy), col)
    mode = any(o['type'] == 'hud' and o['pixels'][0][0] == 2 for o in state)
    player = (objs['player']['x'], objs['player']['y'])
    ghost = (objs['ghost']['x'], objs['ghost']['y']) if 'ghost' in objs else None
    if mode:
        if ghost is None:
            trail = [DEF_START, DEF_SPAWN] if player == DEF_SPAWN else [player, player]
        else:
            mids = [(ghost[0] + dx, ghost[1] + dy) for dx, dy in DIRS.values()]
            mids = [m for m in mids if abs(m[0] - player[0]) + abs(m[1] - player[1]) == 6] or [player]
            trail = [ghost, mids[0], player]
    else:
        trail = [player]
    return dict(n=n, base=base, xl=xl, ya=ya, plate=plate, mode=mode, trail=trail,
                ghost=ghost, start=DEF_START, spawn=DEF_SPAWN, moved=True,
                wall_meta={k: wall[k] for k in wall if k not in ('x', 'y', 'w', 'h', 'pixels', 'name')})


def render_wall(S, bodies):
    px = dict(S['base'])
    if any(box_has(b, S['plate']) for b in bodies):
        px = press(S['base'], S['xl'], S['ya'])
    vis = {p: v for p, v in px.items() if not any(b[0] <= p[0] < b[0] + 5 and b[1] <= p[1] < b[1] + 5 for b in bodies)}
    return px, vis


def build_wall(S, vis, idx):
    xs = [p[0] for p in vis]
    ys = [p[1] for p in vis]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    pix = [[vis.get((x, y), -1) for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]
    o = dict(S['wall_meta'])
    o.update(name='wall_%03d' % idx, x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1, pixels=pix)
    return o


def transition_function(state, action):
    global MEM
    if MEM.get('last') == canon(state):
        S = MEM['S']
    else:
        S = stateless(state)
    objs = [dict(o) for o in state]
    player = next(o for o in objs if o['type'] == 'player')
    counter = next(o for o in objs if o['type'] == 'counter')
    proto_hud = next(o for o in objs if o['type'] == 'hud')
    pos = (player['x'], player['y'])
    n = S['n']
    if n == 0:
        S['start'] = pos
        S['moved'] = False
    aid = action['action_id'] if isinstance(action, dict) else action
    ghost = S['ghost']
    if n > 0 and aid in DIRS:
        dx, dy = DIRS[aid]
        np_ = (pos[0] + dx, pos[1] + dy)
        bodies = [pos] + ([ghost] if ghost else [])
        cur, _ = render_wall(S, bodies)
        ok = 0 <= np_[0] and np_[0] + 5 <= 64 and np_[1] >= 8 and np_[1] + 5 <= 63
        if ok and not box_has(np_, S['plate']):
            ok = not any(box_has(np_, p) for p in cur)
        if ok:
            if not S['moved']:
                S['spawn'] = np_
                S['moved'] = True
            pos = np_
            S['trail'] = S['trail'] + [pos]
            if S['mode'] and len(S['trail']) >= 3:
                ghost = S['trail'][-3]
    elif n > 0 and aid == 5 and pos != S['spawn']:
        huds = [o for o in objs if o['type'] == 'hud' and o['pixels'][0][0] != 2]
        if S['mode']:
            S['mode'] = False
            ghost = None
            for o in huds:
                o['x'] -= 4
            S['trail'] = [S['spawn']]
        else:
            S['mode'] = True
            for o in huds:
                o['x'] += 4
            S['trail'] = [S['start'], S['spawn']]
        pos = S['spawn']
    S['ghost'] = ghost
    player['x'], player['y'] = pos
    S['n'] = n + 1
    if n % 2 == 0:
        w = max(counter['w'] - 1, 0)
        counter['w'] = w
        counter['pixels'] = [counter['pixels'][0][:w]]
    out = [o for o in objs if o['type'] not in ('wall', 'ghost') and not (o['type'] == 'hud' and o['pixels'][0][0] == 2)]
    if S['mode']:
        h = dict(proto_hud)
        h.update(x=1, y=1, w=3, h=3, pixels=[[2, 2, 2], [2, -1, 2], [2, 2, 2]])
        out.append(h)
    bodies = [pos]
    if S['mode'] and ghost is not None:
        bodies.append(ghost)
        if ghost != pos:
            g = dict(player)
            g.update(type='ghost', tags=['ghost', 'snake_tail', 'follower'], x=ghost[0], y=ghost[1],
                     pixels=[[-1 if v == -1 else 2 for v in r] for r in player['pixels']])
            out.append(g)
    _, vis = render_wall(S, bodies)
    out.append(build_wall(S, vis, 0))
    for i, o in enumerate(out):
        o['name'] = '%s_%03d' % (o['type'], i)
    MEM = {'S': S, 'last': canon(out)}
    return out
