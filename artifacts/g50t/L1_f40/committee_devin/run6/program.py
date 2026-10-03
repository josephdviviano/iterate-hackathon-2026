# Mechanics: player (5x5) moves 6px per arrow; blocked by bounds, the HUD band and wall pixels (a rect holding the
# plate centre is walkable); the very first input of a level (counter w=64) is a no-op. Counter loses 1px every 2nd
# action: w = 64 - (n+1)//2 (n = action count, hidden; continuity-gated). Plate (3x3 wall square) covered by player/
# ghost -> the cable arm retracts 6px toward the column. A5 after moving: respawn at SPAWN and toggle ghost mode (on:
# record the life's path, red HUD ring, ghost replays path[k-1] after k moves). Unconfirmed: invisible cell vs no-op.
import json

STEP, SPAWN, PS = 6, (14, 8), 5
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
ORDER = ['hud', 'hud', 'player', 'gate', 'target', 'counter']
_mem = {}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def obj_pixels(o):
    return {(o['x'] + i, o['y'] + j) for j, row in enumerate(o['pixels']) for i, v in enumerate(row) if v >= 0}


def rect(p):
    return {(p[0] + i, p[1] + j) for i in range(PS) for j in range(PS)}


def body(p, color, name, typ, tags):
    px = [[color] * PS for _ in range(PS)]
    px[PS // 2][PS // 2] = -1
    return {'name': name, 'type': typ, 'tags': tags, 'x': p[0], 'y': p[1], 'w': PS, 'h': PS,
            'pixels': px, 'layer': 1, 'visible': True}


def parse_wall(wall_px, occluders):
    """Canonical wall model: plate centre, column x, bar row, unpressed arm (key pixels, bar)."""
    pc = None
    for (x, y) in wall_px:
        if all((x + i, y + j) in wall_px for i in range(3) for j in range(3)) and \
                (x - 1, y + 1) not in wall_px and (x + 3, y + 1) not in wall_px:
            pc = (x + 1, y + 1)
            break
    if pc is None:  # plate hidden under a body standing on it
        for o in occluders:
            c = (o[0] + PS // 2, o[1] + PS // 2)
            if (c[0], o[1] + PS) in wall_px:
                pc = c
    cx = pc[0]
    col = [y for (x, y) in wall_px if x == cx]
    bar_y = max(col)
    arm = {(x, y) for (x, y) in wall_px if x != cx and y > pc[1] + 1}
    d = 1 if min(x for x, _ in arm) < cx else -1
    pressed = any(pc in rect(o) for o in occluders)
    if pressed:
        arm = {(x - d * STEP, y) for (x, y) in arm}
    edge = min(x for x, _ in arm) if d == 1 else max(x for x, _ in arm)
    key = {(x, y) for (x, y) in arm if abs(x - edge) < PS}
    return {'pc': pc, 'cx': cx, 'top': pc[1] + 2, 'bar_y': bar_y, 'key': sorted(key), 'd': d, 'edge': edge}


def wall_pixels(m, pressed):
    pc, cx, d = m['pc'], m['cx'], m['d']
    out = {(pc[0] + i, pc[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1)}
    out |= {(cx, y) for y in range(m['top'], m['bar_y'] + 1)}
    off = d * STEP if pressed else 0
    out |= {(x + off, y) for (x, y) in m['key']}
    e = m['edge'] + off
    rng = range(e, cx) if d == 1 else range(cx + 1, e + 1)
    out |= {(x, m['bar_y']) for x in rng}
    return out


def render_wall(px, template, name):
    if not px:
        return None
    xs, ys = [p[0] for p in px], [p[1] for p in px]
    x0, y0, w, h = min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    o = dict(template)
    o.update(name=name, x=x0, y=y0, w=w, h=h,
             pixels=[[8 if (x0 + i, y0 + j) in px else -1 for i in range(w)] for j in range(h)])
    return o


def transition_function(state, action):
    st = [dict(o) for o in state]
    cont = _mem.get('last') == canon(state)
    by = {}
    for o in st:
        by.setdefault(o['type'], []).append(o)
    player = by['player'][0]
    ppos = (player['x'], player['y'])
    counter = by['counter'][0]
    ghost_o = by.get('ghost', [None])[0]
    wall_o = by['wall'][0]
    huds = by['hud']
    red_hud = next((h for h in huds if h['pixels'][0][0] == 2), None)
    base_huds = [h for h in huds if h is not red_hud]
    occ = [ppos] + ([(ghost_o['x'], ghost_o['y'])] if ghost_o else [])

    if cont:
        n, path, replay, k, model = _mem['n'], _mem['path'], _mem['replay'], _mem['k'], _mem['model']
    else:
        n = 2 * (64 - counter['w'])
        path = [ppos] if ppos != SPAWN else [SPAWN]
        replay = None if red_hud is None else []
        k = 0
        model = parse_wall(obj_pixels(wall_o), occ)
    ghost_on = red_hud is not None
    gpos = (ghost_o['x'], ghost_o['y']) if ghost_o else None
    pressed_before = any(model['pc'] in rect(o) for o in occ)
    wall_now = wall_pixels(model, pressed_before)
    band = max(h['y'] + h['h'] for h in huds)

    def blocked(p):
        if p[0] < 0 or p[0] + PS > 64 or p[1] <= band or p[1] + PS > 63:
            return True
        r = rect(p)
        return model['pc'] not in r and bool(r & wall_now)

    a = action if isinstance(action, int) else action.get('action_id')
    if a in DIRS and n > 0:
        dx, dy = DIRS[a]
        np_ = (ppos[0] + dx * STEP, ppos[1] + dy * STEP)
        if not blocked(np_):
            ppos = np_
            path = path + [ppos]
            if ghost_on:
                k += 1
                if replay:
                    gpos = replay[min(k, len(replay)) - 1]
    elif a == 5 and (len(path) > 1 if cont else ppos != SPAWN):
        if ghost_on:
            ghost_on, replay, gpos = False, None, None
        else:
            ghost_on, replay, gpos = True, path, None
        ppos, path, k = SPAWN, [SPAWN], 0
    n += 1

    w = 64 - (n + 1) // 2
    out = []
    shift = 4 if ghost_on else 0
    for h in sorted(base_huds, key=lambda o: o['name']):
        h = dict(h)
        h['x'] = h['x'] - (4 if red_hud is not None else 0) + shift
        out.append(h)
    p = dict(player)
    p['x'], p['y'] = ppos
    out.append(p)
    out += [dict(o) for o in by['gate'] + by['target']]
    c = dict(counter)
    c['w'], c['pixels'] = w, [[9] * w]
    out.append(c)
    if ghost_on:
        rh = dict(red_hud) if red_hud else dict(base_huds[0], pixels=[[2, 2, 2], [2, -1, 2], [2, 2, 2]])
        rh['x'], rh['y'] = 1, 1
        out.append(rh)
    if ghost_on and gpos is not None:
        out.append(body(gpos, 2, '', 'ghost', ['ghost', 'snake_tail', 'follower']))
    bodies = [ppos] + ([gpos] if ghost_on and gpos else [])
    pressed = any(model['pc'] in rect(b) for b in bodies)
    vis = wall_pixels(model, pressed) - set().union(*[rect(b) for b in bodies])
    wo = render_wall(vis, wall_o, '')
    if wo:
        out.append(wo)
    for i, o in enumerate(out):
        o['name'] = '%s_%03d' % (o['type'], i)
    _mem.update(last=canon(out), n=n, path=path, replay=replay, k=k, model=model)
    return out
