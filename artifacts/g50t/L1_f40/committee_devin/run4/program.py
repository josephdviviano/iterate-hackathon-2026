# Mechanics: player 5x5 steps 6px (A1 up, A2 down, A3 left, A4 right); blocked by board/HUD band and wall px,
#   except a box containing the plate centre (3x3 top block) may overlap the wall. First input of a fresh level (counter w=64) is a no-op.
# Plate covered by player/ghost -> arm (door rows left of the vertical line) retracts 6px right; wall px under layer-1 boxes are cropped.
# A5 off respawn toggles ghost mode, teleports to respawn (first cell reached), trail=[start,respawn]; ghost = trail[-3]; ghost HUD icon at x=1, icons +4.
# Counter loses 1px every 2nd call (hidden parity, continuity-gated). Unconfirmed: exact HUD band limit, respawn without history, gate/target effects.
import json

H = {}
DELTA = {1: (0, -6), 2: (0, 6), 3: (-6, 0), 4: (6, 0)}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def px_set(o):
    return {(o['x'] + c, o['y'] + r) for r, row in enumerate(o['pixels']) for c, v in enumerate(row) if v >= 0}


def box(p):
    return {(p[0] + i, p[1] + j) for i in range(5) for j in range(5)}


def wall_geometry(E):
    xs = {}
    ys = {}
    for x, y in E:
        xs[x] = xs.get(x, 0) + 1
        ys[y] = ys.get(y, 0) + 1
    vx = max(xs, key=lambda k: (xs[k], k))
    ly = max(ys, key=lambda k: (ys[k], k))
    y0 = min(y for _, y in E)
    top = [(x, y) for x, y in E if y < y0 + 3]
    cx = (min(x for x, _ in top) + max(x for x, _ in top)) // 2
    plate = (cx, y0 + 1)
    return vx, ly, plate


def render_wall(E, pressed):
    vx, ly, _ = wall_geometry(E)
    if not pressed:
        return set(E)
    out = set()
    for x, y in E:
        if x < vx and abs(y - ly) <= 2:
            if x + 6 < vx:
                out.add((x + 6, y))
        else:
            out.add((x, y))
    return out


def pressed_by(plate, boxes):
    return any(plate in box(b) for b in boxes if b is not None)


def blocked(p, walls, plate):
    x, y = p
    if x < 0 or x + 5 > 64 or y < 7 or y + 5 > 63:
        return True
    b = box(p)
    if plate in b:
        return False
    return bool(b & walls)


def guess_ghost(g, pl):
    dx, dy = pl[0] - g[0], pl[1] - g[1]
    if dx and dy:
        return (g[0] + (6 if dx > 0 else -6), g[1])
    if abs(dx) + abs(dy) == 12:
        return ((g[0] + pl[0]) // 2, (g[1] + pl[1]) // 2)
    return pl


def transition_function(state, action):
    global H
    cont = bool(H) and canon(state) == H.get('last')
    aid = action['action_id'] if isinstance(action, dict) else action
    by = {}
    for o in state:
        by.setdefault(o['type'], []).append(o)
    huds = by.get('hud', [])
    ghost_hud = [o for o in huds if any(v == 2 for row in o['pixels'] for v in row)]
    main_huds = [o for o in huds if o not in ghost_hud]
    main_huds.sort(key=lambda o: (o['y'], o['x']))
    mode = bool(ghost_hud)
    player = by['player'][0]
    pos = (player['x'], player['y'])
    ghost_o = by.get('ghost', [None])[0]
    ghost = (ghost_o['x'], ghost_o['y']) if ghost_o else None
    counter = by['counter'][0]
    wall_os = by.get('wall', [])
    fresh = counter['w'] == 64
    if cont:
        E, start, respawn, trail, dec = H['E'], H['start'], H['respawn'], H['trail'], H['dec']
    else:
        E = set()
        for o in wall_os:
            E |= px_set(o)
        start = pos
        respawn = None
        trail = None
        dec = True
    if fresh:
        start = pos
    vx, ly, plate = wall_geometry(E) if E else (None, None, (-99, -99))
    was_pressed = pressed_by(plate, [pos, ghost])
    walls_now = render_wall(E, was_pressed) if E else set()

    new_pos, new_ghost, new_mode = pos, ghost, mode
    if aid in DELTA and not fresh:
        d = DELTA[aid]
        cand = (pos[0] + d[0], pos[1] + d[1])
        if not blocked(cand, walls_now, plate):
            new_pos = cand
            if respawn is None and cont:
                respawn = cand
            if mode:
                if trail is not None:
                    trail = trail + [cand]
                    new_ghost = trail[-3] if len(trail) >= 3 else None
                elif ghost is not None:
                    new_ghost = guess_ghost(ghost, pos)
    elif aid == 5:
        rs = respawn if respawn is not None else start
        if pos != rs:
            new_mode = not mode
            new_pos = rs
            new_ghost = None
            trail = [start, rs] if new_mode else None

    if dec:
        cw = counter['w'] - 1
    else:
        cw = counter['w']
    dec = not dec

    pressed = pressed_by(plate, [new_pos, new_ghost])
    wpx = render_wall(E, pressed) if E else set()
    for b in (new_pos, new_ghost):
        if b is not None:
            wpx -= box(b)

    out = []
    base_x = [o['x'] - (4 if mode else 0) for o in main_huds]
    for o, bx in zip(main_huds, base_x):
        n = dict(o)
        n['x'] = bx + (4 if new_mode else 0)
        out.append(n)
    p = dict(player)
    p['x'], p['y'] = new_pos
    out.append(p)
    for t in ('gate', 'target'):
        out.extend(dict(o) for o in by.get(t, []))
    c = dict(counter)
    c['w'] = max(cw, 0)
    c['pixels'] = [[9] * c['w']]
    out.append(c)
    if new_mode:
        src = main_huds[0]
        out.append({'h': src['h'], 'layer': src['layer'], 'name': '', 'tags': list(src['tags']), 'type': 'hud',
                    'visible': True, 'w': src['w'], 'x': min(base_x), 'y': src['y'],
                    'pixels': [[2 if v >= 0 else -1 for v in row] for row in src['pixels']]})
    if new_mode and new_ghost is not None:
        g = dict(player)
        if ghost_o:
            g = dict(ghost_o)
        else:
            g['tags'] = ['ghost', 'snake_tail', 'follower']
            g['type'] = 'ghost'
            g['pixels'] = [[2 if v >= 0 else -1 for v in row] for row in player['pixels']]
        g['x'], g['y'] = new_ghost
        out.append(g)
    if wpx and wall_os:
        w0 = wall_os[0]
        x0 = min(x for x, _ in wpx)
        y0 = min(y for _, y in wpx)
        x1 = max(x for x, _ in wpx)
        y1 = max(y for _, y in wpx)
        n = dict(w0)
        n.update(x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1,
                 pixels=[[8 if (x, y) in wpx else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)])
        out.append(n)
    for o in state:
        if o['type'] not in ('hud', 'player', 'gate', 'target', 'counter', 'ghost', 'wall'):
            out.append(dict(o))
    for i, o in enumerate(out):
        o['name'] = '%s_%03d' % (o['type'], i)
    H = {'last': canon(out), 'E': E, 'start': start, 'respawn': respawn, 'trail': trail, 'dec': dec}
    return out
