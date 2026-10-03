# Mechanics: player 5x5 steps 6 (A1-4); blocked by board/HUD band (y<8) or any wall px in the dest box unless it holds the plate centre.
# First input of a fresh level (counter w=64, n=0) is a no-op. Counter loses 1px on odd calls (n hidden, continuity-gated; fallback n odd).
# Plate (topmost 3x3 of the wall) covered by player/ghost -> wall px left of the column shift +6 toward it (clipped); layer-1 boxes crop the wall.
# A5 off spawn: no ghost -> record (red hud at x=1, other huds x+4, trail=[level start, spawn]); ghost -> clear; both respawn. Ghost = trail 2 moves behind,
# not extracted when it sits on the player (step-43). Unconfirmed: level start/spawn fallback when never observed, left-edge bound, ghost collisions.
import json

MOVES = {1: (0, -6), 2: (0, 6), 3: (-6, 0), 4: (6, 0)}
STEP_TOP = 8
H = {}  # hidden state


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def px_set(o):
    return {(o['x'] + i, o['y'] + j) for j, r in enumerate(o['pixels']) for i, v in enumerate(r) if v != -1}


def find_plate(px):
    sq = [(x, y) for (x, y) in px if all((x + a, y + b) in px for a in range(3) for b in range(3))]
    if not sq:
        return None
    x, y = min(sq, key=lambda p: (p[1], p[0]))
    return (x + 1, y + 1)


def column(px):
    cnt = {}
    for x, _ in px:
        cnt[x] = cnt.get(x, 0) + 1
    return max(cnt, key=lambda x: (cnt[x], x))


def pressed_px(base, plate, lx):
    plate_cells = {(plate[0] + a, plate[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
    out = set()
    for (x, y) in base:
        if x < lx and (x, y) not in plate_cells:
            if x + 6 < lx:
                out.add((x + 6, y))
        else:
            out.add((x, y))
    return out


def covers(pos, pt):
    return pos[0] <= pt[0] < pos[0] + 5 and pos[1] <= pt[1] < pos[1] + 5


def wall_model(wall_obj, bodies):
    """Recover the unpressed wall pixel set (cached when the frame shows it unpressed)."""
    px = px_set(wall_obj)
    plate = find_plate(px)
    if plate is not None and not any(covers(b, plate) for b in bodies) and len(px) >= len(H.get('wall_base', ())):
        H['wall_base'] = px
        H['plate'] = plate
        H['col'] = column(px)
    return H.get('wall_base', px), H.get('plate'), H.get('col', column(px))


def recolor(pixels, c):
    return [[c if v != -1 else -1 for v in r] for r in pixels]


def obj_from_px(proto, px):
    xs = [p[0] for p in px]; ys = [p[1] for p in px]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    color = proto['pixels'][0][0] if proto['pixels'][0][0] != -1 else 8
    for r in proto['pixels']:
        for v in r:
            if v != -1:
                color = v
    pix = [[color if (x0 + i, y0 + j) in px else -1 for i in range(w)] for j in range(h)]
    o = dict(proto); o.update(x=x0, y=y0, w=w, h=h, pixels=pix)
    return o


def stateless(state):
    by = {}
    for o in state:
        by.setdefault(o['type'], []).append(o)
    counter = by['counter'][0]
    w = counter['w']
    n = 0 if w == 64 else 2 * (64 - w) - 1
    player = by['player'][0]
    ghost_mode = any(o['pixels'][0][0] == 2 for o in by.get('hud', []))
    g = by.get('ghost', [])
    pos = (player['x'], player['y'])
    if ghost_mode:
        trail = [(g[0]['x'], g[0]['y']), pos, pos] if g else [pos, pos]
    else:
        trail = []
    if n == 0:
        H['start'] = pos
    return {'n': n, 'ghost_mode': ghost_mode, 'trail': trail}


def transition_function(state, action):
    if H.get('last') is not None and canon(state) == H['last']:
        st = H['dyn']
    else:
        st = stateless(state)
    st = dict(st); st['trail'] = list(st['trail'])
    by = {}
    for o in state:
        by.setdefault(o['type'], []).append(o)
    player = dict(by['player'][0])
    pos = (player['x'], player['y'])
    ghost_pos = st['trail'][-3] if len(st['trail']) >= 3 else None
    bodies = [pos] + ([ghost_pos] if ghost_pos else [])
    wall = by['wall'][0]
    base, plate, lx = wall_model(wall, bodies)
    start = H.get('start', pos)
    spawn = H.get('spawn', (wall['x'], STEP_TOP))
    pressed = plate is not None and any(covers(b, plate) for b in bodies)
    cur_wall = pressed_px(base, plate, lx) if pressed else base
    huds = [dict(o) for o in by.get('hud', []) if o['pixels'][0][0] != 2]
    ghost_hud = [dict(o) for o in by.get('hud', []) if o['pixels'][0][0] == 2]
    ghost_proto = by['ghost'][0] if by.get('ghost') else None

    n = st['n']
    if n == 0:
        pass
    elif action in MOVES:
        dx, dy = MOVES[action]
        nx, ny = pos[0] + dx, pos[1] + dy
        box = {(nx + i, ny + j) for i in range(5) for j in range(5)}
        ok = nx >= 0 and nx + 5 <= 64 and ny >= STEP_TOP and ny + 5 <= 63
        if ok and (box & cur_wall) and not (plate and plate in box):
            ok = False
        if ok:
            if 'spawn' not in H and pos == start:
                H['spawn'] = (nx, ny); spawn = (nx, ny)
            pos = (nx, ny)
            if st['ghost_mode']:
                st['trail'].append(pos)
    elif action == 5 and pos != spawn:
        if st['ghost_mode']:
            st['ghost_mode'] = False; st['trail'] = []
            for o in huds:
                o['x'] -= 4
            ghost_hud = []
        else:
            st['ghost_mode'] = True; st['trail'] = [start, spawn]
            for o in huds:
                o['x'] += 4
            proto = huds[0]
            ghost_hud = [dict(proto, x=1, y=1, pixels=recolor(proto['pixels'], 2))]
        pos = spawn
    st['n'] = n + 1

    counter = dict(by['counter'][0])
    if st['n'] % 2 == 1:
        counter['w'] = max(0, counter['w'] - 1)
        counter['pixels'] = [[9] * counter['w']]
    player['x'], player['y'] = pos
    ghost_pos = st['trail'][-3] if len(st['trail']) >= 3 else None
    if ghost_pos == pos:
        ghost_pos = None
    bodies = [pos] + ([ghost_pos] if ghost_pos else [])
    pressed = plate is not None and any(covers(b, plate) for b in bodies)
    wpx = pressed_px(base, plate, lx) if pressed else set(base)
    wpx = {p for p in wpx if not any(covers(b, p) for b in bodies)}

    out = huds + [player] + [dict(o) for o in by.get('gate', [])] + [dict(o) for o in by.get('target', [])]
    out += [counter] + ghost_hud
    if ghost_pos:
        gp = ghost_proto or dict(player, type='ghost', tags=['ghost', 'snake_tail', 'follower'],
                                 pixels=recolor(player['pixels'], 2))
        out.append(dict(gp, x=ghost_pos[0], y=ghost_pos[1]))
    if wpx:
        out.append(obj_from_px(wall, wpx))
    for i, o in enumerate(out):
        o['name'] = '%s_%03d' % (o['type'], i)
    H['last'] = canon(out)
    H['dyn'] = st
    return out
