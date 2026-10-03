# Mechanics: player (5x5) moves 6px per A1-A4 (up/down/left/right), blocked by the HUD band, screen edges and
# wall pixels (except a wall's pressure plate = isolated solid 3x3); the very first input of a fresh level (counter full) is consumed.
# Wall = rope: plate -> vertical line -> horizontal arm ending in a door; while a player/ghost box covers the plate the arm retracts 6px.
# A5 (player off respawn) toggles ghost mode, respawns at the level's first reached cell; in ghost mode the ghost follows the player's
# trail 2 moves behind (trail restarts as [level start, respawn]); HUD gets a ghost icon. Counter loses 1px every 2nd call (hidden parity).
import json

STEP = 6
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
GHOST_PIX = [[2] * 5, [2] * 5, [2, 2, -1, 2, 2], [2] * 5, [2] * 5]
HUD_GHOST = [[2, 2, 2], [2, -1, 2], [2, 2, 2]]
_H = {}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def pix_set(o):
    out = {}
    for j, row in enumerate(o['pixels']):
        for i, v in enumerate(row):
            if v != -1:
                out[(o['x'] + i, o['y'] + j)] = v
    return out


def box(o):
    return (o['x'], o['y'], o['x'] + o['w'], o['y'] + o['h'])


def in_box(p, b):
    return b[0] <= p[0] < b[2] and b[1] <= p[1] < b[3]


def is_ghost_hud(o):
    return o['type'] == 'hud' and any(v == 2 for r in o['pixels'] for v in r)


# ---------- wall (rope) model ----------
def find_plate(px):
    cores = [p for p in px if all((p[0] + a, p[1] + b) in px for a in (-1, 0, 1) for b in (-1, 0, 1))]
    cs = set(cores)
    for c in cores:
        if not any((c[0] + a, c[1] + b) in cs for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)):
            return c
    return None


def vcol(px):
    cnt = {}
    for (x, y) in px:
        cnt[x] = cnt.get(x, 0) + 1
    return max(cnt, key=lambda k: cnt[k])


def arm_keys(px, plate, xv):
    return [p for p in px if p[0] < xv and p[1] > plate[1] + 1]


def press(rest, plate, sign):
    """Return pixels with the arm retracted (sign=+1) or extended back (sign=-1) by one step."""
    xv = vcol(rest)
    arm = arm_keys(rest, plate, xv)
    out = {p: v for p, v in rest.items() if p not in set(arm)}
    line_rows = {y for (x, y) in arm if x == xv - 1}
    for (x, y) in arm:
        nx = x + sign * STEP
        if nx < xv:
            out[(nx, y)] = rest[(x, y)]
    if sign < 0:
        for y in line_rows:
            for x in range(xv - STEP, xv):
                out[(x, y)] = rest[(xv - 1, y)]
    return out


def recover_wall(w, covers):
    """Stateless: rebuild the rest wall (plate visible, arm extended) from an observed wall."""
    px = pix_set(w)
    plate = find_plate(px)
    if plate is not None or not px:
        return px, plate
    xv = vcol(px)
    top = min(y for (x, y) in px if x == xv)
    for b in covers:
        c = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        if c[0] == xv and b[3] >= top - 1 and c[1] < top:
            col = px[(xv, top)]
            for a in (-1, 0, 1):
                for d in (-1, 0, 1):
                    px[(c[0] + a, c[1] + d)] = col
            for y in range(c[1] + 2, top):
                px[(xv, y)] = col
            return press(px, c, -1), c
    return px, None


def render_wall(w, rest, plate, covers):
    px = dict(rest)
    if plate is not None and any(in_box(plate, b) for b in covers):
        px = press(rest, plate, +1)
    vis = {p: v for p, v in px.items() if not any(in_box(p, b) for b in covers)}
    o = dict(w)
    if not vis:
        o.update(w=0, h=0, pixels=[])
        return o, px
    x0, y0 = min(p[0] for p in vis), min(p[1] for p in vis)
    x1, y1 = max(p[0] for p in vis), max(p[1] for p in vis)
    o.update(x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1,
             pixels=[[vis.get((x, y), -1) for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)])
    return o, px


# ---------- player ----------
def blocked(nb, state_walls, huds):
    if nb[0] < 0 or nb[2] > 64 or nb[3] > 63:
        return True
    band = max((h['y'] + h['h'] for h in huds), default=0)
    if nb[1] <= band:
        return True
    for px, plate in state_walls:
        if plate is not None and in_box(plate, nb):
            continue
        if any(in_box(p, nb) for p in px):
            return True
    return False


def transition_function(state, action):
    global _H
    cont = bool(_H) and canon(state) == _H.get('last')
    H = _H if cont else {}
    aid = action['action_id'] if isinstance(action, dict) else action

    huds = [o for o in state if o['type'] == 'hud' and not is_ghost_hud(o)]
    ghud = [o for o in state if is_ghost_hud(o)]
    player = next(o for o in state if o['type'] == 'player')
    ghost = next((o for o in state if o['type'] == 'ghost'), None)
    walls = [o for o in state if o['type'] == 'wall']
    counter = next((o for o in state if o['type'] == 'counter'), None)
    others = [o for o in state if o['type'] not in ('wall', 'ghost') and not is_ghost_hud(o)]
    mode = 1 if ghud else 0

    covers_before = [box(o) for o in state if o['layer'] >= 1 and o['type'] in ('player', 'ghost')]
    if cont:
        rests = H['rests']
    else:
        rests = [recover_wall(w, covers_before) for w in walls]
    pos = (player['x'], player['y'])
    trail = list(H.get('trail', [pos]))
    origin = H.get('origin')
    tick = H.get('tick', True)
    fresh = counter is not None and counter['w'] == 64 and not origin and len(trail) == 1

    # walls as currently drawn (pressed state from before positions)
    cur = []
    for (rest, plate) in rests:
        if plate is not None and any(in_box(plate, b) for b in covers_before):
            cur.append((press(rest, plate, +1), plate))
        else:
            cur.append((rest, plate))

    player = dict(player)
    ghost_pos = (ghost['x'], ghost['y']) if ghost else None
    if aid in DIRS and not fresh:
        dx, dy = DIRS[aid]
        nb = (pos[0] + dx, pos[1] + dy, pos[0] + dx + player['w'], pos[1] + dy + player['h'])
        if not blocked(nb, cur, huds + ghud):
            pos = (pos[0] + dx, pos[1] + dy)
            trail.append(pos)
            if origin is None and len(trail) >= 2:
                origin = [trail[0], trail[1]]
            if mode == 1:
                ghost_pos = tuple(trail[-3]) if len(trail) >= 3 else ghost_pos
    elif aid == 5 and origin is not None and tuple(origin[1]) != pos:
        mode = 1 - mode
        pos = tuple(origin[1])
        trail = [tuple(origin[0]), tuple(origin[1])]
        ghost_pos = None
    if mode == 0:
        ghost_pos = None
    player['x'], player['y'] = pos

    # HUD: ghost icons prepend on the left, player icons shift right by 4 per icon
    n_old, n_new = len(ghud), mode
    new_huds = []
    for o in others:
        o = dict(o)
        if o['type'] == 'hud':
            o['x'] += 4 * (n_new - n_old)
        if o['type'] == 'player':
            o = player
        if o['type'] == 'counter' and tick:
            o['w'] -= 1
            o['pixels'] = [r[:-1] for r in o['pixels']]
        new_huds.append(o)
    out = new_huds
    if n_new:
        base = min((h['x'] for h in huds), default=1) - 4 * n_old if huds else 1
        out.append({'name': '', 'type': 'hud', 'tags': ['hud'], 'x': base, 'y': 1, 'w': 3, 'h': 3,
                    'pixels': HUD_GHOST, 'layer': 0, 'visible': True})
    if ghost_pos is not None:
        out.append({'name': '', 'type': 'ghost', 'tags': ['ghost', 'snake_tail', 'follower'],
                    'x': ghost_pos[0], 'y': ghost_pos[1], 'w': 5, 'h': 5, 'pixels': GHOST_PIX,
                    'layer': 1, 'visible': True})
    covers_after = [box(o) for o in out if o['type'] in ('player', 'ghost')]
    for w, (rest, plate) in zip(walls, rests):
        o, _ = render_wall(w, rest, plate, covers_after)
        if o['pixels']:
            out.append(o)
    for i, o in enumerate(out):
        o['name'] = '%s_%03d' % (o['type'], i)

    _H = {'last': canon(out), 'rests': rests, 'trail': trail, 'origin': origin, 'tick': not tick}
    return out
