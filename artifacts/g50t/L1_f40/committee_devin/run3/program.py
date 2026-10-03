# Mechanics: player (5x5) moves 6px per A1-4 inside the field enclosed by the L-shaped wall cable; blocked by the plug, field edge, and
# invisible cell (20,14) (hypothesis). Counter bar w=64-n//2, n = all actions incl. a pre-buffer step 0: the counter A5 mixed outcome is parity.
# A5: if moved this life -> player back to SPAWN (14,8, hypothesis), first time records the life path as a ghost (hud box colour 2,
# player hud shifts +4); A5 while a recording exists clears it. Ghost replays the recorded path one cell per effective move (hidden when on
# the player). Actor on the button (3x3 at the cable's far end) pulls the plug 6px along the cable. Wall re-extracted with actor occlusion.
import json

STEP, SPAWN, HIDDEN_BLOCK = 6, (14, 8), {(20, 14)}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
MEM = {}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells(o):
    return {(o['x'] + i, o['y'] + j) for j, r in enumerate(o['pixels']) for i, v in enumerate(r) if v >= 0}


def rect(p, w=5, h=5):
    return {(p[0] + i, p[1] + j) for i in range(w) for j in range(h)}


def plug_cells(px, py, d):
    s = set()
    for j in range(5):
        for i in range(5):
            if j % 2 == 0 or (i >= 1 if d > 0 else i <= 3):
                s.add((px + i, py + j))
    return s


def find_plug(W):
    for (x, y) in sorted(W):
        for d in (1, -1):
            pc = plug_cells(x - (0 if d > 0 else 0), y, d)
            if pc <= W and ((x + 5, y + 2) in W if d > 0 else (x - 1, y + 2) in W):
                if not ((x, y + 1) in W if d > 0 else (x + 4, y + 1) in W):
                    return (x, y), d
    return None, 1


def parse_wall(W, plug, d, actors):
    """Return (rest plug pos, line cells, button centre) from visible wall cells."""
    pc = plug_cells(plug[0], plug[1], d)
    line = W - pc
    button = None
    for (x, y) in sorted(line):
        if all((x + i, y + j) in line for i in range(3) for j in range(3)):
            button = (x + 1, y + 1)
            break
    if button is None:
        for a in actors:
            R = rect(a)
            ends = [p for p in line if any((p[0] + dx, p[1] + dy) in R for dx, dy in DIRS.values())]
            if ends:
                c = (a[0] + 2, a[1] + 2)
                e = ends[0]
                sx = (c[0] > e[0]) - (c[0] < e[0])
                sy = (c[1] > e[1]) - (c[1] < e[1])
                q = e
                while abs(c[0] - q[0]) + abs(c[1] - q[1]) > 2:
                    q = (q[0] + sx, q[1] + sy)
                    line.add(q)
                line |= {(c[0] + i, c[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1)}
                button = c
                break
    pressed = button is not None and any(button in rect(a) for a in actors)
    rest = (plug[0] - STEP * d, plug[1]) if pressed else plug
    if pressed:
        fx = rest[0] + 5 if d > 0 else rest[0] - 1
        for k in range(STEP + 1):
            line.add((fx + k * d, rest[1] + 2))
    return rest, line, button


def build_wall(rest, d, line, button, actors):
    shift = STEP if button is not None and any(button in rect(a) for a in actors) else 0
    P = (rest[0] + shift * d, rest[1])
    front = rest[0] + 5 + shift if d > 0 else rest[0] - 1 - shift
    keep = {p for p in line if (p[0] >= front if d > 0 else p[0] <= front) or p[1] != rest[1] + 2 and
            not (rest[1] <= p[1] < rest[1] + 5)}
    full = keep | plug_cells(P[0], P[1], d)
    occ = set()
    for a in actors:
        occ |= rect(a)
    return full - occ


def field_ok(p, rest, d, line, button):
    if button is None:
        return 0 <= p[0] <= 59 and 0 <= p[1] <= 58
    rows = [q[1] for q in line if q[1] == rest[1] + 2]
    lx = rest[0] if d > 0 else rest[0] - 0
    x0, x1 = (lx, button[0] + 2) if d > 0 else (button[0] - 2, rest[0] + 4)
    y0, y1 = button[1] - 2, rest[1] + 1
    return x0 <= p[0] and p[0] + 4 <= x1 and y0 <= p[1] and p[1] + 4 <= y1


def obj(name, typ, tags, layer, cellset, colour_map=None):
    xs = [c[0] for c in cellset]
    ys = [c[1] for c in cellset]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x + 1, max(ys) - y + 1
    px = [[-1] * w for _ in range(h)]
    for (cx, cy) in cellset:
        px[cy - y][cx - x] = colour_map(cx, cy) if colour_map else 8
    return {'name': name, 'type': typ, 'tags': tags, 'x': x, 'y': y, 'w': w, 'h': h, 'layer': layer,
            'visible': True, 'pixels': px}


def transition_function(state, action):
    aid = action['action_id'] if isinstance(action, dict) else action
    player = next(o for o in state if o['type'] == 'player')
    ghost = next((o for o in state if o['type'] == 'ghost'), None)
    counter = next((o for o in state if o['type'] == 'counter'), None)
    walls = [o for o in state if o['type'] == 'wall']
    ghud = [o for o in state if o['type'] == 'hud' and any(v == 2 for r in o['pixels'] for v in r)]
    pos = (player['x'], player['y'])
    gpos = (ghost['x'], ghost['y']) if ghost else None
    if MEM.get('out') == canon(state):
        n, path, rec, m = MEM['n'], list(MEM['path']), MEM['rec'], MEM['m']
    else:
        n = 2 * (64 - (counter['w'] if counter else 64)) + 1
        m = 0 if pos == SPAWN else 1
        path = [SPAWN] if pos == SPAWN else [SPAWN, pos]
        rec = ([gpos] if gpos else [SPAWN]) if ghud else None
        if gpos:
            m = 0
    actors = [pos] + ([gpos] if gpos else [])
    W = set()
    for o in walls:
        W |= cells(o)
    plug, d = find_plug(W) if W else (None, 1)
    if plug:
        rest, line, button = parse_wall(W, plug, d, actors)
    n += 1
    if aid in DIRS:
        dx, dy = DIRS[aid]
        dest = (pos[0] + dx * STEP, pos[1] + dy * STEP)
        ok = dest not in HIDDEN_BLOCK
        if plug:
            ok = ok and field_ok(dest, rest, d, line, button) and not (rect(dest) & plug_cells(plug[0], plug[1], d))
        if ok:
            pos, m = dest, m + 1
            path.append(dest)
    elif aid == 5 and m > 0:
        rec = None if rec else path
        pos, m, path = SPAWN, 0, [SPAWN]
    gpos = rec[min(m, len(rec) - 1)] if rec else None
    if gpos == pos:
        gpos = None
    keep = [o for o in state if o['type'] not in ('ghost', 'wall') and o not in ghud]
    out = []
    for o in keep:
        o = dict(o)
        if o['type'] == 'player':
            o['x'], o['y'] = pos
        elif o['type'] == 'hud':
            o['x'] = 1 + (4 if rec else 0)
        elif o['type'] == 'counter':
            w = 64 - n // 2
            o['w'], o['pixels'] = w, [[9] * w]
        out.append(o)
    k = len(out)
    extra = []
    if rec:
        extra.append({'name': '', 'type': 'hud', 'tags': ['hud'], 'x': 1, 'y': 1, 'w': 3, 'h': 3, 'layer': 0,
                      'visible': True, 'pixels': [[2, 2, 2], [2, -1, 2], [2, 2, 2]]})
    if gpos:
        extra.append({'name': '', 'type': 'ghost', 'tags': ['ghost', 'snake_tail', 'follower'], 'x': gpos[0],
                      'y': gpos[1], 'w': 5, 'h': 5, 'layer': 1, 'visible': True,
                      'pixels': [[2 if (i, j) != (2, 2) else -1 for i in range(5)] for j in range(5)]})
    if plug:
        vis = build_wall(rest, d, line, button, [pos] + ([gpos] if gpos else []))
        if vis:
            extra.append(obj('', 'wall', list(walls[0]['tags']), 0, vis))
    for o in extra:
        o['name'] = '%s_%03d' % (o['type'], k)
        k += 1
    out += extra
    MEM.update(out=canon(out), n=n, path=path, rec=rec, m=m)
    return out
