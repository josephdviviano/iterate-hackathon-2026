# Mechanics: player (5x5) moves 6 on a 6-cell grid (A1 up,A2 down,A3 left,A4 right), blocked by grid cells holding wall pixels
# (except the switch cell) and bounds x>=14,y>=8. Wall switch (3x3 cap) pressed by player/ghost slides the far-end door block
# one cell toward the wall. A5: no-op if no move since spawn; else if a ghost is recorded -> clear it; else record the trail as a
# ghost (red HUD ring, white ring shifts right), respawn at (14,8); ghost replays the trail one position per successful move.
# Counter shrinks 1 every other action. Unconfirmed: first action (counter full) is a no-op; spawn=(14,8); ghost capacity 1.
import json

G, OX, OY, LEFT, TOP, CAP = 6, 2, 2, 14, 8, 1
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_H = {}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def cells_of(o):
    return {(o['x'] + i, o['y'] + j) for j, r in enumerate(o['pixels']) for i, v in enumerate(r) if v >= 0}


def cell(p):
    return ((p[0] - OX) // G, (p[1] - OY) // G)


def region(c):
    return {(OX + c[0] * G + i, OY + c[1] * G + j) for i in range(G) for j in range(G)}


def shift(cs, d):
    return {(x + d[0] * G, y + d[1] * G) for x, y in cs}


def find_switch(W):
    sq = [(x, y) for x, y in W if all((x + i, y + j) in W for i in range(3) for j in range(3))]
    nbh = lambda x, y: sum((x + i, y + j) in W for i in range(-1, 4) for j in range(-1, 4))
    sq = [s for s in sq if nbh(*s) <= 12]
    return cell((sq[0][0] + 1, sq[0][1] + 1)) if sq else None


def chain_info(W, sw):
    wc = {cell(p) for p in W} | {sw}
    dist, q = {sw: 0}, [sw]
    for c in q:
        for d in MOVES.values():
            n = (c[0] + d[0], c[1] + d[1])
            if n in wc and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    door = max(dist, key=lambda c: (dist[c], c))
    nb = [n for n in dist if dist[n] == dist[door] - 1 and abs(n[0] - door[0]) + abs(n[1] - door[1]) == 1]
    dd = (nb[0][0] - door[0], nb[0][1] - door[1]) if nb else (0, 0)
    return door, dd


def press(W, door, dd):
    R = region(door)
    return (W - R) | shift(W & R, dd)


def unpress(W, door_now, dd):
    R = region(door_now)
    back = shift(W & R, (-dd[0], -dd[1]))
    nxt = region((door_now[0] + dd[0], door_now[1] + dd[1]))
    fill = set()
    for p in R:
        if dd[0] > 0 and (OX + (door_now[0] + 1) * G, p[1]) in W:
            fill.add(p)
        elif dd[0] < 0 and (OX + door_now[0] * G - 1, p[1]) in W:
            fill.add(p)
        elif dd[1] > 0 and (p[0], OY + (door_now[1] + 1) * G) in W:
            fill.add(p)
        elif dd[1] < 0 and (p[0], OY + door_now[1] * G - 1) in W:
            fill.add(p)
    return (W - R) | back | fill


def parse(state):
    m = {'base': [], 'rings': 0, 'ghost': None, 'W': set(), 'occ': []}
    for o in state:
        t = o['type']
        if t == 'wall':
            m['W'] |= cells_of(o)
            m['wall_proto'] = o
        elif t == 'ghost':
            m['ghost'] = (o['x'], o['y'])
            m['ghost_proto'] = o
        elif t == 'hud' and o['h'] == 3 and o['pixels'][0][0] == 2:
            m['rings'] += 1
            m['ring_proto'] = o
        else:
            m['base'].append(o)
            if t == 'player':
                m['player'] = (o['x'], o['y'])
            if t == 'counter':
                m['cw'] = o['w']
    return m


def stateless(m):
    p = m['player']
    occ = [cell(p)] + ([cell(m['ghost'])] if m['ghost'] else [])
    W = m['W']
    sw = find_switch(W)
    pressed = False
    if sw is None:
        wc = {cell(q) for q in W}
        cand = [c for c in occ if any((c[0] + d[0], c[1] + d[1]) in wc for d in MOVES.values())]
        sw = cand[0] if cand else (None if not occ else occ[0])
        pressed = True
    else:
        pressed = sw in occ
    door, dd = chain_info(W, sw)
    home = unpress(W, door, dd) if pressed else W
    if pressed:
        door = (door[0] - dd[0], door[1] - dd[1])
    rec = []
    if m['rings']:
        rec = [{'trail': [m['ghost']] if m['ghost'] else [], 'k': 1 if m['ghost'] else 0}]
    return {'n': 2 * (64 - m.get('cw', 64)), 'spawn': (LEFT, TOP), 'trail': [p],
            'moved': p != (LEFT, TOP), 'rec': rec, 'home': home, 'sw': sw, 'door': door, 'dd': dd}


def ghost_pos(g):
    if g['k'] < 1 or not g['trail']:
        return None
    return g['trail'][min(g['k'] - 1, len(g['trail']) - 1)]


def wall_now(h, occ):
    if h['sw'] in occ:
        return press(h['home'], h['door'], h['dd'])
    return h['home']


def step(h, action, cw):
    aid = action['action_id'] if isinstance(action, dict) else action
    first = cw >= 64
    h['n'] += 1
    p = h['trail'][-1]
    if first:
        return
    if aid in MOVES:
        d = MOVES[aid]
        q = (p[0] + d[0] * G, p[1] + d[1] * G)
        occ = [cell(p)] + [cell(g) for g in map(ghost_pos, h['rec']) if g]
        W = wall_now(h, occ)
        c = cell(q)
        ok = q[0] >= LEFT and q[1] >= TOP and q[0] + 5 <= 64 and q[1] + 5 <= 63
        ok = ok and (c == h['sw'] or not (W & region(c)))
        if ok:
            h['trail'].append(q)
            h['moved'] = True
            for g in h['rec']:
                g['k'] += 1
    elif aid == 5 and h['moved']:
        if len(h['rec']) >= CAP:
            h['rec'] = []
        else:
            h['rec'].append({'trail': list(h['trail']), 'k': 0})
        h['trail'] = [h['spawn']]
        h['moved'] = False


def render(h, m, cw_new):
    p = h['trail'][-1]
    gps = [g for g in map(ghost_pos, h['rec']) if g]
    occ = [cell(p)] + [cell(g) for g in gps]
    W = wall_now(h, occ)
    nr = len(h['rec'])
    out = []
    for o in m['base']:
        o = dict(o)
        if o['type'] == 'player':
            o['x'], o['y'] = p
        elif o['type'] == 'hud':
            o['x'] = 1 + 4 * nr
        elif o['type'] == 'counter':
            o['w'], o['pixels'] = cw_new, [[9] * cw_new]
        out.append(o)
    ring = m.get('ring_proto') or next(o for o in m['base'] if o['type'] == 'hud' and o['h'] == 3)
    for i in range(nr):
        r = dict(ring)
        r['x'], r['y'] = 1 + 4 * i, ring['y']
        r['pixels'] = [[2 if v >= 0 else v for v in row] for row in ring['pixels']]
        out.append(r)
    pl = next(o for o in m['base'] if o['type'] == 'player')
    for g in gps:
        o = dict(m.get('ghost_proto') or pl)
        o.update(type='ghost', tags=['ghost', 'snake_tail', 'follower'], x=g[0], y=g[1], layer=pl['layer'],
                 w=pl['w'], h=pl['h'], pixels=[[2 if v >= 0 else v for v in row] for row in pl['pixels']])
        out.append(o)
    hidden = set()
    for q in [p] + gps:
        hidden |= {(q[0] + i, q[1] + j) for i in range(pl['w']) for j in range(pl['h'])}
    V = W - hidden
    if V and 'wall_proto' in m:
        x0, y0 = min(x for x, _ in V), min(y for _, y in V)
        x1, y1 = max(x for x, _ in V), max(y for _, y in V)
        o = dict(m['wall_proto'])
        o.update(x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1,
                 pixels=[[8 if (x, y) in V else -1 for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)])
        out.append(o)
    for i, o in enumerate(out):
        o['name'] = '%s_%03d' % (o['type'], i)
    return out


def transition_function(state, action):
    m = parse(state)
    cw = m.get('cw', 64)
    h = _H.get('h') if _H.get('last') == canon(state) else None
    if h is None:
        h = stateless(m)
    dec = h['n'] % 2 == 0
    step(h, action, cw)
    out = render(h, m, max(0, cw - 1) if dec else cw)
    _H['h'], _H['last'] = h, canon(out)
    return json.loads(json.dumps(out))
