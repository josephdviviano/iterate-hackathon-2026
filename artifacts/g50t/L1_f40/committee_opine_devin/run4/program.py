# Mechanics: floor 5, void/walls 0, rope 8; player 9 / ghost 2 = 5x5 rings, centre drawn 5; arrows move 6 iff the
# dest box is all floor (8 allowed when the box holds the plate centre). Plate = topmost 3x3 of 8; while a body covers
# it, the 7x7 key tile at the rope's far end slides 6 toward the plate column (vacated cols copy the outer column).
# A5 off spawn toggles record mode: player back to spawn, HUD swaps; in record mode the ghost trails 2 moves behind
# (trail starts [level start, spawn]); counter row 63 gains a 1 from the right on even calls. Unconfirmed: A5 at spawn in off mode.
FLOOR, ROPE, VOID, PCOL, GCOL, SPENT = 5, 8, 0, 9, 2, 1
B = 5
DIRS = {1: (0, -6), 2: (0, 6), 3: (-6, 0), 4: (6, 0)}
MEM = {}


def copyf(f):
    return [list(map(int, r)) for r in f]


def find_plate(S):
    for y in range(62):
        for x in range(62):
            if all(S[y + j][x + i] == ROPE for i in range(3) for j in range(3)):
                return (x + 1, y + 1)
    return None


def find_key(S, plate):
    best, arm = 0, None
    for y in range(64):
        run = 0
        for x in range(64):
            run = run + 1 if S[y][x] == ROPE else 0
            if run > best and not (plate and abs(y - plate[1]) <= 1):
                best, arm = run, y
    if arm is None or plate is None:
        return None
    xs = [x for y in range(arm - 2, arm + 3) if 0 <= y < 64 for x in range(64) if S[y][x] == ROPE]
    lo, hi = min(xs), max(xs)
    if abs(lo - plate[0]) >= abs(hi - plate[0]):
        return (lo - 1, arm, 6)
    return (hi - 5, arm, -6)


def door_open(S, key):
    G = copyf(S)
    if key is None:
        return G
    x0, arm, d = key
    rows = [y for y in range(arm - 3, arm + 4) if 0 <= y < 64]
    for y in rows:
        fill = S[y][x0] if d > 0 else S[y][x0 + 6]
        for i in range(7):
            if 0 <= x0 + i < 64:
                G[y][x0 + i] = fill
        for i in range(7):
            if 0 <= x0 + i + d < 64 and 0 <= x0 + i < 64:
                G[y][x0 + i + d] = S[y][x0 + i]
    return G


def covers(pos, pt):
    return pos is not None and pt is not None and pos[0] <= pt[0] < pos[0] + B and pos[1] <= pt[1] < pos[1] + B


def body_pos(state, typ):
    for o in state:
        if o.get('type') == typ and o.get('visible', True):
            return (o['x'], o['y'])
    return None


def erase(frame, positions):
    S = copyf(frame)
    for p in positions:
        if p:
            for j in range(B):
                for i in range(B):
                    if 0 <= p[1] + j < 64 and 0 <= p[0] + i < 64:
                        S[p[1] + j][p[0] + i] = FLOOR
    return S


def find_spawn(S, ref):
    for y in range(ref[1] % 6, 60, 6):
        for x in range(ref[0] % 6, 60, 6):
            if all(S[y + j][x + i] == FLOOR for i in range(B) for j in range(B)):
                return (x, y)
    return ref


def fallback(state, frame):
    p = body_pos(state, 'player')
    g = body_pos(state, 'ghost')
    S = erase(frame, [p, g])
    if p and p[1] + B < 64 and S[p[1] + B][p[0] + 2] == ROPE:
        cx, cy = p[0] + 2, p[1] + 2
        for j in range(-1, 2):
            for i in range(-1, 2):
                S[cy + j][cx + i] = ROPE
        S[cy + 2][cx] = ROPE
    ones = sum(1 for v in frame[63] if v == SPENT)
    rec = frame[1][1] == GCOL
    spawn = find_spawn(S, p)
    trail = ([g, p] if g else [p]) if rec else []
    return dict(S=S, n=max(0, 2 * ones - 1), rec=rec, trail=trail, start=p if ones == 0 else spawn,
                spawn=spawn, pos=p)


def ring(G, pos, col):
    for j in range(B):
        for i in range(B):
            y, x = pos[1] + j, pos[0] + i
            if 0 <= y < 64 and 0 <= x < 64:
                G[y][x] = FLOOR if (i == 2 and j == 2) else col


def can_enter(base, pos, plate):
    x, y = pos
    if x < 0 or y < 0 or x + B > 64 or y + B > 63:
        return False
    on_plate = covers(pos, plate)
    for j in range(B):
        for i in range(B):
            v = base[y + j][x + i]
            if v != FLOOR and not (on_plate and v == ROPE):
                return False
    return True


def draw_hud(G, rec):
    for j in range(3):
        for i in range(3):
            G[1 + j][1 + i] = VOID if (i == 1 and j == 1) else (GCOL if rec else PCOL)
            G[1 + j][5 + i] = (VOID if (i == 1 and j == 1) else PCOL) if rec else SPENT
    for i in range(3):
        G[5][1 + i] = VOID if rec else PCOL
        G[5][5 + i] = PCOL if rec else VOID


def ghost_of(m):
    return m['trail'][-3] if m['rec'] and len(m['trail']) >= 3 else None


def transition_function(state, action, frame):
    global MEM
    m = MEM.get('m')
    if m is None or MEM.get('last') != frame:
        m = fallback(state, frame)
    m = dict(m, trail=list(m['trail']))
    S = m['S']
    plate = find_plate(S)
    key = find_key(S, plate)
    p = m['pos']
    pressed = covers(p, plate) or covers(ghost_of(m), plate)
    base = door_open(S, key) if pressed else S
    aid = action.get('action_id') if isinstance(action, dict) else action
    changed = False
    if aid in DIRS:
        q = (p[0] + DIRS[aid][0], p[1] + DIRS[aid][1])
        if can_enter(base, q, plate):
            p = q
            if m['rec']:
                m['trail'].append(q)
            changed = True
    elif aid == 5 and p != m['spawn']:
        m['rec'] = not m['rec']
        p = m['spawn']
        m['trail'] = [m['start'], m['spawn']] if m['rec'] else []
        changed = True
    m['pos'] = p
    G = copyf(frame)
    if changed:
        g = ghost_of(m)
        pressed = covers(p, plate) or covers(g, plate)
        G = door_open(S, key) if pressed else copyf(S)
        for r in (0, 63):
            G[r] = list(frame[r])
        for y in range(8):
            G[y][:9] = list(frame[y][:9])
        draw_hud(G, m['rec'])
        if g:
            ring(G, g, GCOL)
        ring(G, p, PCOL)
    if m['n'] % 2 == 0:
        xs = [x for x in range(64) if G[63][x] == PCOL]
        if xs:
            G[63][max(xs)] = SPENT
    m['n'] += 1
    MEM = {'m': m, 'last': copyf(G)}
    return G
