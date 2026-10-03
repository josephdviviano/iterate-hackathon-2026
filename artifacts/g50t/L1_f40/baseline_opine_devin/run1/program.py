# Mechanics: 5x5 player (9) steps 6 cells; a move succeeds iff the dest box is all floor (5) in the terrain
# (actors erased), rope (8) cells allowed only when the box holds the plate centre. Plate = topmost 3x3 rope square;
# any actor on it -> plate drawn as floor and the key tile (densest 5x5 rope window + 1 halo) slides 6 toward the rope column.
# A5 off spawn (spawn = first cell moved to): toggles ghost mode, player -> spawn; ghost (2) = player trail lag 2 from
# [level start, spawn], hidden on player; HUD icons shift +4 with a red copy of the first. Row 63 = (calls+1)//2 ones from right.
# Unconfirmed: win/gate behaviour, ghost blocking moves, multiple recordings, stateless trail/spawn (fallbacks only).
FLOOR, ROPE, PLAYER, GHOST, TICK = 5, 8, 9, 2, 1
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 6
MEM = {}


def copy(f):
    return [list(r) for r in f]


def hud_rows(frame):
    for y, row in enumerate(frame):
        if FLOOR in row:
            return y
    return 0


def box(p):
    return [(p[0] + i, p[1] + j) for j in range(5) for i in range(5)]


def find_ring(frame, colour, top):
    for y in range(top, 59):
        for x in range(60):
            if all(frame[y + j][x + i] == colour for j in range(5) for i in range(5) if (i, j) != (2, 2)):
                return (x, y)
    return None


def erase(frame, actors):
    t = copy(frame)
    for p in actors:
        if p:
            for x, y in box(p):
                t[y][x] = FLOOR
    return t


def rope_parts(t, top):
    cells = {(x, y) for y in range(top, 63) for x in range(64) if t[y][x] == ROPE}
    if not cells:
        return None
    plate = None
    for y in range(top, 61):
        for x in range(62):
            if all((x + i, y + j) in cells for i in range(3) for j in range(3)):
                plate = (x + 1, y + 1)
                break
        if plate:
            break
    best, key = -1, None
    for y in range(top, 59):
        for x in range(60):
            n = sum((x + i, y + j) in cells for i in range(5) for j in range(5))
            if n > best:
                best, key = n, (x, y)
    cols = {}
    for x, _ in cells:
        cols[x] = cols.get(x, 0) + 1
    line = max(cols, key=lambda c: cols[c])
    return {'n': len(cells), 'plate': plate, 'key': key, 'line': line}


def terrain(mem, pressed):
    base, parts = mem['base'], mem['parts']
    if not pressed or not parts or not parts['plate'] or not mem['clean']:
        return copy(base)
    t = copy(base)
    px, py = parts['plate']
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            t[py + j][px + i] = FLOOR
    kx, ky = parts['key']
    dx = STEP if parts['line'] > kx + 2 else -STEP
    tile = [(x, y) for y in range(ky - 1, ky + 6) for x in range(kx - 1, kx + 6)]
    vals = {(x, y): base[y][x] for x, y in tile}
    for x, y in tile:
        t[y][x] = FLOOR
    for (x, y), v in vals.items():
        if 0 <= x + dx < 64:
            t[y][x + dx] = v
    return t


def is_pressed(mem, actors):
    parts = mem['parts']
    if not parts or not parts['plate']:
        return False
    return any(p and parts['plate'] in box(p) for p in actors)


def can_move(t, dest, plate):
    on_plate = plate is not None and plate in box(dest)
    for x, y in box(dest):
        if not (0 <= x < 64 and 0 <= y < 63):
            return False
        v = t[y][x]
        if v != FLOOR and not (v == ROPE and on_plate):
            return False
    return True


def ghost_hud(base_hud):
    h = copy(base_hud)
    seen, comps = set(), []
    for y in range(len(h)):
        for x in range(64):
            if base_hud[y][x] == PLAYER and (x, y) not in seen:
                stack, comp = [(x, y)], []
                seen.add((x, y))
                while stack:
                    cx, cy = stack.pop()
                    comp.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < 64 and 0 <= ny < len(h) and (nx, ny) not in seen and base_hud[ny][nx] == PLAYER:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                comps.append(comp)
    for comp in comps:
        for x, y in comp:
            h[y][x] = 0
    comps.sort(key=lambda c: (min(p[1] for p in c), min(p[0] for p in c)))
    for comp in comps:
        xs, ys = [p[0] for p in comp], [p[1] for p in comp]
        for y in range(min(ys), max(ys) + 1):
            for x in range(min(xs), max(xs) + 1):
                if x + 4 < 64:
                    h[y][x + 4] = PLAYER if (x, y) in comp else 0
    if comps:
        for x, y in comps[0]:
            h[y][x] = GHOST
    return h


def parse(frame):
    top = hud_rows(frame)
    player = find_ring(frame, PLAYER, top)
    ghost = find_ring(frame, GHOST, top)
    ghost_mode = any(GHOST in frame[y] for y in range(top))
    ones = 0
    for x in range(63, -1, -1):
        if frame[63][x] != TICK:
            break
        ones += 1
    base = erase(frame, [player, ghost])
    parts = rope_parts(base, top)
    clean = not is_pressed({'parts': parts}, [player, ghost])
    hud = [list(frame[y]) for y in range(top)]
    if ghost_mode:
        hud = [[(r[x + 4] if x + 4 < 64 and r[x + 4] != GHOST else 0) for x in range(64)] for r in hud]
    st = {'top': top, 'player': player, 'ghost': ghost, 'ghost_mode': ghost_mode,
          'calls': 2 * ones - 1 if ones else 0, 'start': player, 'spawn': None,
          'trail': [p for p in (ghost, player) if p], 'base': base, 'parts': parts, 'clean': clean,
          'hud': hud, 'counter_bg': [v if v != TICK else PLAYER for v in frame[63]]}
    if st['spawn'] is None and player:
        st['spawn'] = top_left_cell(base, player, top)
    return st


def top_left_cell(t, p, top):
    for y in range(top + (p[1] - top) % STEP, 59, STEP):
        for x in range(p[0] % STEP, 60, STEP):
            if can_move(t, (x, y), None):
                return (x, y)
    return p


def step(st, action):
    st['calls'] += 1
    p = st['player']
    if p is None:
        return
    pressed = is_pressed(st, [p, st['ghost']])
    t = terrain(st, pressed)
    plate = st['parts']['plate'] if st['parts'] else None
    if isinstance(action, int) and action in DIRS:
        dx, dy = DIRS[action]
        dest = (p[0] + dx * STEP, p[1] + dy * STEP)
        if can_move(t, dest, plate):
            if not st.get('moved'):
                st['moved'] = True
                st['spawn'] = dest
            st['player'] = dest
            st['trail'].append(dest)
            if st['ghost_mode'] and len(st['trail']) >= 3:
                st['ghost'] = st['trail'][-3]
    elif action == 5 and st['spawn'] and p != st['spawn']:
        st['player'] = st['spawn']
        st['ghost_mode'] = not st['ghost_mode']
        st['ghost'] = None
        st['trail'] = [st['start'], st['spawn']] if st['ghost_mode'] else [st['spawn']]


def draw_ring(f, p, colour):
    for (x, y) in box(p):
        if (x - p[0], y - p[1]) != (2, 2):
            f[y][x] = colour


def render(st):
    g = st['ghost'] if st['ghost_mode'] else None
    if g == st['player']:
        g = None
    f = terrain(st, is_pressed(st, [st['player'], g]))
    hud = ghost_hud(st['hud']) if st['ghost_mode'] else st['hud']
    for y in range(st['top']):
        f[y] = list(hud[y])
    if g:
        draw_ring(f, g, GHOST)
    if st['player']:
        draw_ring(f, st['player'], PLAYER)
    ones = (st['calls'] + 1) // 2
    f[63] = [TICK if x >= 64 - ones else st['counter_bg'][x] for x in range(64)]
    return f


def transition_function(state, action, frame):
    st = MEM.get('st')
    if st is None or MEM.get('last') != frame:
        st = parse(frame)
    elif not st['clean']:
        fresh = parse(frame)
        if fresh['clean']:
            st.update(base=fresh['base'], parts=fresh['parts'], clean=True)
    if st['calls'] == 0 and st['player']:
        st['start'] = st['player']
        st['trail'] = [st['player']]
    step(st, action)
    out = render(st)
    MEM['st'], MEM['last'] = st, out
    return [[int(v) for v in r] for r in out]
