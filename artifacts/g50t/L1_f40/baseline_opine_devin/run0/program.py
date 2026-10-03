# Mechanics: 5x5 player steps 6 cells (A1-4); a move succeeds iff the dest box is all floor (5), or floor/rope if it holds the plate.
# Rope (8): plate atop a line, arm row ending in a key; player/ghost on plate -> key+1-cell halo slide 6 toward line, plate=floor.
# A5 off spawn (top-left fitting lattice cell): toggles ghost mode (record this life's path / clear), respawns, redraws HUD;
#   the ghost (colour 2) retraces the recorded path one step per move; counter row 63: rightmost unspent cell -> 1 on odd calls.
# Unconfirmed: recorded-path vs lag-2 follower ghost (both fit), whether the ghost blocks, A6/A7 effects; stateless fallbacks are guesses.
FLOOR, ROPE, GHOST, SLOT, SPENT, STEP, SZ = 5, 8, 2, 1, 1, 6, 5
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
MEM = {}


def copyf(f):
    return [list(r) for r in f]


def sprites(state):
    out = {}
    for o in state:
        if o.get('type') in ('player', 'ghost') and o.get('visible', True):
            out[o['type']] = o
    return out


def inside(px, py, pos):
    return pos is not None and pos[0] <= px < pos[0] + SZ and pos[1] <= py < pos[1] + SZ


class Rope:
    """Plate / line / arm / key parsed from rope-coloured cells of a background with sprites erased."""

    def __init__(self, B, covers):
        cells = {(x, y) for y in range(63) for x in range(64) if B[y][x] == ROPE}
        self.ok = bool(cells)
        if not self.ok:
            return
        self.line = max(range(64), key=lambda x: sum((x, y) in cells for y in range(63)))
        self.arm = max(range(63), key=lambda y: sum((x, y) in cells for x in range(64)))
        lo = hi = self.arm
        while (self.line, lo - 1) in cells:
            lo -= 1
        while (self.line, hi + 1) in cells:
            hi += 1
        up = abs(lo - self.arm) >= abs(hi - self.arm)
        end = lo if up else hi
        s = -1 if up else 1
        self.plate = (self.line, end - s)
        for c in covers:  # plate hidden under a sprite whose edge touches the line end
            if c and inside(self.line, end + s, c):
                self.plate = (c[0] + 2, c[1] + 2)
                for y in range(min(end, self.plate[1]), max(end, self.plate[1]) + 1):
                    B[y][self.line] = ROPE
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        B[self.plate[1] + dy][self.plate[0] + dx] = ROPE
        a = b = self.line
        while (a - 1, self.arm) in cells:
            a -= 1
        while (b + 1, self.arm) in cells:
            b += 1
        far = a if abs(a - self.line) >= abs(b - self.line) else b
        self.dir = 1 if self.line > far else -1
        x0 = far if self.dir == 1 else far - SZ + 1
        self.key = (x0 - 1, self.arm - 3, SZ + 2, SZ + 2)  # key bbox grown by 1

    def plate_cells(self):
        px, py = self.plate
        return {(px + dx, py + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}

    def pressed_by(self, positions):
        return any(inside(self.plate[0], self.plate[1], p) for p in positions)

    def slide(self, B, sign, fill_beyond):
        x0, y0, w, h = self.key
        d = STEP * self.dir * sign
        old = [[B[y][x] for x in range(x0, x0 + w)] for y in range(y0, y0 + h)]
        for j in range(h):
            for i in range(w):
                x = x0 + i
                if fill_beyond:
                    bx = x0 + w if self.dir == 1 else x0 - 1
                    B[y0 + j][x] = B[y0 + j][bx]
                else:
                    B[y0 + j][x] = FLOOR
        for j in range(h):
            for i in range(w):
                B[y0 + j][x0 + i + d] = old[j][i]

    def press(self, B0):
        B = copyf(B0)
        self.slide(B, 1, False)
        for x, y in self.plate_cells():
            B[y][x] = FLOOR
        return B


def unpressed_background(frame, spr):
    B = copyf(frame)
    covers = []
    for o in spr.values():
        covers.append((o['x'], o['y']))
        for y in range(o['y'], o['y'] + o['h']):
            for x in range(o['x'], o['x'] + o['w']):
                B[y][x] = FLOOR
    rope = Rope(B, covers)
    if rope.ok and rope.pressed_by(covers):
        rope.slide(B, -1, True)
    return B, rope


def fits(B, x, y, plate):
    if x < 0 or y < 0 or x + SZ > 64 or y + SZ > 63:
        return False
    on_plate = any(inside(px, py, (x, y)) for px, py in plate)
    ok = (FLOOR, ROPE) if on_plate else (FLOOR,)
    return all(B[yy][xx] in ok for yy in range(y, y + SZ) for xx in range(x, x + SZ))


def spawn_cell(B0, ref, plate):
    for y in range(ref[1] % STEP, 64, STEP):
        for x in range(ref[0] % STEP, 64, STEP):
            if fits(B0, x, y, plate):
                return (x, y)
    return ref


def draw_sprite(F, B, pos, pixels):
    for j, row in enumerate(pixels):
        for i, v in enumerate(row):
            F[pos[1] + j][pos[0] + i] = B[pos[1] + j][pos[0] + i] if v < 0 else v


def body(colour):
    return [[-1 if (i, j) == (2, 2) else colour for i in range(SZ)] for j in range(SZ)]


def draw_hud(F, ghost_mode, pc):
    for y in range(1, 6):
        for x in range(1, 12):
            F[y][x] = 0
    ring = lambda x0, c: [F[y].__setitem__(x, 0 if (x - x0, y) == (1, 2) else c)
                          for y in range(1, 4) for x in range(x0, x0 + 3)]
    if ghost_mode:
        ring(1, GHOST)
        ring(5, pc)
        bx = 5
    else:
        ring(1, pc)
        for y in range(1, 4):
            for x in range(5, 8):
                F[y][x] = SLOT
        bx = 1
    for x in range(bx, bx + 3):
        F[5][x] = pc


def tick_counter(F, calls):
    if calls % 2 == 1:
        row = F[63]
        left = [x for x in range(64) if row[x] != SPENT]
        if left:
            row[max(left)] = SPENT


def transition_function(state, action, frame):
    spr = sprites(state)
    P = spr.get('player')
    if P is None:
        return copyf(frame)
    pc = max((v for r in P['pixels'] for v in r), default=9)
    pos = (P['x'], P['y'])
    ghost = (spr['ghost']['x'], spr['ghost']['y']) if 'ghost' in spr else None
    ghost_mode = frame[2][1] == GHOST or ghost is not None
    cont = MEM.get('out') == frame
    if not cont:
        zeros = sum(v == SPENT for v in frame[63])
        B0, rope = unpressed_background(frame, spr)
        MEM.clear()
        MEM.update(calls=0 if zeros == 0 else 2 * zeros - 1, B0=B0, life=[pos],
                   rec=None, k=0)
    B0 = MEM['B0']
    rope = Rope(copyf(B0), [])
    plate = rope.plate_cells() if rope.ok else set()
    MEM['calls'] += 1
    spawn = spawn_cell(B0, pos, plate)

    def bg(positions):
        return rope.press(B0) if rope.ok and rope.pressed_by(positions) else B0

    Bcur = bg([pos, ghost])
    if isinstance(action, int) and action in DIRS:
        dx, dy = DIRS[action]
        nx, ny = pos[0] + dx * STEP, pos[1] + dy * STEP
        if fits(Bcur, nx, ny, plate):
            pos = (nx, ny)
            MEM['life'].append(pos)
            if ghost_mode:
                MEM['k'] += 1
                rec = MEM['rec']
                if rec:
                    ghost = rec[min(MEM['k'] - 1, len(rec) - 1)]
    elif action == 5 and pos != spawn:
        if ghost_mode:
            ghost_mode, ghost, MEM['rec'] = False, None, None
        else:
            ghost_mode, MEM['rec'], MEM['k'] = True, MEM['life'], 0
        pos = spawn
        MEM['life'] = [pos]
    F = copyf(frame)
    Bnew = bg([pos, ghost])
    for y in range(7, 63):
        for x in range(64):
            F[y][x] = Bnew[y][x]
    if ghost is not None:
        draw_sprite(F, Bnew, ghost, body(GHOST))
    draw_sprite(F, Bnew, pos, P['pixels'])
    if ghost_mode != (frame[2][1] == GHOST):
        draw_hud(F, ghost_mode, pc)
    tick_counter(F, MEM['calls'])
    MEM['out'] = F
    return copyf(F)
