# Mechanics: arrows move the player 3 cells; player floor = colours {1,4}; if its next rect overlaps a block it KICKS
# (player stays): the chain slides in 3-cell ticks, absorbing blocks it hits, stopped by bounds, non-{1,4,15} floor,
# the player, or a socket interior (bbox inset 1) it does not fit. Hidden bound: colour 15 is ice; once a chain's leading
# line has been on 15, it slides only 3 more ticks (9 cells) past the band (the y=15 stop). Click on a block = possess it,
# old player left as husk (centre 4). HUD row 63: (n+1)//2 zeros from the right. Unconfirmed: friction rule vs socket line.

STEP = 3
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
ICE = 15
SLIDE_AFTER_ICE = 3
_memo = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d, k=STEP):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def cells(r):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            yield x, y


def in_bounds(r, bound):
    return r[0] >= bound[0] and r[1] >= bound[1] and r[0] + r[2] <= bound[0] + bound[2] and r[1] + r[3] <= bound[1] + bound[3]


def interior(s):
    return (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)


class World:
    def __init__(self, state, frame):
        self.frame = frame
        self.player = None
        self.blocks = []
        self.sockets = []
        self.bound = (0, 0, 63, 63)
        for o in state:
            t = o.get("type")
            if t == "player":
                self.player = rect(o)
            elif t == "block":
                self.blocks.append(rect(o))
            elif t == "target":
                self.sockets.append(rect(o))
            elif t == "portal":
                self.bound = rect(o)
        self.covered = set()
        for r in self.blocks + ([self.player] if self.player else []):
            self.covered.update(cells(r))
        self.bg = [row[:] for row in frame]
        for (x, y) in self.covered:
            if 0 <= x < 64 and 0 <= y < 63:
                self.bg[y][x] = self.terrain(x, y)

    # background under movers
    def socket_cell(self, x, y):
        for s in self.sockets:
            if overlap((x, y, 1, 1), s):
                return 1 if overlap((x, y, 1, 1), interior(s)) else 4
        return None

    def free(self, x, y):
        return 0 <= x < 64 and 0 <= y < 63 and (x, y) not in self.covered

    def scan(self, x, y, dx, dy):
        x, y = x + dx, y + dy
        while 0 <= x < 64 and 0 <= y < 63:
            if (x, y) not in self.covered:
                return x, y
            x, y = x + dx, y + dy
        return None

    def terrain(self, x, y):
        s = self.socket_cell(x, y)
        if s is not None:
            return s
        f = self.frame
        L, R, U, D = (self.scan(x, y, -1, 0), self.scan(x, y, 1, 0),
                      self.scan(x, y, 0, -1), self.scan(x, y, 0, 1))
        val = lambda p: None if p is None else f[p[1]][p[0]]
        for a, b in ((L, R), (U, D)):
            if a and b and val(a) == val(b):
                return val(a)
        if L and R:
            v = self.continue_line(x, y, L, R, horizontal=True)
            if v is not None:
                return v
        if U and D:
            v = self.continue_line(x, y, U, D, horizontal=False)
            if v is not None:
                return v
        for p in (L, R, U, D):
            if p:
                return val(p)
        return 1

    def continue_line(self, x, y, a, b, horizontal):
        f = self.frame
        for k in range(1, 64):
            for o in (-k, k):
                if horizontal:
                    yy = y + o
                    if not 0 <= yy < 63:
                        continue
                    if all(self.free(xx, yy) for xx in range(a[0], b[0] + 1)) and \
                            f[yy][a[0]] == f[a[1]][a[0]] and f[yy][b[0]] == f[b[1]][b[0]]:
                        return f[yy][x]
                else:
                    xx = x + o
                    if not 0 <= xx < 64:
                        continue
                    if all(self.free(xx, yy) for yy in range(a[1], b[1] + 1)) and \
                            f[a[1]][xx] == f[a[1]][a[0]] and f[b[1]][xx] == f[b[1]][b[0]]:
                        return f[y][xx]
        return None

    # guards
    def floor_ok(self, r, floor):
        return all(self.bg[y][x] in floor for x, y in cells(r))

    def player_blocked(self, r):
        return not in_bounds(r, self.bound) or not self.floor_ok(r, PLAYER_FLOOR)

    def socket_rejects(self, r):
        for s in self.sockets:
            i = interior(s)
            if overlap(r, i) and r != i:
                return True
        return False

    def block_blocked(self, r):
        return (not in_bounds(r, self.bound) or not self.floor_ok(r, BLOCK_FLOOR)
                or overlap(r, self.player) or self.socket_rejects(r))

    def on_ice(self, r, d):
        if d[0] < 0:
            line = (r[0], r[1], 1, r[3])
        elif d[0] > 0:
            line = (r[0] + r[2] - 1, r[1], 1, r[3])
        elif d[1] < 0:
            line = (r[0], r[1], r[2], 1)
        else:
            line = (r[0], r[1] + r[3] - 1, r[2], 1)
        return any(self.bg[y][x] == ICE for x, y in cells(line))

    # interactions
    def push_chain(self, members, d):
        iced, left = False, SLIDE_AFTER_ICE
        while True:
            while True:
                moved = [shift(self.blocks[i], d) for i in members]
                hit = [j for j, b in enumerate(self.blocks)
                       if j not in members and any(overlap(m, b) for m in moved)]
                if not hit:
                    break
                members = members | set(hit)
            if any(self.block_blocked(m) for m in moved):
                return
            for i in members:
                self.blocks[i] = shift(self.blocks[i], d)
            if any(self.on_ice(self.blocks[i], d) for i in members):
                iced = True
            elif iced:
                left -= 1
                if left == 0:
                    return

    def move(self, action):
        d = DIRS[action]
        nxt = shift(self.player, d)
        hit = {i for i, b in enumerate(self.blocks) if overlap(nxt, b)}
        if hit:
            self.push_chain(hit, d)
        elif not self.player_blocked(nxt):
            self.player = nxt

    def click(self, x, y):
        for i, b in enumerate(self.blocks):
            if overlap((x, y, 1, 1), b):
                old = self.player
                for (cx, cy) in centre_cells(old):
                    self.bg[cy][cx] = 4
                for (px, py) in cells(old):
                    if (px, py) not in centre_cells(old):
                        self.bg[py][px] = self.frame[py][px]
                self.player = b
                del self.blocks[i]
                return

    def render(self, zeros):
        out = [row[:] for row in self.bg]
        for b in self.blocks:
            draw_token(out, b, 5)
        draw_token(out, self.player, 0)
        p = self.player
        for b in self.blocks:
            yspan = b[1] < p[1] + p[3] and p[1] < b[1] + b[3]
            xspan = b[0] < p[0] + p[2] and p[0] < b[0] + b[2]
            if yspan and b[0] + b[2] == p[0]:
                paint(out, (p[0], p[1], 1, p[3]), 0)
            if yspan and b[0] == p[0] + p[2]:
                paint(out, (p[0] + p[2] - 1, p[1], 1, p[3]), 0)
            if xspan and b[1] + b[3] == p[1]:
                paint(out, (p[0], p[1], p[2], 1), 0)
            if xspan and b[1] == p[1] + p[3]:
                paint(out, (p[0], p[1] + p[3] - 1, p[2], 1), 0)
        for x in range(64 - zeros, 64):
            out[63][x] = 0
        return out


def centre_cells(r):
    xs = range(r[0] + (r[2] - 1) // 2, r[0] + r[2] // 2 + 1)
    ys = range(r[1] + (r[3] - 1) // 2, r[1] + r[3] // 2 + 1)
    return [(x, y) for y in ys for x in xs]


def paint(out, r, c):
    for x, y in cells(r):
        if 0 <= x < 64 and 0 <= y < 64:
            out[y][x] = c


def draw_token(out, r, centre):
    paint(out, r, 14)
    for x, y in centre_cells(r):
        out[y][x] = centre


def hud_zeros(frame):
    z = 0
    for x in range(63, -1, -1):
        if frame[63][x] != 0:
            break
        z += 1
    return z


def transition_function(state, action, frame):
    frame = [[int(v) for v in row] for row in frame]
    if _memo["frame"] == frame:
        n = _memo["n"]
    else:
        z = hud_zeros(frame)
        n = 2 * z - 1 if z > 0 else 0
    w = World(state, frame)
    if w.player is not None:
        if isinstance(action, dict):
            if action.get("action_id") == 6:
                w.click(action.get("x", -1), action.get("y", -1))
        elif action in DIRS:
            w.move(action)
    n += 1
    out = w.render((n + 1) // 2)
    _memo["frame"] = [row[:] for row in out]
    _memo["n"] = n
    return out
