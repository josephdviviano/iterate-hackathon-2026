# Mechanics: arrows move the player 3 cells over floor colours {1,4}; if its next rect hits a block it
# kicks instead (player stays). Kicked chains slide 1-cell ticks over {1,4,15}, absorbing blocks they hit;
# chain momentum = 15 ticks per block in the chain (count of objects moving together), stop on wall colour,
# bounds, player, or a non-fitting socket interior. Click on a block: player takes its rect, old player stays
# as a husk (centre 4). HUD row 63: (n+1)//2 zeros, n = actions this level (continuity-gated; else n=2z-1).
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
TICKS_PER_BLOCK = 15
_memo = {"frame": None, "n": 0}


def rect(o):
    return [o["x"], o["y"], o["w"], o["h"]]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    for y in range(r[1], r[1] + r[3]):
        for x in range(r[0], r[0] + r[2]):
            yield x, y


def shifted(r, d, k=1):
    return [r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3]]


class World:
    def __init__(self, state, frame):
        self.frame = frame
        self.player = None
        self.blocks = []
        self.sockets = []
        self.bound = [0, 0, 63, 63]
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
        for r in self.movers():
            self.covered.update(cells(r))

    def movers(self):
        return ([self.player] if self.player else []) + self.blocks

    # ---- terrain under objects ----
    def socket_colour(self, x, y):
        for s in self.sockets:
            if s[0] <= x < s[0] + s[2] and s[1] <= y < s[1] + s[3]:
                edge = x in (s[0], s[0] + s[2] - 1) or y in (s[1], s[1] + s[3] - 1)
                return 4 if edge else 1
        return None

    def free(self, x, y):
        return 0 <= x < 64 and 0 <= y < 64 and (x, y) not in self.covered

    def nearest(self, x, y, dx, dy):
        x, y = x + dx, y + dy
        while 0 <= x < 64 and 0 <= y < 64:
            if (x, y) not in self.covered:
                return (x, y), self.frame[y][x]
            x, y = x + dx, y + dy
        return None, None

    def boundary_side(self, x, y, axis, a, b):
        # a, b: nearest free cells before/after (x,y) along axis; find where the colour boundary lies
        # in parallel free lines and extend it across the covered span.
        (pa, ca), (pb, cb) = a, b
        lo, hi = (pa[1], pb[1]) if axis == "v" else (pa[0], pb[0])
        for dist in range(1, 64):
            for off in (-dist, dist):
                line = []
                for t in range(lo, hi + 1):
                    cx, cy = (x + off, t) if axis == "v" else (t, y + off)
                    if not self.free(cx, cy):
                        line = None
                        break
                    line.append(self.frame[cy][cx])
                if not line or line[0] != ca or line[-1] != cb:
                    continue
                k = next(i for i, v in enumerate(line) if v != ca)
                pos = y if axis == "v" else x
                return ca if pos < lo + k else cb
        return None

    def terrain(self, x, y):
        if (x, y) not in self.covered:
            return self.frame[y][x]
        s = self.socket_colour(x, y)
        if s is not None:
            return s
        L, R = self.nearest(x, y, -1, 0), self.nearest(x, y, 1, 0)
        U, D = self.nearest(x, y, 0, -1), self.nearest(x, y, 0, 1)
        if L[1] is not None and L[1] == R[1]:
            return L[1]
        if U[1] is not None and U[1] == D[1]:
            return U[1]
        for axis, a, b in (("v", U, D), ("h", L, R)):
            if a[0] is not None and b[0] is not None:
                c = self.boundary_side(x, y, axis, a, b)
                if c is not None:
                    return c
        vals = [v for _, v in (L, R, U, D) if v is not None]
        return max(set(vals), key=vals.count) if vals else 1

    # ---- guards ----
    def in_bounds(self, r):
        b = self.bound
        return r[0] >= b[0] and r[1] >= b[1] and r[0] + r[2] <= b[0] + b[2] and r[1] + r[3] <= b[1] + b[3]

    def player_blocked(self, r):
        return not self.in_bounds(r) or any(self.terrain(x, y) not in PLAYER_FLOOR for x, y in cells(r))

    def socket_rejects(self, r):
        for s in self.sockets:
            inner = [s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2]
            if inner[2] > 0 and inner[3] > 0 and overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
                return True
        return False

    def block_blocked(self, r):
        if not self.in_bounds(r) or self.socket_rejects(r):
            return True
        if self.player and overlap(r, self.player):
            return True
        return any(self.terrain(x, y) not in BLOCK_FLOOR for x, y in cells(r))

    # ---- updates ----
    def push_chain(self, seed, d):
        chain = list(seed)
        budget = TICKS_PER_BLOCK * len(chain)
        while True:
            grew = True
            while grew:
                grew = False
                for i, b in enumerate(self.blocks):
                    if i not in chain and any(overlap(shifted(self.blocks[j], d), b) for j in chain):
                        chain.append(i)
                        budget += TICKS_PER_BLOCK
                        grew = True
            if budget <= 0 or any(self.block_blocked(shifted(self.blocks[j], d)) for j in chain):
                return
            for j in chain:
                self.blocks[j] = shifted(self.blocks[j], d)
            budget -= 1

    def arrow(self, d):
        if not self.player:
            return
        nxt = shifted(self.player, d, 3)
        hit = [i for i, b in enumerate(self.blocks) if overlap(nxt, b)]
        if hit:
            self.push_chain(hit, d)
        elif not self.player_blocked(nxt):
            self.player = nxt

    def click(self, x, y):
        for i, b in enumerate(self.blocks):
            if b[0] <= x < b[0] + b[2] and b[1] <= y < b[1] + b[3]:
                husk = self.player
                self.player = self.blocks.pop(i)
                return husk
        return None


def centre_cells(r):
    xs = range(r[0] + (r[2] - 1) // 2, r[0] + r[2] // 2 + 1)
    ys = range(r[1] + (r[3] - 1) // 2, r[1] + r[3] // 2 + 1)
    return [(x, y) for y in ys for x in xs]


def touching_sides(p, blocks):
    sides = set()
    for b in blocks:
        vov = p[1] < b[1] + b[3] and b[1] < p[1] + p[3]
        hov = p[0] < b[0] + b[2] and b[0] < p[0] + p[2]
        if vov and b[0] + b[2] == p[0]:
            sides.add("l")
        if vov and p[0] + p[2] == b[0]:
            sides.add("r")
        if hov and b[1] + b[3] == p[1]:
            sides.add("u")
        if hov and p[1] + p[3] == b[1]:
            sides.add("d")
    return sides


def render(w, old, husk, out):
    for r in old:
        if husk is not None and r == husk:
            continue
        for x, y in cells(r):
            if 0 <= x < 64 and 0 <= y < 64:
                out[y][x] = w.terrain(x, y)
    if husk is not None:
        for x, y in centre_cells(husk):
            out[y][x] = 4
    for b in w.blocks:
        for x, y in cells(b):
            out[y][x] = 14
        for x, y in centre_cells(b):
            out[y][x] = 5
    p = w.player
    if p:
        sides = touching_sides(p, w.blocks)
        for x, y in cells(p):
            edge = (("l" in sides and x == p[0]) or ("r" in sides and x == p[0] + p[2] - 1)
                    or ("u" in sides and y == p[1]) or ("d" in sides and y == p[1] + p[3] - 1))
            out[y][x] = 0 if edge else 14
        for x, y in centre_cells(p):
            out[y][x] = 0


def hud(out, n):
    zeros = (n + 1) // 2
    out[63] = [0 if x >= 64 - zeros else 4 for x in range(64)]


def transition_function(state, action, frame):
    frame = [[int(v) for v in row] for row in frame]
    if _memo["frame"] is not None and frame == _memo["frame"]:
        n = _memo["n"]
    else:
        z = sum(1 for v in frame[63] if v == 0)
        n = 2 * z - 1 if z > 0 else 0
    w = World(state, frame)
    old = [list(r) for r in w.movers()]
    husk = None
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            husk = w.click(action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        w.arrow(DIRS[action])
    out = [row[:] for row in frame]
    render(w, old, husk, out)
    n += 1
    hud(out, n)
    _memo["frame"], _memo["n"] = [row[:] for row in out], n
    return out
