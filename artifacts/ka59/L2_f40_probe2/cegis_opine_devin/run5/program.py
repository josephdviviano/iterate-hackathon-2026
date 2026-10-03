# Mechanics: arrows move the player 3 cells over floor colours {1,4}; if its next rect overlaps a block it
# kicks instead (player stays): the chain slides in 3-cell ticks, absorbing blocks it overlaps, over {1,4,15},
# stopping on colour 2, the portal bounds, the player, or a non-fitting socket interior; a block stops when its
# leading edge reaches a rim line of a socket it fits exactly (step-101 fit; hypothesis, rim lines extend fully).
# Click on a block possesses it (old player left as husk, centre 4). HUD row 63: (n+1)//2 zeros, n = actions.
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
TERRAIN = {1, 2, 15}
TICK = 3
_memo = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shift(r, d, k=1):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def inset(r):
    return (r[0] + 1, r[1] + 1, r[2] - 2, r[3] - 2)


def fits(block, sock):
    return block[2] == sock[2] - 2 and block[3] == sock[3] - 2


def in_bounds(r, bound):
    return r[0] >= bound[0] and r[1] >= bound[1] and r[0] + r[2] <= bound[0] + bound[2] and r[1] + r[3] <= bound[1] + bound[3]


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
        for r in ([self.player] if self.player else []) + self.blocks:
            self.covered.update(cells(r))
        self.bg = [row[:] for row in frame]
        for (x, y) in self.covered:
            self.bg[y][x] = self.terrain_at(x, y)

    def free(self, x, y):
        return 0 <= x < 64 and 0 <= y < 63 and (x, y) not in self.covered

    def neighbour(self, x, y, d):
        k = 1
        while True:
            nx, ny = x + d[0] * k, y + d[1] * k
            if not (0 <= nx < 64 and 0 <= ny < 63):
                return None
            if (nx, ny) not in self.covered:
                return (nx, ny, k)
            k += 1

    def side_consistent(self, x, y, d, k):
        for p in ((d[1], d[0]), (-d[1], -d[0])):
            for off in range(1, 64):
                line = [(x + d[0] * i + p[0] * off, y + d[1] * i + p[1] * off) for i in range(k + 1)]
                if any(not (0 <= cx < 64 and 0 <= cy < 63) for cx, cy in line):
                    break
                if all(self.free(cx, cy) for cx, cy in line):
                    if len({self.frame[cy][cx] for cx, cy in line}) > 1:
                        return False
                    break
        return True

    def terrain_at(self, x, y):
        for s in self.sockets:
            if overlaps((x, y, 1, 1), s):
                return 1 if overlaps((x, y, 1, 1), inset(s)) else 4
        valid, other = [], []
        for d in DIRS.values():
            nb = self.neighbour(x, y, d)
            if nb is None:
                continue
            c = self.frame[nb[1]][nb[0]]
            if c not in TERRAIN:
                continue
            (valid if self.side_consistent(x, y, d, nb[2]) else other).append((nb[2], c))
        pool = valid or other
        if not pool:
            return 1
        votes = {}
        for dist, c in pool:
            votes[c] = votes.get(c, 0) + 1
        best = max(votes.values())
        return min((dist, c) for dist, c in pool if votes[c] == best)[1]

    def colour_ok(self, r, floor):
        return all(self.bg[y][x] in floor for x, y in cells(r))

    def player_blocked(self, r):
        return not in_bounds(r, self.bound) or not self.colour_ok(r, PLAYER_FLOOR)

    def socket_rejects(self, r):
        return any(not fits(r, s) and overlaps(r, inset(s)) for s in self.sockets)

    def block_blocked(self, r):
        if not in_bounds(r, self.bound) or not self.colour_ok(r, BLOCK_FLOOR):
            return True
        if self.player and overlaps(r, self.player):
            return True
        return self.socket_rejects(r)

    def rim_stop(self, r, d):
        lead = r[1] if d[1] < 0 else r[1] + r[3] - 1 if d[1] > 0 else r[0] if d[0] < 0 else r[0] + r[2] - 1
        for s in self.sockets:
            if not fits(r, s):
                continue
            rims = (s[1], s[1] + s[3] - 1) if d[1] else (s[0], s[0] + s[2] - 1)
            if lead in rims:
                return True
        return False

    def push_chain(self, first, d):
        blocks = self.blocks[:]
        chain = {first}
        while True:
            changed = True
            while changed:
                changed = False
                for i, b in enumerate(blocks):
                    if i not in chain and any(overlaps(shift(blocks[j], d, TICK), b) for j in chain):
                        chain.add(i)
                        changed = True
            moved = {i: shift(blocks[i], d, TICK) for i in chain}
            if any(self.block_blocked(r) for r in moved.values()):
                break
            for i, r in moved.items():
                blocks[i] = r
            if any(self.rim_stop(r, d) for r in moved.values()):
                break
        self.blocks = blocks

    def step(self, action):
        self.husk = None
        if isinstance(action, int) and action in DIRS and self.player:
            d = DIRS[action]
            nxt = shift(self.player, d, TICK)
            hit = [i for i, b in enumerate(self.blocks) if overlaps(nxt, b)]
            if hit:
                self.push_chain(hit[0], d)
            elif not self.player_blocked(nxt):
                self.player = nxt
        elif isinstance(action, dict) and action.get("action_id") == 6:
            cx, cy = action.get("x", -1), action.get("y", -1)
            for i, b in enumerate(self.blocks):
                if overlaps((cx, cy, 1, 1), b):
                    self.husk = self.player
                    self.player = b
                    del self.blocks[i]
                    break

    def render(self):
        out = [row[:] for row in self.bg]
        if self.husk:
            hx, hy, hw, hh = self.husk
            for x, y in cells(self.husk):
                out[y][x] = self.frame[y][x]
            for x, y in centre_cells(self.husk):
                out[y][x] = 4
        for b in self.blocks:
            draw_token(out, b, 5)
        if self.player:
            draw_token(out, self.player, 0)
            px, py, pw, ph = self.player
            for b in self.blocks:
                hov = b[0] < px + pw and px < b[0] + b[2]
                vov = b[1] < py + ph and py < b[1] + b[3]
                if hov and b[1] + b[3] == py:
                    for x in range(px, px + pw):
                        out[py][x] = 0
                if hov and b[1] == py + ph:
                    for x in range(px, px + pw):
                        out[py + ph - 1][x] = 0
                if vov and b[0] + b[2] == px:
                    for y in range(py, py + ph):
                        out[y][px] = 0
                if vov and b[0] == px + pw:
                    for y in range(py, py + ph):
                        out[y][px + pw - 1] = 0
        return out


def centre_cells(r):
    x, y, w, h = r
    xs = range(x + (w - 1) // 2, x + w // 2 + 1)
    ys = range(y + (h - 1) // 2, y + h // 2 + 1)
    return [(cx, cy) for cy in ys for cx in xs]


def draw_token(out, r, centre):
    for x, y in cells(r):
        out[y][x] = 14
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
    if _memo["frame"] is not None and frame == _memo["frame"]:
        n = _memo["n"]
    else:
        z = hud_zeros(frame)
        n = 2 * z - 1 if z > 0 else 0
    world = World(state, frame)
    world.step(action)
    out = world.render()
    n += 1
    z = (n + 1) // 2
    out[63] = [0 if x >= 64 - z else frame[63][x] if frame[63][x] != 0 else 4 for x in range(64)]
    _memo["frame"] = [row[:] for row in out]
    _memo["n"] = n
    return out
