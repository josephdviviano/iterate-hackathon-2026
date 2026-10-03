# Mechanics: arrows move the player 3px over floor {1,4}; a dest overlapping blocks kicks them instead
# (player stays): the hit chain slides in 1px ticks over {1,4,15}, collecting blocks it hits, until a
# member meets another colour/edge, the player, or a socket interior it does not fit. Click on a block:
# player takes its rect, old player stays as an e/4 husk. HUD row 63: ceil(n/2) zeros, n = actions.
# Unconfirmed: after crossing colour 15 a slide stops 9px later (fits steps 63/101; alt: fitting-socket line).
FLOOR_PLAYER = {1, 4}
FLOOR_BLOCK = {1, 4, 15}
ICE = 15
POST_ICE_RUN = 9
RIM, BLOCK_CORE, PLAYER_CORE, HUSK_CORE, HUD_BG, HUD_SPENT = 14, 5, 0, 4, 4, 0
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_mem = {}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def shift(r, d, k=1):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def interior(s):
    return (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)


def in_bounds(r, bound):
    return r[0] >= bound[0] and r[1] >= bound[1] and r[0] + r[2] <= bound[0] + bound[2] and r[1] + r[3] <= bound[1] + bound[3]


def estimate_bg(frame, covered, sockets):
    bg = [row[:] for row in frame]
    H, W = len(frame), len(frame[0])
    free = lambda x, y: 0 <= x < W and 0 <= y < H and (x, y) not in covered

    def span_ok(x0, x1, y0, y1, horiz):
        # the boundary-free test: nearest fully uncovered parallel lines show no colour change
        for sgn in (-1, 1):
            k = (y0 if horiz else x0) + sgn
            while 0 <= k < (H if horiz else W):
                line = [(x, k) for x in range(x0, x1 + 1)] if horiz else [(k, y) for y in range(y0, y1 + 1)]
                if all(free(a, b) for a, b in line):
                    if len({frame[b][a] for a, b in line}) > 1:
                        return False
                    break
                k += sgn
        return True

    for (x, y) in covered:
        sk = [s for s in sockets if overlaps(s, (x, y, 1, 1))]
        if sk:
            bg[y][x] = 1 if overlaps(interior(sk[0]), (x, y, 1, 1)) else 4
            continue
        cands = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            k = 1
            while 0 <= x + dx * k < W and 0 <= y + dy * k < H and not free(x + dx * k, y + dy * k):
                k += 1
            nx, ny = x + dx * k, y + dy * k
            if not free(nx, ny):
                continue
            if dy == 0:
                ok = span_ok(min(x, nx), max(x, nx), y, y, True)
            else:
                ok = span_ok(x, x, min(y, ny), max(y, ny), False)
            cands.append((not ok, k, frame[ny][nx]))
        if cands:
            valid = [c for bad, _, c in cands if not bad]
            pool = valid or [c for _, _, c in cands]
            bg[y][x] = max(pool, key=lambda c: (pool.count(c), -min(k for b, k, cc in cands if cc == c)))
    return bg


def core_idx(n):
    return range((n - 1) // 2, n // 2 + 1)


def paint_token(out, r, core, zero_sides=()):
    x0, y0, w, h = r
    for (x, y) in cells(r):
        out[y][x] = RIM
    for j in core_idx(h):
        for i in core_idx(w):
            out[y0 + j][x0 + i] = core
    for side in zero_sides:
        if side in ("L", "R"):
            xx = x0 if side == "L" else x0 + w - 1
            for y in range(y0, y0 + h):
                out[y][xx] = PLAYER_CORE
        else:
            yy = y0 if side == "U" else y0 + h - 1
            for x in range(x0, x0 + w):
                out[yy][x] = PLAYER_CORE


def touching_sides(p, blocks):
    sides = []
    for b in blocks:
        vert = p[1] < b[1] + b[3] and b[1] < p[1] + p[3]
        horiz = p[0] < b[0] + b[2] and b[0] < p[0] + p[2]
        if vert and b[0] + b[2] == p[0]:
            sides.append("L")
        if vert and p[0] + p[2] == b[0]:
            sides.append("R")
        if horiz and b[1] + b[3] == p[1]:
            sides.append("U")
        if horiz and p[1] + p[3] == b[1]:
            sides.append("D")
    return sides


class World:
    def __init__(self, state, frame, bg):
        self.player = next((rect(o) for o in state if o["type"] == "player"), None)
        self.blocks = [rect(o) for o in state if o["type"] == "block"]
        self.sockets = [rect(o) for o in state if o["type"] == "target"]
        portal = next((rect(o) for o in state if o["type"] == "portal"), None)
        self.bound = portal or (0, 0, len(frame[0]), len(frame))
        self.bg = bg

    def player_blocked(self, r):
        return not in_bounds(r, self.bound) or any(self.bg[y][x] not in FLOOR_PLAYER for x, y in cells(r))

    def socket_rejects(self, r, d):
        for s in self.sockets:
            inn = interior(s)
            if overlaps(r, inn):
                fits = (r[2], r[3]) == (inn[2], inn[3]) and (r[1] == inn[1] if d[1] == 0 else r[0] == inn[0])
                if not fits:
                    return True
        return False

    def leading(self, r, d):
        x0, y0, w, h = r
        if d[0]:
            xx = x0 + w - 1 if d[0] > 0 else x0
            return [(xx, y) for y in range(y0, y0 + h)]
        yy = y0 + h - 1 if d[1] > 0 else y0
        return [(x, yy) for x in range(x0, x0 + w)]

    def block_blocked(self, r, d, run):
        if not in_bounds(r, self.bound) or (self.player and overlaps(r, self.player)):
            return True
        if any(self.bg[y][x] not in FLOOR_BLOCK for x, y in cells(r)):
            return True
        if self.socket_rejects(r, d):
            return True
        lead_ice = any(self.bg[y][x] == ICE for x, y in self.leading(r, d))
        return run is not None and not lead_ice and run >= POST_ICE_RUN

    def push_chain(self, seed, d):
        moving = set(seed)
        runs = {i: None for i in moving}
        while True:
            grow = True
            while grow:
                grow = False
                nxt = [shift(self.blocks[i], d) for i in moving]
                for j, b in enumerate(self.blocks):
                    if j not in moving and any(overlaps(n, b) for n in nxt):
                        moving.add(j)
                        runs[j] = None
                        grow = True
            if any(self.block_blocked(shift(self.blocks[i], d), d, runs[i]) for i in moving):
                return
            for i in moving:
                n = shift(self.blocks[i], d)
                if any(self.bg[y][x] == ICE for x, y in self.leading(n, d)):
                    runs[i] = 0
                elif runs[i] is not None:
                    runs[i] += 1
                self.blocks[i] = n

    def arrow(self, d):
        if self.player is None:
            return
        dest = shift(self.player, d, 3)
        hit = [i for i, b in enumerate(self.blocks) if overlaps(dest, b)]
        if hit:
            self.push_chain(hit, d)
        elif not self.player_blocked(dest):
            self.player = dest

    def click(self, x, y):
        for i, b in enumerate(self.blocks):
            if overlaps(b, (x, y, 1, 1)):
                if self.player:
                    paint_token(self.bg, self.player, HUSK_CORE)
                self.player = b
                del self.blocks[i]
                return

    def render(self):
        out = [row[:] for row in self.bg]
        for b in self.blocks:
            paint_token(out, b, BLOCK_CORE)
        if self.player:
            paint_token(out, self.player, PLAYER_CORE, touching_sides(self.player, self.blocks))
        return out


def hud_render(out, n):
    row = out[-1]
    z = (n + 1) // 2
    W = len(row)
    for x in range(W):
        row[x] = HUD_SPENT if x >= W - z else HUD_BG


def transition_function(state, action, frame):
    covered = set()
    for o in state:
        if o["type"] in ("player", "block"):
            covered.update(cells(rect(o)))
    sockets = [rect(o) for o in state if o["type"] == "target"]
    if _mem.get("frame") == frame:
        bg, n = [r[:] for r in _mem["bg"]], _mem["n"]
    else:
        bg = estimate_bg(frame, covered, sockets)
        z = sum(1 for v in frame[-1] if v == HUD_SPENT)
        n = 2 * z - 1 if z else 0
    w = World(state, frame, bg)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            w.click(action["x"], action["y"])
    elif action in DIRS:
        w.arrow(DIRS[action])
    n += 1
    out = w.render()
    hud_render(out, n)
    _mem.update(frame=[r[:] for r in out], bg=[r[:] for r in w.bg], n=n)
    return out
