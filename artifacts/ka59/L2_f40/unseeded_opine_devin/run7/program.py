# Mechanics: arrows move the 3px player over floor colours {1,4}; if its next rect overlaps a block it
# kicks instead: the block chain slides 1 cell per tick (closure over touched blocks) over {1,4,15}
# until a wall, the board edge, the player or a non-fitting socket interior (target inset 1) stops it.
# Click on a block: player takes that block's rect, old player stays as an inert husk (e rim, 4 centre).
# HUD row 63: zeros from the right = (k+1)//2, k = actions this level (continuity-gated; unconfirmed: fitting-socket entry).
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
SPRITE = {14, 0, 5}
STEP = 3
N = 63
_mem = {"frame": None, "k": 0}


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def in_bounds(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= N and r[1] + r[3] <= N


def background(frame, movers, sockets):
    covered = set()
    for r in movers:
        covered.update(cells(r))
    bg = {}
    for (x, y) in covered:
        val = None
        for s in sockets:
            if s[0] <= x < s[0] + s[2] and s[1] <= y < s[1] + s[3]:
                border = x in (s[0], s[0] + s[2] - 1) or y in (s[1], s[1] + s[3] - 1)
                val = 4 if border else 1
        if val is None:
            votes = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cx, cy = x + dx, y + dy
                while 0 <= cx < 64 and 0 <= cy < 64:
                    c = frame[cy][cx]
                    if (cx, cy) not in covered and c not in SPRITE:
                        votes.append(c)
                        break
                    cx, cy = cx + dx, cy + dy
            val = max(votes, key=votes.count) if votes else 1
        bg[(x, y)] = val
    return bg


class World:
    def __init__(self, state, frame):
        self.frame = frame
        self.player = None
        self.blocks = []
        self.sockets = []
        for o in state:
            r = [o["x"], o["y"], o["w"], o["h"]]
            if o["type"] == "player":
                self.player = r
            elif o["type"] == "block":
                self.blocks.append(r)
            elif o["type"] == "target":
                self.sockets.append(r)
        self.movers = self.blocks + ([self.player] if self.player else [])
        self.bg = background(frame, self.movers, self.sockets)
        self.husk = None

    def terrain(self, x, y):
        return self.bg.get((x, y), self.frame[y][x])

    def floor_ok(self, r, floor):
        return in_bounds(r) and all(self.terrain(x, y) in floor for x, y in cells(r))

    def socket_rejects(self, r):
        for s in self.sockets:
            inner = [s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2]
            if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
                return True
        return False

    def block_blocked(self, r):
        return (not self.floor_ok(r, BLOCK_FLOOR) or self.socket_rejects(r)
                or (self.player is not None and overlap(r, self.player)))

    def push_chain(self, seeds, dx, dy):
        moved = False
        while True:
            chain = list(seeds)
            grew = True
            while grew:
                grew = False
                for b in self.blocks:
                    if b in chain:
                        continue
                    if any(overlap([c[0] + dx, c[1] + dy, c[2], c[3]], b) for c in chain):
                        chain.append(b)
                        grew = True
            if any(self.block_blocked([c[0] + dx, c[1] + dy, c[2], c[3]]) for c in chain):
                return moved
            for c in chain:
                c[0] += dx
                c[1] += dy
            moved = True

    def move(self, dx, dy):
        p = self.player
        nxt = [p[0] + dx * STEP, p[1] + dy * STEP, p[2], p[3]]
        hit = [b for b in self.blocks if overlap(nxt, b)]
        if hit:
            self.push_chain(hit, dx, dy)
        elif self.floor_ok(nxt, PLAYER_FLOOR):
            p[0], p[1] = nxt[0], nxt[1]

    def click(self, x, y):
        for b in self.blocks:
            if b[0] <= x < b[0] + b[2] and b[1] <= y < b[1] + b[3]:
                self.husk = list(self.player)
                self.player[:] = b
                self.blocks.remove(b)
                return

    def render(self):
        out = [row[:] for row in self.frame]
        for (x, y), v in self.bg.items():
            out[y][x] = v
        if self.husk:
            draw_token(out, self.husk, 4)
        for b in self.blocks:
            draw_token(out, b, 5)
        if self.player:
            p = self.player
            draw_token(out, p, 0)
            for b in self.blocks:
                ov_y = p[1] < b[1] + b[3] and b[1] < p[1] + p[3]
                ov_x = p[0] < b[0] + b[2] and b[0] < p[0] + p[2]
                if ov_y and b[0] + b[2] == p[0]:
                    paint(out, [p[0], p[1], 1, p[3]], 0)
                if ov_y and p[0] + p[2] == b[0]:
                    paint(out, [p[0] + p[2] - 1, p[1], 1, p[3]], 0)
                if ov_x and b[1] + b[3] == p[1]:
                    paint(out, [p[0], p[1], p[2], 1], 0)
                if ov_x and p[1] + p[3] == b[1]:
                    paint(out, [p[0], p[1] + p[3] - 1, p[2], 1], 0)
        return out


def paint(out, r, c):
    for x, y in cells(r):
        if 0 <= x < 64 and 0 <= y < 64:
            out[y][x] = c


def draw_token(out, r, centre):
    paint(out, r, 14)
    xs = range(r[0] + (r[2] - 1) // 2, r[0] + r[2] // 2 + 1)
    ys = range(r[1] + (r[3] - 1) // 2, r[1] + r[3] // 2 + 1)
    for y in ys:
        for x in xs:
            out[y][x] = centre


def hud_zeros(frame):
    return sum(1 for v in frame[63] if v == 0)


def draw_hud(out, k):
    z = (k + 1) // 2
    for x in range(64):
        if out[63][x] in (0, 4):
            out[63][x] = 0 if x >= 64 - z else 4


def transition_function(state, action, frame):
    z = hud_zeros(frame)
    if _mem["frame"] is not None and _mem["frame"] == frame:
        k = _mem["k"]
    else:
        k = 0 if z == 0 else 2 * z - 1
    w = World(state, frame)
    if isinstance(action, dict):
        if action.get("action_id") == 6 and w.player:
            w.click(action["x"], action["y"])
    elif action in DIRS and w.player:
        w.move(*DIRS[action])
    out = w.render()
    k += 1
    draw_hud(out, k)
    _mem["frame"] = [row[:] for row in out]
    _mem["k"] = k
    return out
