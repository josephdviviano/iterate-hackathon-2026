# Mechanics: blocking first - a player step (3 cells) is a no-op when its next rect leaves the portal bbox or hits a
# frame colour outside {1,4} (e.g. wall 2); else if it overlaps a block it kicks: the chain slides 1 cell/tick over
# {1,4,15} (closure over hit blocks) until a wall, bound, the player or a non-fitting socket interior blocks it.
# Click on a block: player takes its rect, old player stays an inert husk; HUD row 63 = (n+1)//2 zeros, n carried on continuity.
# Unconfirmed: fitting-socket completion, player touching husk, actions 2/5/7 beyond plain moves.
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
HUD_ROW = 63
MEM = {"frame": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def shift(r, d, k=1):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def interior(s):
    return (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)


class World:
    def __init__(self, state, frame):
        self.frame = frame
        self.player = None
        self.blocks = []
        self.sockets = []
        self.bound = (0, 0, 64, HUD_ROW)
        for o in state:
            if not o.get("visible", True):
                continue
            if o["type"] == "player":
                self.player = rect(o)
            elif o["type"] == "block":
                self.blocks.append(rect(o))
            elif o["type"] == "target":
                self.sockets.append(rect(o))
            elif o["type"] == "portal":
                self.bound = rect(o)
        self.bg = self.terrain()

    def terrain(self):
        f = self.frame
        bg = [row[:] for row in f]
        movers = list(self.blocks) + ([self.player] if self.player else [])
        cov = set()
        for r in movers:
            cov.update(cells(r))
        for (x, y) in cov:
            v = None
            for s in self.sockets:
                if overlap((x, y, 1, 1), s):
                    v = 1 if overlap((x, y, 1, 1), interior(s)) else 4
                    break
            if v is None:
                votes = {}
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cx, cy = x + dx, y + dy
                    while 0 <= cx < 64 and 0 <= cy < HUD_ROW and (cx, cy) in cov:
                        cx, cy = cx + dx, cy + dy
                    if 0 <= cx < 64 and 0 <= cy < HUD_ROW:
                        c = f[cy][cx]
                        votes[c] = votes.get(c, 0) + 1
                v = max(votes, key=lambda c: votes[c]) if votes else 1
            bg[y][x] = v
        return bg

    def in_bound(self, r):
        b = self.bound
        return r[0] >= b[0] and r[1] >= b[1] and r[0] + r[2] <= b[0] + b[2] and r[1] + r[3] <= b[1] + b[3]

    def on_floor(self, r, floor):
        return self.in_bound(r) and all(self.bg[y][x] in floor for (x, y) in cells(r))

    def socket_rejects(self, r):
        for s in self.sockets:
            i = interior(s)
            if overlap(r, i) and (r[2], r[3]) != (i[2], i[3]):
                return True
        return False

    def block_blocked(self, r):
        return (not self.on_floor(r, BLOCK_FLOOR) or self.socket_rejects(r)
                or (self.player is not None and overlap(r, self.player)))

    def push_chain(self, seed, d):
        while True:
            moving = set(seed)
            while True:
                hit = {j for j, b in enumerate(self.blocks) if j not in moving
                       and any(overlap(shift(self.blocks[i], d), b) for i in moving)}
                if not hit:
                    break
                moving |= hit
            if any(self.block_blocked(shift(self.blocks[i], d)) for i in moving):
                return
            for i in moving:
                self.blocks[i] = shift(self.blocks[i], d)
            seed = moving

    def move_player(self, d):
        nxt = shift(self.player, d, 3)
        hit = [j for j, b in enumerate(self.blocks) if overlap(nxt, b)]
        if hit:
            self.push_chain(hit, d)
        elif self.on_floor(nxt, PLAYER_FLOOR):
            self.player = nxt

    def click(self, x, y):
        for j, b in enumerate(self.blocks):
            if overlap((x, y, 1, 1), b):
                if self.player:
                    draw_token(self.bg, self.player, 4)
                self.player = b
                del self.blocks[j]
                return

    def render(self, zeros):
        out = [row[:] for row in self.bg]
        for b in self.blocks:
            draw_token(out, b, 5)
        if self.player:
            draw_token(out, self.player, 0)
            x, y, w, h = self.player
            for b in self.blocks:
                vert = b[1] < y + h and y < b[1] + b[3]
                horiz = b[0] < x + w and x < b[0] + b[2]
                if vert and b[0] + b[2] == x:
                    paint(out, (x, y, 1, h), 0)
                if vert and b[0] == x + w:
                    paint(out, (x + w - 1, y, 1, h), 0)
                if horiz and b[1] + b[3] == y:
                    paint(out, (x, y, w, 1), 0)
                if horiz and b[1] == y + h:
                    paint(out, (x, y + h - 1, w, 1), 0)
        row = [4] * 64
        for k in range(min(zeros, 64)):
            row[63 - k] = 0
        out[HUD_ROW] = row
        return out


def paint(g, r, c):
    for (x, y) in cells(r):
        if 0 <= x < 64 and 0 <= y < 64:
            g[y][x] = c


def draw_token(g, r, centre):
    x, y, w, h = r
    paint(g, r, 14)
    paint(g, (x + (w - 1) // 2, y + (h - 1) // 2, w // 2 - (w - 1) // 2 + 1, h // 2 - (h - 1) // 2 + 1), centre)


def hud_zeros(frame):
    return sum(1 for v in frame[HUD_ROW] if v == 0)


def transition_function(state, action, frame):
    if MEM["frame"] is not None and frame == MEM["frame"]:
        n = MEM["n"]
    else:
        z = hud_zeros(frame)
        n = 2 * z - 1 if z > 0 else 0
    w = World(state, frame)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            w.click(action["x"], action["y"])
    elif action in DIRS and w.player:
        w.move_player(DIRS[action])
    n += 1
    out = w.render((n + 1) // 2)
    MEM["frame"] = [row[:] for row in out]
    MEM["n"] = n
    return out
