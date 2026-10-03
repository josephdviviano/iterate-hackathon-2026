# Mechanics: player (3-step arrows) walks on bg colours other than 2/15 inside the portal bbox; a step whose rect
# overlaps a block is a kick: player stays, the block chain (closure of overlapped blocks) slides 1 cell per tick
# until any member hits colour 2, the bounds, the player, or a socket interior (inset 1) it does not fit exactly.
# Click on a block: player takes its rect, old player stays as an inert husk (e ring, 4 centre). Sprites: e fill,
# centre 5 (blocks) / 0 (player, plus every side touching a block). HUD row 63: ceil(n/2) zeros, n = actions this
# level. Unconfirmed: n parity is hidden (continuity-gated, fallback n=2*zeros); fitting-socket fill and A5/A7 unseen.
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_WALLS = (2, 15)
BLOCK_WALLS = (2,)
SPRITE = 14
_memo = {"frame": None, "bg": None, "n": 0}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def overlaps(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def cells(r):
    return [(x, y) for y in range(r[1], r[1] + r[3]) for x in range(r[0], r[0] + r[2])]


def shift(r, d, k=1):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def centre(n):
    return range((n - 1) // 2, n // 2 + 1)


class World:
    def __init__(self, state, frame):
        self.player = None
        self.blocks, self.sockets = [], []
        self.bounds = (0, 0, 63, 63)
        for o in state:
            t = o.get("type")
            if t == "player":
                self.player = rect(o)
            elif t == "block":
                self.blocks.append(rect(o))
            elif t == "target":
                self.sockets.append(rect(o))
            elif t == "portal":
                self.bounds = rect(o)

    def in_bounds(self, r):
        b = self.bounds
        return b[0] <= r[0] and b[1] <= r[1] and r[0] + r[2] <= b[0] + b[2] and r[1] + r[3] <= b[1] + b[3]

    def hits_colour(self, bg, r, walls):
        return any(bg[y][x] in walls for x, y in cells(r))

    def socket_rejects(self, r):
        for s in self.sockets:
            inner = (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)
            if overlaps(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
                return True
        return False

    def block_blocked(self, bg, r):
        return (not self.in_bounds(r) or self.hits_colour(bg, r, BLOCK_WALLS)
                or (self.player and overlaps(r, self.player)) or self.socket_rejects(r))

    def push_chain(self, bg, seeds, d):
        while True:
            chain = set(seeds)
            grew = True
            while grew:
                grew = False
                for i in list(chain):
                    nr = shift(self.blocks[i], d)
                    for j, b in enumerate(self.blocks):
                        if j not in chain and overlaps(nr, b):
                            chain.add(j)
                            grew = True
            if any(self.block_blocked(bg, shift(self.blocks[i], d)) for i in chain):
                return
            for i in chain:
                self.blocks[i] = shift(self.blocks[i], d)

    def step_player(self, bg, d):
        if self.player is None:
            return
        nr = shift(self.player, d, 3)
        hit = [i for i, b in enumerate(self.blocks) if overlaps(nr, b)]
        if hit:
            self.push_chain(bg, hit, d)
        elif self.in_bounds(nr) and not self.hits_colour(bg, nr, PLAYER_WALLS):
            self.player = nr

    def click(self, bg, x, y):
        for i, b in enumerate(self.blocks):
            if b[0] <= x < b[0] + b[2] and b[1] <= y < b[1] + b[3]:
                if self.player:
                    draw_token(bg, self.player, 4)
                self.player = b
                del self.blocks[i]
                return

    def touching_sides(self):
        p = self.player
        sides = set()
        for b in self.blocks:
            yo = b[1] < p[1] + p[3] and p[1] < b[1] + b[3]
            xo = b[0] < p[0] + p[2] and p[0] < b[0] + b[2]
            if yo and b[0] + b[2] == p[0]:
                sides.add("L")
            if yo and p[0] + p[2] == b[0]:
                sides.add("R")
            if xo and b[1] + b[3] == p[1]:
                sides.add("U")
            if xo and p[1] + p[3] == b[1]:
                sides.add("D")
        return sides


def draw_token(f, r, core):
    x0, y0, w, h = r
    for x, y in cells(r):
        if 0 <= x < 64 and 0 <= y < 64:
            f[y][x] = SPRITE
    for j in centre(h):
        for i in centre(w):
            f[y0 + j][x0 + i] = core


def background(world, frame):
    covered = set()
    for r in world.blocks + ([world.player] if world.player else []):
        covered.update(cells(r))
    bg = [row[:] for row in frame]
    for x, y in covered:
        val = None
        for s in world.sockets:
            if s[0] <= x < s[0] + s[2] and s[1] <= y < s[1] + s[3]:
                border = x in (s[0], s[0] + s[2] - 1) or y in (s[1], s[1] + s[3] - 1)
                val = 4 if border else 1
        if val is None:
            votes = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cx, cy = x + dx, y + dy
                while 0 <= cx < 64 and 0 <= cy < 63:
                    if (cx, cy) not in covered and frame[cy][cx] not in (SPRITE, 0, 5, 4):
                        votes.append(frame[cy][cx])
                        break
                    cx, cy = cx + dx, cy + dy
            val = max(set(votes), key=votes.count) if votes else 1
        bg[y][x] = val
    return bg


def render(world, bg, n):
    f = [row[:] for row in bg]
    for b in world.blocks:
        draw_token(f, b, 5)
    if world.player:
        p = world.player
        draw_token(f, p, 0)
        x0, y0, w, h = p
        for s in world.touching_sides():
            for x, y in cells(p):
                if (s == "L" and x == x0) or (s == "R" and x == x0 + w - 1) or \
                   (s == "U" and y == y0) or (s == "D" and y == y0 + h - 1):
                    f[y][x] = 0
    zeros = (n + 1) // 2
    for k in range(64):
        x = 63 - k
        if f[63][x] in (0, 4):
            f[63][x] = 0 if k < zeros else 4
    return f


def transition_function(state, action, frame):
    world = World(state, frame)
    if _memo["frame"] is not None and frame == _memo["frame"]:
        bg, n = [row[:] for row in _memo["bg"]], _memo["n"]
    else:
        bg = background(world, frame)
        n = 2 * sum(1 for v in frame[63] if v == 0)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            world.click(bg, action.get("x", -1), action.get("y", -1))
    elif action in DIRS:
        world.step_player(bg, DIRS[action])
    n += 1
    out = render(world, bg, n)
    _memo.update(frame=[row[:] for row in out], bg=[row[:] for row in bg], n=n)
    return out
