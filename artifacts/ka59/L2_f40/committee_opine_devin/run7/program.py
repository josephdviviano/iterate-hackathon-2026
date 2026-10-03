# Mechanics: player (3-cell steps) walks on frame colours {1,4}; a step whose rect overlaps a block
# kicks it instead (player stays): the chain slides 1 cell per tick, absorbing blocks it hits, and stops
# on colour 2, the portal bound, the player, or a socket interior (inset 1) it does not exactly fit.
# Click on a block: player takes its rect, old player stays as an inert husk (centre 4). HUD row 63:
# (n+1)//2 zeros from the right, n = actions this level (carried on frame continuity). Unconfirmed: socket fill.
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
WALL = 2
STEP = 3
HUD_ROW = 63
_mem = {"frame": None, "n": 0}


def rect_cells(r):
    x, y, w, h = r
    return [(i, j) for j in range(y, y + h) for i in range(x, x + w)]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shifted(r, dx, dy):
    return (r[0] + dx, r[1] + dy, r[2], r[3])


def inside(r, bound):
    return bound[0] <= r[0] and bound[1] <= r[1] and r[0] + r[2] <= bound[0] + bound[2] and r[1] + r[3] <= bound[1] + bound[3]


def touching(a, b):
    """Sides of rect a that are adjacent to rect b with perpendicular overlap."""
    sides = set()
    xo = a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
    yo = a[1] < b[1] + b[3] and b[1] < a[1] + a[3]
    if xo and b[1] + b[3] == a[1]:
        sides.add("top")
    if xo and a[1] + a[3] == b[1]:
        sides.add("bottom")
    if yo and b[0] + b[2] == a[0]:
        sides.add("left")
    if yo and a[0] + a[2] == b[0]:
        sides.add("right")
    return sides


class World:
    def __init__(self, state, frame):
        self.frame = frame
        self.player = None
        self.blocks = []
        self.sockets = []
        self.bound = (0, 0, 63, 63)
        for o in state:
            r = (o["x"], o["y"], o["w"], o["h"])
            t = o.get("type")
            if t == "player":
                self.player = r
            elif t == "block":
                self.blocks.append(r)
            elif t == "target":
                self.sockets.append(r)
            elif t == "portal":
                self.bound = r
        self.bg = self.terrain()

    def terrain(self):
        """Frame with mover sprites erased to the terrain beneath them."""
        bg = [row[:] for row in self.frame]
        covered = set()
        for r in self.movers():
            covered.update(rect_cells(r))
        for (x, y) in covered:
            bg[y][x] = self.erase_colour(x, y, covered)
        return bg

    def movers(self):
        return ([self.player] if self.player else []) + self.blocks

    def erase_colour(self, x, y, covered):
        for s in self.sockets:
            if overlap((x, y, 1, 1), s):
                inner = (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)
                return 1 if overlap((x, y, 1, 1), inner) else 4
        votes = {}
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cx, cy = x + dx, y + dy
            while 0 <= cx < 64 and 0 <= cy < HUD_ROW and (cx, cy) in covered:
                cx, cy = cx + dx, cy + dy
            if 0 <= cx < 64 and 0 <= cy < HUD_ROW:
                c = self.frame[cy][cx]
                if c in (1, 2, 15):
                    votes[c] = votes.get(c, 0) + 1
        return max(votes, key=lambda c: votes[c]) if votes else 1

    # ---- guards ----
    def player_blocked(self, r):
        if not inside(r, self.bound):
            return True
        return any(self.bg[y][x] not in PLAYER_FLOOR for x, y in rect_cells(r))

    def socket_rejects(self, r):
        for s in self.sockets:
            inner = (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)
            if overlap(r, inner) and r != inner:
                return True
        return False

    def block_blocked(self, r):
        if not inside(r, self.bound):
            return True
        if self.player and overlap(r, self.player):
            return True
        if any(self.bg[y][x] not in BLOCK_FLOOR for x, y in rect_cells(r)):
            return True
        return self.socket_rejects(r)

    # ---- interactions ----
    def push_chain(self, seeds, dx, dy):
        chain = set(seeds)
        while True:
            while True:
                nxt = {i: shifted(self.blocks[i], dx, dy) for i in chain}
                extra = {j for j, b in enumerate(self.blocks) if j not in chain
                         and any(overlap(nr, b) for nr in nxt.values())}
                if not extra:
                    break
                chain |= extra
            if any(self.block_blocked(nr) for nr in nxt.values()):
                return
            for i, nr in nxt.items():
                self.blocks[i] = nr

    def move_player(self, action):
        dx, dy = DIRS[action]
        dest = shifted(self.player, dx * STEP, dy * STEP)
        hit = [i for i, b in enumerate(self.blocks) if overlap(dest, b)]
        if hit:
            self.push_chain(hit, dx, dy)
        elif not self.player_blocked(dest):
            self.player = dest

    def click(self, x, y):
        for i, b in enumerate(self.blocks):
            if overlap((x, y, 1, 1), b):
                self.leave_husk()
                self.player = b
                del self.blocks[i]
                return

    def leave_husk(self):
        px, py, pw, ph = self.player
        for (x, y) in rect_cells(self.player):
            self.bg[y][x] = self.frame[y][x]
        for (x, y) in centre_cells(self.player):
            self.bg[y][x] = 4

    # ---- render ----
    def render(self):
        out = [row[:] for row in self.bg]
        for b in self.blocks:
            draw_token(out, b, 5)
        if self.player:
            draw_token(out, self.player, 0)
            px, py, pw, ph = self.player
            sides = set()
            for b in self.blocks:
                sides |= touching(self.player, b)
            for (x, y) in rect_cells(self.player):
                if ("top" in sides and y == py) or ("bottom" in sides and y == py + ph - 1) or \
                   ("left" in sides and x == px) or ("right" in sides and x == px + pw - 1):
                    out[y][x] = 0
        return out


def centre_range(start, n):
    return range(start + (n - 1) // 2, start + n // 2 + 1)


def centre_cells(r):
    return [(x, y) for y in centre_range(r[1], r[3]) for x in centre_range(r[0], r[2])]


def draw_token(out, r, centre):
    for (x, y) in rect_cells(r):
        out[y][x] = 14
    for (x, y) in centre_cells(r):
        out[y][x] = centre


def draw_hud(out, n):
    zeros = (n + 1) // 2
    for x in range(64):
        if out[HUD_ROW][x] in (0, 4):
            out[HUD_ROW][x] = 0 if x >= 64 - zeros else 4


def hud_zeros(frame):
    return sum(1 for v in frame[HUD_ROW] if v == 0)


def transition_function(state, action, frame):
    if _mem["frame"] is not None and _mem["frame"] == frame:
        n = _mem["n"]
    else:
        z = hud_zeros(frame)
        n = 2 * z - 1 if z else 0
    w = World(state, frame)
    if isinstance(action, int) and action in DIRS and w.player:
        w.move_player(action)
    elif isinstance(action, dict) and action.get("action_id") == 6:
        w.click(action["x"], action["y"])
    out = w.render()
    n += 1
    draw_hud(out, n)
    _mem["frame"] = [row[:] for row in out]
    _mem["n"] = n
    return [[int(v) for v in row] for row in out]
