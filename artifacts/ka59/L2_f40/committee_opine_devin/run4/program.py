# Mechanics: kick/slide token game. Arrows move the player 3 cells over floor colours {1,4} inside 0..62;
# a player step that would overlap blocks kicks them instead (player stays): the push chain slides 1 cell per
# tick, absorbing blocks it hits, until any member meets a non-{1,4,15} cell, the bound, the player, or a socket
# interior (inset 1) it does not exactly fit. Click on a block: player takes its rect, old player stays as an e/4 husk.
# HUD row 63: ceil(n/2) zeros from the right, n = actions this level (parity hidden; carried on frame continuity).
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
BOUND = 62
HUD_ROW = 63
_memo = {"frame": None, "n": 0, "bg": None}


def cells(r):
    x, y, w, h = r
    return [(i, j) for j in range(y, y + h) for i in range(x, x + w)]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d, k=1):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


def in_bounds(r):
    return r[0] >= 0 and r[1] >= 0 and r[0] + r[2] - 1 <= BOUND and r[1] + r[3] - 1 <= BOUND


class World:
    def __init__(self, state, frame, bg_memo):
        self.frame = frame
        self.player = None
        self.blocks = []
        self.sockets = []
        for o in state:
            r = (o["x"], o["y"], o["w"], o["h"])
            if o["type"] == "player":
                self.player = r
            elif o["type"] == "block":
                self.blocks.append(r)
            elif o["type"] == "target":
                self.sockets.append(r)
        self.bg = self.background(bg_memo)

    def sprite_cells(self):
        s = set()
        for r in self.blocks + ([self.player] if self.player else []):
            s.update(cells(r))
        return s

    def socket_colour(self, x, y):
        for sx, sy, sw, sh in self.sockets:
            if sx <= x < sx + sw and sy <= y < sy + sh:
                edge = x in (sx, sx + sw - 1) or y in (sy, sy + sh - 1)
                return 4 if edge else 1
        return None

    def background(self, memo):
        bg = [row[:] for row in self.frame]
        covered = self.sprite_cells()
        for (x, y) in covered:
            if memo is not None:
                bg[y][x] = memo[y][x]
                continue
            c = self.socket_colour(x, y)
            if c is None:
                votes = []
                for dx, dy in DIRS.values():
                    i, j = x + dx, y + dy
                    while 0 <= i < 64 and 0 <= j < 64 and (i, j) in covered:
                        i, j = i + dx, j + dy
                    if 0 <= i < 64 and 0 <= j < 64 and self.frame[j][i] in (1, 2, 15):
                        votes.append(self.frame[j][i])
                c = max(set(votes), key=votes.count) if votes else 1
            bg[y][x] = c
        return bg

    # guards
    def player_blocked(self, r):
        return not in_bounds(r) or any(self.bg[j][i] not in PLAYER_FLOOR for i, j in cells(r))

    def socket_rejects(self, r):
        for sx, sy, sw, sh in self.sockets:
            inner = (sx + 1, sy + 1, sw - 2, sh - 2)
            if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
                return True
        return False

    def block_blocked(self, r):
        if not in_bounds(r) or overlap(r, self.player):
            return True
        if any(self.bg[j][i] not in BLOCK_FLOOR for i, j in cells(r)):
            return True
        return self.socket_rejects(r)

    def push_chain(self, seeds, d):
        chain = set(seeds)
        while True:
            grew = True
            while grew:
                grew = False
                for k, b in enumerate(self.blocks):
                    if k not in chain and any(overlap(shift(self.blocks[c], d), b) for c in chain):
                        chain.add(k)
                        grew = True
            if any(self.block_blocked(shift(self.blocks[c], d)) for c in chain):
                return
            for c in chain:
                self.blocks[c] = shift(self.blocks[c], d)

    def step(self, action):
        if isinstance(action, dict):
            if action.get("action_id") == 6:
                self.click(action["x"], action["y"])
            return
        d = DIRS.get(action)
        if d is None or self.player is None:
            return
        nxt = shift(self.player, d, 3)
        hit = [k for k, b in enumerate(self.blocks) if overlap(nxt, b)]
        if hit:
            self.push_chain(hit, d)
        elif not self.player_blocked(nxt):
            self.player = nxt

    def click(self, x, y):
        for k, b in enumerate(self.blocks):
            if b[0] <= x < b[0] + b[2] and b[1] <= y < b[1] + b[3]:
                if self.player:
                    self.paint(self.bg, self.player, 4)
                self.player = b
                del self.blocks[k]
                return

    @staticmethod
    def paint(grid, r, centre):
        x, y, w, h = r
        for i, j in cells(r):
            grid[j][i] = 14
        for j in range(y + (h - 1) // 2, y + h // 2 + 1):
            for i in range(x + (w - 1) // 2, x + w // 2 + 1):
                grid[j][i] = centre

    def touching(self, b):
        px, py, pw, ph = self.player
        bx, by, bw, bh = b
        sides = set()
        if py < by + bh and by < py + ph:
            if bx + bw == px:
                sides.add(3)
            if px + pw == bx:
                sides.add(4)
        if px < bx + bw and bx < px + pw:
            if by + bh == py:
                sides.add(1)
            if py + ph == by:
                sides.add(2)
        return sides

    def render(self):
        out = [row[:] for row in self.bg]
        for b in self.blocks:
            self.paint(out, b, 5)
        if self.player:
            self.paint(out, self.player, 0)
            px, py, pw, ph = self.player
            sides = set()
            for b in self.blocks:
                sides |= self.touching(b)
            for i, j in cells(self.player):
                if (1 in sides and j == py) or (2 in sides and j == py + ph - 1) or \
                        (3 in sides and i == px) or (4 in sides and i == px + pw - 1):
                    out[j][i] = 0
        return out


def hud_zeros(frame):
    z = 0
    for x in range(63, -1, -1):
        if frame[HUD_ROW][x] != 0:
            break
        z += 1
    return z


def transition_function(state, action, frame):
    continuous = _memo["frame"] is not None and frame == _memo["frame"]
    if continuous:
        n = _memo["n"]
    else:
        z = hud_zeros(frame)
        n = 2 * z - 1 if z else 0
    w = World(state, frame, _memo["bg"] if continuous else None)
    w.step(action)
    out = w.render()
    n += 1
    z = (n + 1) // 2
    for x in range(64 - z, 64):
        out[HUD_ROW][x] = 0
    _memo.update(frame=[row[:] for row in out], n=n, bg=[row[:] for row in w.bg])
    return out
