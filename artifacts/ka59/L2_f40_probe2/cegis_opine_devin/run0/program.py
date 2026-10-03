# Player: arrows move 3 over colours {1,4}; next rect overlapping a block = KICK (player stays). Blocks: kicked chain slides 3/tick
# over {1,4,15}, absorbing hit blocks, stopped by colour 2 / bounds / player / non-fitting socket interior (inset 1) or momentum:
# unlimited until a member touches 15, then 2 ticks, refilled while on 15 (step-101 stop 15 cells up). Click block: player takes
# its rect, old player stays as husk (centre 4). HUD row 63 = (n+1)//2 zeros, n continuity-gated (fallback 2z-1). Unconfirmed:
# momentum vs a weaker 6x6-kicker alternative (both fit), fitting-socket entry, A5/A7.
STEP = 3
DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
PLAYER_FLOOR = {1, 4}
BLOCK_FLOOR = {1, 4, 15}
ICE = 15
MOMENTUM = 2
H = W = 64
_memo = {"frame": None, "n": 0, "bg": None}


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def cells(r):
    x, y, w, h = r
    return [(i, j) for j in range(y, y + h) for i in range(x, x + w)]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def shift(r, d):
    return (r[0] + d[0], r[1] + d[1], r[2], r[3])


def hud_zeros(frame):
    z = 0
    for x in range(W - 1, -1, -1):
        if frame[H - 1][x] != 0:
            break
        z += 1
    return z


class World:
    def __init__(self, state, frame, bg):
        self.player = None
        self.blocks = []
        self.sockets = []
        self.bound = (0, 0, W - 1, H - 1)
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
        self.bg = bg if bg is not None else self.infer_bg(frame)

    def sprites(self):
        return ([self.player] if self.player else []) + self.blocks

    def infer_bg(self, frame):
        covered = set()
        for r in self.sprites():
            covered.update(cells(r))
        bg = [row[:] for row in frame]
        for (x, y) in covered:
            bg[y][x] = self.terrain_at(frame, covered, x, y)
        return bg

    def terrain_at(self, frame, covered, x, y):
        for s in self.sockets:
            sx, sy, sw, sh = s
            if sx <= x < sx + sw and sy <= y < sy + sh:
                rim = x in (sx, sx + sw - 1) or y in (sy, sy + sh - 1)
                return 4 if rim else 1
        votes = {}
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            i, j = x + dx, y + dy
            while 0 <= i < W and 0 <= j < H - 1:
                if (i, j) not in covered and frame[j][i] in (1, 2, 15):
                    votes[frame[j][i]] = votes.get(frame[j][i], 0) + 1
                    break
                i, j = i + dx, j + dy
        if not votes:
            return 1
        return max(votes, key=lambda c: (votes[c], c == 1, c == 15))

    def in_bound(self, r):
        bx, by, bw, bh = self.bound
        return r[0] >= bx and r[1] >= by and r[0] + r[2] <= bx + bw and r[1] + r[3] <= by + bh

    def on_floor(self, r, floor):
        return all(self.bg[j][i] in floor for (i, j) in cells(r))

    def on_ice(self, r):
        return any(self.bg[j][i] == ICE for (i, j) in cells(r))

    def player_blocked(self, r):
        return not self.in_bound(r) or not self.on_floor(r, PLAYER_FLOOR)

    def socket_rejects(self, r):
        for sx, sy, sw, sh in self.sockets:
            inner = (sx + 1, sy + 1, sw - 2, sh - 2)
            if overlap(r, inner) and (r[2], r[3]) != (inner[2], inner[3]):
                return True
        return False

    def block_blocked(self, r):
        return (not self.in_bound(r) or not self.on_floor(r, BLOCK_FLOOR)
                or (self.player and overlap(r, self.player)) or self.socket_rejects(r))

    def push_chain(self, seed, d):
        group = set(seed)
        momentum = None
        while True:
            grew = True
            while grew:
                grew = False
                moved = [shift(self.blocks[i], d) for i in group]
                for k, b in enumerate(self.blocks):
                    if k not in group and any(overlap(m, b) for m in moved):
                        group.add(k)
                        grew = True
            if momentum is not None and momentum <= 0:
                return
            moved = {i: shift(self.blocks[i], d) for i in group}
            if any(self.block_blocked(r) for r in moved.values()):
                return
            for i, r in moved.items():
                self.blocks[i] = r
            if any(self.on_ice(r) for r in moved.values()):
                momentum = MOMENTUM
            elif momentum is not None:
                momentum -= 1

    def arrow(self, d):
        if not self.player:
            return
        nxt = shift(self.player, d)
        hit = [k for k, b in enumerate(self.blocks) if overlap(nxt, b)]
        if hit:
            self.push_chain(hit, d)
        elif not self.player_blocked(nxt):
            self.player = nxt

    def click(self, x, y):
        for k, b in enumerate(self.blocks):
            if b[0] <= x < b[0] + b[2] and b[1] <= y < b[1] + b[3]:
                if self.player:
                    self.leave_husk(self.player)
                self.player = b
                del self.blocks[k]
                return

    def leave_husk(self, r):
        for (i, j), c in self.sprite_cells(r, True).items():
            self.bg[j][i] = 4 if c == 0 and self.is_centre(r, i, j) else c

    @staticmethod
    def is_centre(r, i, j):
        x, y, w, h = r
        return (w - 1) // 2 <= i - x <= w // 2 and (h - 1) // 2 <= j - y <= h // 2

    def touching_sides(self, r):
        x, y, w, h = r
        sides = set()
        for b in self.blocks:
            bx, by, bw, bh = b
            vert = bx < x + w and x < bx + bw
            horiz = by < y + h and y < by + bh
            if vert and by + bh == y:
                sides.add("top")
            if vert and by == y + h:
                sides.add("bottom")
            if horiz and bx + bw == x:
                sides.add("left")
            if horiz and bx == x + w:
                sides.add("right")
        return sides

    def sprite_cells(self, r, is_player):
        x, y, w, h = r
        out = {}
        sides = self.touching_sides(r) if is_player else set()
        for (i, j) in cells(r):
            c = 14
            if self.is_centre(r, i, j):
                c = 0 if is_player else 5
            if is_player and ((j == y and "top" in sides) or (j == y + h - 1 and "bottom" in sides)
                              or (i == x and "left" in sides) or (i == x + w - 1 and "right" in sides)):
                c = 0
            out[(i, j)] = c
        return out

    def render(self, n):
        out = [row[:] for row in self.bg]
        for b in self.blocks:
            for (i, j), c in self.sprite_cells(b, False).items():
                out[j][i] = c
        if self.player:
            for (i, j), c in self.sprite_cells(self.player, True).items():
                out[j][i] = c
        z = (n + 1) // 2
        for x in range(W):
            out[H - 1][x] = 0 if x >= W - z else 4
        return out


def transition_function(state, action, frame):
    frame = [[int(c) for c in row] for row in frame]
    if _memo["frame"] is not None and frame == _memo["frame"]:
        n, bg = _memo["n"], [row[:] for row in _memo["bg"]]
    else:
        z = hud_zeros(frame)
        n, bg = (2 * z - 1 if z else 0), None
    world = World(state, frame, bg)
    if isinstance(action, dict):
        if action.get("action_id") == 6:
            world.click(int(action.get("x", -1)), int(action.get("y", -1)))
    elif action in DIRS:
        world.arrow(DIRS[action])
    n += 1
    out = world.render(n)
    _memo.update(frame=[row[:] for row in out], n=n, bg=[row[:] for row in world.bg])
    return out
