# Mechanics: arrows move the player 3 cells; the player is blocked by frame colours 2/15 or the portal bbox. If its next
# rect overlaps blocks, those blocks are kicked (player stays) and slide 3/tick as a closed push chain until a member hits
# colour 2, the bound, the player, or a non-fitting socket interior; a sliding block also stops once its leading edge
# reaches a rim line of a socket it fits exactly (step-101 stop at y15). Click on a block = player takes its rect, old
# player stays as an inert husk (centre 4). HUD row 63 = ceil(n/2) zeros from the right (n hidden; rim-stop unconfirmed).
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
STEP = 3
_last = {"frame": None, "n": 0}


def cells(r):
    x, y, w, h = r
    return [(cx, cy) for cy in range(y, y + h) for cx in range(x, x + w)]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def shift(r, d, k=STEP):
    return (r[0] + d[0] * k, r[1] + d[1] * k, r[2], r[3])


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
        for r in self.movers():
            self.covered.update(cells(r))

    def movers(self):
        return ([self.player] if self.player else []) + self.blocks

    def in_bound(self, r):
        b = self.bound
        return r[0] >= b[0] and r[1] >= b[1] and r[0] + r[2] <= b[0] + b[2] and r[1] + r[3] <= b[1] + b[3]

    def colour_hit(self, r, bad):
        for cx, cy in cells(r):
            if (cx, cy) not in self.covered and self.frame[cy][cx] in bad:
                return True
        return False

    def player_blocked(self, r):
        return not self.in_bound(r) or self.colour_hit(r, (2, 15))

    def fits(self, s, b):
        return s[2] - 2 == b[2] and s[3] - 2 == b[3]

    def socket_rejects(self, b):
        for s in self.sockets:
            inner = (s[0] + 1, s[1] + 1, s[2] - 2, s[3] - 2)
            if overlap(b, inner) and not self.fits(s, b):
                return True
        return False

    def block_blocked(self, b):
        return (not self.in_bound(b) or self.colour_hit(b, (2,))
                or (self.player and overlap(b, self.player)) or self.socket_rejects(b))

    def rim_stop(self, b, d):
        lead = (b[1] if d[1] < 0 else b[1] + b[3] - 1) if d[0] == 0 else (b[0] if d[0] < 0 else b[0] + b[2] - 1)
        for s in self.sockets:
            if not self.fits(s, b):
                continue
            rims = (s[1], s[1] + s[3] - 1) if d[0] == 0 else (s[0], s[0] + s[2] - 1)
            if lead in rims:
                return True
        return False

    def push_chain(self, seed, d):
        moving = set(seed)
        moved = False
        while True:
            changed = True
            while changed:
                changed = False
                for i in list(moving):
                    nr = shift(self.blocks[i], d)
                    for j, o in enumerate(self.blocks):
                        if j not in moving and overlap(nr, o):
                            moving.add(j)
                            changed = True
            new = {i: shift(self.blocks[i], d) for i in moving}
            if any(self.block_blocked(r) for r in new.values()):
                return moved
            for i, r in new.items():
                self.blocks[i] = r
            moved = True
            if any(self.rim_stop(r, d) for r in new.values()):
                return moved

    def arrow(self, a):
        d = DIRS[a]
        nr = shift(self.player, d)
        if self.player_blocked(nr):
            return
        hit = [i for i, b in enumerate(self.blocks) if overlap(nr, b)]
        if hit:
            self.push_chain(hit, d)
        else:
            self.player = nr

    def click(self, x, y):
        for i, b in enumerate(self.blocks):
            if b[0] <= x < b[0] + b[2] and b[1] <= y < b[1] + b[3]:
                husk = self.player
                self.player = b
                del self.blocks[i]
                return husk
        return None


def centre_idx(n):
    return range((n - 1) // 2, n // 2 + 1)


def background(world, x, y):
    for s in world.sockets:
        if s[0] <= x < s[0] + s[2] and s[1] <= y < s[1] + s[3]:
            edge = x in (s[0], s[0] + s[2] - 1) or y in (s[1], s[1] + s[3] - 1)
            return 4 if edge else 1
    f = world.frame
    found = []
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cx, cy = x + dx, y + dy
        while 0 <= cx < 64 and 0 <= cy < 63 and (cx, cy) in world.covered:
            cx, cy = cx + dx, cy + dy
        if 0 <= cx < 64 and 0 <= cy < 63 and f[cy][cx] in (1, 15):
            found.append(f[cy][cx])
    if not found:
        return 1
    return max((1, 15), key=lambda c: (found.count(c), c == found[0]))


def paint_token(out, r, centre):
    x, y, w, h = r
    for cx, cy in cells(r):
        out[cy][cx] = 14
    for j in centre_idx(h):
        for i in centre_idx(w):
            out[y + j][x + i] = centre


def touching(p, b):
    px, py, pw, ph = p
    bx, by, bw, bh = b
    vert = px < bx + bw and bx < px + pw
    horz = py < by + bh and by < py + ph
    return {"top": vert and by + bh == py, "bottom": vert and by == py + ph,
            "left": horz and bx + bw == px, "right": horz and bx == px + pw}


def paint_player(out, p, blocks):
    paint_token(out, p, 0)
    x, y, w, h = p
    for b in blocks:
        t = touching(p, b)
        for i in range(w):
            if t["top"]:
                out[y][x + i] = 0
            if t["bottom"]:
                out[y + h - 1][x + i] = 0
        for j in range(h):
            if t["left"]:
                out[y + j][x] = 0
            if t["right"]:
                out[y + j][x + w - 1] = 0


def hud(out, n):
    z = (n + 1) // 2
    for x in range(64):
        out[63][x] = 0 if x >= 64 - z else 4


def transition_function(state, action, frame):
    zeros = sum(1 for v in frame[63] if v == 0)
    if _last["frame"] is not None and frame == _last["frame"]:
        n = _last["n"]
    else:
        n = 0 if zeros == 0 else 2 * zeros - 1
    world = World(state, frame)
    out = [list(row) for row in frame]
    for cx, cy in world.covered:
        if cy < 63:
            out[cy][cx] = background(world, cx, cy)
    husk = None
    a = action["action_id"] if isinstance(action, dict) else action
    if world.player is not None:
        if a in DIRS:
            world.arrow(a)
        elif a == 6:
            husk = world.click(action.get("x", -1), action.get("y", -1))
    if husk:
        paint_token(out, husk, 4)
    for b in world.blocks:
        paint_token(out, b, 5)
    if world.player:
        paint_player(out, world.player, world.blocks)
    n += 1
    hud(out, n)
    out = [[int(v) for v in row] for row in out]
    _last["frame"] = [list(row) for row in out]
    _last["n"] = n
    return out
