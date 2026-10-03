# Arrows move the player three cells; contacting a block kicks a sliding chain.
# Frame colours 1/4 are player floor; blocks also cross 15, but never colour 2.
# Mismatched sockets stop blocks at interiors; exact-fit entry is unconfirmed.
# Clicking possesses a block, leaving a colour-4-core husk; touching edges are 0.
# HUD spends one cell per two actions; isolated parity and actions 5/7 are uncertain.
import json
from collections import Counter

_last_frame = None
_last_state = None
_count = 0
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def state_key(state):
    return tuple(sorted(json.dumps(o, sort_keys=True) for o in state))


def cells(o):
    return ((x, y) for y in range(o['y'], o['y'] + o['h'])
            for x in range(o['x'], o['x'] + o['w']))


def overlaps(a, b):
    return (a['x'] < b['x'] + b['w'] and b['x'] < a['x'] + a['w']
            and a['y'] < b['y'] + b['h'] and b['y'] < a['y'] + a['h'])


def shifted(o, dx, dy):
    return dict(o, x=o['x'] + dx, y=o['y'] + dy)


def socket_rejects(block, socket):
    inside = dict(socket, x=socket['x'] + 1, y=socket['y'] + 1,
                  w=socket['w'] - 2, h=socket['h'] - 2)
    return (overlaps(block, inside)
            and (block['w'], block['h']) != (inside['w'], inside['h']))


def draw_token(frame, obj, core):
    for x, y in cells(obj):
        cx, cy = x - obj['x'], y - obj['y']
        centre = ((obj['w'] - 1) // 2 <= cx <= obj['w'] // 2
                  and (obj['h'] - 1) // 2 <= cy <= obj['h'] // 2)
        frame[y][x] = core if centre else 14


def touching_edges(player, block):
    x, y, w, h = (player[k] for k in ('x', 'y', 'w', 'h'))
    bx, by, bw, bh = (block[k] for k in ('x', 'y', 'w', 'h'))
    if max(y, by) < min(y + h, by + bh):
        if bx + bw == x:
            yield ((x, yy) for yy in range(y, y + h))
        if x + w == bx:
            yield ((x + w - 1, yy) for yy in range(y, y + h))
    if max(x, bx) < min(x + w, bx + bw):
        if by + bh == y:
            yield ((xx, y) for xx in range(x, x + w))
        if y + h == by:
            yield ((xx, y + h - 1) for xx in range(x, x + w))


class World:
    def __init__(self, state, frame):
        self.objects = [dict(o) for o in state]
        self.player = next((o for o in self.objects if o['type'] == 'player'), None)
        self.blocks = [o for o in self.objects if o['type'] == 'block']
        self.sockets = [o for o in self.objects if o['type'] == 'target']
        self.bg = [row[:] for row in frame]
        movers = self.blocks + ([self.player] if self.player else [])
        covered = {p for o in movers for p in cells(o)}
        for x, y in covered:
            self.bg[y][x] = self.terrain(x, y, frame, covered)

    def terrain(self, x, y, frame, covered):
        for s in self.sockets:
            if s['x'] <= x < s['x'] + s['w'] and s['y'] <= y < s['y'] + s['h']:
                return 4 if (x in (s['x'], s['x'] + s['w'] - 1)
                             or y in (s['y'], s['y'] + s['h'] - 1)) else 1
        nearby = []
        for dx, dy in DIRS.values():
            xx, yy = x + dx, y + dy
            while 0 <= xx < 64 and 0 <= yy < 63:
                if (xx, yy) not in covered and frame[yy][xx] in (1, 2, 15):
                    nearby.append(frame[yy][xx])
                    break
                xx, yy = xx + dx, yy + dy
        return Counter(nearby).most_common(1)[0][0] if nearby else 1

    def terrain_allows(self, obj, is_player):
        floor = (1, 4) if is_player else (1, 4, 15)
        return all(0 <= x < 64 and 0 <= y < 63 and self.bg[y][x] in floor
                   for x, y in cells(obj))

    def block_allowed(self, obj):
        return (self.terrain_allows(obj, False)
                and not overlaps(obj, self.player)
                and not any(socket_rejects(obj, s) for s in self.sockets))

    def update_blocks(self, hit, dx, dy):
        chain = set(hit)
        while True:
            while True:
                candidates = {i: shifted(self.blocks[i], dx, dy) for i in chain}
                added = {j for j, b in enumerate(self.blocks) if j not in chain
                         and any(overlaps(a, b) for a in candidates.values())}
                if not added:
                    break
                chain.update(added)
            if not all(self.block_allowed(o) for o in candidates.values()):
                return
            for i, o in candidates.items():
                self.blocks[i].update(x=o['x'], y=o['y'])

    def update_player(self, action):
        if self.player is None:
            return
        if isinstance(action, dict) and action.get('action_id') == 6:
            click = dict(x=action['x'], y=action['y'], w=1, h=1)
            chosen = next((b for b in self.blocks if overlaps(click, b)), None)
            if chosen is not None:
                draw_token(self.bg, self.player, 4)
                self.player.update({k: chosen[k] for k in ('x', 'y', 'w', 'h')})
                self.blocks.remove(chosen)
                self.objects.remove(chosen)
        elif isinstance(action, int) and action in DIRS:
            dx, dy = DIRS[action]
            dest = shifted(self.player, 3 * dx, 3 * dy)
            hit = [i for i, b in enumerate(self.blocks) if overlaps(dest, b)]
            if hit:
                self.update_blocks(hit, dx, dy)
            elif self.terrain_allows(dest, True):
                self.player.update(x=dest['x'], y=dest['y'])

    def render(self):
        result = [row[:] for row in self.bg]
        for b in self.blocks:
            draw_token(result, b, 5)
        if self.player is not None:
            draw_token(result, self.player, 0)
            for b in self.blocks:
                for edge in touching_edges(self.player, b):
                    for x, y in edge:
                        result[y][x] = 0
        for i, b in enumerate(sorted(self.blocks, key=lambda b: (b['y'], b['x']))):
            prefix, sep, suffix = b.get('name', '').rpartition('_')
            if sep and suffix.isdigit():
                b['name'] = prefix + '_' + str(i)
        return result


def transition_function(state, action, frame):
    global _last_frame, _last_state, _count
    continuing = frame == _last_frame and state_key(state) == _last_state
    count = _count if continuing else 2 * frame[-1].count(0)
    world = World(state, frame)
    world.update_player(action)
    result = world.render()
    count += 1
    used = min(64, (count + 1) // 2)
    result[-1] = [4] * (64 - used) + [0] * used
    _count = count
    _last_frame = [row[:] for row in result]
    _last_state = state_key(world.objects)
    return result
