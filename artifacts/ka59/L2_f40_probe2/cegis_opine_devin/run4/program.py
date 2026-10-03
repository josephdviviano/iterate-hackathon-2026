# Arrows move the player three cells; contacting a block kicks a sliding chain.
# Colour 2 blocks all tokens; colour 15 blocks only the player; targets reject wrong shapes.
# A kick stops at a matching target's directional rim line; transverse alignment is unconfirmed.
# Clicking possesses a block, leaving a husk; touching player edges are black.
# HUD counts every action at half rate with continuity-gated parity; actions 5/7 are unobserved.

def matching_rim_stops(block, direction, targets):
    dx, dy = direction
    for target in targets:
        if (block['w'], block['h']) != (target['w'] - 2, target['h'] - 2):
            continue
        if dy < 0 and block['y'] == target['y'] + target['h'] - 1:
            return True
        if dy > 0 and block['y'] + block['h'] - 1 == target['y']:
            return True
        if dx < 0 and block['x'] == target['x'] + target['w'] - 1:
            return True
        if dx > 0 and block['x'] + block['w'] - 1 == target['x']:
            return True
    return False


def overlaps(a, b):
    return (a['x'] < b['x'] + b['w'] and b['x'] < a['x'] + a['w']
            and a['y'] < b['y'] + b['h'] and b['y'] < a['y'] + a['h'])


def shifted(obj, dx, dy):
    return dict(obj, x=obj['x'] + dx, y=obj['y'] + dy)


def cells(obj):
    for y in range(obj['y'], obj['y'] + obj['h']):
        for x in range(obj['x'], obj['x'] + obj['w']):
            yield x, y


def freeze(value):
    if isinstance(value, dict):
        return tuple((k, freeze(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


def state_key(state):
    return tuple(sorted(freeze(o) for o in state))


class World:
    def __init__(self, state, frame, background=None):
        self.state = state
        self.frame = frame
        self.player = next(dict(o) for o in state if o['type'] == 'player')
        self.blocks = [dict(o) for o in state if o['type'] == 'block']
        self.targets = [o for o in state if o['type'] == 'target']
        self.portal = next((o for o in state if o['type'] == 'portal'),
                           dict(x=0, y=0, w=64, h=63))
        occupied = set()
        for obj in [self.player] + self.blocks:
            occupied.update(cells(obj))
        self.background = [row[:] for row in frame]
        for x, y in occupied:
            if background is not None:
                colour = background[y][x]
            else:
                colour = self.terrain_under(x, y, occupied)
            self.background[y][x] = colour

    def terrain_under(self, x, y, occupied):
        for target in self.targets:
            if (target['x'] <= x < target['x'] + target['w']
                    and target['y'] <= y < target['y'] + target['h']):
                rim = (x in (target['x'], target['x'] + target['w'] - 1)
                       or y in (target['y'], target['y'] + target['h'] - 1))
                return 4 if rim else 1
        votes = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            xx, yy = x + dx, y + dy
            while 0 <= xx < 64 and 0 <= yy < 63:
                if (xx, yy) not in occupied:
                    colour = self.frame[yy][xx]
                    if colour in (1, 2, 15):
                        votes.append(colour)
                        break
                xx, yy = xx + dx, yy + dy
        return max((1, 15, 2), key=lambda c: votes.count(c))

    def terrain_blocks(self, obj, allowed):
        p = self.portal
        if (obj['x'] < p['x'] or obj['y'] < p['y']
                or obj['x'] + obj['w'] > p['x'] + p['w']
                or obj['y'] + obj['h'] > p['y'] + p['h']):
            return True
        return any(self.background[y][x] not in allowed for x, y in cells(obj))

    def target_rejects(self, block):
        for target in self.targets:
            interior = dict(target, x=target['x'] + 1, y=target['y'] + 1,
                            w=target['w'] - 2, h=target['h'] - 2)
            if overlaps(block, interior) and (block['w'], block['h']) != (interior['w'], interior['h']):
                return True
        return False

    def player_blocked(self, player):
        return self.terrain_blocks(player, (1, 4))

    def block_blocked(self, block):
        return (self.terrain_blocks(block, (1, 4, 15))
                or overlaps(block, self.player) or self.target_rejects(block))

    def update_blocks(self, hit, direction):
        dx, dy = direction
        group = set(hit)
        for tick in range(64):
            for substep in range(3):
                while True:
                    extra = {j for i in group for j, b in enumerate(self.blocks)
                             if j not in group and overlaps(shifted(self.blocks[i], dx, dy), b)}
                    if not extra:
                        break
                    group.update(extra)
                moved = {i: shifted(self.blocks[i], dx, dy) for i in group}
                if any(self.block_blocked(b) for b in moved.values()):
                    return
                for i, b in moved.items():
                    self.blocks[i] = b
            if any(matching_rim_stops(self.blocks[i], direction, self.targets) for i in group):
                return

    def update_player(self, action):
        directions = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        if isinstance(action, int) and action in directions:
            dx, dy = directions[action]
            destination = shifted(self.player, 3 * dx, 3 * dy)
            hit = [i for i, b in enumerate(self.blocks) if overlaps(destination, b)]
            if hit:
                self.update_blocks(hit, (dx, dy))
            elif not self.player_blocked(destination):
                self.player = destination
        elif isinstance(action, dict) and action.get('action_id') == 6:
            for i, b in enumerate(self.blocks):
                if (b['x'] <= action['x'] < b['x'] + b['w']
                        and b['y'] <= action['y'] < b['y'] + b['h']):
                    self.leave_husk()
                    self.player = dict(self.player, **{k: b[k] for k in ('x', 'y', 'w', 'h')})
                    self.blocks.pop(i)
                    break

    def leave_husk(self):
        p = self.player
        for x, y in cells(p):
            self.background[y][x] = self.frame[y][x]
        self.draw_centre(self.background, p, 4)

    @staticmethod
    def draw_centre(frame, obj, colour):
        for yy in range((obj['h'] - 1) // 2, obj['h'] // 2 + 1):
            for xx in range((obj['w'] - 1) // 2, obj['w'] // 2 + 1):
                frame[obj['y'] + yy][obj['x'] + xx] = colour

    def touching_sides(self):
        p = self.player
        sides = set()
        for b in self.blocks:
            horizontal = p['x'] < b['x'] + b['w'] and b['x'] < p['x'] + p['w']
            vertical = p['y'] < b['y'] + b['h'] and b['y'] < p['y'] + p['h']
            if horizontal and b['y'] + b['h'] == p['y']:
                sides.add('top')
            if horizontal and p['y'] + p['h'] == b['y']:
                sides.add('bottom')
            if vertical and b['x'] + b['w'] == p['x']:
                sides.add('left')
            if vertical and p['x'] + p['w'] == b['x']:
                sides.add('right')
        return sides

    def render(self, actions):
        result = [row[:] for row in self.background]
        for obj in self.blocks + [self.player]:
            for x, y in cells(obj):
                result[y][x] = 14
            self.draw_centre(result, obj, 0 if obj['type'] == 'player' else 5)
        p = self.player
        for side in self.touching_sides():
            if side in ('top', 'bottom'):
                y = p['y'] if side == 'top' else p['y'] + p['h'] - 1
                for x in range(p['x'], p['x'] + p['w']):
                    result[y][x] = 0
            else:
                x = p['x'] if side == 'left' else p['x'] + p['w'] - 1
                for y in range(p['y'], p['y'] + p['h']):
                    result[y][x] = 0
        used = min(64, (actions + 1) // 2)
        result[63] = [4] * (64 - used) + [0] * used
        return result

    def predicted_state(self):
        blocks = sorted(self.blocks, key=lambda b: (b['y'], b['x']))
        for i, b in enumerate(blocks):
            b['name'] = 'token_' + str(i)
        return [dict(o) for o in self.state if o['type'] not in ('player', 'block')] + [self.player] + blocks


_memory = None


def transition_function(state, action, frame):
    global _memory
    continuous = (_memory is not None and _memory['frame'] == frame
                  and _memory['state'] == state_key(state))
    zeros = sum(c == 0 for c in frame[63])
    actions = _memory['actions'] if continuous else max(0, 2 * zeros - 1)
    background = _memory['background'] if continuous else None
    world = World(state, frame, background)
    world.update_player(action)
    actions += 1
    result = world.render(actions)
    _memory = dict(frame=[row[:] for row in result], state=state_key(world.predicted_state()),
                   actions=actions, background=world.background)
    return result
