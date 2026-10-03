# Fuel-maze: player 5x5 moves on a 5-grid; A1-4 = y-5, y+5, x-5, x+5. Every
# action burns 2 budget: step_bar w-=2, x=55-w, tag=str(w); blocked moves burn
# too. Invisible walls {(39,5),(34,10)} block; cell (9,5) is a portal to (34,5).
# A refuel ring fully covered is consumed and refills the bar to 42; other
# layer-0 objects fully covered are occluded and restored when uncovered
# (hidden memory gated on continuity). Unconfirmed: what unlocks the chamber —
# a move whose landing rect overlaps a 'chamber' object is rejected, no burn.
import json, copy

WALLS = {(39, 5), (34, 10)}
PORTAL = {(9, 5): (34, 5)}
MOVES = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
MEM = {'last': None, 'occ': []}


def canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def covered(o, px, py, pw, ph):
    return (px <= o['x'] and py <= o['y']
            and o['x'] + o['w'] <= px + pw and o['y'] + o['h'] <= py + ph)


def overlaps(o, px, py, pw, ph):
    return px < o['x'] + o['w'] and o['x'] < px + pw \
        and py < o['y'] + o['h'] and o['y'] < py + ph


def transition_function(state, action):
    state = copy.deepcopy(state)
    if MEM['last'] is not None and canon(state) == canon(MEM['last']):
        occ = MEM['occ']
    else:
        occ = []
    out = state
    player = next((o for o in out if o.get('type') == 'player'), None)
    refuel = False
    if player is None:
        out.extend(occ)
        occ = []
    else:
        pw, ph = player['w'], player['h']
        dx, dy = MOVES.get(action if isinstance(action, int) else -1, (0, 0))
        nx, ny = player['x'] + dx, player['y'] + dy
        if (nx, ny) not in WALLS and 0 <= nx <= 58 and 0 <= ny <= 58:
            nx, ny = PORTAL.get((nx, ny), (nx, ny))
            chamber = any(overlaps(o, nx, ny, pw, ph) for o in out
                          if 'chamber' in o.get('tags', ()))
            if chamber:
                # locked exit chamber: whole action rejected, budget kept
                MEM['last'] = out
                MEM['occ'] = occ
                return out
            player['x'], player['y'] = nx, ny
        px, py = player['x'], player['y']
        new_occ = []
        for o in occ:
            if covered(o, px, py, pw, ph):
                new_occ.append(o)
            else:
                out.append(o)
        for o in list(out):
            if o is player or o.get('layer', 0) != 0:
                continue
            if covered(o, px, py, pw, ph):
                out.remove(o)
                if o.get('type') == 'refuel':
                    refuel = True
                else:
                    new_occ.append(o)
        occ = new_occ
    for o in out:
        if o.get('type') == 'counter' and 'budget' in o.get('tags', ()):
            w = 42 if refuel else o['w'] - 2
            o['w'], o['x'], o['tags'] = w, 55 - w, ['budget', str(w)]
    MEM['last'] = out
    MEM['occ'] = occ
    return out
