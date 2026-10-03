# program.py
# Mechanics inferred:
# - Actions 1‑4 move the player 5 cells (up, down, left, right).
# - Every action costs 2 budget units: step_bar width -=2, x +=2, budget tag decreases.
# - Colliding with a refuel or target removes it and resets the budget (width=42, x=13, budget=42).
#   Removed targets respawn when the budget is reset.
# - Colliding with a button removes it (no budget reset).
# - Rendering: floor colour = '3'.
#   player pixels use their numeric colour (12→'c', 9→'9', etc.).
#   step_bar = 'b', refuel = 'c', target = 'e', button = 'd'.
# - Objects are drawn after clearing the whole previous layer.

import copy

# Global cache of the original target objects (for respawn after a reset)
_ORIGINAL_TARGETS = None
_MAX_BUDGET = 42
_STEP = 2  # budget decrement per action


def _rect_overlap(ax, ay, aw, ah, bx, by, bw, bh):
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _clear_rect(grid, x, y, w, h, fill='3'):
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            if 0 <= yy < len(grid) and 0 <= xx < len(grid[0]):
                grid[yy][xx] = fill


def _draw_rect(grid, x, y, w, h, colour):
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            if 0 <= yy < len(grid) and 0 <= xx < len(grid[0]):
                grid[yy][xx] = colour


def _draw_player(grid, obj):
    px, py = obj['x'], obj['y']
    for dy, row in enumerate(obj['pixels']):
        for dx, col in enumerate(row):
            colour = format(col, 'x')  # int -> hex digit
            _draw_rect(grid, px + dx, py + dy, 1, 1, colour)


def _clone_state(state):
    return [copy.deepcopy(o) for o in state]


def transition_function(state, action, frame):
    """
    Predict the after‑frame given the before‑frame, the object list and the action.
    """
    global _ORIGINAL_TARGETS

    # ----------------------------------------------------------------------
    # 1.  Convert the incoming frame (list of strings) to a mutable grid.
    # ----------------------------------------------------------------------
    grid = [list(row) for row in frame]

    # ----------------------------------------------------------------------
    # 2.  Work on a copy of the object list – we will mutate it to obtain the
    #     after‑state.
    # ----------------------------------------------------------------------
    objs = _clone_state(state)

    # ----------------------------------------------------------------------
    # 3.  Identify special objects.
    # ----------------------------------------------------------------------
    player = next(o for o in objs if o['type'] == 'player')
    step_bar = next(o for o in objs if o['name'] == 'step_bar')
    # keep a list of other objects (may be removed/added)
    others = [o for o in objs if o not in (player, step_bar)]

    # ----------------------------------------------------------------------
    # 4.  Apply movement based on the action.
    # ----------------------------------------------------------------------
    dx = dy = 0
    if action == 1:   # up
        dy = -5
    elif action == 2:  # down
        dy = 5
    elif action == 3:  # left
        dx = -5
    elif action == 4:  # right
        dx = 5
    # actions 5,7,6 (click) are not used in the observed data – they cause no move.
    player['x'] += dx
    player['y'] += dy

    # ----------------------------------------------------------------------
    # 5.  Collision handling.
    # ----------------------------------------------------------------------
    reset_budget = False
    removed = set()          # names of objects that disappear this step
    for obj in others:
        if _rect_overlap(player['x'], player['y'], player['w'], player['h'],
                         obj['x'], obj['y'], obj['w'], obj['h']):
            if obj['type'] in ('refuel', 'target'):
                reset_budget = True
                removed.add(obj['name'])
            elif obj['type'] == 'button':
                removed.add(obj['name'])
    # filter out removed objects
    others = [o for o in others if o['name'] not in removed]

    # ----------------------------------------------------------------------
    # 6.  Budget (step_bar) update.
    # ----------------------------------------------------------------------
    if reset_budget:
        step_bar['w'] = _MAX_BUDGET
        step_bar['x'] = 13
        step_bar['tags'] = ['budget', str(_MAX_BUDGET)]
    else:
        step_bar['w'] = max(0, step_bar['w'] - _STEP)
        step_bar['x'] = step_bar['x'] + _STEP
        new_budget = max(0, int(step_bar['tags'][1]) - _STEP)
        step_bar['tags'] = ['budget', str(new_budget)]

    # ----------------------------------------------------------------------
    # 7.  Respawn targets after a reset (only the original ones).
    # ----------------------------------------------------------------------
    if _ORIGINAL_TARGETS is None:
        # store the original target objects from the very first call
        _ORIGINAL_TARGETS = [o for o in objs if o['type'] == 'target']

    if reset_budget:
        # any original target not currently present should be added back
        present_names = {o['name'] for o in others}
        for tgt in _ORIGINAL_TARGETS:
            if tgt['name'] not in present_names:
                # clone to avoid mutating the stored original
                others.append(copy.deepcopy(tgt))

    # ----------------------------------------------------------------------
    # 8.  Rendering.
    # ----------------------------------------------------------------------
    # 8a. Clear everything that can change (player, step_bar and all other
    #     dynamic objects) to floor colour.
    _clear_rect(grid, player['x'] - dx, player['y'] - dy,
                player['w'], player['h'], '3')
    _clear_rect(grid, step_bar['x'] - (_STEP if not reset_budget else 0),
                step_bar['y'], step_bar['w'] + (_STEP if not reset_budget else 0),
                step_bar['h'], '3')
    for o in others:
        _clear_rect(grid, o['x'], o['y'], o['w'], o['h'], '3')

    # 8b. Draw static objects that never change (walls, floor etc.) are already
    #     present in the copied grid – we only need to draw the dynamic ones.

    # draw step_bar
    _draw_rect(grid, step_bar['x'], step_bar['y'],
               step_bar['w'], step_bar['h'], 'b')

    # draw other dynamic objects
    for o in others:
        colour = '3'  # default fallback
        if o['type'] == 'refuel':
            colour = 'c'
        elif o['type'] == 'target':
            colour = 'e'
        elif o['type'] == 'button':
            colour = 'd'
        else:
            # unknown dynamic type – leave as floor
            continue
        _draw_rect(grid, o['x'], o['y'], o['w'], o['h'], colour)

    # draw player on top
    _draw_player(grid, player)

    # ----------------------------------------------------------------------
    # 9.  Convert grid back to list‑of‑strings format.
    # ----------------------------------------------------------------------
    return [''.join(row) for row in grid]
