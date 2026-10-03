# transition_function implements the observed mechanics:
# - Action 1: move player up 5 cells; if at top (y==5) jump right 25 cells.
# - Action 2: move player down 5 cells; if at bottom (y==45) jump left 25 cells.
# - Action 3: move player left 5 cells; if at left edge (x==9) jump down 25 cells.
# - Action 4: move player right 5 cells; if at right edge (x==54) jump down 25 cells.
# - Each action costs 2 budget points. The step_bar counter shows the budget
#   (width = budget) and moves right 2 cells per step. When it would move
#   past x=35 it resets to x=13, budget=42 and consumes one refuel ring.
# - When budget reaches 8 the recolor button disappears and the HUD glyph
#   progress changes from "12" to "9".
# - The diamond target toggles (appears/disappears) on every vertical move
#   (action 1 or 2) while the player is in column x==49.
# - The player leaves a trail: previous 5×5 cells become colour 'c',
#   new cells become colour '9'.

import copy

# persistent memory across calls
_last_state = None          # deep‑copied state we returned last time
_last_player = None        # (x, y) of player in that state
_original_objects = {}     # name → object dict from the very first state
_diamond_template = None   # stored diamond object for rebirth


def _find(obj_list, **kw):
    for o in obj_list:
        if all(o.get(k) == v for k, v in kw.items()):
            return o
    return None


def transition_function(state, action, frame=None):
    global _last_state, _last_player, _original_objects, _diamond_template

    # ------------------------------------------------------------------
    # continuity check – if the incoming state does not match what we
    # returned last time, forget all cached data and initialise from this
    # state.
    # ------------------------------------------------------------------
    if _last_state is None or len(state) != len(_last_state):
        _reset_memory(state)
    else:
        # crude check: compare player position
        cur_player = _find(state, type='player')
        if cur_player is None or (cur_player['x'], cur_player['y']) != _last_player:
            _reset_memory(state)

    # work on copies so we never mutate the caller's data
    new_state = copy.deepcopy(state)
    player = _find(new_state, type='player')
    step_bar = _find(new_state, name='step_bar')
    hud_glyph = _find(new_state, name='hud_glyph')
    recolor_token = _find(new_state, name='recolor_token')
    diamond = _find(new_state, name='diamond')
    # ------------------------------------------------------------------
    # 1. PLAYER MOVEMENT
    # ------------------------------------------------------------------
    old_x, old_y = player['x'], player['y']
    act = action if not isinstance(action, dict) else action.get('action_id')
    if act == 1:          # up
        if player['y'] > 5:
            player['y'] -= 5
        else:  # at top edge – jump right
            player['x'] += 25
    elif act == 2:        # down
        if player['y'] < 45:
            player['y'] += 5
        else:  # bottom edge – jump left (not seen but symmetric)
            player['x'] -= 25
    elif act == 3:        # left
        if player['x'] > 9:
            player['x'] -= 5
        else:  # left edge – jump down
            player['y'] += 25
    elif act == 4:        # right
        if player['x'] < 54:
            player['x'] += 5
        else:  # right edge – jump down
            player['y'] += 25
    # actions 5 and 6 are unused in the observed data
    # ------------------------------------------------------------------
    # 2. UPDATE FRAME (if supplied)
    # ------------------------------------------------------------------
    if frame is not None:
        # paint old footprint with colour 'c'
        for dy in range(player['h']):
            for dx in range(player['w']):
                yy = old_y + dy
                xx = old_x + dx
                if 0 <= yy < 64 and 0 <= xx < 64:
                    frame[yy][xx] = 'c'
        # paint new footprint with colour '9'
        for dy in range(player['h']):
            for dx in range(player['w']):
                yy = player['y'] + dy
                xx = player['x'] + dx
                if 0 <= yy < 64 and 0 <= xx < 64:
                    frame[yy][xx] = '9'

    # ------------------------------------------------------------------
    # 3. BUDGET COUNTER (step_bar)
    # ------------------------------------------------------------------
    # extract current budget from tags
    cur_budget = int(step_bar['tags'][0][1])
    new_budget = cur_budget - 2
    new_x = step_bar['x'] + 2

    reset_occurred = False
    if new_x > 35:                     # overflow → reset
        reset_occurred = True
        new_budget = 42
        new_x = 13

    step_bar['w'] = new_budget
    step_bar['x'] = new_x
    step_bar['tags'] = [['budget', str(new_budget)]]

    # ------------------------------------------------------------------
    # 4. REFUEL RING CONSUMPTION (on reset)
    # ------------------------------------------------------------------
    if reset_occurred:
        # remove the first refuel object we can find
        refuel = _find(new_state, type='refuel')
        if refuel:
            new_state.remove(refuel)

    # ------------------------------------------------------------------
    # 5. RECOLOR TOKEN DISAPPEARANCE (budget reaches 8)
    # ------------------------------------------------------------------
    if new_budget == 8 and recolor_token:
        new_state.remove(recolor_token)
        # update HUD glyph progress from 12 → 9
        if hud_glyph and hud_glyph['tags']:
            # replace the numeric progress entry (second element)
            for i, tag in enumerate(hud_glyph['tags']):
                if isinstance(tag, list) and tag[0] == 'progress':
                    hud_glyph['tags'][i] = ['progress', '9', 'pending']
                    break

    # ------------------------------------------------------------------
    # 6. DIAMOND TOGGLE (vertical moves while player in column 49)
    # ------------------------------------------------------------------
    if act in (1, 2) and player['x'] == 49:
        if diamond:
            # disappear
            new_state.remove(diamond)
        else:
            # re‑appear using the stored template
            if _diamond_template:
                new_state.append(copy.deepcopy(_diamond_template))

    # ------------------------------------------------------------------
    # 7. STORE continuity data for the next call
    # ------------------------------------------------------------------
    _last_state = copy.deepcopy(new_state)
    _last_player = (player['x'], player['y'])

    # Return the appropriate thing
    if frame is not None:
        return frame
    return new_state


def _reset_memory(state):
    """Initialise the persistent caches from a fresh state."""
    global _last_state, _last_player, _original_objects, _diamond_template
    _last_state = None
    _last_player = None
    _original_objects = {}
    _diamond_template = None
    for obj in state:
        _original_objects[obj['name']] = copy.deepcopy(obj)
        if obj['name'] == 'diamond':
            _diamond_template = copy.deepcopy(obj)
