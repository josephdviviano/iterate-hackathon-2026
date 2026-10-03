# Mechanics: player (5x5, layer 1) moves 5 cells per A1-4 (up/down/left/right); invisible maze walls
# block moves (cells inferred from blocked moves) and a portal cell warps the player. Each accepted
# action costs 2 step_bar units (x = 55 - w, tag = str(w)); a refuel ring fully covered is consumed and
# refills the bar to 42; layer-0 targets covered by the player are occluded (remembered while states are
# continuous). Guard: a move into the legend's (locked chamber's) cell while hud_glyph is 'pending' is
# rejected outright and costs no step. Unconfirmed: what unlocks the chamber; the unseen step-133 jump.
import copy, json

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALL_CELLS = {(39, 5), (34, 10)}          # invisible maze cells (player top-left) inferred from blocks
PORTALS = {(9, 5): (34, 5)}               # entering the key cell lands on the value cell
BAR_RIGHT, BAR_FULL, BAR_COST = 55, 42, 2

_mem = {"last": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _box(o):
    return o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"]


def _covers(p, o):
    px0, py0, px1, py1 = _box(p)
    ox0, oy0, ox1, oy1 = _box(o)
    return px0 <= ox0 and py0 <= oy0 and ox1 <= px1 and oy1 <= py1


def _find(state, **kw):
    return [o for o in state if all(o.get(k) == v for k, v in kw.items())]


# ---- guards -----------------------------------------------------------------
def blocked_by_wall(dest):
    return dest in WALL_CELLS


def chamber_locked(state, player, dest):
    """Destination cell holds the legend (exit chamber) and the HUD key is still pending."""
    moved = dict(player, x=dest[0], y=dest[1])
    legends = [o for o in state if "legend" in o.get("tags", [])]
    pending = any("pending" in o.get("tags", []) for o in state if o["type"] == "counter")
    return pending and any(_covers(moved, l) for l in legends)


# ---- per-type updates -------------------------------------------------------
def update_player(state, player, action):
    """Returns (new player, accepted). accepted=False means the action is ignored entirely."""
    if action not in MOVES:
        return player, True
    dx, dy = MOVES[action]
    dest = (player["x"] + dx, player["y"] + dy)
    if chamber_locked(state, player, dest):
        return player, False
    if blocked_by_wall(dest):
        return player, True
    dest = PORTALS.get(dest, dest)
    return dict(player, x=dest[0], y=dest[1]), True


def set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = bar["tags"][:-1] + [str(w)]
    return bar


def transition_function(state, action):
    state = copy.deepcopy(state)
    hidden = _mem["hidden"] if _mem["last"] == _canon(state) else []
    hidden = copy.deepcopy(hidden)
    act = action["action_id"] if isinstance(action, dict) else action

    players = _find(state, type="player")
    if not players:
        return state
    player = players[0]
    new_player, accepted = update_player(state, player, act)
    if not accepted:
        _mem["last"], _mem["hidden"] = _canon(state), hidden
        return state

    out = []
    bar = None
    refuel = False
    for o in state + hidden:
        if o is player:
            out.append(new_player)
            continue
        if o["type"] == "refuel" and _covers(new_player, o):
            refuel = True
            continue
        if o["name"] == "step_bar":
            bar = o
        out.append(o)
    if bar is not None:
        set_bar(bar, BAR_FULL if refuel else bar["w"] - BAR_COST)

    visible, new_hidden = [], []
    for o in out:
        if o is not new_player and o["type"] == "target" and o.get("layer", 0) < new_player["layer"] \
                and _covers(new_player, o):
            new_hidden.append(o)
        else:
            visible.append(o)
    _mem["last"], _mem["hidden"] = _canon(visible), new_hidden
    return visible
