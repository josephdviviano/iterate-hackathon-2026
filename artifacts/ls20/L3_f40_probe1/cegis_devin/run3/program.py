# Mechanics: player 5x5 moves 5 cells per A1-4 (up/down/left/right); hidden walls {(39,5),(34,10)} block entry
#   (bump still costs budget); entering (9,5) warps to (34,5). A move leaving the playfield (board edge or the HUD
#   band from the legend/counters' top row down) is rejected entirely: no move, no budget spent (step-134 case).
#   step_bar -2 per spent action (x=55-w, tag str(w)); fully covered ring consumed -> bar refilled to 42.
#   Covered target occluded, restored when uncovered (continuity-gated memory). Unconfirmed: bar at 0, HUD band = bound.
import copy

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BOARD = 64
BAR_RIGHT, BAR_FULL, BAR_COST = 55, 42, 2

_memory = {"last": None, "hidden": []}


def _canon(state):
    return sorted(repr(sorted(o.items())) for o in state)


def _covers(p, o):
    return (p["x"] <= o["x"] and o["x"] + o["w"] <= p["x"] + p["w"]
            and p["y"] <= o["y"] and o["y"] + o["h"] <= p["y"] + p["h"])


def _is_hud(o):
    return o["type"] == "counter" or "legend" in o.get("tags", [])


def _in_playfield(x, y, p, state):
    band_top = min([o["y"] for o in state if _is_hud(o)] + [BOARD])
    return x >= 0 and y >= 0 and x + p["w"] <= BOARD and y + p["h"] <= band_top


def _set_bar(bar, w):
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def transition_function(state, action):
    state = copy.deepcopy(state)
    if _memory["last"] is not None and _canon(state) == _memory["last"]:
        hidden = copy.deepcopy(_memory["hidden"])
    else:
        hidden = []
    aid = action["action_id"] if isinstance(action, dict) else action
    player = next((o for o in state if o["type"] == "player"), None)
    bar = next((o for o in state if o["name"] == "step_bar"), None)
    if player is None or aid not in MOVES:
        return _remember(state, hidden, bar, spend=bar is not None)
    dx, dy = MOVES[aid]
    nx, ny = player["x"] + dx, player["y"] + dy
    if not _in_playfield(nx, ny, player, state):
        return _remember(state, hidden, bar, spend=False)
    if (nx, ny) not in WALLS:
        nx, ny = PORTALS.get((nx, ny), (nx, ny))
        player["x"], player["y"] = nx, ny
    out = [player]
    refuel = False
    for o in state + hidden:
        if o is player:
            continue
        if o["type"] == "refuel" and _covers(player, o):
            refuel = True
            continue
        out.append(o)
    hidden = []
    shown = []
    for o in out:
        if o is not player and o["type"] == "target" and not _is_hud(o) and _covers(player, o):
            hidden.append(o)
        else:
            shown.append(o)
    if bar is not None:
        if refuel:
            _set_bar(bar, BAR_FULL)
            return _remember(shown, hidden, bar, spend=False)
    return _remember(shown, hidden, bar, spend=True)


def _remember(state, hidden, bar, spend):
    if spend and bar is not None:
        _set_bar(bar, max(0, bar["w"] - BAR_COST))
    _memory["last"] = _canon(state)
    _memory["hidden"] = copy.deepcopy(hidden)
    return state
