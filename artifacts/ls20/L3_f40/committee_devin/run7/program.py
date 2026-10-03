# Mechanics: player (5x5 'block') moves 5 cells per A1-4 (up/down/left/right); invisible maze walls
# block entering cells {(39,5),(34,10)}; entering (9,5) warps to (34,5) (portal). Every action costs
# step_bar 2 (w-=2, x=55-w, tag=str(w)); a refuel ring fully covered by the player is consumed and
# refills the bar to 42. Other markers fully covered by the player are occluded (not extracted) and
# reappear when uncovered (continuity-gated memory). Unconfirmed: bar at 0, board bounds, A5-7 effects.
import copy, json

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BAR_RIGHT, BAR_FULL, BAR_COST = 55, 42, 2
LO, HI = 0, 59

_mem = {"last": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _covered(o, p):
    return (p["x"] <= o["x"] and o["x"] + o["w"] <= p["x"] + p["w"]
            and p["y"] <= o["y"] and o["y"] + o["h"] <= p["y"] + p["h"])


def _move_player(p, action):
    if action not in MOVES:
        return
    dx, dy = MOVES[action]
    nx, ny = p["x"] + dx, p["y"] + dy
    if (nx, ny) in WALLS or not (LO <= nx and nx + p["w"] - 1 <= HI and LO <= ny and ny + p["h"] - 1 <= HI):
        return
    nx, ny = PORTALS.get((nx, ny), (nx, ny))
    p["x"], p["y"] = nx, ny


def _set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def transition_function(state, action):
    hidden = _mem["hidden"] if _mem["last"] is not None and _canon(state) == _mem["last"] else []
    objs = copy.deepcopy(state) + copy.deepcopy(hidden)
    aid = action["action_id"] if isinstance(action, dict) else action
    player = next(o for o in objs if o["type"] == "player")
    bar = next((o for o in objs if o["name"] == "step_bar"), None)
    _move_player(player, aid)
    if bar is not None:
        _set_bar(bar, bar["w"] - BAR_COST)
    out, new_hidden = [], []
    for o in objs:
        if o is not player and o["type"] in ("refuel", "target") and o.get("layer", 0) < player["layer"] \
                and _covered(o, player):
            if o["type"] == "refuel":
                if bar is not None:
                    _set_bar(bar, BAR_FULL)
                continue
            new_hidden.append(o)
            continue
        out.append(o)
    _mem["last"], _mem["hidden"] = _canon(out), new_hidden
    return out
