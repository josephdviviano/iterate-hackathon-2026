# Mechanics: player (5x5 block) moves 5 cells per A1-4 (up/down/left/right) unless the target cell is an
# invisible maze wall {(39,5),(34,10)} or off-board; entering (9,5) teleports to portal exit (34,5).
# step_bar counter loses 2 width per action (x = 55 - w, tag = str(w)); a refuel ring fully covered by the
# player is consumed and refills the bar to 42. Targets fully covered by the player are occluded (not
# extracted) and reappear when uncovered (continuity-gated memory). Unconfirmed: board bounds, bar at 0, A5-7.
import copy
import json

MOVES = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BAR_FULL, BAR_RIGHT, BAR_COST = 42, 55, 2
LO, HI = 0, 59

_memory = {"last": None, "hidden": []}


def _key(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def covered(obj, player):
    return (player["x"] <= obj["x"] and obj["x"] + obj["w"] <= player["x"] + player["w"]
            and player["y"] <= obj["y"] and obj["y"] + obj["h"] <= player["y"] + player["h"])


def move_player(player, action):
    if action not in MOVES:
        return
    dx, dy = MOVES[action]
    nx, ny = player["x"] + dx, player["y"] + dy
    if (nx, ny) in WALLS or not (LO <= nx <= HI and LO <= ny <= HI):
        return
    nx, ny = PORTALS.get((nx, ny), (nx, ny))
    player["x"], player["y"] = nx, ny


def set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [t for t in bar["tags"] if not t.isdigit()] + [str(w)]


def transition_function(state, action):
    state = copy.deepcopy(state)
    hidden = _memory["hidden"] if _memory["last"] == _key(state) else []
    aid = action["action_id"] if isinstance(action, dict) else action
    objs = state + copy.deepcopy(hidden)
    player = next((o for o in objs if o["type"] == "player"), None)
    bar = next((o for o in objs if o["type"] == "counter" and "budget" in o.get("tags", [])), None)
    if player is not None:
        move_player(player, aid)
    out, new_hidden, refuel = [], [], False
    for o in objs:
        if player is not None and o is not player and o["type"] == "refuel" and covered(o, player):
            refuel = True
            continue
        if player is not None and o["type"] == "target" and o.get("layer", 0) < player["layer"] \
                and covered(o, player):
            new_hidden.append(o)
            continue
        out.append(o)
    if bar is not None:
        set_bar(bar, BAR_FULL if refuel else bar["w"] - BAR_COST)
    _memory["last"] = _key(out)
    _memory["hidden"] = new_hidden
    return out
