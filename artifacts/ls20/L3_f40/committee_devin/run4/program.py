# Mechanics: player (5x5) moves 5 cells per action 1-4 (up/down/left/right) unless the destination is a
# hidden maze wall cell (walls are not extracted); stepping onto (9,5) warps the player to (34,5).
# step_bar counter: every action costs 2 (w-2, x+2, tag=str(w)); a refuel ring covered by the player is
# consumed and refills the bar to 42 (x=13). Targets covered by the player are occluded (dropped) and
# reappear when uncovered (continuity-gated hidden memory). Unconfirmed: bar-empty reset, other warps/walls.
import copy, json

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALLS = {(39, 5), (34, 10)}
WARPS = {(9, 5): (34, 5)}
BAR_FULL, BAR_COST = 42, 2

_mem = {"last": None, "hidden": []}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _covers(p, o):
    return (p["x"] <= o["x"] and o["x"] + o["w"] <= p["x"] + p["w"]
            and p["y"] <= o["y"] and o["y"] + o["h"] <= p["y"] + p["h"])


def _move_player(p, action):
    if action not in MOVES:
        return
    dx, dy = MOVES[action]
    dest = (p["x"] + dx, p["y"] + dy)
    if dest in WALLS:
        return
    dest = WARPS.get(dest, dest)
    p["x"], p["y"] = dest


def _set_bar(bar, w):
    right = bar["x"] + bar["w"]
    w = max(0, w)
    bar["w"], bar["x"] = w, right - w
    bar["tags"] = [bar["tags"][0], str(w)]


def transition_function(state, action):
    state = copy.deepcopy(state)
    hidden = []
    if _mem["last"] is not None and _canon(state) == _mem["last"]:
        hidden = copy.deepcopy(_mem["hidden"])
    objs = state + hidden
    aid = action["action_id"] if isinstance(action, dict) else action

    player = next((o for o in objs if o["type"] == "player"), None)
    bar = next((o for o in objs if o["type"] == "counter" and "budget" in o.get("tags", [])), None)
    if player is not None:
        _move_player(player, aid)
    refuel = False
    out, new_hidden = [], []
    for o in objs:
        covered = player is not None and o is not player and _covers(player, o)
        if covered and o["type"] == "refuel":
            refuel = True
            continue
        if covered and o["type"] == "target":
            new_hidden.append(o)
            continue
        out.append(o)
    if bar is not None:
        _set_bar(bar, BAR_FULL if refuel else bar["w"] - BAR_COST)

    _mem["last"] = _canon(out)
    _mem["hidden"] = copy.deepcopy(new_hidden)
    return out
