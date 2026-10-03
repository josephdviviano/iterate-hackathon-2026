# Mechanics: player (5x5, layer 1) moves 5 cells per action (A1 up, A2 down, A3 left, A4 right) on a hidden maze;
# blocked cells {(39,5),(34,10)} make the move a no-op, entering portal cell (9,5) teleports to (34,5).
# step_bar spends 2 per action (w-2, x=55-w, tag=str(w)); a refuel ring fully covered by the player is consumed and refills w=42.
# Targets fully covered by the player are occluded (not extracted) and restored when uncovered (continuity-gated memory).
# Chamber guard: a move whose destination covers a 'chamber' target is refused outright (no move, no budget spent); why unconfirmed.
import copy
import json

MOVES = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
BLOCKED = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BAR_RIGHT, BAR_FULL, BAR_COST = 55, 42, 2

_memory = {"last": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _covers(p, o, px, py):
    return (px <= o["x"] and py <= o["y"] and
            o["x"] + o["w"] <= px + p["w"] and o["y"] + o["h"] <= py + p["h"])


def _is_chamber(o):
    return o.get("type") == "target" and "chamber" in o.get("tags", [])


def _set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def transition_function(state, action):
    state = copy.deepcopy(state)
    continuous = _memory["last"] is not None and _canon(state) == _memory["last"]
    hidden = copy.deepcopy(_memory["hidden"]) if continuous else []
    aid = action.get("action_id") if isinstance(action, dict) else action

    player = next((o for o in state if o.get("type") == "player"), None)
    bar = next((o for o in state if o.get("name") == "step_bar"), None)
    if player is None:
        return _finish(state, hidden)

    px, py = player["x"], player["y"]
    if aid in MOVES:
        dx, dy = MOVES[aid]
        nx, ny = px + dx, py + dy
        if any(_is_chamber(o) and _covers(player, o, nx, ny) for o in state + hidden):
            return _finish(state, hidden)
        if (nx, ny) in BLOCKED:
            nx, ny = px, py
        nx, ny = PORTALS.get((nx, ny), (nx, ny))
        px, py = nx, ny
    player["x"], player["y"] = px, py

    refuel = False
    out = []
    for o in state + hidden:
        if o is player or o is bar:
            continue
        covered = _covers(player, o, px, py)
        if o.get("type") == "refuel":
            if covered:
                refuel = True
                continue
            out.append(o)
        elif covered:
            continue
        else:
            out.append(o)
    new_hidden = [o for o in state + hidden
                  if o is not player and o is not bar and o.get("type") != "refuel"
                  and _covers(player, o, px, py)]

    if bar is not None:
        _set_bar(bar, BAR_FULL if refuel else bar["w"] - BAR_COST)
        out.append(bar)
    out.append(player)
    return _finish(out, new_hidden)


def _finish(state, hidden):
    _memory["last"] = _canon(state)
    _memory["hidden"] = copy.deepcopy(hidden)
    return state
