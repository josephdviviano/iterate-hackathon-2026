# Mechanics: player 5x5 moves +-5 (A1-4); invisible wall cells block moves (still cost 2);
# entering portal cell (9,5) warps to (34,5). Every accepted action burns 2 budget on
# step_bar (w-=2, x=55-w, tag=str(w)); a refuel ring fully covered by the player is consumed
# and refills the bar to 42. A move into a 'chamber' target (legend) is refused for free:
# whole state unchanged. Covered layer-0 targets are occluded and restored via gated memory.
import copy, json

WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
DELTA = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
BAR_RIGHT, BAR_FULL, COST = 55, 42, 2

_mem = {"last": None, "hidden": []}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _covers(p, o):
    return (p["x"] <= o["x"] and p["y"] <= o["y"] and
            o["x"] + o["w"] <= p["x"] + p["w"] and o["y"] + o["h"] <= p["y"] + p["h"])


def _overlaps(x, y, w, h, o):
    return x < o["x"] + o["w"] and o["x"] < x + w and y < o["y"] + o["h"] and o["y"] < y + h


def _is_chamber(o):
    return o.get("type") == "target" and "chamber" in o.get("tags", [])


def _set_bar(bar, w):
    w = max(0, w)
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [bar["tags"][0], str(w)]


def transition_function(state, action):
    state = copy.deepcopy(state)
    hidden = copy.deepcopy(_mem["hidden"]) if (
        _mem["last"] is not None and _canon(state) == _mem["last"]) else []
    aid = action["action_id"] if isinstance(action, dict) else action
    player = next((o for o in state if o.get("type") == "player"), None)
    bar = next((o for o in state if o.get("name") == "step_bar"), None)

    if player is not None and aid in DELTA:
        dx, dy = DELTA[aid]
        nx, ny = player["x"] + dx, player["y"] + dy
        if any(_is_chamber(o) and _overlaps(nx, ny, player["w"], player["h"], o)
               for o in state + hidden):
            _mem["last"], _mem["hidden"] = _canon(state), hidden
            return state
        if (nx, ny) not in WALLS:
            nx, ny = PORTALS.get((nx, ny), (nx, ny))
            player["x"], player["y"] = nx, ny

    if bar is not None:
        _set_bar(bar, bar["w"] - COST)

    out = []
    for o in state:
        if player is not None and o is not player and _covers(player, o):
            if o.get("type") == "refuel":
                if bar is not None:
                    _set_bar(bar, BAR_FULL)
                continue
            if o.get("layer", 0) < player.get("layer", 1):
                hidden.append(o)
                continue
        out.append(o)
    still = []
    for o in hidden:
        if player is not None and _covers(player, o):
            still.append(o)
        else:
            out.append(o)
    _mem["last"], _mem["hidden"] = _canon(out), still
    return out
