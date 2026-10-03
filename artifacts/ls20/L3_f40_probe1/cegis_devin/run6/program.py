# Mechanics: player (5x5, layer 1) moves 5 cells per A1-4 (up/down/left/right) through an invisible maze:
#   hidden wall cells {(39,5),(34,10)} block (no move, still costs), portal cell (9,5) warps to (34,5).
# step_bar budget: every action costs 2 (w-=2, x=55-w, tag=str(w)); a refuel ring fully covered is consumed
#   and refills the bar to 42. A covered target (diamond) is occluded by the layer-1 player and reappears when
#   uncovered (continuity-gated memory). Guard: a move into a 'chamber' target (legend) is refused and free.
# Unconfirmed: what unlocks the chamber; actions 5/6/7 assumed to only cost budget; maze beyond seen cells.
import copy, json

MOVES = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BAR_RIGHT, BAR_FULL, COST = 55, 42, 2

_mem = {"last": None, "hidden": []}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _overlap(a, b):
    return (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
            and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"])


def _covers(a, b):
    return (a["x"] <= b["x"] and b["x"] + b["w"] <= a["x"] + a["w"]
            and a["y"] <= b["y"] and b["y"] + b["h"] <= a["y"] + a["h"])


def _enters_chamber(dest, objs):
    return any(o["type"] == "target" and "chamber" in o.get("tags", []) and _overlap(dest, o)
               for o in objs)


def _set_bar(bar, w):
    w = max(0, min(BAR_FULL, w))
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [t for t in bar["tags"] if not t.isdigit()] + [str(w)]


def transition_function(state, action):
    s = copy.deepcopy(state)
    hidden = _mem["hidden"] if _mem["last"] is not None and _canon(state) == _mem["last"] else []
    aid = action["action_id"] if isinstance(action, dict) else action
    player = next((o for o in s if o["type"] == "player"), None)
    bar = next((o for o in s if o["name"] == "step_bar"), None)

    if player is not None and aid in MOVES:
        dx, dy = MOVES[aid]
        dest = dict(player, x=player["x"] + dx, y=player["y"] + dy)
        if _enters_chamber(dest, s + hidden):
            _mem["last"], _mem["hidden"] = _canon(s), hidden
            return s
        pos = (dest["x"], dest["y"])
        if pos not in WALLS:
            pos = PORTALS.get(pos, pos)
            player["x"], player["y"] = pos

    if bar is not None:
        _set_bar(bar, bar["w"] - COST)

    if player is not None:
        for o in [o for o in s if o["type"] == "refuel" and _covers(player, o)]:
            s.remove(o)
            if bar is not None:
                _set_bar(bar, BAR_FULL)
        pool = [o for o in s if o["type"] == "target" and "chamber" not in o.get("tags", [])] + hidden
        s = [o for o in s if o not in pool]
        hidden = [o for o in pool if _covers(player, o)]
        s += [o for o in pool if not _covers(player, o)]

    _mem["last"], _mem["hidden"] = _canon(s), hidden
    return s
