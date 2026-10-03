# Mechanics: player (5x5, layer 1) steps 5 px per arrow (A1 up, A2 down, A3 left, A4 right); cells in WALLS
#   are invisible blockers; entering the portal cell (9,5) warps to (34,5). Every action burns 2 from step_bar
#   (w-=2, x=55-w, tag=str(w)), including wall bumps. A refuel ring fully covered by the player is consumed
#   and refills the bar to 42. A collect target covered by the player is occluded (kept in continuity-gated
#   memory, re-shown when uncovered). A chamber target (legend) is solid and bumping it is rejected free (no burn).
import copy, json

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BAR_RIGHT, BAR_FULL, BAR_COST = 55, 42, 2

_mem = {"last": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _rect(o):
    return (o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"])


def _covers(outer, inner):
    a, b = _rect(outer), _rect(inner)
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _overlaps(a, b):
    ra, rb = _rect(a), _rect(b)
    return ra[0] < rb[2] and rb[0] < ra[2] and ra[1] < rb[3] and rb[1] < ra[3]


def is_chamber(o):
    return o["type"] == "target" and "chamber" in o.get("tags", [])


def is_collect(o):
    return o["type"] == "target" and "collect" in o.get("tags", [])


def is_bar(o):
    return o["type"] == "counter" and "budget" in o.get("tags", [])


def enters_chamber(dest, objs):
    return any(is_chamber(o) and _overlaps(dest, o) for o in objs)


def set_bar(bar, w):
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [t if not t.lstrip("-").isdigit() else str(w) for t in bar["tags"]]


def transition_function(state, action):
    hidden = _mem["hidden"] if _mem["last"] is not None and _canon(state) == _mem["last"] else []
    objs = copy.deepcopy(state) + copy.deepcopy(hidden)
    aid = action["action_id"] if isinstance(action, dict) else action
    player = next((o for o in objs if o["type"] == "player"), None)
    bar = next((o for o in objs if is_bar(o)), None)

    # player update
    if player is not None and aid in MOVES:
        dx, dy = MOVES[aid]
        dest = dict(player, x=player["x"] + dx, y=player["y"] + dy)
        if enters_chamber(dest, objs):
            _mem["last"], _mem["hidden"] = _canon(state), hidden
            return copy.deepcopy(state)
        if (dest["x"], dest["y"]) not in WALLS:
            dest["x"], dest["y"] = PORTALS.get((dest["x"], dest["y"]), (dest["x"], dest["y"]))
            player["x"], player["y"] = dest["x"], dest["y"]

    # counter update: every action burns budget
    if bar is not None:
        set_bar(bar, max(0, bar["w"] - BAR_COST))

    out, new_hidden = [], []
    for o in objs:
        if player is not None and o is not player and o["type"] == "refuel" and _covers(player, o):
            if bar is not None:
                set_bar(bar, BAR_FULL)
            continue
        if player is not None and is_collect(o) and _covers(player, o):
            new_hidden.append(o)
            continue
        out.append(o)
    _mem["last"], _mem["hidden"] = _canon(out), new_hidden
    return out
