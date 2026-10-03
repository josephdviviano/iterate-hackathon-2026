# Mechanics: player (5x5) moves 5 cells per action (1 up, 2 down, 3 left, 4 right) on a grid with an
# invisible maze: moves into WALL_CELLS are blocked (A4/A2 no_change at (34,5)), PORTALS teleport the player.
# Every action costs 2 budget (step_bar w-=2, x+=2, tag=str(w)); a refuel ring fully covered by the player is
# consumed and refills step_bar to full (w=42, x=13). Layer-0 objects fully covered by the player are occluded
# (vanish) and reappear when uncovered (continuity-gated memory). Unconfirmed: budget-0 reset, diamond/legend goal.
import copy
import json

STEP = 5
MOVES = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
WALL_CELLS = {(39, 5), (34, 10)}
PORTALS = {(9, 5): (34, 5)}
BAR_FULL_W, BAR_RIGHT = 42, 55
BAR_COST = 2

_memory = {"last": None, "hidden": []}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _covers(p, o):
    return (p["x"] <= o["x"] and o["x"] + o["w"] <= p["x"] + p["w"]
            and p["y"] <= o["y"] and o["y"] + o["h"] <= p["y"] + p["h"])


def _blocked(x, y, p):
    if (x, y) in WALL_CELLS:
        return True
    return x < 0 or y < 0 or x + p["w"] > 64 or y + p["h"] > 64


def _move_player(p, action):
    if action not in MOVES:
        return
    dx, dy = MOVES[action]
    nx, ny = p["x"] + dx, p["y"] + dy
    if _blocked(nx, ny, p):
        return
    nx, ny = PORTALS.get((nx, ny), (nx, ny))
    p["x"], p["y"] = nx, ny


def _set_bar(bar, w):
    bar["w"] = w
    bar["x"] = BAR_RIGHT - w
    bar["tags"] = [t if not t.isdigit() else str(w) for t in bar["tags"]]


def transition_function(state, action):
    hidden = []
    if _memory["last"] is not None and _canon(state) == _memory["last"]:
        hidden = copy.deepcopy(_memory["hidden"])
    objs = copy.deepcopy(state) + hidden
    act = action["action_id"] if isinstance(action, dict) else action

    players = [o for o in objs if o["type"] == "player"]
    for p in players:
        _move_player(p, act)

    bar = next((o for o in objs if o["type"] == "counter" and "budget" in o["tags"]), None)
    if bar is not None:
        _set_bar(bar, max(0, bar["w"] - BAR_COST))

    out, new_hidden = [], []
    for o in objs:
        if o["type"] == "player":
            out.append(o)
            continue
        cover = any(_covers(p, o) and p["layer"] > o["layer"] for p in players)
        if o["type"] == "refuel" and cover:
            if bar is not None:
                _set_bar(bar, BAR_FULL_W)
            continue
        if cover:
            new_hidden.append(o)
        else:
            out.append(o)
    _memory["last"] = _canon(out)
    _memory["hidden"] = new_hidden
    return out
