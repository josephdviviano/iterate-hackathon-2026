# Mechanics: block mode = two cyan 4x4 blocks; ACTION1/2 move both up/down 4, ACTION3/4 move them apart/together (mirrored).
# Moves are per-block, blocked by board bounds, a hidden maze of 4x4 cells and marker cells (marker xy - 1).
# ACTION6 on a marker arms it (blocks -> color_1 players, marker -> color_11 active); ACTION1-4 then move the active marker;
# clicking a player returns to block mode; clicking the active marker/a block is a no-op. color_0 bars width floor(3(n+1)/7),
# n = actions since level start (hidden, continuity-gated; fallback smallest n). Hypothesis: maze cells inferred from failed moves only.
import json

MAZE = {(14, 38), (6, 18), (38, 14)}
LO, HI = 2, 58
_mem = {"canon": None, "n": 0, "left_xy": None}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _bar_width(state):
    for o in state:
        if o["type"] == "wall" and "color_0" in o["tags"]:
            return o["w"]
    return 0


def _width_for(n):
    return (3 * (n + 1)) // 7


def _min_n(w):
    n = 0
    while _width_for(n) < w:
        n += 1
    return n


def _parse(state):
    blocks, players, markers, scenery = [], [], [], []
    for o in state:
        t = o["type"]
        if t == "player" and "block" in o["tags"]:
            blocks.append(o)
        elif t == "player":
            players.append(o)
        elif t == "marker":
            markers.append(o)
        elif not (t == "wall" and "color_0" in o["tags"]):
            scenery.append(o)
    return blocks, players, markers, scenery


def _free(cell, occupied):
    x, y = cell
    return LO <= x <= HI and LO <= y <= HI and cell not in MAZE and cell not in occupied


def _inside(o, x, y):
    return o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]


def _render(mode, pieces, markers, scenery, n):
    # pieces: [(x, y, side)] ; markers: [(x, y, active)]
    objs = []
    for s in scenery:
        objs.append(dict(s))
    w = _width_for(n)
    if w > 0:
        for x, y in ((64 - w, 0), (0, 63)):
            objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                         "visible": True, "w": w, "x": x, "y": y})
    for x, y, act in markers:
        c = "color_11" if act else "color_9"
        objs.append({"h": 2, "layer": 1, "tags": [c, "active" if act else "inactive", "marker"],
                     "type": "marker", "visible": True, "w": 2, "x": x, "y": y})
    if mode == "armed":
        for x, y, _ in pieces:
            objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                         "visible": True, "w": 4, "x": x, "y": y})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    start = 2 if mode == "block" else 0
    for i, o in enumerate(objs):
        o["name"] = "%s_%s_%d" % (o["type"] if o["type"] != "player" else "player",
                                  o["tags"][0].split("_")[1], start + i)
    if mode == "block":
        for x, y, side in pieces:
            objs.append({"h": 4, "layer": 1, "name": "block_" + side, "tags": ["cyan", "block", "player"],
                         "type": "player", "visible": True, "w": 4, "x": x, "y": y,
                         "pixels": [[10] * 4 for _ in range(4)]})
    return objs


def transition_function(state, action):
    continuous = _mem["canon"] is not None and _canon(state) == _mem["canon"]
    n = _mem["n"] if continuous else _min_n(_bar_width(state))
    blocks, players, markers, scenery = _parse(state)
    mode = "block" if blocks else "armed"
    if mode == "block":
        pieces = [(b["x"], b["y"], b["name"].split("_")[1]) for b in blocks]
    else:
        ps = sorted((p["x"], p["y"]) for p in players)
        lxy = _mem["left_xy"] if continuous else None
        if lxy not in ps:
            lxy = ps[0]
        pieces = [(x, y, "left" if (x, y) == lxy else "right") for x, y in ps]
    mk = [(m["x"], m["y"], "active" in m["tags"]) for m in markers]
    aid = action["action_id"] if isinstance(action, dict) else action
    dirs = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}

    if aid in dirs and mode == "block":
        dx, dy = dirs[aid]
        occ = {(x - 1, y - 1) for x, y, _ in mk}
        new = []
        for x, y, side in pieces:
            sdx = dx if side == "left" or aid in (1, 2) else -dx
            cell = (x + sdx, y + dy)
            new.append(cell + (side,) if _free(cell, occ) else (x, y, side))
        pieces = new
    elif aid in dirs and mode == "armed":
        dx, dy = dirs[aid]
        occ = {(x - 1, y - 1) for x, y, a in mk if not a} | {(x, y) for x, y, _ in pieces}
        new = []
        for x, y, a in mk:
            if a and _free((x - 1 + dx, y - 1 + dy), occ):
                x, y = x + dx, y + dy
            new.append((x, y, a))
        mk = new
    elif aid == 6:
        cx, cy = action["x"], action["y"]
        hit_m = [i for i, m in enumerate(markers) if _inside(m, cx, cy)]
        hit_p = [p for p in players if _inside(p, cx, cy)]
        if hit_m and not mk[hit_m[0]][2]:
            mk = [(x, y, i == hit_m[0]) for i, (x, y, _) in enumerate(mk)]
            mode = "armed"
        elif hit_p and mode == "armed":
            mk = [(x, y, False) for x, y, _ in mk]
            mode = "block"

    out = _render(mode, pieces, mk, scenery, n + 1)
    _mem["canon"] = _canon(out)
    _mem["n"] = n + 1
    _mem["left_xy"] = next(((x, y) for x, y, s in pieces if s == "left"), None)
    return out
