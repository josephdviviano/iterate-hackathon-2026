# Mechanics: block mode - A1/A2 move both cyan blocks y-4/y+4, A3 apart / A4 together (mirrored x+-4); each block moves alone, blocked by bounds [2,58], maze cells, marker cells (marker xy-1) and the other block.
# Clicking a marker arms: blocks -> color_1 'armed' players, clicked marker -> color_11 active; armed A1-A4 move the active marker +-4 (blocked by maze/markers/players/bounds); click another marker switches, click a player disarms; other clicks / A5 / A7 are no-ops.
# Timer: every action increments a hidden counter a; bars wall_0 (top-right, bottom-left) have w = 3(a+1)//7. Names = type_color_rank by (y,x): block mode ranks start at 2 (block_left/right take 0,1), armed mode at 0.
# Hidden state (counter a, left-player position) is used only when the incoming state equals the last returned state; fallback a = 0 without bars, else the largest a with that width.
# Unconfirmed: the invisible maze {(14,38),(6,18),(38,14),(22,18)} is inferred from blocked moves (no pixels show it); block-block collision and A7 are unobserved.
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
_mem = {"out": None, "a": 0, "left": None}


def _canon(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def _parse(state):
    blocks, players, markers, scenery, bar = {}, [], [], [], 0
    for o in state:
        tags = o.get("tags", [])
        if o["type"] == "player" and "block" in tags:
            blocks[o["name"]] = (o["x"], o["y"])
        elif o["type"] == "player":
            players.append((o["x"], o["y"]))
        elif o["type"] == "marker":
            markers.append([o["x"], o["y"], "active" in tags])
        elif o["type"] == "wall" and "color_0" in tags:
            bar = max(bar, o["w"])
        else:
            scenery.append(dict(o))
    return blocks, players, markers, scenery, bar


def _free(cell, blocked):
    x, y = cell
    return LO <= x <= HI and LO <= y <= HI and cell not in MAZE and cell not in blocked


def _hit(o, x, y, size):
    return o[0] <= x < o[0] + size and o[1] <= y < o[1] + size


def _step(blocks, players, markers, action, left):
    mcells = {(m[0] - 1, m[1] - 1) for m in markers}
    if isinstance(action, dict):
        cx, cy = action.get("x"), action.get("y")
        if blocks:
            for m in markers:
                if _hit(m, cx, cy, 2):
                    m[2] = True
                    left = blocks["block_left"]
                    return {}, [blocks["block_left"], blocks["block_right"]], markers, left
            return blocks, players, markers, left
        for m in markers:
            if _hit(m, cx, cy, 2):
                if not m[2]:
                    for k in markers:
                        k[2] = k is m
                return blocks, players, markers, left
        for p in players:
            if _hit(p, cx, cy, 4):
                for k in markers:
                    k[2] = False
                lp = left if left in players else min(players)
                rp = [q for q in players if q != lp]
                rp = rp[0] if rp else lp
                return {"block_left": lp, "block_right": rp}, [], markers, None
        return blocks, players, markers, left
    d = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}.get(action)
    if d is None:
        return blocks, players, markers, left
    if blocks:
        dx, dy = d
        moves = {"block_left": (dx, dy), "block_right": (-dx, dy)}
        new = {}
        for name, (x, y) in blocks.items():
            other = [p for n, p in blocks.items() if n != name]
            tx, ty = x + moves[name][0], y + moves[name][1]
            new[name] = (tx, ty) if _free((tx, ty), mcells | set(other)) else (x, y)
        if new["block_left"] == new["block_right"]:
            new = dict(blocks)
        return new, players, markers, left
    for m in markers:
        if m[2]:
            others = {(k[0] - 1, k[1] - 1) for k in markers if k is not m}
            tgt = (m[0] - 1 + d[0], m[1] - 1 + d[1])
            if _free(tgt, others | set(players)):
                m[0], m[1] = tgt[0] + 1, tgt[1] + 1
    return blocks, players, markers, left


def _render(blocks, players, markers, scenery, a):
    w = min(64, 3 * (a + 1) // 7)
    objs = [dict(o) for o in scenery]
    for x, y, act in markers:
        c = "11" if act else "9"
        objs.append({"type": "marker", "tags": ["color_" + c, "active" if act else "inactive", "marker"],
                     "x": x, "y": y, "w": 2, "h": 2, "layer": 1, "visible": True, "_c": c})
    for x, y in players:
        objs.append({"type": "player", "tags": ["color_1", "armed", "player"],
                     "x": x, "y": y, "w": 4, "h": 4, "layer": 1, "visible": True, "_c": "1"})
    if w > 0:
        for x, y in ((64 - w, 0), (0, 63)):
            objs.append({"type": "wall", "tags": ["color_0", "wall"], "x": x, "y": y,
                         "w": w, "h": 1, "layer": 0, "visible": True, "_c": "0"})
    for o in objs:
        if "_c" not in o:
            o["_c"] = next(t[6:] for t in o["tags"] if t.startswith("color_"))
    start = 2 if blocks else 0
    objs.sort(key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(objs):
        o["name"] = "%s_%s_%d" % (o["type"], o.pop("_c"), start + i)
    for name, (x, y) in blocks.items():
        objs.append({"name": name, "type": "player", "tags": ["cyan", "block", "player"],
                     "x": x, "y": y, "w": 4, "h": 4, "layer": 1, "visible": True,
                     "pixels": [[10] * 4 for _ in range(4)]})
    return objs


def transition_function(state, action):
    blocks, players, markers, scenery, bar = _parse(state)
    if _mem["out"] is not None and _canon(state) == _mem["out"]:
        a, left = _mem["a"], _mem["left"]
    else:
        a = 0 if bar == 0 else (7 * bar + 3) // 3
        left = None
    blocks, players, markers, left = _step(blocks, players, markers, action, left)
    a += 1
    out = _render(blocks, players, markers, scenery, a)
    _mem.update(out=_canon(out), a=a, left=left)
    return out
