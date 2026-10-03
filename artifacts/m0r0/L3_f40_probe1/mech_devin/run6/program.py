# Mechanics: two cyan blocks move together (A1/A2 y-+4; A3/A4 mirrored x: left -+4, right +-4), each alone if its target cell is in bounds [2,58] and free of maze cells, marker cells (marker xy-1) and the other block.
# Click on an inactive marker arms: blocks become color_1 'armed' players, the marker turns active (color_11); armed A1-A4 steer the active marker by 4 (blocked by maze, markers, players, bounds).
# Armed: click another marker = switch active, click a player = disarm (players back to blocks); other clicks, A5, A7 are no-ops. Names = type_color_rank by (y,x), from 2 in block mode (blocks own 0,1), from 0 armed.
# Timer: a = actions since level start (hidden, continuity-gated); two color_0 bars of width 3(a+1)//7 at (64-w,0) and (0,63). Fallback a = 0 if no bar else largest a with that width.
# Unconfirmed: maze cell (22,18) is inferred only from step 124 (block_left A4 blocked with no visible cause); further maze cells unseen.
import json

MAZE = {(14, 38), (6, 18), (38, 14), (22, 18)}
LO, HI = 2, 58
_mem = {"out": None, "a": 0, "left": None}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _bar_w(a):
    return 3 * (a + 1) // 7


def _parse(state):
    blocks, players, markers, scenery, bar = {}, [], [], [], 0
    for o in state:
        tags = o.get("tags", [])
        if "block" in tags:
            blocks[o["name"]] = (o["x"], o["y"])
        elif "armed" in tags:
            players.append((o["x"], o["y"]))
        elif o["type"] == "marker":
            markers.append([o["x"], o["y"], "active" in tags])
        elif "color_0" in tags:
            if o["y"] == 0:
                bar = o["w"]
        else:
            scenery.append(o)
    return blocks, players, markers, scenery, bar


def _free(cell, blocked):
    x, y = cell
    return LO <= x <= HI and LO <= y <= HI and cell not in MAZE and cell not in blocked


def _hit(px, py, x, y, size):
    return x <= px < x + size and y <= py < y + size


def _render(blocks, players, markers, scenery, a):
    objs = []
    for o in scenery:
        objs.append(dict(o))
    w = _bar_w(a)
    if w > 0:
        for bx, by in ((64 - w, 0), (0, 63)):
            objs.append({"h": 1, "layer": 0, "tags": ["color_0", "wall"], "type": "wall",
                         "visible": True, "w": w, "x": bx, "y": by, "_c": "wall_0"})
    for mx, my, act in markers:
        c = 11 if act else 9
        objs.append({"h": 2, "layer": 1, "tags": ["color_%d" % c, "active" if act else "inactive", "marker"],
                     "type": "marker", "visible": True, "w": 2, "x": mx, "y": my, "_c": "marker_%d" % c})
    for px, py in players:
        objs.append({"h": 4, "layer": 1, "tags": ["color_1", "armed", "player"], "type": "player",
                     "visible": True, "w": 4, "x": px, "y": py, "_c": "player_1"})
    for o in objs:
        if "_c" not in o:
            o["_c"] = o["name"].rsplit("_", 1)[0]
    objs.sort(key=lambda o: (o["y"], o["x"]))
    start = 2 if blocks else 0
    for i, o in enumerate(objs):
        o["name"] = "%s_%d" % (o.pop("_c"), start + i)
    for name, (bx, by) in blocks.items():
        objs.append({"h": 4, "layer": 1, "name": name, "pixels": [[10] * 4 for _ in range(4)],
                     "tags": ["cyan", "block", "player"], "type": "player", "visible": True,
                     "w": 4, "x": bx, "y": by})
    return objs


def transition_function(state, action):
    blocks, players, markers, scenery, bar = _parse(state)
    if _mem["out"] is not None and _canon(state) == _mem["out"]:
        a, left = _mem["a"], _mem["left"]
    else:
        a = 0 if bar == 0 else (7 * bar + 3) // 3
        left = None
    if isinstance(action, dict):
        aid, cx, cy = action.get("action_id", 6), action.get("x", -1), action.get("y", -1)
    else:
        aid, cx, cy = action, -1, -1
    mcells = {(m[0] - 1, m[1] - 1) for m in markers}
    dirs = {1: (0, -4), 2: (0, 4), 3: (-4, 0), 4: (4, 0)}

    if blocks and aid in dirs:
        dx, dy = dirs[aid]
        moves = {"block_left": (dx, dy), "block_right": (-dx, dy)}
        new = {}
        for name, (bx, by) in blocks.items():
            mdx, mdy = moves.get(name, (dx, dy))
            tgt = (bx + mdx, by + mdy)
            others = {p for n, p in blocks.items() if n != name}
            new[name] = tgt if _free(tgt, mcells | others) else (bx, by)
        blocks = new
    elif players and aid in dirs:
        dx, dy = dirs[aid]
        for i, m in enumerate(markers):
            if m[2]:
                tgt = (m[0] - 1 + dx, m[1] - 1 + dy)
                blocked = {(o[0] - 1, o[1] - 1) for j, o in enumerate(markers) if j != i} | set(players)
                if _free(tgt, blocked):
                    m[0], m[1] = tgt[0] + 1, tgt[1] + 1
    elif aid == 6:
        hit_m = next((m for m in markers if _hit(cx, cy, m[0], m[1], 2)), None)
        if blocks:
            if hit_m is not None and not hit_m[2]:
                for m in markers:
                    m[2] = m is hit_m
                left = blocks.get("block_left")
                players = [p for p in blocks.values()]
                blocks = {}
        else:
            hit_p = next((p for p in players if _hit(cx, cy, p[0], p[1], 4)), None)
            if hit_m is not None and not hit_m[2]:
                for m in markers:
                    m[2] = m is hit_m
            elif hit_p is not None:
                ps = sorted(players)
                if left not in ps:
                    left = ps[0]
                right = next(p for p in ps if p != left) if len(ps) > 1 else left
                blocks = {"block_left": left, "block_right": right}
                players = []
                for m in markers:
                    m[2] = False
                left = None
    a += 1
    out = _render(blocks, players, markers, scenery, a)
    _mem.update(out=_canon(out), a=a, left=left)
    return out
