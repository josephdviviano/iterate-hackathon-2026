# Mechanics: two modes. BLOCK mode: cyan blocks (left/right half) move by 4; ACTION1/2 move both up/down,
# ACTION3/4 move them mirrored (3 = apart, 4 = together). Clicking a marker enters ARMED mode: blocks become
# color-1 players, that marker becomes active (color 11) and ACTION1-4 move it by 4; clicking another marker
# switches, clicking a player returns to BLOCK mode. Moves are blocked by board/half bounds, markers, players and
# hidden wall cells (UNCONFIRMED: invisible maze cells inferred from blocked moves). Top/bottom bars grow floor(3(n+1)/7).
import json

STEP = 4
HIDDEN_WALL_CELLS = {(14, 38), (6, 18), (38, 14)}  # top-left of 4x4 cells; not visible in the extracted objects
_mem = {"last": None, "n": 0}


def _canon(s):
    return sorted(json.dumps(o, sort_keys=True) for o in s)


def _color(o):
    for t in o.get("tags", []):
        if t.startswith("color_"):
            return int(t[6:])
    return None


def _is_bar(o):
    return o["type"] == "wall" and _color(o) == 0 and o["h"] == 1


def _bar_width(n):
    return (3 * (n + 1)) // 7


def _guess_n(width):
    n = 0
    while _bar_width(n) < width:
        n += 1
    return n


def _inside(o, px, py):
    return o["x"] <= px < o["x"] + o["w"] and o["y"] <= py < o["y"] + o["h"]


def _overlap(ax, ay, aw, ah, bx, by, bw, bh):
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _hits_hidden_wall(x, y, w, h):
    return any(_overlap(x, y, w, h, cx, cy, 4, 4) for cx, cy in HIDDEN_WALL_CELLS)


# ---- per-type guards ----
def _block_can_move(b, nx, ny, others):
    left = b["x"] < 32
    if ny < 1 or ny + b["h"] > 63:
        return False
    if left and (nx < 0 or nx + b["w"] > 32):
        return False
    if not left and (nx < 32 or nx + b["w"] > 64):
        return False
    if _hits_hidden_wall(nx, ny, b["w"], b["h"]):
        return False
    return not any(_overlap(nx, ny, b["w"], b["h"], o["x"], o["y"], o["w"], o["h"]) for o in others)


def _marker_can_move(m, nx, ny, others):
    cx, cy = nx - 1, ny - 1
    if cx < 0 or cy < 1 or cx + 4 > 64 or cy + 4 > 63:
        return False
    if _hits_hidden_wall(cx, cy, 4, 4):
        return False
    return not any(_overlap(cx, cy, 4, 4, o["x"], o["y"], o["w"], o["h"]) for o in others)


# ---- per-type updates ----
def _to_armed(p):
    return {"h": p["h"], "layer": p["layer"], "name": "", "tags": ["color_1", "armed", "player"],
            "type": "player", "visible": True, "w": p["w"], "x": p["x"], "y": p["y"]}


def _to_block(p):
    return {"h": p["h"], "layer": p["layer"], "name": "", "tags": ["cyan", "block", "player"],
            "type": "player", "visible": True, "w": p["w"], "x": p["x"], "y": p["y"],
            "pixels": [[10] * p["w"] for _ in range(p["h"])]}


def _set_marker(m, active):
    m["tags"] = ["color_11", "active", "marker"] if active else ["color_9", "inactive", "marker"]


def _make_bar(top, width):
    return {"h": 1, "layer": 0, "name": "", "tags": ["color_0", "wall"], "type": "wall", "visible": True,
            "w": width, "x": 64 - width if top else 0, "y": 0 if top else 63}


def _rename(objs):
    blocks = [o for o in objs if "block" in o.get("tags", [])]
    for b in blocks:
        b["name"] = "block_left" if b["x"] < 32 else "block_right"
    rest = sorted((o for o in objs if o not in blocks), key=lambda o: (o["y"], o["x"]))
    for i, o in enumerate(rest):
        o["name"] = "%s_%s_%d" % (o["type"], _color(o), i + len(blocks))
    return objs


def _step(state, action):
    objs = [dict(o) for o in state]
    players = [o for o in objs if o["type"] == "player"]
    markers = [o for o in objs if o["type"] == "marker"]
    armed = any("armed" in p["tags"] for p in players)
    active = next((m for m in markers if "active" in m["tags"]), None)
    aid = action["action_id"] if isinstance(action, dict) else action

    if aid == 6:
        px, py = action["x"], action["y"]
        hit_m = next((m for m in markers if _inside(m, px, py)), None)
        hit_p = next((p for p in players if _inside(p, px, py)), None)
        if hit_m is not None and hit_m is not active:
            for m in markers:
                _set_marker(m, m is hit_m)
            if not armed:
                objs = [_to_armed(o) if o in players else o for o in objs]
        elif armed and hit_m is None and hit_p is not None:
            for m in markers:
                _set_marker(m, False)
            objs = [_to_block(o) if o in players else o for o in objs]
    elif aid in (1, 2, 3, 4):
        dx, dy = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}[aid]
        if armed:
            if active is not None:
                nx, ny = active["x"] + dx, active["y"] + dy
                others = [o for o in players + markers if o is not active]
                if _marker_can_move(active, nx, ny, others):
                    active["x"], active["y"] = nx, ny
        else:
            moves = []
            for b in players:
                mdx = dx if b["x"] < 32 else -dx
                moves.append((b, b["x"] + mdx, b["y"] + dy))
            for b, nx, ny in moves:
                others = markers + [p for p in players if p is not b]
                if _block_can_move(b, nx, ny, others):
                    b["x"], b["y"] = nx, ny
    return objs


def transition_function(state, action):
    bars = [o for o in state if _is_bar(o)]
    width = max([o["w"] for o in bars] or [0])
    if _mem["last"] is not None and _canon(state) == _mem["last"]:
        n = _mem["n"]
    else:
        n = _guess_n(width)
    objs = [o for o in _step(state, action) if not _is_bar(o)]
    n += 1
    w = _bar_width(n)
    if w > 0:
        objs += [_make_bar(True, w), _make_bar(False, w)]
    out = _rename(objs)
    _mem["last"] = _canon(out)
    _mem["n"] = n
    return out
