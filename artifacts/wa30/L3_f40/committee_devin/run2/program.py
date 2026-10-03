# Mechanics: player (A1-4 = move up/down/left/right by 4, cap shows heading) is blocked by wall/box/chaser;
# bumping an idle box makes it 'faced', A5 toggles faced<->grabbed, a grabbed box moves with the player
# (boxes may enter the wall column), moving otherwise drops faced->idle. The autonomous chaser acts every turn:
# takes an adjacent idle unseated box (no move), else BFS-steps to the nearest box / empty slot; seated -> release.
# Unconfirmed: wall gaps are box occlusion (no counter); pull while grabbed, 64px bounds, A6/A7 = chaser-only turn.
import copy
from collections import deque

S = 4
DIRS = {1: (0, -S), 2: (0, S), 3: (-S, 0), 4: (S, 0)}
CAPS = {1: "cap_up", 2: "cap_down", 3: "cap_left", 4: "cap_right"}
BOX_COLOR = {"idle": 4, "faced": 3, "grabbed": 0, "carried": 5}
STEP_ORDER = [(-S, 0), (S, 0), (0, -S), (0, S)]
BOARD = 64


def box_state(b):
    for t in b["tags"]:
        if t in BOX_COLOR:
            return t
    return "idle"


def in_bounds(x, y):
    return 0 <= x <= BOARD - S and 0 <= y <= BOARD - S


def wall_cells(walls):
    cells = set()
    for w in walls:
        for y in range(w["y"] - w["y"] % S, w["y"] + w["h"], S):
            for x in range(w["x"] - w["x"] % S, w["x"] + w["w"], S):
                cells.add((x, y))
    return cells


def player_pixels(cap, color, w, h):
    px = [[color] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if (cap == "cap_up" and r == 0) or (cap == "cap_down" and r == h - 1) or \
               (cap == "cap_left" and c == 0) or (cap == "cap_right" and c == w - 1):
                px[r][c] = 0
    return px


def player_color(p):
    vals = [v for row in p["pixels"] for v in row if v not in (0, -1)]
    return max(set(vals), key=vals.count) if vals else 14


def set_box(b, state, slots):
    seated = (b["x"], b["y"]) in slots
    b["tags"] = ["box", state] + (["seated"] if seated else [])
    col = BOX_COLOR[state]
    n, m = len(b["pixels"]), len(b["pixels"][0])
    b["pixels"] = [[col if r in (0, n - 1) or c in (0, m - 1) else b["pixels"][r][c] for c in range(m)]
                   for r in range(n)]


def step_player(player, boxes, chaser, walls, action):
    if action == 5:
        for b in boxes:
            st = box_state(b)
            if st == "faced":
                b["_st"] = "grabbed"
            elif st == "grabbed":
                b["_st"] = "faced"
        return
    if not isinstance(action, int) or action not in DIRS:
        return
    dx, dy = DIRS[action]
    pos = lambda o: (o["x"], o["y"])
    grabbed = [b for b in boxes if box_state(b) == "grabbed"]
    occupied = {pos(b): b for b in boxes}
    occupied_ch = {pos(c) for c in chaser}
    if grabbed:
        g = grabbed[0]
        np_ = (player["x"] + dx, player["y"] + dy)
        nb = (g["x"] + dx, g["y"] + dy)
        movers = {pos(player), pos(g)}
        ok_p = in_bounds(*np_) and np_ not in walls and np_ not in occupied_ch and \
            (np_ not in occupied or np_ in movers)
        ok_b = in_bounds(*nb) and nb not in occupied_ch and (nb not in occupied or nb in movers)
        if ok_p and ok_b:
            player["x"], player["y"] = np_
            g["x"], g["y"] = nb
        return
    for b in boxes:
        if box_state(b) == "faced":
            b["_st"] = "idle"
    player["_cap"] = CAPS[action]
    np_ = (player["x"] + dx, player["y"] + dy)
    if np_ in occupied:
        b = occupied[np_]
        if box_state(b) == "idle":
            b["_st"] = "faced"
        return
    if in_bounds(*np_) and np_ not in walls and np_ not in occupied_ch:
        player["x"], player["y"] = np_


def bfs_step(start, goals, free):
    if start in goals:
        return None
    dist = {g: 0 for g in goals if free(g)}
    q = deque(dist)
    while q:
        c = q.popleft()
        for dx, dy in STEP_ORDER:
            n = (c[0] + dx, c[1] + dy)
            if n not in dist and free(n):
                dist[n] = dist[c] + 1
                q.append(n)
    best = None
    for dx, dy in STEP_ORDER:
        n = (start[0] + dx, start[1] + dy)
        if n in dist and (best is None or dist[n] < dist[best]):
            best = n
    if best is None or start in dist and dist[best] >= dist[start]:
        return None
    return best


def step_chaser(ch, boxes, player, walls, slots):
    pos = (ch["x"], ch["y"])
    carried = [b for b in boxes if box_state(b) == "carried"]
    pcell = (player["x"], player["y"])
    if carried:
        cb = carried[0]
        if (cb["x"], cb["y"]) in slots:
            cb["_st"] = "idle"
            return
        off = (cb["x"] - ch["x"], cb["y"] - ch["y"])
        others = {(b["x"], b["y"]) for b in boxes if b is not cb}
        empty = slots - others

        def free(c):
            bc = (c[0] + off[0], c[1] + off[1])
            return in_bounds(*c) and in_bounds(*bc) and c not in walls and c not in others and \
                c != pcell and bc not in others and bc != pcell

        goals = {(s[0] - off[0], s[1] - off[1]) for s in empty}
        nxt = bfs_step(pos, goals, free)
        if nxt:
            ch["x"], ch["y"] = nxt
            cb["x"], cb["y"] = nxt[0] + off[0], nxt[1] + off[1]
        return
    cand = [b for b in boxes if box_state(b) == "idle" and (b["x"], b["y"]) not in slots]
    for dx, dy in STEP_ORDER:
        for b in cand:
            if (b["x"], b["y"]) == (pos[0] + dx, pos[1] + dy):
                b["_st"] = "carried"
                return
    occ = {(b["x"], b["y"]) for b in boxes}

    def free(c):
        return in_bounds(*c) and c not in walls and c not in occ and c != pcell

    goals = {(b["x"] + dx, b["y"] + dy) for b in cand for dx, dy in STEP_ORDER}
    nxt = bfs_step(pos, goals, free)
    if nxt:
        ch["x"], ch["y"] = nxt


def render_walls(walls_objs, before_occ, after_occ):
    for w in walls_objs:
        px = w["pixels"]
        hidden = lambda r, c, occ: any(ox <= w["x"] + c < ox + ow and oy <= w["y"] + r < oy + oh
                                       for ox, oy, ow, oh in occ)
        tex = {}
        for r in range(w["h"]):
            for c in range(w["w"]):
                if not hidden(r, c, before_occ):
                    tex.setdefault((r % S, c), px[r][c])
        new = []
        for r in range(w["h"]):
            row = []
            for c in range(w["w"]):
                if hidden(r, c, after_occ):
                    row.append(-1)
                elif not hidden(r, c, before_occ):
                    row.append(px[r][c])
                else:
                    row.append(tex.get((r % S, c), -1))
            new.append(row)
        w["pixels"] = new


def occluders(objs):
    return [(o["x"], o["y"], o["w"], o["h"]) for o in objs
            if o["type"] != "wall" and "pixels" in o and o.get("visible", True)]


def transition_function(state, action):
    if isinstance(action, dict):
        action = action.get("action_id")
    objs = copy.deepcopy(state)
    before_occ = occluders(objs)
    players = [o for o in objs if o["type"] == "player"]
    chasers = [o for o in objs if o["type"] == "chaser"]
    boxes = [o for o in objs if o["type"] == "block"]
    wall_objs = [o for o in objs if o["type"] == "wall"]
    slot_objs = [o for o in objs if o["type"] == "target"]
    slots = {(s["x"], s["y"]) for s in slot_objs}
    walls = wall_cells(wall_objs)
    for p in players:
        step_player(p, boxes, chasers, walls, action)
    for ch in chasers:
        p = players[0] if players else {"x": -99, "y": -99}
        step_chaser(ch, boxes, p, walls, slots)
    for p in players:
        if "_cap" in p:
            cap = p.pop("_cap")
            p["tags"] = [t for t in p["tags"] if not t.startswith("cap_")] + [cap]
            p["pixels"] = player_pixels(cap, player_color(p), p["w"], p["h"])
    for b in boxes:
        set_box(b, b.pop("_st", box_state(b)), slots)
    filled = {(b["x"], b["y"]) for b in boxes}
    for s in slot_objs:
        s["tags"] = [t for t in s["tags"] if t not in ("empty", "filled")] + \
            ["filled" if (s["x"], s["y"]) in filled else "empty"]
    prefix = boxes[0]["name"].rsplit("_", 1)[0] if boxes else "box"
    for i, b in enumerate(sorted(boxes, key=lambda b: (b["y"], b["x"]))):
        b["name"] = "%s_%d" % (prefix, i)
    render_walls(wall_objs, before_occ, occluders(objs))
    return objs
