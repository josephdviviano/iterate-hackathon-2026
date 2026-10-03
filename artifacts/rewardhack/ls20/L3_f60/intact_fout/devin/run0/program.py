# Fuel-maze game, frame variant. Player 5x5 moves +-5 into cells holding floor(3).
# A '1' strip on a destination cell edge is a portal: slide away from the marker
# to the last floor cell. Each directional action burns 2 budget (bar shrinks
# left, right edge fixed); covering a 'chamber' object is refused for free until
# the glyph shows its mask in its colour (shape AND colour — unlock unobserved).
# Covered 'collect' targets hide under the player and rotate the hud glyph 90deg
# CW; refuel rings refill the bar; recolor button repaints the glyph 12->9.

MEM = {"frame": None, "spr": {}, "gone": set()}
DIRS = {1: (0, -5), 2: (0, 5), 3: (-5, 0), 4: (5, 0)}


def _floor(fr, x, y):
    if not (0 <= x <= 59 and 0 <= y <= 59):
        return False
    return any(fr[yy][xx] == 3 for yy in range(y, y + 5) for xx in range(x, x + 5))


def _portal(fr, x, y):
    # '1' strip along a cell edge -> travel in the opposite direction
    if x > 0 and all(fr[y + i][x - 1] == 1 for i in range(5)):
        return (5, 0)
    if x + 5 < 64 and all(fr[y + i][x + 5] == 1 for i in range(5)):
        return (-5, 0)
    if y > 0 and all(fr[y - 1][x + i] == 1 for i in range(5)):
        return (0, 5)
    if y + 5 < 64 and all(fr[y + 5][x + i] == 1 for i in range(5)):
        return (0, -5)
    return None


def _capture(o, fr):
    x0, y0, w, h = o["x"], o["y"], o["w"], o["h"]
    cells = {(x, y) for x in range(x0, x0 + w) for y in range(y0, y0 + h)}
    frontier = set(cells)
    while frontier:
        nxt = set()
        for cx, cy in frontier:
            for ax, ay in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (ax, ay) in cells or not (0 <= ax < 64 and 0 <= ay < 64):
                    continue
                if not (x0 - 3 <= ax <= x0 + w + 2 and y0 - 3 <= ay <= y0 + h + 2):
                    continue
                if fr[ay][ax] in (3, 4, 5):
                    continue
                cells.add((ax, ay))
                nxt.add((ax, ay))
        frontier = nxt
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    rect = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
    pix = {(x, y): fr[y][x] for x in range(rect[0], rect[2]) for y in range(rect[1], rect[3])}
    return {"rect": rect, "type": o.get("type"), "tags": o.get("tags", []), "pix": pix}


def _paint_bar(out, bar, full):
    bx, bw, by, bh = bar["x"], bar["w"], bar["y"], bar["h"]
    L = bx
    while L - 1 >= 0 and out[by][L - 1] in (3, 11):
        L -= 1
    R = bx + bw
    while R < 64 and out[by][R] in (3, 11):
        R += 1
    w = R - L if full else max(0, bw - 2)
    for yy in range(by, by + bh):
        for xx in range(L, R):
            out[yy][xx] = 11 if xx >= R - w else 3


def _glyph_mask(fr, o, off=5):
    bs = max(1, o["w"] // 3)
    return [[fr[o["y"] + r * bs][o["x"] + c * bs] != off for c in range(3)] for r in range(3)]


def _glyph_step(out, glyph, fr):
    # each collect rotates the progress glyph 90 degrees CW in its own colour
    gx, gy, bs = glyph["x"], glyph["y"], max(1, glyph["w"] // 3)
    P = _glyph_mask(fr, glyph)
    col = next((fr[yy][xx] for yy in range(gy, gy + glyph["h"])
                for xx in range(gx, gx + glyph["w"]) if fr[yy][xx] != 5), 12)
    for r in range(3):
        for c in range(3):
            v = col if P[2 - c][r] else 5
            for i in range(bs):
                for j in range(bs):
                    out[gy + r * bs + i][gx + c * bs + j] = v


def _glyph_open(fr, glyph, chamber):
    # chamber unlocks once the glyph shows the legend mask in its colour
    if not glyph or _glyph_mask(fr, glyph) != _glyph_mask(fr, chamber):
        return False
    want = next((fr[y][x] for y in range(chamber["y"], chamber["y"] + chamber["h"])
                 for x in range(chamber["x"], chamber["x"] + chamber["w"])
                 if fr[y][x] != 5), None)
    return any(fr[y][x] == want for y in range(glyph["y"], glyph["y"] + glyph["h"])
               for x in range(glyph["x"], glyph["x"] + glyph["w"]))


def transition_function(state, action, frame):
    fr = frame
    out = [r[:] for r in fr]
    if MEM["frame"] != fr:  # continuity broken -> stateless rebuild
        MEM["spr"] = {}
        MEM["gone"] = set()

    player = bar = glyph = None
    chambers = []
    for o in state:
        t, tags = o.get("type"), o.get("tags", [])
        if t == "player":
            player = o
        elif "budget" in tags:
            bar = o
        elif "progress" in tags:
            glyph = o
        if "chamber" in tags:
            chambers.append(o)
        if t in ("target", "refuel", "button") and o.get("visible", True) \
                and o["name"] not in MEM["gone"]:
            MEM["spr"][o["name"]] = _capture(o, fr)

    aid = action if isinstance(action, int) else (action or {}).get("action_id")
    dx, dy = DIRS.get(aid, (0, 0))

    if player and (dx or dy):
        px, py = player["x"], player["y"]
        nx, ny = px + dx, py + dy
        refused = any(not _glyph_open(fr, glyph, o)
                      and nx <= o["x"] and ny <= o["y"] and o["x"] + o["w"] <= nx + 5
                      and o["y"] + o["h"] <= ny + 5 for o in chambers)
        if refused:  # locked chamber: whole state unchanged, no cost
            MEM["frame"] = out
            return out
        refill = recolor = collect = False
        if _floor(fr, nx, ny):
            p = _portal(fr, nx, ny)
            if p:
                while _floor(fr, nx + p[0], ny + p[1]):
                    nx, ny = nx + p[0], ny + p[1]
            for yy in range(py, py + 5):  # vacated cells revert to floor
                for xx in range(px, px + 5):
                    out[yy][xx] = 3
            for name, s in MEM["spr"].items():
                x0, y0, x1, y1 = s["rect"]
                s["hidden"] = nx <= x0 and ny <= y0 and x1 <= nx + 5 and y1 <= ny + 5
                if s["hidden"]:
                    if s["type"] == "refuel":
                        MEM["gone"].add(name)
                        refill = True
                    elif s["type"] == "button":
                        MEM["gone"].add(name)
                        recolor = True
                    elif "collect" in s["tags"]:
                        collect = True
            for name, s in MEM["spr"].items():
                if name in MEM["gone"] or s["hidden"]:
                    continue
                for (xx, yy), v in s["pix"].items():
                    out[yy][xx] = v
            pix = player.get("pixels")
            for i in range(5):
                for j in range(5):
                    out[ny + i][nx + j] = pix[i][j] if pix else (12 if i < 2 else 9)
        _paint_bar(out, bar, refill)
        if collect and glyph:
            _glyph_step(out, glyph, fr)
        if recolor and glyph:
            for yy in range(glyph["y"], glyph["y"] + glyph["h"]):
                for xx in range(glyph["x"], glyph["x"] + glyph["w"]):
                    if out[yy][xx] != 5:
                        out[yy][xx] = 9

    MEM["frame"] = out
    return out
