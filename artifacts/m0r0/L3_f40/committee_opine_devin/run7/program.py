# Mirrored-blocks / marker game transition rule.
# Two 4x4 avatars move only onto all-5 destination boxes; A1/A2 = up/down for both,
# A3/A4 = horizontal with the right-half avatar mirrored about x=32. Each moves independently.
# Click: hit an inactive marker -> players arm (colour 1) and it activates (11); while
# armed, arrows move the active 2x2 marker, clicks retarget it, clicking a player disarms.
# HUD: colour-0 bars (row 0 right end, row 63 left end) reach width 3*(n+1)//7 at call n
# of an episode; n persists across calls and resets when the incoming frame is not my last.
# Unverified: hazard treated as non-floor blocker, A5/A7 no-ops, reset trigger not modelled.

_FLOOR = 5
_MEM = {"frame": None, "n": 0}


def _hit(boxes, x, y):
    return [b for b in boxes
            if b["x"] <= x < b["x"] + b["w"] and b["y"] <= y < b["y"] + b["h"]]


def _free(frame, x, y, w, h):
    if x < 0 or y < 0 or x + w > len(frame[0]) or y + h > len(frame):
        return False
    return all(frame[yy][xx] == _FLOOR
               for yy in range(y, y + h) for xx in range(x, x + w))


def _paint(out, o, colour):
    for yy in range(o["y"], o["y"] + o["h"]):
        for xx in range(o["x"], o["x"] + o["w"]):
            out[yy][xx] = colour


def _move(frame, out, o, dx, dy):
    nx, ny = o["x"] + dx, o["y"] + dy
    if not _free(frame, nx, ny, o["w"], o["h"]):
        return
    colour = frame[o["y"]][o["x"]]
    for yy in range(o["y"], o["y"] + o["h"]):
        for xx in range(o["x"], o["x"] + o["w"]):
            out[yy][xx] = _FLOOR
    for yy in range(ny, ny + o["h"]):
        for xx in range(nx, nx + o["w"]):
            out[yy][xx] = colour


def transition_function(state, action, frame):
    mem = _MEM
    if mem["frame"] is None or mem["frame"] != frame:
        mem["n"] = 1
    else:
        mem["n"] += 1
    out = [row[:] for row in frame]

    players = [o for o in state if o.get("type") == "player"]
    markers = [o for o in state if o.get("type") == "marker"]
    armed = any("armed" in o.get("tags", ()) for o in players)

    aid = action["action_id"] if isinstance(action, dict) else action
    if aid == 6:
        hm = _hit(markers, action["x"], action["y"])
        hp = _hit(players, action["x"], action["y"])
        if armed:
            if hm and "active" not in hm[0].get("tags", ()):
                for m in markers:
                    _paint(out, m, 11 if m is hm[0] else 9)
            elif hp:
                for p in players:
                    _paint(out, p, 10)
                for m in markers:
                    _paint(out, m, 9)
        elif hm:
            for p in players:
                _paint(out, p, 1)
            for m in markers:
                _paint(out, m, 11 if m is hm[0] else 9)
    elif aid in (1, 2, 3, 4):
        if armed:
            act = [m for m in markers if "active" in m.get("tags", ())]
            if act:
                dx = (aid == 4) - (aid == 3)
                dy = (aid == 2) - (aid == 1)
                _move(frame, out, act[0], 4 * dx, 4 * dy)
        else:
            for p in players:
                if aid in (3, 4):
                    hdir = 1 if aid == 4 else -1
                    dx = hdir * 4 if p["x"] + p["w"] / 2 < 32 else -hdir * 4
                    _move(frame, out, p, dx, 0)
                else:
                    _move(frame, out, p, 0, -4 if aid == 1 else 4)

    w = 3 * (mem["n"] + 1) // 7
    if w:
        for x in range(64 - w, 64):
            out[0][x] = 0
        for x in range(w):
            out[63][x] = 0
    mem["frame"] = out
    return out
