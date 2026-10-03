# Crane: A1/A2 move ring+beam y -/+6 carrying blocks overlapping the beam rect
# + chain-push; cancelled if a block leaves y[2,58] or hits the ring. A4/A3
# beam w -/+6 in [1,43]: hooked (next/collected at tip-5 on beam rows) drags
# all on-beam blocks +-6, else extend pushes swept target; blocked groups let
# beam slide alone. Tags: hooked->collected; next sticky. Clicks are no-ops.


def transition_function(state, action):
    objs = [dict(o) for o in state]
    ring = beam = None
    blocks = []
    for o in objs:
        t = o.get("type")
        if t == "player":
            ring = o
        elif t == "arm":
            beam = o
        elif t == "block":
            blocks.append(o)
    if ring is None or beam is None or not isinstance(action, int):
        return objs

    def hit(a0, a1, b0, b1):
        return a0 < b1 and b0 < a1

    def on_rows(b, by=None):
        by = beam["y"] if by is None else by
        return hit(b["y"], b["y"] + b["h"], by, by + beam["h"])

    def status(b):
        for t in b["tags"]:
            if t in ("todo", "next", "collected"):
                return t
        return "todo"

    def set_status(b, s):
        ts = [s if t in ("todo", "next", "collected") else t for t in b["tags"]]
        if not any(t in ("todo", "next", "collected") for t in ts):
            ts.append(s)
        b["tags"] = ts

    def hits_ring(nx, ny, ry=None):
        ry = ring["y"] if ry is None else ry
        return (hit(nx, nx + 4, ring["x"], ring["x"] + ring["w"]) and
                hit(ny, ny + 4, ry, ry + ring["h"]))

    def propagate(movers, dx, dy):
        changed = True
        while changed:
            changed = False
            for i, b in enumerate(blocks):
                if i in movers:
                    continue
                for j in movers:
                    m = blocks[j]
                    if (hit(b["x"], b["x"] + 4, m["x"] + dx, m["x"] + dx + 4)
                            and hit(b["y"], b["y"] + 4,
                                    m["y"] + dy, m["y"] + dy + 4)):
                        movers.add(i)
                        changed = True
                        break

    def valid(movers, dx, dy, ry=None):
        for i in movers:
            b = blocks[i]
            nx, ny = b["x"] + dx, b["y"] + dy
            if not (0 <= nx <= 48 and 2 <= ny <= 58):
                return False
            if hits_ring(nx, ny, ry):
                return False
        return True

    def apply(movers, dx, dy):
        for i in movers:
            blocks[i]["x"] += dx
            blocks[i]["y"] += dy

    def retag():
        onb = [b for b in blocks if on_rows(b)]
        tip = beam["x"] + beam["w"]
        coll = set()
        for b in onb:
            if (status(b) in ("next", "collected") and b["x"] == tip - 5
                    and hit(b["x"], b["x"] + 4, beam["x"], tip)
                    and all(o is b or status(o) == "collected" for o in onb)):
                coll.add(id(b))
        leftover = [b for b in blocks
                    if status(b) == "collected" and id(b) not in coll]
        prevn = [b for b in blocks
                 if status(b) == "next" and id(b) not in coll]
        if leftover:
            nxt = max(leftover, key=lambda b: b["x"])
        elif prevn:
            nxt = prevn[0]
        else:
            pool = [b for b in blocks if id(b) not in coll]
            nxt = max(pool, key=lambda b: b["x"]) if pool else None
        for b in blocks:
            set_status(b, "collected" if id(b) in coll
                       else ("next" if b is nxt else "todo"))

    if action in (1, 2):
        dy = -6 if action == 1 else 6
        nry, nby = ring["y"] + dy, beam["y"] + dy
        if not (0 <= nry and nry + ring["h"] <= 64):
            return objs
        movers = set()
        for i, b in enumerate(blocks):
            if (hit(b["x"], b["x"] + 4, beam["x"], beam["x"] + beam["w"])
                    and (on_rows(b) or on_rows(b, nby))):
                movers.add(i)
        for i, b in enumerate(blocks):
            if i not in movers and hits_ring(b["x"], b["y"], nry):
                movers.add(i)
        propagate(movers, 0, dy)
        if not valid(movers, 0, dy, nry):
            return objs
        apply(movers, 0, dy)
        ring["y"], beam["y"] = nry, nby
        retag()
        return objs

    if action in (3, 4):
        dw = 6 if action == 4 else -6
        nw = beam["w"] + dw
        if not (1 <= nw <= 43):
            return objs
        tip = beam["x"] + beam["w"]
        hooked = any(status(b) in ("next", "collected") and on_rows(b)
                     and b["x"] == tip - 5
                     and hit(b["x"], b["x"] + 4, beam["x"], tip)
                     for b in blocks)
        if hooked:
            movers = set(i for i, b in enumerate(blocks)
                         if on_rows(b) and hit(b["x"], b["x"] + 4,
                                               beam["x"],
                                               max(tip, beam["x"] + nw)))
        elif action == 4:
            movers = set(i for i, b in enumerate(blocks)
                         if status(b) in ("next", "collected") and on_rows(b)
                         and hit(b["x"], b["x"] + 4, tip, beam["x"] + nw))
        else:
            movers = set()
        propagate(movers, dw, 0)
        if valid(movers, dw, 0):
            apply(movers, dw, 0)
        beam["w"] = nw
        beam["pixels"] = [[2 if i % 3 == 1 else 1 for i in range(nw)],
                          [2 if i % 3 == 0 else 1 for i in range(nw)]]
        retag()
        return objs

    return objs
