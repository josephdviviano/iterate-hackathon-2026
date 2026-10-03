# Crane/skewer game: ring (player) + beam (arm, h=2, x=ring.x+5, y=ring.y+2, w=1+6k<=43) + 4x4 blocks.
# A1/A2: ring+beam y-/+6, blocks on the beam are carried, blocks hit get pushed (chain); any block
#   leaving y in [2,58] or ring leaving the board cancels the whole move. A4/A3: beam w +/-6.
# Hooked = next/collected block at beam tip-4: blocks on beam follow the tip (group blocked -> beam alone).
# Unhooked extend pushes swept blocks +6 (chain, x<=48 else beam slides under). Tags: hooked next alone
#   -> collected, rightmost todo -> next; unhooked collected reverts. Hypothesis: board 64, clicks no-op.
import copy

STEP, MAXW, BXMAX, BYMIN, BYMAX = 6, 43, 48, 2, 58


def beam_pixels(w):
    return [[2 if i % 3 == 1 else 1 for i in range(w)], [2 if i % 3 == 0 else 1 for i in range(w)]]


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def rect(o):
    return (o["x"], o["y"], o["w"], o["h"])


def tagstate(b):
    for t in ("collected", "next", "todo"):
        if t in b["tags"]:
            return t
    return None


def set_tag(b, t):
    b["tags"] = [x for x in b["tags"] if x not in ("collected", "next", "todo")] + [t]


def push_chain(blocks, seeds, dx, dy, fixed):
    """Move seed blocks by (dx,dy), pushing any block they hit; returns moved set or None if blocked."""
    moved, frontier = set(), list(seeds)
    while frontier:
        i = frontier.pop()
        if i in moved:
            continue
        moved.add(i)
        b = blocks[i]
        nr = (b["x"] + dx, b["y"] + dy, b["w"], b["h"])
        for j, c in enumerate(blocks):
            if j not in moved and overlap(nr, rect(c)):
                frontier.append(j)
    for i in moved:
        b = blocks[i]
        nx, ny = b["x"] + dx, b["y"] + dy
        if not (BYMIN <= ny <= BYMAX) or nx > BXMAX or overlap((nx, ny, b["w"], b["h"]), fixed):
            return None
    return moved


def transition_function(state, action):
    st = copy.deepcopy(state)
    ring = next((o for o in st if o["type"] == "player"), None)
    beam = next((o for o in st if o["type"] == "arm"), None)
    blocks = [o for o in st if o["type"] == "block"]
    if ring is None or beam is None or not isinstance(action, int) or action not in (1, 2, 3, 4):
        return st
    on_beam = [i for i, b in enumerate(blocks) if overlap(rect(b), rect(beam))]
    tip = beam["x"] + beam["w"] - 1
    hooked = any(tagstate(blocks[i]) in ("next", "collected") and blocks[i]["x"] == tip - 4 for i in on_beam)
    ringr = rect(ring)

    if action in (1, 2):
        dy = -STEP if action == 1 else STEP
        if ring["y"] + dy < 0 or ring["y"] + dy + ring["h"] > 64:
            return st
        nbeam = (beam["x"], beam["y"] + dy, beam["w"], beam["h"])
        seeds = set(on_beam) | {i for i, b in enumerate(blocks) if overlap(rect(b), nbeam)}
        moved = push_chain(blocks, seeds, 0, dy, (ring["x"], ring["y"] + dy, ring["w"], ring["h"]))
        if moved is None:
            return st
        for i in moved:
            blocks[i]["y"] += dy
        ring["y"] += dy
        beam["y"] += dy
    else:
        dw = STEP if action == 4 else -STEP
        nw = beam["w"] + dw
        if nw < 1 or nw > MAXW:
            return st
        if hooked:
            ok = all(blocks[i]["x"] + dw <= BXMAX and not overlap(
                (blocks[i]["x"] + dw, blocks[i]["y"], 4, 4), ringr) for i in on_beam)
            if ok:
                for i in on_beam:
                    blocks[i]["x"] += dw
            elif dw > 0:
                return st
        elif dw > 0:
            swept = (tip + 1, beam["y"], dw, beam["h"])
            seeds = {i for i, b in enumerate(blocks) if overlap(rect(b), swept)}
            moved = push_chain(blocks, seeds, dw, 0, ringr) if seeds else set()
            for i in moved or ():
                blocks[i]["x"] += dw
        beam["w"] = nw
        beam["pixels"] = beam_pixels(nw)

    update_tags(blocks, beam)
    return st


def update_tags(blocks, beam):
    tip = beam["x"] + beam["w"] - 1
    on_beam = [b for b in blocks if overlap(rect(b), rect(beam))]
    for b in blocks:
        s = tagstate(b)
        hk = b in on_beam and b["x"] == tip - 4
        if s == "next" and hk and len(on_beam) == 1:
            set_tag(b, "collected")
            todo = [c for c in blocks if tagstate(c) == "todo"]
            if todo:
                set_tag(max(todo, key=lambda c: c["x"]), "next")
            return
        if s == "collected" and not hk:
            for c in blocks:
                if tagstate(c) == "next":
                    set_tag(c, "todo")
            set_tag(b, "next")
            return
