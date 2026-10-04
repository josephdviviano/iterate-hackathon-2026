# Mechanics: word-edit game. A3/A4 move the cursor (colour-0 bracket sprite) to the previous/next
# editable glyph slot with wraparound. A1/A2 step the glyph under the cursor forward/back through a
# fixed 7-glyph font cycle (ink 5 on 7), each slot drawing it under its own fixed rotation, read
# from the slot's current glyph. HUD row 63 turns 1->4 from the right: ceil(actions/2) cells.
# Hypotheses: slot rotation ties (symmetric glyphs) broken by the slot's reference-word rotation vs
# the cyan legend minus 90 deg; action count is hidden (parity), carried only while frames continue.

FONT = [  # cycle order; '#' = ink
    "#####|.#..#|.#..#|.####|....#",
    "..#..|#####|#.#.#|#####|..#..",
    "####.|#..##|#...#|##..#|.####",
    "..#..|#####|#...#|#...#|#####",
    "#####|#...#|#####|.#.#.|.###.",
    "..###|..#.#|#####|#.#..|###..",
    "#####|#.#.#|#.###|#...#|#####",
]
FONT = [tuple(tuple(int(c == "#") for c in row) for row in g.split("|")) for g in FONT]
INK, PAPER, HUD_OFF, HUD_ON, BG = 5, 7, 1, 4, 3

_last = {"frame": None, "n": 0}


def rot(m, k):
    for _ in range(k % 4):
        m = tuple(zip(*m[::-1]))
    return tuple(tuple(r) for r in m)


def d4(m):
    out, a = [], m
    for k in range(4):
        out.append((k, False, a))
        out.append((k, True, tuple(r[::-1] for r in a)))
        a = rot(a, 1)
    return out


def read_bits(frame, o, ink):
    return tuple(tuple(int(frame[o["y"] + j][o["x"] + i] == ink) for i in range(o["w"])) for j in range(o["h"]))


def ref_rotation(state, frame, slot):
    """Rotation of the reference word above `slot` relative to its cyan legend glyph, or None."""
    refs = [o for o in state if o["type"] == "target" and o["x"] == slot["x"]]
    legends = [o for o in state if o["type"] == "legend" and "cyan" in o.get("tags", [])]
    if not refs:
        return None
    r = read_bits(frame, refs[0], INK)
    for lg in legends:
        for k, flip, m in d4(read_bits(frame, lg, INK)):
            if not flip and m == r:
                return k
    return None


def glyph_step(state, frame, out, slot, delta):
    cur = read_bits(frame, slot, INK)
    cands = [(i, k) for i, g in enumerate(FONT) for k in range(4) if rot(g, k) == cur]
    if not cands:
        return
    pr = ref_rotation(state, frame, slot)
    prefer = None if pr is None else (pr + 3) % 4
    cands.sort(key=lambda c: (c[1] != prefer, c[1]))
    i, k = cands[0]
    new = rot(FONT[(i + delta) % len(FONT)], k)
    for j in range(slot["h"]):
        for x in range(slot["w"]):
            out[slot["y"] + j][slot["x"] + x] = INK if new[j][x] else PAPER


def cursor_step(state, frame, out, cursor, delta):
    slots = sorted({o["x"] for o in state if o["type"] == "glyph"})
    if cursor["x"] not in slots:
        return
    nx = slots[(slots.index(cursor["x"]) + delta) % len(slots)]
    cells = [(cursor["x"] + i, cursor["y"] + j) for j in range(cursor["h"]) for i in range(cursor["w"])
             if frame[cursor["y"] + j][cursor["x"] + i] == 0]
    for x, y in cells:
        out[y][x] = BG
    for x, y in cells:
        out[y][x - cursor["x"] + nx] = 0


def hud_step(frame, out, counter):
    y = counter["y"]
    lit = sum(1 for v in frame[y] if v == HUD_ON)
    n = _last["n"] if _last["frame"] == frame else max(0, 2 * lit - 1)
    n += 1
    want = (n + 1) // 2
    for i in range(min(want, 64)):
        out[y][63 - i] = HUD_ON
    return n


def transition_function(state, action, frame):
    out = [list(r) for r in frame]
    aid = action["action_id"] if isinstance(action, dict) else action
    cursor = next((o for o in state if o["type"] == "player"), None)
    counter = next((o for o in state if o["type"] == "counter"), None)
    if cursor is not None:
        slot = next((o for o in state if o["type"] == "glyph" and o["x"] == cursor["x"]), None)
        if aid in (1, 2) and slot is not None:
            glyph_step(state, frame, out, slot, 1 if aid == 1 else -1)
        elif aid in (3, 4):
            cursor_step(state, frame, out, cursor, -1 if aid == 3 else 1)
    n = hud_step(frame, out, counter) if counter is not None else 0
    _last["frame"], _last["n"] = [list(r) for r in out], n
    return out
