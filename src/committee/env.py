"""The environment contract, shared by every build.

Three modes. "objects": ``transition_function(state, action) -> state``,
admitted by exact object replay, every field of every object. "frame": the
same rule with the 64x64 before frame as a third argument, which is where
OPINE-World's rule reads walls, floor and hazards that the extractor does
not emit. "frame_out": ``transition_function(state, action, frame) -> frame``,
OPINE-World's rule itself; admission is frame equality, cell by cell, and the
game's released extractor run on the predicted frame gives the object view
for the effect-row analyses. That view is exact only where the extractor is
stateless, so ``engine_source`` checks it against the whole recording and
refuses the games where it is not (two of the 25 released extractors keep
state across frames). The object list is an input in every mode; in
"frame_out" it is read only. The engine source never enters a workspace.
"""

from __future__ import annotations

import json
import re

from .loader import Transition, build_buffer, final_engine_source, load_bundle, run_extractor
from .matrix import effect_signature, pair_objects

MODES = ("objects", "frame", "frame_out")

# Source patterns the verifier rejects before running a program; the workspace checker applies the same list.
FORBIDDEN = [r"\bopen\s*\(", r"\bpickle\b", r"\bsubprocess\b", r"\bimport\s+socket\b", r"\bfrom\s+socket\s+import\b",
             r"\burllib\b",
             r"\brequests\b", r"\b__import__\b", r"\bimportlib\b", r"\beval\s*\(", r"\bexec\s*\(",
             r"replay", r"buffer\.json"]


def static_violations(source: str) -> list[str]:
    return [p for p in FORBIDDEN if re.search(p, source)]


def mode_name(frame: bool = False, frame_out: bool = False) -> str:
    if frame and frame_out:
        raise ValueError("--frame and --frame-out name different modes; give one")
    return "frame_out" if frame_out else "frame" if frame else "objects"


def takes_frame(mode: str) -> bool:
    _check(mode)
    return mode != "objects"


def returns_frame(mode: str) -> bool:
    _check(mode)
    return mode == "frame_out"


def _check(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; one of {MODES}")


def observed(t: Transition, mode: str) -> list:
    """What a prediction is compared with: the after objects, or in frame_out the after frame."""
    return t.after_grid if returns_frame(mode) else t.after_objs


def _sorted_json(objs: list) -> list[str]:
    return sorted(json.dumps(o, sort_keys=True) for o in objs)


_GATE: dict[str, str | ValueError] = {}


def _stateless_engine(game: str) -> str:
    """The released engine, after a check that a fresh extraction of every recorded after frame
    equals the recording's objects. Where it does not, the extractor carried state between frames
    and the object view of a predicted frame cannot be exact. One pass per game per process,
    whichever way it comes out."""
    if game not in _GATE:
        engine = final_engine_source(load_bundle(game))
        ts = build_buffer(game)
        fresh = run_extractor(engine, [t.after_grid for t in ts], fresh=True)
        bad = sum(not (isinstance(f, list) and _sorted_json(f) == _sorted_json(t.after_objs)) for f, t in zip(fresh, ts))
        _GATE[game] = engine if not bad else ValueError(
            f"{game}: the released extractor keeps state across frames; a fresh extraction differs from the "
            f"recording on {bad} of {len(ts)} transitions, so frame_out has no exact object view here")
    result = _GATE[game]
    if isinstance(result, ValueError):
        raise result
    return result


def engine_source(game: str, mode: str) -> str | None:
    """The released engine, needed only to extract objects from predicted frames."""
    return _stateless_engine(game) if returns_frame(mode) else None


CONTRACT = """\
# Task: write the transition rule of an unknown ARC-AGI-3 game

You see a sequence of observed transitions of one level of a game. Each
transition is (before state, action, after state). A state is a list of object
dicts produced by a fixed extractor; the schema is whatever the objects carry
(name, type, tags, x, y, w, h, layer, visible, pixels, ...).

Write `transition_function(state, action)` in `program.py`:

- `state`: list of object dicts. `action`: an int (1 to 5, 7) or
  `{"action_id": 6, "x": int, "y": int}` for a click at grid cell (x, y).
- Return the predicted after state as a list of object dicts with the same
  schema. Object order does not matter. Every field of every object must match
  the observed after state exactly, including names and pixels.
- The checker calls your function once per transition, in order, always with
  the observed before state. You may keep hidden state across calls, but gate
  it on continuity: if the incoming state is not the state you last returned,
  fall back to a stateless default.

Rules:

1. Model the mechanics. Factor the rule by object type: one update rule per
   type, named guards for interactions between types. Prefer the simplest rule
   that explains every transition and would generalise to unseen layouts.
2. Never tabulate observed states. Do not map a state or step index to its
   successor. Do not read `buffer.json` or any file from `program.py`. A
   source that contains `open(`, `pickle`, `exec(`, `eval(`, `importlib`,
   `__import__`, `subprocess`, `socket`, `urllib`, `requests`, or the words
   `replay` or `buffer.json` is rejected; `check.py` reports it.
3. Only the Python standard library is available to `program.py`.
4. Run `python3 check.py` to see which transitions fail and how. Iterate until
   it prints ALL PASS, or until you are convinced the remaining failures need
   observations you do not have. Then stop.
5. Keep `program.py` self-contained and under 400 lines. Put a 5-line header
   comment that states the mechanics you implemented and any hypothesis you
   could not confirm.

Files: `transitions.md` (readable diffs), `buffer.json` (full states),
`program.py` (your code), `check.py` (the checker).
"""

_SIGNATURE_LINE = "Write `transition_function(state, action)` in `program.py`:"
_STATE_BULLET = """- `state`: list of object dicts. `action`: an int (1 to 5, 7) or
  `{"action_id": 6, "x": int, "y": int}` for a click at grid cell (x, y).
"""
_RETURN_BULLET = """- Return the predicted after state as a list of object dicts with the same
  schema. Object order does not matter. Every field of every object must match
  the observed after state exactly, including names and pixels.
"""
FRAME_BULLET = """- `frame`: the 64x64 before frame, a list of 64 rows of 64 colour indices
  (ints 0 to 15), `frame[y][x]`. The object extractor does not emit walls,
  floor, hazards or teleporter pads; they are visible in the frame. Read
  geometry from the frame as a rule over cell colours, for example "a move
  is blocked when the cell ahead has the wall colour". Do not list
  coordinates of observed positions.
"""
FRAME_RETURN_BULLET = """- Return the predicted 64x64 after frame in the same form as `frame`, every
  cell a Python int. The checker compares it with the observed after frame
  cell by cell; every cell must match. The object list is given for reading
  only. Render moves, pickups and HUD changes by writing cells.
"""
for _anchor in (_SIGNATURE_LINE, _STATE_BULLET, _RETURN_BULLET):
    assert _anchor in CONTRACT


def contract(mode: str = "objects") -> str:
    """The task text, with the signature and the return bullet in the form the mode fixes."""
    text = CONTRACT
    if takes_frame(mode):
        text = text.replace(_SIGNATURE_LINE, _SIGNATURE_LINE.replace("(state, action)", "(state, action, frame)"))
        text = text.replace(_STATE_BULLET, _STATE_BULLET + FRAME_BULLET)
    if returns_frame(mode):
        text = text.replace(_RETURN_BULLET, FRAME_RETURN_BULLET)
    return text


def signature(mode: str = "objects") -> str:
    return "transition_function(state, action, frame)" if takes_frame(mode) else "transition_function(state, action)"


def call_expr(mode: str = "objects") -> str:
    """The call a checker or runner makes on one buffer row `r`."""
    args = 'copy.deepcopy(r["before"]), r["action"]' + (', copy.deepcopy(r["frame"])' if takes_frame(mode) else "")
    return f"mod.transition_function({args})"


_CHECK = r'''
import json, copy, importlib.util, traceback, sys, re
rows = json.load(open("buffer.json"))
bad = [p for p in PATTERNS if re.search(p, open("program.py").read())]
if bad:
    print("FORBIDDEN: program.py matches " + ", ".join(bad) + "; the verifier rejects such a program"); sys.exit(1)
spec = importlib.util.spec_from_file_location("program", "program.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    print("LOAD ERROR\n" + traceback.format_exc()[-1500:]); sys.exit(1)
HEX = "0123456789abcdef"
def canon(s): return sorted(json.dumps(o, sort_keys=True) for o in s)
def describe(pred, actual):
    P, A = set(canon(pred)), set(canon(actual))
    extra, missing = sorted(P - A), sorted(A - P)
    lines = []
    for m in missing[:4]: lines.append("  expected but not predicted: " + m[:300])
    for e in extra[:4]: lines.append("  predicted but not observed:  " + e[:300])
    return "\n".join(lines)
def is_frame(p, want):
    return (isinstance(p, list) and len(p) == len(want)
            and all(isinstance(row, list) and len(row) == len(want[0]) and all(type(v) is int for v in row) for row in p))
def describe_frame(pred, want):
    if not is_frame(pred, want):
        return "  prediction is not a %dx%d list of lists of ints" % (len(want[0]), len(want))
    diff = [(x, y, want[y][x], pred[y][x]) for y in range(len(want)) for x in range(len(want[0])) if want[y][x] != pred[y][x]]
    shown = ", ".join("(%d,%d) want %s got %s" % (x, y, HEX[w], HEX[p] if 0 <= p < 16 else p) for x, y, w, p in diff[:10])
    return "  %d cells differ: %s" % (len(diff), shown)
n_pass, fails = 0, []
for i, r in enumerate(rows):
    try:
        pred = json.loads(json.dumps(CALL))
    except Exception:
        fails.append((i, r, "EXCEPTION\n" + traceback.format_exc()[-600:])); continue
COMPARE
print(f"{n_pass}/{len(rows)} UNIT replay exactly")
for i, r, d in fails[:6]:
    print(f"\nFAIL transition {i} (step {r['step']}) action {r['action']}\n{d}")
if not fails: print("ALL PASS")
'''

_COMPARE_OBJECTS = '''    if isinstance(pred, list) and canon(pred) == canon(r["after"]): n_pass += 1
    else: fails.append((i, r, describe(pred if isinstance(pred, list) else [], r["after"])))'''

_COMPARE_FRAMES = '''    if is_frame(pred, r["after_frame"]) and pred == r["after_frame"]: n_pass += 1
    else: fails.append((i, r, describe_frame(pred, r["after_frame"])))'''


def check_script(mode: str = "objects") -> str:
    """The workspace checker: exact object replay, or in frame_out exact frame replay."""
    frames = returns_frame(mode)
    return (_CHECK.replace("CALL", call_expr(mode)).replace("PATTERNS", repr(FORBIDDEN))
            .replace("COMPARE", _COMPARE_FRAMES if frames else _COMPARE_OBJECTS)
            .replace("UNIT", "frames" if frames else "transitions"))


CHECK_SCRIPT = check_script("objects")

_STUBS = {
    "objects": '"""Mechanics: (fill in)\n"""\n\n\ndef transition_function(state, action):\n    return state\n',
    "frame": '"""Mechanics: (fill in)\n"""\n\n\ndef transition_function(state, action, frame):\n    return state\n',
    "frame_out": '"""Mechanics: (fill in)\n"""\n\n\ndef transition_function(state, action, frame):\n'
                 '    return [row[:] for row in frame]\n',
}


def stub(mode: str = "objects") -> str:
    _check(mode)
    return _STUBS[mode]


STUB = stub("objects")


def buffer_rows(train: list[Transition], mode: str = "objects") -> list[dict]:
    """The rows of buffer.json: what the checker replays. Frames only in the modes that use them."""
    rows = []
    for t in train:
        r = {"step": t.step, "action": t.action, "before": t.before_objs, "after": t.after_objs}
        if takes_frame(mode):
            r["frame"] = t.before_grid
        if returns_frame(mode):
            r["after_frame"] = t.after_grid
        rows.append(r)
    return rows


def render_transitions(train: list[Transition]) -> str:
    out = ["# Observed transitions (in order)\n"]
    out.append("## Initial state\n")
    out.append("```json\n" + json.dumps(train[0].before_objs, sort_keys=True) + "\n```\n")
    for i, t in enumerate(train):
        out.append(f"## transition {i} (step {t.step})  action = {json.dumps(t.action)}\n")
        unchanged = 0
        for bo, ao in pair_objects(t.before_objs, t.after_objs):
            sig = effect_signature(bo, ao)
            if sig == "no_change":
                unchanged += 1
                continue
            if bo is None:
                out.append(f"- born: {json.dumps(ao, sort_keys=True)}")
            elif ao is None:
                out.append(f"- gone: {json.dumps(bo, sort_keys=True)}")
            else:
                changes = {k: [bo.get(k), ao.get(k)] for k in sig.split(",")}
                out.append(f"- {bo.get('name')} ({bo.get('type')}): {json.dumps(changes)}")
        out.append(f"- {unchanged} objects unchanged\n")
    return "\n".join(out)


HEX = "0123456789abcdef"


def render_frame(grid: list[list[int]]) -> str:
    return "\n".join("".join(HEX[v] for v in row) for row in grid)


def changed_cells(a: list[list[int]], b: list[list[int]]) -> list[tuple[int, int, int, int]]:
    return [(x, y, a[y][x], b[y][x]) for y in range(len(a)) for x in range(len(a[0])) if a[y][x] != b[y][x]]


def render_frames(train: list[Transition]) -> str:
    out = ["\n# Frames\n", "Initial before frame, one hex digit per cell, row y from top, column x from left:\n",
           "```\n" + render_frame(train[0].before_grid) + "\n```\n",
           "Cells that change per transition (x, y, before, after), up to 12 shown:\n"]
    for i, t in enumerate(train):
        cells = changed_cells(t.before_grid, t.after_grid)
        shown = ", ".join(f"({x},{y},{HEX[a]}>{HEX[b]})" for x, y, a, b in cells[:12])
        out.append(f"- transition {i}: {len(cells)} cells changed" + (f": {shown}" if cells else ""))
    return "\n".join(out) + "\n"


def render(train: list[Transition], mode: str = "objects") -> str:
    """transitions.md: object diffs, plus the frames in the modes that give them."""
    return render_transitions(train) + (render_frames(train) if takes_frame(mode) else "")


def is_frame(p, want: list[list[int]]) -> bool:
    return (isinstance(p, list) and len(p) == len(want)
            and all(isinstance(row, list) and len(row) == len(want[0]) and all(type(v) is int for v in row) for row in p))


def frame_diff(pred, want: list[list[int]], limit: int = 10) -> str:
    if not is_frame(pred, want):
        return f"  prediction is not a {len(want[0])}x{len(want)} list of lists of ints"
    diff = [(x, y, want[y][x], pred[y][x]) for y in range(len(want)) for x in range(len(want[0])) if want[y][x] != pred[y][x]]
    shown = ", ".join(f"({x},{y}) want {HEX[w]} got {HEX[p] if 0 <= p < 16 else p}" for x, y, w, p in diff[:limit])
    return f"  {len(diff)} cells differ: {shown}"


def extract_objects(engine_src: str, frames: list) -> list:
    """The released extractor on each predicted frame; None where there is no valid frame or the
    extractor fails. The extractor is reloaded before every frame, so no state carries over from
    the recording or from other predictions."""
    valid = [i for i, f in enumerate(frames) if f is not None]
    objs: list = [None] * len(frames)
    if valid:
        for i, e in zip(valid, run_extractor(engine_src, [frames[i] for i in valid], fresh=True)):
            objs[i] = e if isinstance(e, list) else None
    return objs
