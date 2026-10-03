"""Load an OPINE-World replay bundle into an object-level transition buffer.

A bundle holds one frame group per environment step: optional animation ticks,
then the settled frame. The action label sits on the first frame of the group
that it produced, so the settled frame of step i is the state after action i,
and the transition for step i is (settled[i-1], action[i], settled[i]).

Objects come from the game's released extractor, run in a subprocess. The
extractor is frozen input data; this project studies the transition layer only.
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = ROOT / "external" / "opine-world" / "docs" / "replay_data"
CACHE_DIR = ROOT / "cache"

RESET = 0
CLICK = 6

_CLICK_RE = re.compile(r"click\((\d+),(\d+)\)")

_EXTRACT_RUNNER = """
import sys, json, importlib.util
spec = importlib.util.spec_from_file_location("ge", sys.argv[1])
ge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ge)
grids = json.load(open(sys.argv[2]))
fresh = len(sys.argv) > 4 and sys.argv[4] == "fresh"
out = []
for g in grids:
    try:
        if fresh:
            spec.loader.exec_module(ge)
        out.append(ge.extract_objects(g))
    except Exception as e:
        out.append({"error": repr(e)[:200]})
json.dump(out, open(sys.argv[3], "w"))
"""


@dataclass
class Step:
    index: int
    action_label: str | None
    grid: list[list[int]]


@dataclass
class Transition:
    step: int
    level: int
    action_id: int
    click: tuple[int, int] | None
    level_advance: bool
    before_grid: list[list[int]]
    after_grid: list[list[int]]
    before_objs: list[dict]
    after_objs: list[dict]

    @property
    def action(self) -> int | dict:
        """The action in the form the engine contract takes."""
        if self.action_id == CLICK and self.click is not None:
            return {"action_id": CLICK, "x": self.click[0], "y": self.click[1]}
        return self.action_id

    @property
    def action_key(self) -> str:
        return "RESET" if self.action_id == RESET else f"ACTION{self.action_id}"


def list_games() -> list[str]:
    return sorted(p.name.split(".")[0] for p in BUNDLE_DIR.glob("*.json.gz"))


def load_bundle(game: str) -> dict:
    with gzip.open(BUNDLE_DIR / f"{game}.json.gz") as f:
        return json.load(f)


def final_engine_source(bundle: dict) -> str:
    lines: list[str] = []
    for e in bundle["engines"]:
        if "t" in e:
            lines = e["t"].split("\n")
        else:
            for start, del_count, new in sorted(e["o"], key=lambda o: -o[0]):
                lines[start : start + del_count] = new
    return "\n".join(lines)


def settled_steps(bundle: dict) -> list[Step]:
    grid: list[list[int]] | None = None
    steps: dict[int, Step] = {}
    for fr in bundle["frames"]:
        if "g" in fr:
            grid = [row[:] for row in fr["g"]]
        assert grid is not None
        for x, y, v in fr.get("d", []):
            grid[y][x] = v
        s = fr["s"]
        if s not in steps:
            steps[s] = Step(index=s, action_label=None, grid=grid)
        if fr.get("a"):
            steps[s].action_label = fr["a"]
        steps[s].grid = [row[:] for row in grid]
    return [steps[i] for i in sorted(steps)]


def parse_action(label: str) -> tuple[int, tuple[int, int] | None]:
    if label == "RESET":
        return RESET, None
    m = _CLICK_RE.fullmatch(label)
    if m:
        return CLICK, (int(m.group(1)), int(m.group(2)))
    return int(label.removeprefix("ACTION")), None


def completion_steps(bundle: dict) -> set[int]:
    """Step indices whose settled frame closed a level."""
    frame_step = [fr["s"] for fr in bundle["frames"]]
    return {frame_step[f] for f in bundle["level_steps"]}


def run_extractor(engine_src: str, grids: list[list[list[int]]], timeout_s: float = 600, fresh: bool = False) -> list:
    """Objects of each grid, in one process and in order, as the recording was extracted. With
    fresh, the engine module is reloaded before every grid so no extractor state carries over."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "engine.py").write_text(engine_src)
        (tmp_path / "run.py").write_text(_EXTRACT_RUNNER)
        (tmp_path / "grids.json").write_text(json.dumps(grids))
        proc = subprocess.run(
            [sys.executable, "run.py", "engine.py", "grids.json", "out.json"] + (["fresh"] if fresh else []),
            cwd=tmp, capture_output=True, text=True, timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"extractor failed: {proc.stderr[-500:]}")
        return json.loads((tmp_path / "out.json").read_text())


def build_buffer(game: str, use_cache: bool = True) -> list[Transition]:
    cache = CACHE_DIR / game / "buffer.json.gz"
    if use_cache and cache.exists():
        with gzip.open(cache) as f:
            rows = json.load(f)
        return [Transition(**{**r, "click": tuple(r["click"]) if r["click"] else None}) for r in rows]

    bundle = load_bundle(game)
    steps = settled_steps(bundle)
    objs = run_extractor(final_engine_source(bundle), [s.grid for s in steps])
    closers = completion_steps(bundle)

    transitions: list[Transition] = []
    level = 1
    for i in range(1, len(steps)):
        label = steps[i].action_label
        if label is None:
            continue
        action_id, click = parse_action(label)
        before, after = objs[i - 1], objs[i]
        if isinstance(before, dict) or isinstance(after, dict):
            continue
        transitions.append(Transition(
            step=i, level=level, action_id=action_id, click=click,
            level_advance=i in closers,
            before_grid=steps[i - 1].grid, after_grid=steps[i].grid,
            before_objs=before, after_objs=after,
        ))
        if i in closers:
            level += 1

    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt") as f:
        json.dump([asdict(t) for t in transitions], f)
    return transitions


def temporal_split(transitions: list[Transition], level: int, train_frac: float = 0.6,
                   test_level: int | None = None, train_n: int | None = None,
                   test_n: int | None = None, terrain: bool = False) -> tuple[list[Transition], list[Transition]]:
    """Train on the first fraction of a level, or on its first train_n transitions when given.
    Test on the rest of that level (capped at test_n when given), or on all of another level
    when test_level is given. RESET and level-closing transitions are dropped from both sides:
    their outcome is a new layout, not a function of the state, so a program can only pass
    them by storing the layout. With terrain, the static terrain of the train frames is appended
    to every state as constant objects (committee.terrain)."""
    def modellable(ts: list[Transition]) -> list[Transition]:
        return [t for t in ts if t.action_id != RESET and not t.level_advance]

    in_level = modellable([t for t in transitions if t.level == level])
    n_train = min(len(in_level), train_n) if train_n else max(1, round(len(in_level) * train_frac))
    train = in_level[:n_train]
    if test_level is None:
        test = in_level[n_train:]
    else:
        test = modellable([t for t in transitions if t.level == test_level])
    if test_n:
        test = test[:test_n]
    if terrain:
        from .terrain import add_terrain, static_terrain

        objs = static_terrain(train)
        train, test = add_terrain(train, objs), add_terrain(test, objs)
    return train, test


def summary(transitions: list[Transition]) -> dict:
    by_level: dict[int, int] = {}
    by_action: dict[str, int] = {}
    for t in transitions:
        by_level[t.level] = by_level.get(t.level, 0) + 1
        by_action[t.action_key] = by_action.get(t.action_key, 0) + 1
    return {
        "n_transitions": len(transitions),
        "by_level": dict(sorted(by_level.items())),
        "by_action": dict(sorted(by_action.items())),
        "n_level_advance": sum(t.level_advance for t in transitions),
        "types": sorted({str(o.get("type")) for t in transitions for o in t.before_objs}),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build and summarise a game's transition buffer.")
    parser.add_argument("game")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    transitions = build_buffer(args.game, use_cache=not args.no_cache)
    print(json.dumps(summary(transitions), indent=1))


if __name__ == "__main__":
    main()
