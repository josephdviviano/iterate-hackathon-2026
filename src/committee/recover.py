"""Recover Devin sessions whose local runner lost them.

A session keeps running on Devin after the runner loses the network. This lists
the finished committee sessions that no stored run references, matches each one
to a condition and a run index (by the seed text in its prompt, and by replay on
the candidate train splits when the seed does not decide), and writes the run as
the runner would have.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .env import engine_source, mode_name
from .experiment import ARTIFACTS, check_mode, condition_dir, describe, pack_run, write_run
from .loader import Transition, build_buffer, temporal_split
from .seeds import make_seeds
from .synth_devin import API, _headers, devin_result
from .verify import run_program

HEADINGS = re.compile(r"\n# (?:Workspace|transitions\.md|check\.py|buffer\.json|Finish)\n")


@dataclass
class Target:
    game: str
    level: int
    condition: str
    mode: str
    train: list[Transition]
    test: list[Transition]
    base: Path
    seeds: list[str | None]
    missing: list[int]
    engine_src: str | None = None
    label: str = field(init=False)

    def __post_init__(self) -> None:
        self.label = f"{self.game} L{self.level} {self.condition}"


def seed_of(prompt: str) -> str | None:
    """The seed hypothesis a session was given, read back from its prompt."""
    _, sep, rest = prompt.partition("\n# Hypothesis to build on\n\n")
    return HEADINGS.split(rest, 1)[0].strip() if sep else None


def fit(target: Target, seed: str | None) -> int | None:
    """The lowest missing run of the target that this seed belongs to, or None."""
    for i in target.missing:
        want = target.seeds[i]
        if (want is None) == (seed is None) and (want is None or want.strip() == seed.strip()):
            return i
    return None


def assign(orphans: list[dict], targets: list[Target], replay=None) -> list[tuple[dict, Target, int]]:
    """Give each orphan (oldest first) a run. When more than one target takes its seed, the one whose
    train split the program replays best wins; a tie leaves the orphan unassigned."""
    out = []
    for o in orphans:
        fits = [(t, i) for t in targets for i in [fit(t, o["seed"])] if i is not None]
        if len(fits) > 1 and replay is not None:
            scored = sorted(((replay(o["source"], t), t, i) for t, i in fits), key=lambda x: -x[0])
            fits = [scored[0][1:]] if scored[0][0] > scored[1][0] else []
        if not fits:
            o["unassigned"] = True
            continue
        t, i = fits[0]
        t.missing.remove(i)
        out.append((o, t, i))
    return out


def targets_from(specs: list[str], train_frac: float, mode: str) -> list[Target]:
    """Specs are GAME:LEVEL:CONDITION:RUNS, with a fifth field `seeded` for a seeded batch."""
    out = []
    for spec in specs:
        game, level, condition, runs, *rest = spec.split(":")
        level, runs = int(level), int(runs)
        train, test = temporal_split(build_buffer(game), level, train_frac)
        base = condition_dir(game, level, train_frac, condition)
        check_mode(base, mode)
        seeds = list(make_seeds(train, runs)) if rest == ["seeded"] else [None] * runs
        missing = [i for i in range(runs) if not (base / f"run{i}" / "meta.json").exists()]
        out.append(Target(game, level, condition, mode, train, test, base, seeds, missing, engine_source(game, mode)))
    return out


def stored_session_ids() -> set[str]:
    ids = set()
    for p in ARTIFACTS.rglob("run*/meta.json"):
        try:
            ids.add(json.loads(p.read_text()).get("synth", {}).get("session_id"))
        except (ValueError, AttributeError):
            continue
    return ids - {None}


def orphan_sessions(client, since: datetime, limit: int = 100) -> list[dict]:
    """Finished committee sessions created after `since` that no stored run references, oldest first."""
    stored = stored_session_ids()
    r = client.get(f"{API}/sessions", headers=_headers(), params={"limit": limit})
    r.raise_for_status()
    body = r.json()
    sessions = body.get("sessions", body) if isinstance(body, dict) else body
    out = []
    for s in sessions:
        created = datetime.fromisoformat(s["created_at"].replace("Z", "+00:00"))
        if created < since or "committee" not in (s.get("tags") or []) or s["session_id"] in stored:
            continue
        g = client.get(f"{API}/session/{s['session_id']}", headers=_headers())
        g.raise_for_status()
        full = g.json()
        status = full.get("status_enum") or full.get("status")
        structured = full.get("structured_output") or {}
        if status not in ("finished", "expired") and not structured.get("program"):
            print(f"{s['session_id']}: {status}, not collected yet", flush=True)
            continue
        first = next((m for m in full.get("messages") or [] if m.get("type") == "initial_user_message"), {})
        updated = datetime.fromisoformat(full["updated_at"].replace("Z", "+00:00"))
        out.append({"session_id": s["session_id"], "created": created, "status": status, "out": structured,
                    "source": structured.get("program") or "", "seed": seed_of(first.get("message") or ""),
                    "wall_s": round((updated - created).total_seconds(), 1)})
    return sorted(out, key=lambda o: o["created"])


def recover(specs: list[str], train_frac: float, mode: str, since: datetime, dry_run: bool = False) -> list[dict]:
    import httpx

    client = httpx.Client(timeout=60)
    targets = targets_from(specs, train_frac, mode)
    orphans = orphan_sessions(client, since)
    print(f"{len(orphans)} orphan sessions; missing runs: "
          + ", ".join(f"{t.label} {t.missing}" for t in targets), flush=True)
    cache: dict[tuple[str, str], float] = {}

    def replay(source: str, t: Target) -> float:
        key = (source, t.label)
        if key not in cache:
            v = run_program(source, t.train, [], mode=t.mode, engine_src=t.engine_src)
            cache[key] = sum(v.train_pass) / max(1, len(v.train_pass))
        return cache[key]

    metas = []
    for o, t, i in assign(orphans, targets, replay):
        if dry_run:
            print(f"{t.label} run{i} <- {o['session_id']} ({o['status']}, seed={'yes' if o['seed'] else 'no'})")
            continue
        meta = {"backend": "devin", "mode": mode, "session_id": o["session_id"],
                "session_url": f"https://app.devin.ai/sessions/{o['session_id'].removeprefix('devin-')}",
                "status": o["status"], "wall_s": o["wall_s"], "nudges": None, "recovered": True}
        result = devin_result(o["out"], meta, t.train, t.seeds[i], mode)
        verdict = run_program(result.source, t.train, t.test, mode=mode, engine_src=t.engine_src)
        metas.append(write_run(t.base / f"run{i}", pack_run(result, verdict)))
        print(describe(metas[-1], f"{t.label} run{i}"), flush=True)
    for o in orphans:
        if o.get("unassigned"):
            print(f"{o['session_id']}: no run takes it (seed={'yes' if o['seed'] else 'no'})")
    return metas


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write the runs of Devin sessions the runner lost.")
    parser.add_argument("specs", nargs="+", help="GAME:LEVEL:CONDITION:RUNS[:seeded]")
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--frame", action="store_true")
    parser.add_argument("--frame-out", action="store_true")
    parser.add_argument("--since", default=None, help="ISO time; default is 12 hours ago")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    since = (datetime.fromisoformat(args.since).astimezone(timezone.utc) if args.since
             else datetime.now(timezone.utc) - timedelta(hours=12))
    if "DEVIN_API_KEY" not in os.environ:
        raise SystemExit("DEVIN_API_KEY is not set")
    recover(args.specs, args.train_frac, mode_name(args.frame, args.frame_out), since, args.dry_run)


if __name__ == "__main__":
    main()
