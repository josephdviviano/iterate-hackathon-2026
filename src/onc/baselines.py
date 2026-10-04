"""Evaluate every registered ONC-AGI agent on the toy worlds, one scorecard per mode.

Run ``uv run python -m onc.baselines [--store PATH] [--out artifacts/onc/baselines]``.
The command writes ``<out>.json`` (one record per agent and mode) and ``<out>.md``
(one table per mode, sorted by discovery score). Toy-world scores are not results.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from onc_agi.adapters.agents import BASELINES, CHEATERS, make_agent
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import Mode, Scorecard, Tier, WorldScore
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import evaluate

BOOTSTRAP_DRAWS = 200
NOT_INSTALLED = "not installed"

COLUMNS = (
    ("discovery_score", "DS"),
    ("discovery_score_unfloored", "unfloored"),
    ("interval", "95% CI"),
    ("find", "Find"),
    ("find_signed", "Find signed"),
    ("restraint", "Restraint"),
    ("strict", "Strict"),
    ("leak_rate", "Leak"),
    ("abstention_on_signal", "Abstain on signal"),
    ("restraint_on_null", "Restrain on null"),
    ("mean_data_cost", "Cost"),
    ("analysis_regret", "Analysis regret"),
    ("acquisition_gap", "Acq. gap"),
    ("seconds", "s"),
)


def agent_names(mode: Mode) -> list[str]:
    names = ["oracle", "random", *BASELINES, *CHEATERS]
    if mode is Mode.SEQUENTIAL:
        names += [f"seq_{b}" for b in BASELINES]
    return names


def world_record(score: WorldScore) -> dict[str, Any]:
    return {
        "world_id": score.world_id,
        "is_null": score.is_null,
        "find": score.find,
        "find_signed": score.find_signed,
        "abstained": score.abstained,
        "restrained": score.restrained,
        "leaked": score.leaked,
        "efficiency": score.efficiency,
        "spent": score.spent,
    }


def record(name: str, mode: Mode, card: Scorecard, seconds: float) -> dict[str, Any]:
    alignment = card.alignment
    return {
        "agent": name,
        "mode": mode.value,
        "status": "ok",
        "discovery_score": card.discovery_score,
        "discovery_score_unfloored": card.discovery_score_unfloored,
        "interval": [card.interval.low, card.interval.high],
        "find": card.find,
        "find_signed": card.find_signed,
        "restraint": card.restraint,
        "strict": card.strict_discovery_score,
        "leak_rate": card.leak_rate,
        "abstention_on_signal": card.abstention_on_signal,
        "restraint_on_null": card.restraint_on_null,
        "mean_data_cost": card.mean_data_cost,
        "alignment": None
        if alignment is None
        else {"analysis_regret": alignment.analysis_regret, "acquisition_gap": alignment.acquisition_gap},
        "seconds": seconds,
        "worlds": [world_record(s) for s in card.worlds],
    }


def run(store: FileWorldStore, bootstrap_draws: int = BOOTSTRAP_DRAWS, first: int | None = None) -> dict[str, Any]:
    ids = store.world_ids(Tier.PUBLIC_TRAIN)[:first] if first else store.world_ids(Tier.PUBLIC_TRAIN)
    by_mode: dict[Mode, list[str]] = {m: [] for m in Mode}
    for wid in ids:
        by_mode[store.world(wid).card.mode].append(wid)
    start = time.perf_counter()
    records: list[dict[str, Any]] = []
    for mode, worlds in by_mode.items():
        if not worlds:
            continue
        for name in agent_names(mode):
            t0 = time.perf_counter()
            try:
                card, _ = evaluate(
                    make_agent(name, store),
                    store,
                    Tier.PUBLIC_TRAIN,
                    world_ids=worlds,
                    bootstrap_draws=bootstrap_draws,
                )
            except (ImportError, ArenaError, RuntimeError) as exc:
                status = NOT_INSTALLED if isinstance(exc, ImportError) else "error"
                records.append({"agent": name, "mode": mode.value, "status": status, "error": str(exc)})
                print(f"{mode.value:11s} {name:24s} {status}: {exc}")
                continue
            rec = record(name, mode, card, time.perf_counter() - t0)
            records.append(rec)
            print(f"{mode.value:11s} {name:24s} DS {_cell(rec, 'discovery_score')}  {rec['seconds']:.1f}s")
    return {
        "store": os.path.relpath(store.root),
        "tier": Tier.PUBLIC_TRAIN.value,
        "bootstrap_draws": bootstrap_draws,
        "worlds": {m.value: w for m, w in by_mode.items() if w},
        "wall_time_s": time.perf_counter() - start,
        "records": records,
    }


def _cell(rec: dict[str, Any], column: str) -> str:
    if column == "interval":
        low, high = rec["interval"]
        return "n/a" if low is None else f"[{low:+.2f}, {high:+.2f}]"
    if column in ("analysis_regret", "acquisition_gap"):
        value = None if rec["alignment"] is None else rec["alignment"][column]
    else:
        value = rec[column]
    if value is None:
        return "n/a"
    if column == "mean_data_cost":
        return f"{value:.0f}"
    if column == "seconds":
        return f"{value:.1f}"
    return f"{value:.3f}"


def _sort_key(rec: dict[str, Any]) -> tuple[int, float, float]:
    if rec["status"] != "ok":
        return (1, 0.0, 0.0)
    return (0, -(rec["discovery_score"] or 0.0), -(rec["discovery_score_unfloored"] or 0.0))


def markdown(report: dict[str, Any]) -> str:
    lines = ["# ONC-AGI toy-world scorecards", ""]
    lines.append(
        f"Store `{report['store']}`, tier `{report['tier']}`, bootstrap draws {report['bootstrap_draws']}, "
        f"wall time {report['wall_time_s']:.1f} s. Toy-world scores are not results."
    )
    for mode, worlds in report["worlds"].items():
        store_worlds = [r for r in report["records"] if r["mode"] == mode and r["status"] == "ok"]
        n_null = sum(w["is_null"] for w in store_worlds[0]["worlds"]) if store_worlds else 0
        lines += ["", f"## {mode} ({len(worlds)} worlds, {len(worlds) - n_null} signal, {n_null} null)", ""]
        lines.append("| agent | " + " | ".join(label for _, label in COLUMNS) + " |")
        lines.append("|---|" + "---|" * len(COLUMNS))
        for rec in sorted((r for r in report["records"] if r["mode"] == mode), key=_sort_key):
            if rec["status"] != "ok":
                cells = [rec["status"] if rec["status"] == NOT_INSTALLED else f"error: {rec['error']}"]
                cells += ["" for _ in COLUMNS[1:]]
            else:
                cells = [_cell(rec, column) for column, _ in COLUMNS]
            lines.append(f"| {rec['agent']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", help="world store directory (default: the shipped toy worlds)")
    parser.add_argument("--out", default="artifacts/onc/baselines", help="output path without extension")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--first", type=int, help="the first N worlds in store order (the benchmark's first-pass set)")
    args = parser.parse_args(argv)
    store = FileWorldStore(Path(args.store) if args.store else fixture_store())
    report = run(store, bootstrap_draws=args.bootstrap_draws, first=args.first)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    out.with_suffix(".md").write_text(markdown(report))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')} in {report['wall_time_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
