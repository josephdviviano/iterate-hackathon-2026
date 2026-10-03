"""Paired same-host timing report for an interleaved Modal sweep.

    python research/paired_timing.py research/sweeps/<sweep>.toml

Reads ``<results>/interleaved/h<host>-b<block>-<config_id>/`` (written by
``modal_a100.py::interleave``). The first config of the sweep is the control. For every
host-block, each arm's mean timed time is compared with the control's mean in that same
host-block; the report gives the mean relative difference over host-blocks with its standard
error, plus pooled accuracy, so host-to-host variance cancels.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import sweep as sweeplib


def trials(run_dir: Path) -> list[dict]:
    harness_dir = sweeplib.latest_harness_dir(run_dir)
    if harness_dir is None:
        return []
    lines = (harness_dir / "trials.jsonl").read_text().splitlines()
    return [t for t in map(json.loads, lines) if t.get("status") == "ok"]


def main(path: str) -> None:
    spec = sweeplib.Sweep.load(Path(path).resolve())
    ids = [spec.run_dir(p).name for p in spec.configs]
    cells: dict[tuple[str, str], list[dict]] = {}
    for run_dir in sorted((spec.root / "interleaved").glob("h*-b*-*")):
        host, block, config_id = run_dir.name.split("-", 2)
        cells[(f"{host}-{block}", config_id)] = trials(run_dir)
    blocks = sorted({b for b, _ in cells})
    control = ids[0]
    diffs: dict[str, list[float]] = defaultdict(list)
    for block in blocks:
        base = cells.get((block, control))
        if not base:
            continue
        base_time = statistics.mean(t["total_timed_time"] for t in base)
        for config_id in ids:
            arm = cells.get((block, config_id))
            if arm:
                arm_time = statistics.mean(t["total_timed_time"] for t in arm)
                diffs[config_id].append(100 * (arm_time / base_time - 1))
    print(f"{spec.name}: control {control}, host-blocks {blocks}")
    print("| config_id | params | n | time_s | d_time_% | se_% | acc |")
    print("|---|---|---|---|---|---|---|")
    base_params = spec.configs[0]
    for params, config_id in zip(spec.configs, ids, strict=True):
        pooled = [t for (_, c), ts in cells.items() if c == config_id for t in ts]
        if not pooled:
            continue
        delta = {k: v for k, v in params.items() if base_params.get(k) != v}
        d = diffs[config_id]
        se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        time = statistics.mean(t["total_timed_time"] for t in pooled)
        acc = statistics.mean(t["accuracy"] for t in pooled)
        mean_d = statistics.mean(d) if d else float("nan")
        print(
            f"| {config_id} | {json.dumps(delta)} | {len(pooled)} | {time:.3f} | "
            f"{mean_d:+.2f} | {se:.2f} | {acc:.4f} |"
        )
    gpus = {}
    for row in sweeplib.read_json(spec.root / "interleaved.json") or []:
        gpus[f"h{row['host']}"] = row["gpu"].split(",")[0] + "," + row["gpu"].split(",")[1]
    print("\nper host-block time change vs control (%):")
    for block in blocks:
        host = block.split("-")[0]
        changes = []
        for config_id in ids[1:]:
            arm, base = cells.get((block, config_id)), cells.get((block, control))
            if arm and base:
                ratio = statistics.mean(t["total_timed_time"] for t in arm) / statistics.mean(
                    t["total_timed_time"] for t in base
                )
                changes.append(f"{100 * (ratio - 1):+.2f}")
        print(f"  {block} [{gpus.get(host, '?')}]: {' '.join(changes)}")


if __name__ == "__main__":
    main(sys.argv[1])
