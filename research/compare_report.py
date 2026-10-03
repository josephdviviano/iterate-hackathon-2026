"""Report a same-host paired comparison of submissions run by ``modal_a100.py::compare``.

    python research/compare_report.py research/comparisons/<name>.toml

The first arm is the reference. Time differences are paired within host-block (same container,
alternating arm order), then averaged over host-blocks, overall and per GPU variant.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def trials(run_dir: Path) -> list[dict]:
    files = sorted(run_dir.glob("harness/*/*/trials.jsonl"))
    if not files:
        return []
    lines = files[-1].read_text().splitlines()
    return [t for t in map(json.loads, lines) if t.get("status") == "ok"]


def mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.nan
    return statistics.mean(values), se


def main(path: str) -> None:
    config = tomllib.loads(Path(path).read_text())
    root = REPO / "results" / "comparisons" / config["name"]
    rows = json.loads((root / "comparison.json").read_text())
    labels = [arm["label"] for arm in config["arms"]]
    reference = labels[0]
    gpu_of = {row["host"]: row["gpu"].split(",")[0] for row in rows}
    cells = {}
    for row in rows:
        key = (row["host"], row["block"], row["label"])
        cells[key] = trials(root / f"h{row['host']}-b{row['block']}-{row['label']}")
    blocks = sorted({(h, b) for h, b, _ in cells})
    print(f"{config['name']}: reference {reference}; host GPUs {sorted(set(gpu_of.values()))}")
    print("| arm | trials | acc % (SE) | time s | d_time % vs ref (SE) | d_time % PCIe only |")
    print("|---|---|---|---|---|---|")
    for label in labels:
        pooled = [t for (h, b, lab), ts in cells.items() if lab == label for t in ts]
        if not pooled:
            print(f"| {label} | 0 | failed | | | |")
            continue
        acc, acc_se = mean_se([100 * t["accuracy"] for t in pooled])
        time = statistics.mean(t["total_timed_time"] for t in pooled)
        diffs: dict[str, list[float]] = defaultdict(list)
        for h, b in blocks:
            arm, ref = cells.get((h, b, label)), cells.get((h, b, reference))
            if arm and ref:
                ratio = statistics.mean(t["total_timed_time"] for t in arm) / statistics.mean(
                    t["total_timed_time"] for t in ref
                )
                diffs["all"].append(100 * (ratio - 1))
                if "PCIe" in gpu_of[h]:
                    diffs["pcie"].append(100 * (ratio - 1))
        d, d_se = mean_se(diffs["all"])
        p, p_se = mean_se(diffs["pcie"])
        print(
            f"| {label} | {len(pooled)} | {acc:.2f} ({acc_se:.2f}) | {time:.3f} | "
            f"{d:+.2f} ({d_se:.2f}) | {p:+.2f} ({p_se:.2f}) n={len(diffs['pcie'])} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
