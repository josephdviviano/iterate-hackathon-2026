"""End-to-end tests for research/sweep.py on CPU with the harness's synthetic data."""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "research" / "sweep.py"

SPEC = """
name = "{name}"
submission = "submissions/team_segal"
python = "{python}"
seeds = [1, 2]
devices = []
slots_per_device = 2
harness_args = ["--device", "cpu", "--synthetic"]
results_root = "{root}"

[base]
batch_size = 16

[[grid]]
epochs = [1, 2]

[[configs]]
epochs = 1
batch_size = 128   # exceeds the 64 synthetic images, so prepare fails
"""


@pytest.fixture
def sweep_file(tmp_path, monkeypatch):
    monkeypatch.setenv("C100_SLOT_DIR", str(tmp_path / "slots"))
    root = tmp_path / "results"
    path = tmp_path / "tiny.toml"
    path.write_text(SPEC.format(name="tiny", python=sys.executable, root=root))
    return path, root / "tiny"


def sweep(*args):
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        [sys.executable, str(SWEEP), *map(str, args)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def table_without_times(root: Path) -> list[dict]:
    """Collated rows minus local wall time, which legitimately varies between reruns."""
    with (root / "table.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return [{k: v for k, v in row.items() if k != "mean_time_local"} for row in rows]


def launches(root: Path) -> list[str]:
    events = [json.loads(line) for line in (root / "sweep.log").read_text().splitlines()]
    return [e["config_id"] for e in events if e["event"] == "launch"]


def test_sweep_runs_records_failure_resumes_and_collates(sweep_file):
    path, root = sweep_file
    first = sweep("run", path)
    assert first.returncode == 0, first.stderr
    status = json.loads(sweep("status", path).stdout)
    assert status == {"sweep": "tiny", "configs": 3, "complete": 2, "failed": 1}

    table = sweep("collate", path)
    assert table.returncode == 0, table.stderr
    with (root / "table.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert sorted(r["status"] for r in rows) == ["complete", "complete", "failed"]
    failed = next(r for r in rows if r["status"] == "failed")
    assert "batch_size 128 exceeds" in failed["error"]
    assert all(r["n"] == "2" for r in rows if r["status"] == "complete")
    original = table_without_times(root)

    # Simulate an interrupted run: one completed config loses its record and results.
    victim = root / "runs" / next(r["config_id"] for r in rows if r["status"] == "complete")
    (victim / "record.json").unlink()
    shutil.rmtree(victim / "harness")
    before = launches(root)
    resumed = sweep("run", path)
    assert resumed.returncode == 0, resumed.stderr
    assert launches(root)[len(before) :] == [victim.name]

    sweep("collate", path)
    assert table_without_times(root) == original


def test_dry_run_lists_pending_and_rejects_bad_specs(sweep_file, tmp_path):
    path, _ = sweep_file
    dry = sweep("run", path, "--dry-run")
    assert dry.returncode == 0 and len(dry.stdout.splitlines()) == 3
    bad = tmp_path / "bad.toml"
    bad.write_text("seeds = [1, 1]\n[[configs]]\nepochs = 1\n")
    result = sweep("status", bad)
    assert result.returncode != 0 and "distinct integers" in result.stderr
