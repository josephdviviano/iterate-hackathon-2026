"""Sweeps run from a frozen source snapshot that live recipe edits do not touch."""

import json

import pytest
from test_sweep import REPO, sweep
from test_sweep import sweep_file as _sweep_file

sweep_file = pytest.fixture(_sweep_file.__wrapped__)


def test_sweep_runs_from_a_snapshot_that_live_edits_do_not_touch(sweep_file):
    path, root = sweep_file
    assert sweep("run", path).returncode == 0
    snapshot_status = json.loads(sweep("status", path).stdout)
    helper = REPO / "submissions" / "team_segal" / "_live_edit_probe.py"
    helper.write_text("# temporary live edit\n")
    try:
        assert json.loads(sweep("status", path).stdout) == snapshot_status
        assert not (root / "source" / helper.name).exists()
    finally:
        helper.unlink()
