"""Contract tests for the hill-climb levers (Muon mode, progressive resizing) on CPU."""

import pytest
import torch
from test_recipe import SMALL, SUBMISSION, trained_state

from benchmark.api import BuildContext
from benchmark.config import RunConfig
from benchmark.data import synthetic_split
from benchmark.harness import run_submission
from benchmark.worker import load_submission

LEVERS = {
    "muon": SMALL | {"optimizer": "muon", "head_norm": True, "lookahead": False},
    "resize": SMALL | {"res_schedule": [[0.0, 16], [0.5, 24], [0.75, 32]]},
    "stages": SMALL | {"stage_depths": [2, 3, 3], "pool_first": [False, True, True]},
    "amax-pool": SMALL | {"pool_impl": "amax", "block_depth": 3},
    "select": SMALL | {"select_fraction": 0.5, "selector_widths": [16, 32, 32]},
    "freeze": SMALL | {"freeze_schedule": [[0.5, 1], [0.75, 2]], "res_schedule": [[0.6, 24]]},
    "muon-resnet9": SMALL
    | {"arch": "resnet9", "width_mult": 0.25, "optimizer": "muon", "lookahead": False},
}


@pytest.mark.parametrize("name", sorted(LEVERS))
def test_lever_resets_and_runs_through_the_harness(tmp_path, name):
    torch.set_num_threads(2)
    module = load_submission(SUBMISSION)
    state = module.build(BuildContext(torch.device("cpu"), LEVERS[name]))
    data = synthetic_split(train=True)
    first = trained_state(module, state, data, 7)
    trained_state(module, state, data, 8)
    repeated = trained_state(module, state, data, 7)
    assert all(torch.equal(first[k], repeated[k]) for k in first)

    config = RunConfig(device="cpu", synthetic=True, n_trials=2, accuracy_target=None)
    _, summary = run_submission(
        SUBMISSION,
        data_root=tmp_path / "unused",
        results_root=tmp_path / "results",
        seeds=[1, 2],
        parameters=LEVERS[name],
        config=config,
    )
    assert summary["complete"] is True, summary["run_error"]


def test_invalid_lever_parameters_fail_loudly():
    module = load_submission(SUBMISSION)
    bad_parameters = [
        {"optimizer": "adam"},
        {"res_schedule": [[0.5, 16], [0.0, 32]]},
        {"res_schedule": [[0.0, 40]]},
        {"freeze_schedule": [[0.5, 3]]},
        {"compile_mode": "fast"},
        {"select_fraction": 0.0},
        {"stage_depths": [2, 4, 3]},
        {"pool_first": [True, False]},
        {"pool_impl": "avg"},
    ]
    for bad in bad_parameters:
        with pytest.raises(ValueError):
            module.build(BuildContext(torch.device("cpu"), bad))
