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
    "coverage": SMALL
    | {
        "activation": "silu",
        "whiten_kernel": 3,
        "head_pool": "maxmean",
        "brightness": 0.14,
        "contrast": 0.13,
        "mixup_alpha": 0.4,
        "lookahead_flush": True,
        "bn_recal_batches": 2,
    },
    "hiverge": SMALL
    | {
        "optimizer": "muon",
        "head_norm": True,
        "lookahead": False,
        "muon_coefficients": "hiverge",
        "muon_renorm": "hiverge",
        "muon_decoupled_wd": 1.0418e-6,
        "brightness": 0.14,
        "contrast": 0.13,
        "jitter_mode": "hiverge",
        "global_pool": "flatmax",
        "res_schedule": [[0.0, 24], [0.5, 32]],
        "resize_in_model": True,
    },
    "representation": SMALL
    | {
        "rep_branch": True,
        "residual_start": 0.3,
        "residual_ramp": 0.2,
        "loss": "poly1",
        "coarse_aux_weight": 0.5,
    },
    "squentropy": SMALL | {"loss": "squentropy", "block_depth": 3},
    "convmixer": SMALL
    | {"arch": "convmixer", "convmixer_dim": 32, "convmixer_depth": 2, "translate": 2},
    "celu": SMALL | {"activation": "celu", "epochs": 2.3},
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
        {"activation": "relu6"},
        {"whiten_kernel": 4},
        {"mixup_until": 1.5},
        {"loss": "focal"},
        {"residual_start": 1.0},
    ]
    for bad in bad_parameters:
        with pytest.raises(ValueError):
            module.build(BuildContext(torch.device("cpu"), bad))
