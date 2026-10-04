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
    "structure": SMALL
    | {
        "snapshot_fracs": [0.6, 0.8],
        "exit_weight": 0.3,
        "exit_eval_weight": 0.5,
        "head_refit_lambda": 1e-3,
        "refit_samples": 64,
        "lr_shape": "wsd",
        "lookahead_every": 3,
        "lookahead_power": 2.0,
    },
    "ensemble": SMALL | {"members": 2, "widths": [16, 32, 48], "lr_shape": "cosine"},
    "soft-targets": SMALL
    | {"ols_alpha": 0.5, "pskd_alpha": 0.5, "ls_end": 0.0, "head_expand": 64, "block_depth": 3},
    "head": SMALL
    | {
        "pool_overlap": True,
        "stage3_ceil": True,
        "cosine_head_scale": 16.0,
        "cosine_subcenters": 2,
        "head_mean_init": True,
        "head_init_samples": 64,
    },
    "dynamics": SMALL
    | {
        "res_schedule": [[0.0, 24], [0.5, 32]],
        "master_fp32": True,
        "clean_tail_epochs": 1,
        "head_lr_mult": 2.0,
        "head_wd_mult": 0.5,
        "conv_wd_mult": 0.5,
        "init_gain": 0.6,
        "switch_momentum_scale": 0.5,
        "res_blend_steps": 4,
        "soft_pool_tau": 0.5,
    },
    "etf": SMALL | {"etf_head": True, "scaling_factor": 0.33},
    "systems": SMALL | {"skip_discarded": True, "whiten_grad_off": True, "compile_loss": True},
    "prune-easy": SMALL | {"prune_frac": 0.3, "prune_start": 1},
    "prune-soft": SMALL | {"prune_frac": 0.2, "prune_mode": "soft", "prune_score": "grad"},
    "prune-split": SMALL | {"prune_frac": 0.2, "prune_mode": "split"},
    "narrow": SMALL
    | {
        "widths": [32, 64, 64],
        "block_depth": 3,
        "res_schedule": [[0.0, 24], [0.5, 32]],
        "switch_widths": [24, 48, 32],
    },
    "narrow-norm": SMALL
    | {
        "widths": [32, 64, 64],
        "res_schedule": [[0.0, 24], [0.5, 32]],
        "switch_widths": [32, 40, 48],
        "narrow_saliency": "norm",
    },
    "budget": SMALL
    | {
        "res_schedule": [[0.0, 24], [0.5, 32]],
        "stage1_cooldown": [0.4, 0.75],
        "freeze_schedule": [[0.75, 1]],
        "bias_scaler_final": 32.0,
        "lr_shape": "wsd_sqrt",
        "order": "balanced",
    },
    "post-add": SMALL | {"block_depth": 3, "post_add_activation": True},
    "init-dct": SMALL | {"block_depth": 3, "conv_init": "dct", "residual_gate_init": 0.0},
    "init-zero": SMALL | {"widths": [24, 40, 64], "conv_init": "zero"},
    "init-orthogonal": SMALL | {"conv_init": "orthogonal", "residual_gate_init": 0.25},
    "round3": SMALL
    | {
        "block_depth": 3,
        "bias_wd_mult": 0.25,
        "square_init": "dirac+dct",
        "square_beta": 0.5,
        "head_center": True,
        "whiten_eps": 5e-3,
        "stage1_cooldown": [0.4, 0.7],
        "stage1_lr_mult": 1.3,
    },
    "thin-weightfreeze": SMALL
    | {
        "block_depth": 3,
        "res_schedule": [[0.0, 24], [0.5, 32]],
        "whiten_bias_epochs": 0.5,
        "thin_window": [0.4, 0.8],
        "thin_groups": 2,
        "weight_freeze": [[0, 0.3, 0.6, "all"], [2, 0.5, 0.8, "conv1"]],
    },
    "crop": SMALL | {"crop_schedule": [[0.3, 28], [0.7, 0]], "block_depth": 3},
    "pointwise-stage3": SMALL
    | {
        "block_depth": 3,
        "square_kernels": [3, 3, 1],
        "residual_kernels": [3, 3, 1],
        "stage3_ceil": True,
    },
    "centre-tap": SMALL
    | {"block_depth": 3, "res_schedule": [[0.0, 20], [0.5, 32]], "centre_tap_max": 2},
    "snoo-bnfreeze": SMALL
    | {
        "block_depth": 3,
        "lookahead_outer_momentum": 0.5,
        "bn_freeze_frac": 0.8,
        "stage1_cooldown": [0.4, 0.6],
        "freeze_schedule": [[0.6, 1]],
    },
    "round7": SMALL
    | {
        "block_depth": 3,
        "stage_depths": [3, 2, 3],
        "stage_skips": [False, True, False],
        "res_schedule": [[0.0, 24], [0.5, 32]],
        "ls_low": 0.2,
        "identity_scale": 0.577,
        "translate": 2,
        "translate_low": 3,
    },
    "batch-ramp": SMALL | {"batch_schedule": [[0, 8], [1, 16]], "block_depth": 3},
    "floor-base": SMALL | {"lr_decay_end": 0.9, "lookahead_base": 0.97},
    "final-blend-fp32": SMALL | {"lookahead_final_decay": 0.5, "eval_fp32": True},
    "square-kaiming": SMALL | {"square_init": "dirac+kaiming", "square_beta": 0.25},
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
        {"members": 0},
        {"snapshot_fracs": [1.0]},
        {"lr_shape": "step"},
        {"order": "sorted"},
        {"ols_alpha": -0.1},
        {"cosine_subcenters": 0},
        {"init_gain": 0.0},
        {"prune_frac": 1.0},
        {"stage1_cooldown": [0.8, 0.5]},
        {"conv_init": "gabor"},
        {"bias_scaler_final": 0.0},
        {"switch_widths": [128, 384, 700]},
        {"switch_widths": [64, 64, 64], "members": 2},
        {"prune_mode": "random"},
        {"clean_tail_epochs": -1},
        {"compile_loss": True, "members": 2},
    ]
    for bad in bad_parameters:
        with pytest.raises(ValueError):
            module.build(BuildContext(torch.device("cpu"), bad))
