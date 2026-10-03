"""Contract tests for the converged submission in submissions/team_segal (CPU, synthetic)."""

from pathlib import Path

import pytest
import torch

from benchmark.api import BuildContext
from benchmark.config import RunConfig
from benchmark.data import synthetic_split
from benchmark.harness import run_submission
from benchmark.worker import load_submission, seed_everything

REPO = Path(__file__).resolve().parents[2]
SUBMISSION = REPO / "submissions" / "team_segal"
LAB = REPO / "research" / "lab_recipe"
SMALL = {"widths": [16, 32, 48], "batch_size": 16, "epochs": 2.25}
# The lab substrate's parameters that reproduce the submission's converged defaults.
LAB_EQUIVALENT = SMALL | {
    "block_depth": 3,
    "translate": 2,
    "res_schedule": [[0.0, 20], [0.5, 32]],
    "scaling_factor": 1 / 6,
    "global_pool": "flatmax",
    "whiten_grad_off": True,
}


def train_states(directory: Path, parameters: dict, seeds: list[int]) -> list[dict]:
    torch.set_num_threads(2)
    module = load_submission(directory)
    state = module.build(BuildContext(torch.device("cpu"), parameters))
    data = synthetic_split(train=True)
    results = []
    for seed in seeds:
        seed_everything(seed)
        module.prepare(state, data, seed)
        model = module.train(state)
        results.append({k: v.clone() for k, v in model.state_dict().items()})
    return results


def test_defaults_are_the_converged_recipe():
    load_submission(SUBMISSION)
    from benchmark._submission.config import RecipeConfig

    config = RecipeConfig()
    assert config.widths == (128, 384, 640) and config.epochs == 8.25
    assert config.res_schedule == ((0.0, 20), (0.5, 32)) and config.translate == 2
    assert config.scaling_factor == pytest.approx(1 / 6)
    assert config.compile and config.fused_sgd and config.compile_mode == "max-autotune"


def test_training_is_bit_identical_to_the_lab_substrate():
    seeds = [3, 4, 3]
    simplified = train_states(SUBMISSION, SMALL, seeds)
    lab = train_states(LAB, LAB_EQUIVALENT, seeds)
    for ours, theirs in zip(simplified, lab, strict=True):
        assert ours.keys() == theirs.keys()
        assert all(torch.equal(ours[k], theirs[k]) for k in ours)
    assert all(torch.equal(simplified[0][k], simplified[2][k]) for k in simplified[0])
    assert any(not torch.equal(simplified[0][k], simplified[1][k]) for k in simplified[0])


def test_unknown_parameters_fail_loudly():
    module = load_submission(SUBMISSION)
    bad_parameters = [
        {"arch": "resnet9"},
        {"widths": [64, 256]},
        {"res_schedule": [[0.5, 16], [0.0, 32]]},
    ]
    for bad in bad_parameters:
        with pytest.raises(ValueError):
            module.build(BuildContext(torch.device("cpu"), bad))


def test_official_harness_smoke(tmp_path):
    config = RunConfig(device="cpu", synthetic=True, n_trials=2, accuracy_target=None)
    _, summary = run_submission(
        SUBMISSION,
        data_root=tmp_path / "unused",
        results_root=tmp_path / "results",
        seeds=[1, 2],
        parameters=SMALL,
        config=config,
    )
    assert summary["complete"] is True, summary["run_error"]
