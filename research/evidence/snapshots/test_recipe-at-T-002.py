"""Contract tests for the team_segal recipe substrate (CPU, synthetic data)."""

from pathlib import Path

import pytest
import torch

from benchmark.api import BuildContext
from benchmark.config import RunConfig
from benchmark.data import synthetic_split
from benchmark.harness import run_submission
from benchmark.worker import load_submission, seed_everything

SUBMISSION = Path(__file__).resolve().parents[2] / "submissions" / "team_segal"
SMALL = {"batch_size": 16, "epochs": 2}
VARIANTS = {
    "airbench": SMALL,
    "airbench96-shape": SMALL
    | {"widths": [32, 64, 64], "block_depth": 3, "translate": 4, "cutout": 4},
    "resnet9": SMALL | {"arch": "resnet9", "width_mult": 0.25, "flip": "random"},
}


@pytest.fixture(scope="module")
def module():
    torch.set_num_threads(2)
    return load_submission(SUBMISSION)


def trained_state(module, state, data, seed):
    seed_everything(seed)
    module.prepare(state, data, seed)
    model = module.train(state)
    return {key: value.clone() for key, value in model.state_dict().items()}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_repeat_seed_reproduces_after_an_intervening_trial(module, name):
    state = module.build(BuildContext(torch.device("cpu"), VARIANTS[name]))
    data = synthetic_split(train=True)
    original = data.images.clone()
    first = trained_state(module, state, data, 42)
    different = trained_state(module, state, data, 43)
    repeated = trained_state(module, state, data, 42)
    assert any(not torch.equal(first[k], different[k]) for k in first)
    assert all(torch.equal(first[k], repeated[k]) for k in first)
    assert torch.equal(data.images, original)


def test_unknown_or_invalid_parameters_fail_loudly(module):
    with pytest.raises(ValueError, match="Unknown recipe parameters"):
        module.build(BuildContext(torch.device("cpu"), {"bogus": 1}))
    with pytest.raises(ValueError, match="widths"):
        module.build(BuildContext(torch.device("cpu"), {"widths": [64, 256]}))
    with pytest.raises(ValueError, match="batch_size"):
        state = module.build(BuildContext(torch.device("cpu"), {"batch_size": 128}))
        module.prepare(state, synthetic_split(train=True), 0)


def test_defaults_are_the_documented_airbench94_reference(module):
    assert callable(module.build)  # the fixture loads the import bridge used below
    from benchmark._submission.config import RecipeConfig

    config = RecipeConfig()
    assert config.arch == "airbench" and config.block_depth == 2
    assert config.scaled_widths == (64, 256, 256)
    assert (config.epochs, config.batch_size, config.label_smoothing) == (10.0, 1024, 0.2)
    assert RecipeConfig(width_mult=1.5).scaled_widths == (96, 384, 384)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_official_harness_smoke(tmp_path, name):
    config = RunConfig(device="cpu", synthetic=True, n_trials=2, accuracy_target=None)
    _, summary = run_submission(
        SUBMISSION,
        data_root=tmp_path / "unused",
        results_root=tmp_path / "results",
        seeds=[1, 2],
        parameters=VARIANTS[name],
        config=config,
    )
    assert summary["complete"] is True, summary["run_error"]
