"""Exactness of the systems levers that must not change the computation (CPU, fp32)."""

import importlib

import pytest
import torch
from test_recipe import SMALL, SUBMISSION, trained_state

from benchmark.api import BuildContext
from benchmark.data import synthetic_split
from benchmark.worker import load_submission

BASE = SMALL | {"block_depth": 3, "translate": 2, "global_pool": "flatmax"}


@pytest.fixture(scope="module")
def module():
    torch.set_num_threads(2)
    return load_submission(SUBMISSION)


def _sibling(module, name):
    return importlib.import_module(f"{module.__package__}.{name}")


def _trained(module, parameters, seed=5):
    state = module.build(BuildContext(torch.device("cpu"), parameters))
    return trained_state(module, state, synthetic_split(train=True), seed)


@pytest.mark.parametrize(
    "lever", [{"whiten_grad_off": True}, {"compile_loss": True}, {"master_fp32": True}]
)
def test_lever_leaves_training_bit_identical(module, lever):
    control = _trained(module, BASE)
    candidate = _trained(module, BASE | lever)
    assert all(torch.equal(control[k], candidate[k]) for k in control)


@pytest.mark.parametrize("size", [32, 20])
def test_skip_discarded_matches_full_convolution(module, size):
    model_module, config_module = _sibling(module, "model"), _sibling(module, "config")
    torch.manual_seed(0)
    control = model_module.make_model(
        config_module.RecipeConfig.from_parameters(BASE), torch.device("cpu")
    )
    candidate = model_module.make_model(
        config_module.RecipeConfig.from_parameters(BASE | {"skip_discarded": True}),
        torch.device("cpu"),
    )
    candidate.load_state_dict(control.state_dict())
    x = torch.rand(4, 3, size, size)
    for training in (True, False):
        control.train(training)
        candidate.train(training)
        torch.testing.assert_close(candidate(x), control(x), rtol=1e-4, atol=1e-4)


def test_master_weights_keep_fp32_precision(module):
    train_module = _sibling(module, "train")
    layer = torch.nn.Linear(4, 4, bias=False).half()
    masters = train_module.MasterWeights(layer)
    master = masters.of(layer.weight)
    assert master.dtype == torch.float32 and master is not layer.weight
    layer.weight.grad = torch.full_like(layer.weight, 1e-3)
    masters.load_grads()
    assert layer.weight.grad is None and master.grad.dtype == torch.float32
    before = master.detach().clone()
    with torch.no_grad():
        master.sub_(1e-6)  # below fp16 resolution: survives in the master only
    masters.sync()
    assert not torch.equal(master.detach(), before)
    assert torch.equal(layer.weight, master.detach().half())


def test_narrowing_to_full_width_is_bit_identical(module):
    parameters = BASE | {"widths": [32, 64, 64], "res_schedule": [[0.0, 24], [0.5, 32]]}
    control = _trained(module, parameters)
    for saliency in ("taylor", "random"):
        lever = {"switch_widths": [32, 64, 64], "narrow_saliency": saliency}
        candidate = _trained(module, parameters | lever)
        assert all(torch.equal(control[k], candidate[k]) for k in control)


def test_narrowing_keeps_the_selected_channels(module):
    narrowing = _sibling(module, "narrowing")
    keep = [torch.tensor([0, 2]), torch.tensor([1]), torch.tensor([0, 3])]
    conv = torch.arange(4 * 3 * 1 * 1).view(4, 3, 1, 1)
    assert narrowing.sliced("groups.1.conv1.weight", conv, keep).shape == (1, 2, 1, 1)
    assert narrowing.sliced("groups.2.conv2.weight", torch.zeros(4, 4, 3, 3), keep).shape == (
        2,
        2,
        3,
        3,
    )
    assert narrowing.sliced("head.weight", torch.zeros(100, 4), keep).shape == (100, 2)
    assert narrowing.sliced("whiten.weight", torch.zeros(24, 3, 2, 2), keep).shape == (24, 3, 2, 2)


def test_freezing_a_cooled_stage_is_bit_identical(module):
    cooled = BASE | {"res_schedule": [[0.0, 24], [0.5, 32]], "stage1_cooldown": [0.3, 0.6]}
    control = _trained(module, cooled)
    candidate = _trained(module, cooled | {"freeze_schedule": [[0.6, 1]]})
    assert all(torch.equal(control[k], candidate[k]) for k in control)


def test_freezing_two_cooled_stages_is_bit_identical(module):
    cooled = BASE | {
        "res_schedule": [[0.0, 24], [0.5, 32]],
        "stage1_cooldown": [0.3, 0.5],
        "stage2_cooldown": [0.4, 0.7],
        # The freeze also takes the whitening conv out of autograd, so (as in the recipe, which
        # freezes the whitening bias at 3 epochs) it must already be frozen.
        "whiten_bias_epochs": 0.5,
    }
    control = _trained(module, cooled)
    candidate = _trained(module, cooled | {"freeze_schedule": [[0.5, 1], [0.7, 2]]})
    assert all(torch.equal(control[k], candidate[k]) for k in control)
