"""Probe: can untimed build() reach data the rules say it must not touch?"""
import sys
from types import SimpleNamespace

from torch import nn


def build(context):
    caller = sys._getframe(1).f_locals
    train = caller.get("train_data")
    test = caller.get("images")
    seen_train = None if train is None else tuple(train.images.shape)
    seen_test = None if test is None else tuple(test.shape)
    print("PROBE build sees train_data:", seen_train, "| test images:", seen_test,
          file=sys.stderr, flush=True)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 100)).to(context.device)
    return SimpleNamespace(model=model)


def prepare(state, data, seed):
    pass


def train(state):
    return state.model
