"""Conformal coverage, the policy parameterisation and the coverage-only dial."""

import numpy as np

from onc.agent import Policy
from onc.conformal import aci, binary_scores
from onc.hacks import train_dial
from onc.train import to_policy, to_theta


def test_aci_holds_the_target_and_still_commits_on_calibrated_streams():
    rng = np.random.default_rng(0)
    p = rng.random(3000)
    y = (rng.random(3000) < p).astype(int)
    result = aci(binary_scores(p), y, alpha=0.1)
    assert abs(result.coverage - 0.9) < 0.03 and 0.2 < result.committed < 0.9


def test_policy_parameters_round_trip_and_sharpness_keeps_the_half_point():
    policy = Policy(tau=0.4, stop_threshold=0.05, spend_cap=0.7, sharpness=3.0)
    back = to_policy(to_theta(policy), Policy())
    assert all(abs(getattr(back, k) - getattr(policy, k)) < 1e-9 for k in ("tau", "stop_threshold", "spend_cap", "sharpness"))
    assert policy.report(0.5) == 0.5 and policy.report(0.8) > 0.8 and Policy(sharpness=0.5).report(0.8) < 0.8


def test_coverage_only_reward_widens_the_set_until_nothing_is_committed():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    pairs = [(float(v), int(rng.random() < v)) for v in p]
    final = train_dial(pairs, iterations=40)[-1]
    assert final["coverage"] > 0.98 and final["committed"] < 0.05
