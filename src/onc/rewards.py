"""Per-episode reward terms for the ONC committee agent.

All functions are pure. ``R = r_task + lam_cal * r_cal + lam_dis * r_dis``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from onc_agi.core.schema import AnswerKey, WorldScore
from onc_agi.services.scoring import score_world

PI_NULL = 0.2
PI_SIGNAL = 0.8


def r_task(score: WorldScore) -> float:
    """The benchmark score of one world, written as a reward.

    A listed leak already sets ``find_signed`` to the empty-list value.
    """
    if score.is_null:
        return float(score.restrained) * score.efficiency / PI_NULL
    return score.find_signed * score.efficiency - float(score.abstained) / PI_SIGNAL


def feature_credit(ranking: Sequence[str], key: AnswerKey) -> dict[str, bool]:
    """For each listed feature: True if the raw recovery drops when the feature is removed.

    The real scorer is used, so equivalence sets, cluster deduplication, neutral
    removal and joint rules apply. On a null world every feature is False.
    """
    listed = list(ranking)
    if key.is_null:
        return dict.fromkeys(listed, False)
    full = score_world(listed, key, chance=(0.0, 0.0)).raw_recovery
    credit: dict[str, bool] = {}
    for feature in listed:
        rest = [f for f in listed if f != feature]
        credit[feature] = score_world(rest, key, chance=(0.0, 0.0)).raw_recovery < full
    return credit


def brier(p: float, outcome: bool | int) -> float:
    return (p - float(outcome)) ** 2


def _mean_brier(pairs: Sequence[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    return sum(brier(p, y) for p, y in pairs) / len(pairs)


def r_cal(
    p_signal: float,
    is_signal: bool,
    p_driver: Mapping[str, float],
    credit: Mapping[str, bool],
    held_out: Sequence[tuple[float, int]] = (),
) -> float:
    """Negative Brier score over the signal claim, the listed drivers and held-out patients.

    ``held_out`` holds (predicted probability, outcome) pairs for patients that were
    predicted before their recruit batch was revealed. A term with no entries adds 0.
    """
    drivers = [(p_driver[f], float(earned)) for f, earned in credit.items()]
    return -brier(p_signal, is_signal) - _mean_brier(drivers) - _mean_brier(list(held_out))


def r_dis(u_prev: float, u_now: float) -> float:
    """Potential-based shaping: the drop in committee disagreement over one step."""
    return u_prev - u_now


def shaping_total(us: Sequence[float], resynthesis_steps: Collection[int] = ()) -> float:
    """Sum of ``r_dis`` over disagreement values ``u_0 .. u_T``.

    Step ``t`` gives ``u_t``. A resynthesis step earns 0 and the next step is
    measured from ``u_t``. Without resynthesis the sum telescopes to ``u_0 - u_T``.
    """
    skip = set(resynthesis_steps)
    return sum(r_dis(us[t - 1], us[t]) for t in range(1, len(us)) if t not in skip)


def combine(r_task: float, r_cal: float, r_dis: float, lam_cal: float = 0.5, lam_dis: float = 0.25) -> float:
    return r_task + lam_cal * r_cal + lam_dis * r_dis


def ece(probs: Sequence[float], outcomes: Sequence[bool | int], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must have the same length")
    if not probs:
        return 0.0
    sums = [[0.0, 0.0, 0] for _ in range(bins)]
    for p, y in zip(probs, outcomes, strict=True):
        b = min(int(p * bins), bins - 1)
        sums[b][0] += p
        sums[b][1] += float(y)
        sums[b][2] += 1
    return sum(abs(ps - ys) for ps, ys, n in sums if n) / len(probs)
