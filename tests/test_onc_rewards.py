"""Tests for the ONC reward terms (handoff section 5)."""

from __future__ import annotations

import random

import pytest
from onc_agi.adapters.agents import CHEATERS, make_agent
from onc_agi.adapters.cli import SMOKE_AGENTS, fixture_store
from onc_agi.core.schema import Tier
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import evaluate

from onc.rewards import ece, feature_credit, r_cal, r_task, shaping_total


@pytest.fixture(scope="module")
def store() -> FileWorldStore:
    return FileWorldStore(fixture_store())


def test_shaping_telescopes() -> None:
    rng = random.Random(0)
    for _ in range(20):
        us = [rng.random() for _ in range(rng.randint(2, 12))]
        assert abs(shaping_total(us) - (us[0] - us[-1])) < 1e-12
        t = rng.randint(1, len(us) - 1)
        segments = shaping_total(us[:t]) + shaping_total(us[t:])
        assert abs(shaping_total(us, resynthesis_steps={t}) - segments) < 1e-12
    monotone = [1.0, 0.8, 0.6, 0.4, 0.2]
    farmed = [1.0, 0.1, 0.9, 0.0, 0.2]
    assert abs(shaping_total(farmed) - shaping_total(monotone)) < 1e-12


def test_r_cal_is_proper() -> None:
    grid = [i / 10 for i in range(11)]
    for truth in (0.2, 0.5, 0.8):

        def expected(q: float) -> float:
            return sum(
                w * r_cal(q, bool(y), {"f": q}, {"f": bool(y)}, [(q, y)])
                for y, w in ((1, truth), (0, 1 - truth))
            )

        assert max(grid, key=expected) == pytest.approx(truth)
    assert ece([0.2, 0.2, 0.2, 0.2, 0.2], [1, 0, 0, 0, 0]) == pytest.approx(0.0)
    assert ece([1.0, 1.0, 1.0, 1.0], [1, 0, 1, 0]) == pytest.approx(0.5)


def test_feature_credit_on_toy_worlds(store: FileWorldStore) -> None:
    def other(key, exclude: set[str]) -> str:
        return next(f for f in sorted(key.clusters) if f not in exclude and f not in key.reject_set)

    key = store.answer_key("toy-driver-full")
    assert feature_credit(["zhj2", other(key, {"zhj2"})], key) == {"zhj2": True, other(key, {"zhj2"}): False}
    leak = store.answer_key("toy-leak-full")
    assert feature_credit(["sjr2", "lxh7"], leak) == {"sjr2": True, "lxh7": False}
    joint = store.answer_key("toy-interaction-full")
    assert feature_credit(["ggf1", "qnx9"], joint) == {"ggf1": True, "qnx9": True}
    assert feature_credit(["ggf1"], joint) == {"ggf1": False}
    stand_in = store.answer_key("toy-stand-in-full")
    assert feature_credit(["prw7", "nhr3"], stand_in)["nhr3"] is False
    neutral = store.answer_key("toy-neutral-full")
    assert feature_credit(["njs1", "mrd3"], neutral) == {"njs1": False, "mrd3": True}
    null = store.answer_key("toy-null-a-full")
    assert not any(feature_credit(sorted(null.clusters)[:3], null).values())


def test_r_task_orders_smoke_agents_as_the_scorecard(store: FileWorldStore) -> None:
    """Mean per-episode r_task against discovery_score_unfloored over the 20 toy worlds.

    The reward is additive over episodes and the scorecard is a product, so they
    cannot agree for every agent set. On the smoke agents they agree on every
    strictly ordered pair (Kendall tau-b 0.886, no discordant pair). The scorecard
    ties random, giant_list, always_empty and leak_exploiter at 0 because one
    factor is 0; the reward separates them (0 > -0.057 > -0.058 > -0.165).
    """
    mean_r: dict[str, float] = {}
    unfloored: dict[str, float] = {}
    for name in SMOKE_AGENTS:
        card, results = evaluate(make_agent(name, store), store, Tier.PUBLIC_TRAIN, bootstrap_draws=50)
        mean_r[name] = sum(r_task(r.score) for r in results) / len(results)
        unfloored[name] = card.discovery_score_unfloored
    names = list(SMOKE_AGENTS)
    discordant = [
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if (mean_r[a] - mean_r[b]) * (unfloored[a] - unfloored[b]) < 0
    ]
    assert discordant == []
    honest = [n for n in names if n not in CHEATERS and n != "random"]
    floor = max(mean_r[n] for n in names if n in CHEATERS or n == "random")
    assert floor <= 0.0
    assert floor < min(mean_r[n] for n in honest)
    assert max(mean_r, key=mean_r.get) == "oracle"
