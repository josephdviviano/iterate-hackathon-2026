"""Core rules of the ONC committee: leak filter, null membership, disagreement, submission."""

import numpy as np
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.schema import Assay, Timing
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.engine import Episode
from onc_agi.services.kit import run_episode

from onc.agent import CommitteeAgent, Policy
from onc.committee import Committee, build_committee
from onc.hypotheses import Design, Hypothesis, features_from

STORE = FileWorldStore(fixture_store())


def _world(world_id: str):
    world = STORE.world(world_id)
    card = world.card
    return card, features_from(card, card.feature_ids(), world.x, world.y, world.stratum)


def test_leak_filter_removes_post_outcome_before_any_hypothesis():
    card, feats = _world("toy-leak-full")
    post = {f.feature_id for f in card.features if f.timing is Timing.POST_OUTCOME}
    assert post and not post & set(feats.ids)
    committee = build_committee(feats)
    assert not post & set(committee.ranking())


def test_null_is_always_a_member_and_p_signal_is_its_complement():
    _, feats = _world("toy-driver-full")
    committee = build_committee(feats)
    nulls = [w for h, w in zip(committee.members, committee.weights) if h.is_null]
    assert len(nulls) == 1 and committee.p_signal == 1.0 - nulls[0]


def _member(name: str, drivers: tuple[str, ...]) -> Hypothesis:
    return Hypothesis(name, Design(drivers, drivers), np.zeros(len(drivers)), 0.0, 0.5)


def test_disagreement_is_normalised_entropy_over_driver_sets():
    clusters = {"a": 0, "b": 1, "c": 2}
    one = Committee([_member("x", ("a",))], np.array([1.0]), clusters)
    two = Committee([_member("x", ("a",)), _member("y", ("b",))], np.array([0.5, 0.5]), clusters)
    same = Committee([_member("x", ("a",)), _member("y", ("a",))], np.array([0.5, 0.5]), clusters)
    assert one.disagreement == 0.0 and abs(two.disagreement - 1.0) < 1e-12 and same.disagreement == 0.0


def test_submission_rule_abstains_below_half_and_orders_by_p_driver():
    clusters = {"a": 0, "b": 1, "n": 2}
    members = [_member("null", ()), _member("x", ("a", "b")), _member("y", ("b",))]
    claim = Committee(members, np.array([0.3, 0.3, 0.4]), clusters)
    assert claim.ranking(tau=0.25) == ("b", "a") and claim.ranking(tau=0.5) == ("b",)
    assert Committee(members, np.array([0.55, 0.45, 0.0]), clusters).ranking(tau=0.25) == ()


def test_p_driver_counts_one_vote_per_cluster_per_member():
    clusters = {"a": 0, "a2": 0, "b": 1}
    committee = Committee([_member("x", ("a", "a2", "b"))], np.array([1.0]), clusters)
    assert committee.p_driver() == {"a": 1.0, "b": 1.0}


def test_sequential_agent_never_buys_or_lists_a_post_outcome_feature():
    world = STORE.world("toy-leak-seq")
    post = {f.feature_id for f in world.card.features if f.timing is Timing.POST_OUTCOME}
    bought: list[str] = []

    class Recorder:
        def record(self, event):
            if event.action_kind == "assay":
                bought.extend(Assay.model_validate_json(event.action_json).feature_ids)

    result = run_episode(CommitteeAgent(Policy()), Episode(world), recorder=Recorder())
    assert bought and not post & set(bought) and not post & set(result.ranking)
