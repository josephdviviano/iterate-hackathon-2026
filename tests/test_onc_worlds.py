import json
from collections import Counter

import numpy as np
import pytest
from onc_agi.adapters.agents import make_agent
from onc_agi.core.schema import Mode, Tier, Timing
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services import scoring
from onc_agi.services.kit import evaluate

from onc.worlds import ROLES, build_world, main, oracle_ranking, split_ids

TYPE_COUNTS = {"expression": 10, "copy_number": 3, "protein": 2, "clinical": 2, "derived": 2, "lab": 4}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("dev")
    assert main(["--out", str(root), "--n-per-role", "1", "--seed", "0"]) == 0
    return FileWorldStore(root), json.loads((root / "manifest.json").read_text())["worlds"]


def test_same_seed_same_world_and_other_seed_other_world():
    role = next(r for r in ROLES if r.name == "driver")
    draws = [build_world(role, 0, Mode.FULL_ACCESS, seed=s, n=60, effect=1.0, prevalence=0.35) for s in (1, 1, 2)]
    (a, ka, sa), (b, kb, sb), (c, _, _) = draws
    assert a.x.tobytes() == b.x.tobytes() and a.y.tobytes() == b.y.tobytes() and sa == sb
    assert ka == kb and a.card == b.card
    assert a.x.tobytes() != c.x.tobytes()


def test_shape_and_manifest(store):
    files, records = store
    assert len(records) == 24 and Counter(r["role"] for r in records)["null"] == 4
    for world_id in files.world_ids(Tier.PUBLIC_TRAIN):
        card, key = files.card(world_id), files.answer_key(world_id)
        assert card.n_pool == 240 and Counter(f.data_type for f in card.features) == TYPE_COUNTS
        post = {f.feature_id for f in card.features if f.timing is Timing.POST_OUTCOME}
        assert len(post) == 3 and post == set(key.reject_set)
        assert set(key.clusters) == set(key.strata) == set(card.feature_ids())
        assert Counter(key.strata.values()).most_common()[-1][1] >= 4


def test_oracle_finds_every_signal_world_and_nulls_restrain(store):
    files, records = store
    for r in records:
        key = files.answer_key(r["world_id"])
        files.world(r["world_id"])
        if r["role"] == "null":
            assert key.depth == 0 and scoring.score_world([], key).restrained
        else:
            assert key.depth > 0 and scoring.score_world(oracle_ranking(key), key).find > 1 - 1e-9


def test_interaction_has_no_marginal_effect(store):
    files, records = store
    world_id = next(r["world_id"] for r in records if r["role"] == "interaction" and r["mode"] == "full_access")
    world, key = files.world(world_id), files.answer_key(world_id)
    yc = (world.y - world.y.mean()) / world.y.std()
    for part in key.groups[0].parts:
        col = world.x[:, world.column(part.true_feature)]
        z = abs(float((col - col.mean()) / col.std() @ yc)) / np.sqrt(len(yc))
        assert z < key.detection_threshold
    card, _ = evaluate(make_agent("univariate_bh"), files, Tier.PUBLIC_TRAIN, world_ids=[world_id], with_alignment=False)
    assert card.worlds[0].find == 0.0


def test_split_is_deterministic_disjoint_and_stratified():
    records = [
        {"world_id": f"dev-{role}-{i:02d}-{tag}", "role": role, "mode": mode}
        for role in ("null", "driver", "module")
        for i in range(4)
        for mode, tag in (("full_access", "full"), ("sequential", "seq"))
    ]
    train, held = split_ids(records, held_out_fraction=0.25, seed=3)
    assert (train, held) == split_ids(records, held_out_fraction=0.25, seed=3)
    assert not set(train) & set(held) and sorted(train + held) == sorted(r["world_id"] for r in records)
    held_keys = Counter((r["role"], r["mode"]) for r in records if r["world_id"] in held)
    assert len(held) == 6 and set(held_keys.values()) == {1}
    assert split_ids(records, held_out_fraction=0.25, seed=4)[1] != held
