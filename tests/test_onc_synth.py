"""Synthesized programs enter the committee as templates and cannot escape the checker."""

import json

import pytest
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.schema import Tier
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import evaluate

from onc.hypotheses import features_from
from onc.synth import SynthCommitteeAgent, check, compile_program, load_programs

GOOD = """
def design(feats):
    t = [abs(float(np.corrcoef(feats.x[:, j], feats.y)[0, 1])) for j in range(len(feats.ids))]
    best = feats.ids[int(np.argmax(t))]
    return Design(drivers=(best,), columns=(best,))
"""


def _world():
    store = FileWorldStore(fixture_store())
    wid = next(w for w in store.world_ids(Tier.PUBLIC_TRAIN) if w == "toy-driver-full")
    data = store.world(wid)
    return store, wid, data.card, features_from(data.card, data.card.feature_ids(), data.x, data.y, data.stratum)


def test_checker_admits_valid_programs_and_rejects_bad_ones():
    _, _, _, feats = _world()
    assert check(compile_program(GOOD), feats) is None
    with pytest.raises(ValueError):
        compile_program("import os\n" + GOOD)
    report = check(compile_program("def design(feats):\n    return Design(drivers=('not_a_feature',))\n"), feats)
    assert report and "outside feats.ids" in report
    report = check(compile_program("def design(feats):\n    return Design(drivers=tuple(feats.ids[:5]))\n"), feats)
    assert report and "at most" in report


def test_cached_program_becomes_a_voting_member(tmp_path):
    store, wid, card, _ = _world()
    p = tmp_path / "qwen" / wid / "analyst.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"ok": True, "source": GOOD}))
    assert list(load_programs(wid, "qwen", 8, tmp_path)) == ["synth:analyst"]
    agent = SynthCommitteeAgent(k=1, cache=tmp_path)
    _, results = evaluate(agent, store, Tier.PUBLIC_TRAIN, world_ids=[wid], with_alignment=False, bootstrap_draws=10)
    names = {m["name"] for m in agent.logs[wid].members}
    assert "synth:analyst" in names and results[0].ranking
