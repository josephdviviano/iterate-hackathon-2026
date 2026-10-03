import numpy as np

from bioprot.data import function_sequence, split_pseudocode, strip_definitions
from bioprot.metrics import aurc, bootstrap, coverage_at_risk, oracle_aurc, random_aurc, selective_risk_at
from bioprot.prompts import parse_confidence, parse_verdict
from bioprot.score import normalized_levenshtein, precision_recall, symmetric_distance
from bioprot.uncertainty import self_consistency

SRC = "def a(x):\n    pass\n\ndef b(y):\n    pass\n\n# steps\nr = a(x=1)\nfor i in range(2):\n    b(y=r)\nb(y=2)\n"


def test_sequence_parsers_agree_on_statement_calls():
    defs, body = split_pseudocode(SRC)
    assert [d.split("(")[0] for d in defs] == ["def a", "def b"]
    assert function_sequence(body) == (["a", "b", "b"], "ast")
    broken = body.replace("b(y=2)", "b(y=10ul)")  # invalid literal, as in 16 ground-truth files
    assert function_sequence(broken) == (["a", "b", "b"], "regex")
    assert strip_definitions("```python\n" + SRC + "```").strip().startswith("# steps")


def test_scores_follow_the_reference_definitions():
    assert normalized_levenshtein(["a", "b", "c", "d"], ["a", "b"]) == 1.0
    assert normalized_levenshtein([], ["a", "b"]) == 1.0
    assert precision_recall(["a", "a", "b"], ["a", "b", "b", "c"]) == (0.5, 2 / 3)
    assert symmetric_distance(["a"], ["a", "b", "c", "d"]) == 0.75


def test_risk_coverage_orders_oracle_signal_random():
    rng = np.random.default_rng(0)
    risk = rng.integers(0, 2, 200).astype(float)
    unc = risk + rng.normal(0, 0.8, 200)  # informative but noisy
    tb = np.arange(200)
    a, o, r = aurc(risk, unc, tb), oracle_aurc(risk, tb), random_aurc(risk)
    assert o < a < r
    assert selective_risk_at(risk, risk, tb, 0.5) == 0.0 or risk.mean() > 0.5
    assert coverage_at_risk(risk, risk, tb, 0.0) == np.mean(risk == 0)
    lo, hi = bootstrap(np.repeat(np.arange(40), 5), lambda idx: aurc(risk[idx], unc[idx], tb[idx]), n_boot=100)
    assert lo <= a <= hi


def test_self_consistency_is_leave_one_out_mean():
    rows = [{"protocol_id": "p", "sample_idx": i, "parsed_function_sequence": s}
            for i, s in enumerate([["a", "b"], ["a", "b"], ["c", "d"]])]
    sc = self_consistency(rows)
    assert sc[("p", 0)]["self_consistency"] == 0.5 and sc[("p", 2)]["self_consistency"] == 1.0
    assert abs(sc[("p", 0)]["self_consistency_protocol"] - 2 / 3) < 1e-9


def test_elicitation_parsers_separate_abstain_from_decline():
    assert parse_confidence("CONFIDENCE: 85") == {"confidence": 0.85, "abstained": False, "declined": False}
    assert parse_confidence("ABSTAIN: the description names no reagents")["abstained"]
    assert parse_confidence("I can't help with that request.")["declined"]
    assert parse_verdict("FAIL\nThe lysis step is missing.") == "FAIL" and parse_verdict("**PASS**") == "PASS"


def test_conformal_sets_hold_coverage_and_commit_when_the_score_separates():
    from bioprot.conformal import aci, quantile_level, split_conformal

    rng = np.random.default_rng(1)
    acceptable = rng.random(400) < 0.4
    u = np.clip(np.where(acceptable, 0.2, 0.8) + rng.normal(0, 0.15, 400), 0, 1)
    groups = np.repeat(np.arange(80), 5)
    assert quantile_level(9, 0.1) == 1.0 and abs(quantile_level(99, 0.1) - 0.909) < 1e-3
    s = split_conformal(u, acceptable, groups, alpha=0.1, n_splits=50)
    assert 0.88 <= s["coverage"] <= 0.97 and s["committed"] > 0.5 and s["accuracy_when_committed"] > 0.85
    o = aci(u, acceptable, alpha=0.1)
    assert 0.85 <= o["coverage"] <= 0.95


def test_committee_vote_entropy_and_agreed_prefix():
    from bioprot.committee import build, common_prefix, step_gate
    from bioprot.data import Protocol

    p = Protocol("p", "t", "d", None, "", ("def a(x):\n    pass", "def b(x):\n    pass", "def c(x):\n    pass"), ("a", "b", "c"))
    rows = [{"model": "m", "sample_idx": i, "parsed_function_sequence": s}
            for i, s in enumerate([["a", "b", "c"], ["a", "b", "c"], ["a", "c"], ["a", "zz"], []])]
    c = build(rows, p)
    assert len(c.members) == 3 and c.n_total == 5          # the undefined call and the empty plan are not admitted
    assert c.plan == ["a", "b", "c"] and abs(c.share - 2 / 3) < 1e-9 and c.n_distinct == 2
    assert abs(c.disagreement - (-(2 / 3) * np.log(2 / 3) - (1 / 3) * np.log(1 / 3)) / np.log(3)) < 1e-9
    assert c.agreed_prefix == 1 and common_prefix([["a", "b"], ["a", "b"]]) == 2
    g = step_gate(c, ["a", "x", "c"])
    assert (g["first_error"], g["first_disagreement"], g["wrong_executed"], g["wrong_if_all"]) == (1, 1, 0, 2)
    assert g["disagreement_not_after_error"] is True
