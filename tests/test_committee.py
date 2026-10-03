from committee.committee import Committee, Member, auroc, description_length

A = [{"name": "p", "x": 1}]
B = [{"name": "p", "x": 2}]


def _member(name, preds, src="def transition_function(s, a):\n    return s\n"):
    return Member(name=name, source=src, test_preds=preds, length=description_length(src))


def test_description_length_ignores_prose_and_rewards_short_code():
    short = "def transition_function(s, a):\n    return s\n"
    prose = '"""' + "x" * 500 + '"""\n' + short.replace("return s", "# comment\n    return s")
    longer = short + "\ndef helper(z):\n    return [o for o in z if o.get('x', 0) > 3]\n"
    assert description_length(prose) <= description_length(short) + 4
    assert description_length(longer) > description_length(short)


def test_weights_favor_shorter_members_and_votes_measure_disagreement():
    m1 = _member("short", [A, A])
    m2 = _member("long", [A, B], src="def transition_function(s, a):\n    q = [dict(o) for o in s]\n    return q\n")
    c = Committee([m1, m2], lam=0.1)
    assert c.weights[0] > c.weights[1]
    v0, v1 = c.votes()
    assert v0.disagreement == 0.0 and v0.n_distinct == 1
    assert 0.0 < v1.disagreement <= 1.0 and v1.n_distinct == 2
    assert v1.prediction == c.vote(1).prediction
    ev = c.evaluate([A, B])
    assert ev["vote_accuracy"] == 0.5 and ev["simplest_accuracy"] == 0.5
    assert ev["auroc_disagreement_vs_error"] == 1.0
    assert ev["auroc_uniform_disagreement_vs_error"] == 1.0
    assert c.uniform_disagreement(0) == 0.0 and c.uniform_disagreement(1) == 1.0
    rel = {r["bin"]: r for r in ev["reliability"]}
    assert rel["unanimous"]["n"] == 1 and rel["unanimous"]["error_rate"] == 0.0
    assert sum(r["n"] for r in ev["reliability"]) == 2 and rel["low"]["n"] + rel["medium"]["n"] + rel["high"]["n"] == 1


def test_auroc():
    assert auroc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auroc([0.5, 0.5, 0.5], [True, False, False]) == 0.5
    assert auroc([0.1, 0.9], [True, False]) == 0.0
    assert auroc([0.1, 0.9], [True, True]) is None


def test_extract_code_takes_the_longest_python_block_or_bare_source():
    from committee.synth_api import extract_code

    text = "intro\n```python\nx = 1\n```\nthen\n```\ndef transition_function(s, a):\n    return s\n```\n"
    assert extract_code(text).startswith("def transition_function")
    assert extract_code("def transition_function(s, a):\n    return s") is not None
    assert extract_code("no code here") is None


def test_cegis_moves_the_falsifying_probe_into_train_and_names_the_refuted_predictions():
    from committee.cegis import counterexample_text, observed_probes, split_after_probes
    from committee.loader import Transition

    A = [{"name": "p", "type": "player", "x": 1, "y": 0}]
    B = [{"name": "p", "type": "player", "x": 2, "y": 0}]
    C = [{"name": "p", "type": "player", "x": 3, "y": 0}]
    before = [{"name": "p", "type": "player", "x": 0, "y": 0}]
    test = [Transition(10 + i, 1, 1, None, False, [], [], before, after) for i, after in enumerate([A, A, B])]
    train = [Transition(1, 1, 1, None, False, [], [], before, A)]
    # members agree on transitions 0 and 1 and split on 2, where none of them is right
    members = [Member(f"m{i}", "def transition_function(s,a): return s", preds, length=10)
               for i, preds in enumerate([[A, A, C], [A, A, C], [A, A, A]])]
    probes = observed_probes(members, test)
    assert probes == [2]
    train2, test2 = split_after_probes(train, test, probes)
    assert [t.step for t in train2] == [1, 12] and [t.step for t in test2] == [10, 11]
    text = counterexample_text(members, test, probes)
    assert "step 12" in text and "2 program(s) predicted" in text and "1 program(s) predicted" in text
