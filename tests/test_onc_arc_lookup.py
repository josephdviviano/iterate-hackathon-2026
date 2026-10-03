from committee.loader import Transition
from committee.verify import run_program
from onc.arc_lookup import lookup_source, perturbation_admission, perturbations

RULE = "import copy\n\ndef transition_function(state, action):\n    s = copy.deepcopy(state)\n" \
       "    for o in s:\n        if action == 1 and o['type'] == 'player':\n            o['x'] += 1\n    return s\n"
WALLS = [{"name": f"wall_{i}", "type": "wall", "x": 10 + 3 * i, "y": 7} for i in range(3)]


def _rows(xs):
    return [Transition(i + 1, 1, 1, None, False, [], [],
                       WALLS + [{"name": "p", "type": "player", "x": x, "y": 0}],
                       WALLS + [{"name": "p", "type": "player", "x": x + 1, "y": 0}])
            for i, x in enumerate(xs)]


def test_lookup_table_passes_exact_replay_and_predicts_nothing_held_out():
    train, test = _rows(range(4)), _rows(range(20, 23))
    table = run_program(lookup_source(train), train, test)
    assert table.consistent and table.test_accuracy == 0.0
    rule = run_program(RULE, train, test)
    assert rule.consistent and rule.test_accuracy == 1.0


def test_perturbation_rule_rejects_the_table_and_admits_the_rule():
    train = _rows(range(4))
    edits = perturbations(train[0].before_objs, train[0].after_objs)
    assert len(edits) == 9 and all(s != train[0].before_objs for _, s in edits)
    admitted, info = perturbation_admission(lookup_source(train), train)
    assert not admitted and info["identity_share"] == 1.0 and info["n_perturbations"] == 36
    admitted, info = perturbation_admission(RULE, train)
    assert admitted and info["identity_share"] == 0.0
