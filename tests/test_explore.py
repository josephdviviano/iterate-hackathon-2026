from committee.committee import Member
from committee.explore import simulate
from committee.loader import Transition

A = [{"name": "p", "type": "player", "x": 1, "y": 0}]
B = [{"name": "p", "type": "player", "x": 2, "y": 0}]
C = [{"name": "p", "type": "player", "x": 3, "y": 0}]


def _test_set():
    before = [{"name": "p", "type": "player", "x": 0, "y": 0}]
    return [Transition(i, 1, 1, None, False, [], [], before, after) for i, after in enumerate([A, A, B])]


def test_disagreement_probe_collapses_committee_in_one_step_and_random_may_not():
    # members agree on transitions 0 and 1, split on transition 2; only m1 is right there.
    m1 = Member("m1", "def transition_function(s,a): return s", [A, A, B], length=10)
    m2 = Member("m2", "def transition_function(s,a): return s", [A, A, C], length=12)
    m3 = Member("m3", "def transition_function(s,a): return s", [A, A, A], length=14)
    test = _test_set()
    tr = simulate([m1, m2, m3], test, "disagreement", lam=0.01)
    assert tr.probes[0] == 2 and tr.members_left[:2] == [3, 1] and tr.probes_to_collapse == 1
    assert tr.falsified_at is None
    # a member set that is wrong everywhere on the probed transition gets falsified
    tr2 = simulate([m2, m3], test, "disagreement", lam=0.01)
    assert tr2.falsified_at == 1 and tr2.members_left[-1] == 0 and tr2.probes_to_collapse is None
