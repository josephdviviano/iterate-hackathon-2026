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


def test_disputed_probes_rank_by_disagreement_and_describe_outcomes():
    from committee.active import disputed_probes, targeted_seed

    before = [{"name": "p", "type": "player", "x": 0, "y": 0, "w": 1, "h": 1}]
    probes = [Transition(10, 1, 1, None, False, [], [], before, []), Transition(11, 1, 2, None, False, [], [], before, [])]
    moved = [{"name": "p", "type": "player", "x": 1, "y": 0, "w": 1, "h": 1}]
    preds = [[before, before], [before, moved], [before, None]]   # probe 0 unanimous, probe 1 three-way
    d = disputed_probes(probes, preds, top=5)
    assert [x["step"] for x in d] == [11] and d[0]["disagreement"] == 1.0
    texts = [t for _, t in d[0]["outcomes"]]
    assert "nothing changes" in texts and any("x 0->1" in t for t in texts) and "error" in texts
    seed = targeted_seed(d, ["rule A", ""], 3)
    assert "step 11" in seed and "program 0: rule A" in seed and "program 1" not in seed


def test_oracle_headroom_and_best_survivor_track_the_member_that_is_right():
    from committee.selection import headroom

    m1 = Member("m1", "", [A, A, B], length=10)   # right on every transition
    m3 = Member("m3", "", [A, A, A], length=14)   # wrong on transition 2
    m4 = Member("m4", "", [B, B, B], length=9)    # right on transition 2 only
    test = _test_set()
    h = headroom([m1, m3], test)
    assert abs(h["mean_member"] - 5 / 6) < 1e-9 and (h["best_member"], h["oracle"]) == (1.0, 1.0)
    h2 = headroom([m3, m4], test)   # complementary members: the oracle exceeds the best member
    assert (h2["best_member"], h2["oracle"]) == (2 / 3, 1.0)
    tr = simulate([m1, m3], test, "disagreement", lam=0.0)
    assert tr.members_left[:2] == [2, 1] and tr.best_error_left[:2] == [0.0, 0.0]
