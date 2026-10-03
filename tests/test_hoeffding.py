from fractions import Fraction as F

import pytest

from hoeffding.families import exact_n2, family_best
from hoeffding.problem import Instance, bernoulli_value, exact_n1, hoeffding_bound
from hoeffding.verify import Rejected, certify, tail_probability


def test_certificate_is_exact_and_bounded_by_walls():
    inst = Instance(3, F(2, 5), F(3, 5))
    bern = certify([0, 1], [F(3, 5), F(2, 5)], inst)
    assert bern.value == bernoulli_value(inst) == F(27, 125)
    assert not bern.repaired
    best, fam = family_best(inst)
    assert bern.value < best.value < hoeffding_bound(inst)


def test_n1_and_n2_match_the_proven_answers():
    i1 = Instance(1, F(1, 2), F(1, 4))
    v, atoms, w = exact_n1(i1)
    assert certify(atoms, w, i1).value == v == F(2, 3)
    i2 = Instance(2, F(3, 5), F(3, 5))  # Meester: support {t/2, 1}, value ((1-m)/(1-t/2))^2
    assert abs(float(exact_n2(i2).value) - ((1 - 0.6) / (1 - 0.3)) ** 2) < 1e-9


def test_rejections_and_repair():
    inst = Instance(2, F(1, 2), F(1, 2))
    with pytest.raises(Rejected):
        certify([0, 1.0000001], [0.5, 0.5], inst)
    with pytest.raises(Rejected):
        certify([0, F(1, 4)], [0.5, 0.5], inst)  # mean 1/2 unreachable
    c = certify([0, 0.3, 1], [0.1, 0.5, 0.4], inst)  # floats, repaired weights
    assert c.repaired and sum(c.weights) == 1 and sum(w * a for w, a in zip(c.weights, c.atoms)) == F(1, 2)
    assert c.value == tail_probability(c.atoms, c.weights, 2, F(1, 2))


def test_two_tier_bracket_and_policy():
    import random
    from hoeffding.linear import LinearInstance, bellec_fritz_law, certify_linear
    from hoeffding.tiers import decide, estimate_linear
    from hoeffding.tiers import estimate_hoeffding
    # ties count in P(S_n <= t); an exact float tie at t is inside the event, not doubt
    hi = Instance(6, F(3, 5), F(9, 10))
    atoms, ws = [0, F(9, 10), 1], [F(339, 1000), F(610, 1000), F(51, 1000)]
    eh, ch = estimate_hoeffding(atoms, ws, hi), certify(atoms, ws, hi)
    assert eh.value < 0.002 and 0.0179 < float(ch.value) <= eh.upper   # the tie mass is the value
    inst = LinearInstance((1, 1, 1, -2))
    a, w = bellec_fritz_law(F(474346, 10**6), 8)
    est, cert = estimate_linear(a, w, inst), certify_linear(a, w, inst)
    assert est.value <= float(cert.value) <= est.upper          # strict event: the exact value lies in the bracket
    assert est.tie_mass > 0.05                                   # a float 0 is doubt, not a tie: see tiers.py
    near = LinearInstance((1, 1, -1))   # exact sum -1e-12 is inside the strict event; floats cannot tell
    na = [F(1, 10), F(2, 10), F(3, 10) + F(1, 10**12)]
    e2, c2 = estimate_linear(na, [1, 1, 1], near), certify_linear(na, [1, 1, 1], near)
    assert e2.tie_mass > 0.05 and float(c2.value) > e2.value + 0.05 and float(c2.value) <= e2.upper
    rng = random.Random(0)
    assert decide(est, None, 0.0, rng).action == "certify"       # no incumbent: certify
    assert decide(est, est.upper - 1e-6, 0.0, rng).action == "certify"  # could win: certify
    assert decide(est, est.upper, 0.0, rng).action == "skip"            # can only tie the best: skip
    assert decide(est, est.upper + 1e-6, 1.0, rng).action == "audit"    # audit rate 1: audit


def test_integer_certifier_matches_fraction_convolution():
    import random
    from hoeffding.linear import LinearInstance, value_of
    rng = random.Random(3)
    for c in ((1, 1, 1, -2), (2, -1, -1), (1, 2, -3)):
        atoms = sorted({F(rng.randrange(0, 2**20), 2**20) + F(rng.randrange(0, 7), 7 * 2**40) for _ in range(9)})
        weights = [F(rng.randrange(1, 50), 97) for _ in atoms]
        tot = sum(weights); weights = [w / tot for w in weights]
        # reference: plain Fraction convolution
        d = {F(0): F(1)}
        for ci in c:
            nd = {}
            for s, p in d.items():
                for a, w in zip(atoms, weights):
                    nd[s + ci * a] = nd.get(s + ci * a, F(0)) + p * w
            d = nd
        ref_lt = sum((p for s, p in d.items() if s < 0), F(0)); ref_eq = d.get(F(0), F(0))
        lt, eq = value_of(atoms, weights, c)
        assert (lt, eq) == (ref_lt, ref_eq)


def test_prioritizer_learns_from_outcomes_and_tracks_success():
    import random
    from types import SimpleNamespace
    from hoeffding.prioritize import Prioritizer, Record
    pr = Prioritizer()
    parent = SimpleNamespace(name="p", confidences=[0.01])
    for i in range(6):  # refine succeeds, explore fails
        pr.record(Record(i, f"r{i}", "refine", "p", 0.398, 0.5, 1e-4)); pr.resolve(f"r{i}", 0.398 + 1e-4)
        pr.record(Record(i, f"e{i}", "explore", "p", 0.398, 0.5, 1e-4)); pr.resolve(f"e{i}", 0.398)
    rep = pr.report()
    assert rep["by_lane"]["refine"]["successes"] == 6 and rep["by_lane"]["explore"]["successes"] == 0
    assert rep["success_rate"] == 0.5 and abs(rep["brier"] - 0.25) < 1e-9
    picks = [pr.propose([parent], ["refine", "explore"], 1, random.Random(s), lambda p: 0.01)[0]["lane"] for s in range(50)]
    assert picks.count("refine") > 40  # the posterior favours the lane that delivered, with some exploration left
    assert pr.propose([parent], ["refine", "explore"], 2, random.Random(0), lambda p: 0.01)[1]["lane"] == "explore"
    # an untried lane is forced into the plan ahead of a lane with a strong record
    plan = pr.propose([parent], ["refine", "explore", "deck"], 1, random.Random(0), lambda p: 0.01)
    assert plan[0]["lane"] == "deck" and plan[0]["forced"]
