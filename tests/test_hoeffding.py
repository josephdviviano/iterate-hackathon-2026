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
