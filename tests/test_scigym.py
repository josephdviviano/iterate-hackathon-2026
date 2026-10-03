import numpy as np

from scigym.committee import candidates, jaccard_distance, rms, vote
from scigym.data import load_systems
from scigym.env import Environment, Experiment, Trajectory, smape
from scigym.model import EPS, Hypothesis, fit, parse, truth_text


def _system():
    return load_systems(limit=4)[3]  # BIOMD0000000036, 5 reactions, 4 species


def test_true_structure_is_admitted_and_a_wrong_one_is_not():
    s = _system()
    env = Environment(s)
    obs = [(Experiment("observe"), env.run(Experiment("observe")))]
    truth = fit(parse(truth_text(s), s), s, obs)
    assert truth.admitted and truth.max_error < 0.01
    wrong = fit(parse("\n".join(f"R{i}: {sp} -> ; k{i} * {sp}" for i, sp in enumerate(s.species, 1)), s), s, obs)
    assert not wrong.admitted and wrong.max_error > EPS


def test_parse_reads_modifiers_stoichiometry_and_new_parameters():
    s = next(x for x in load_systems(limit=4) if len(x.species) >= 4)
    a, b, c, d = s.species[:4]
    h = parse(f"```text\nR1: 2 {a} + {b} -> {c} [{d}] ; V * {d} * {a} / (K + {a})\nparams: V = 2\n```", s)
    assert h.reactions[0].reactants == (a, a, b) and h.reactions[0].products == (c,) and h.reactions[0].modifiers == (d,)
    assert h.params == {"V": 2.0, "K": 1.0}


def test_rms_vote_and_candidates():
    k = lambda r, p: (tuple(sorted(r)), tuple(sorted(p)))
    class R:  # a reaction with a key, as data.Reaction exposes it
        def __init__(self, r, p): self.key = k(r, p)
    truth = (R(["a"], ["b"]), R(["b"], ["c"]), R(["b"], ["c"]))
    out = rms([k(["a"], ["b"]), k(["b"], ["c"]), k(["x"], ["y"])], truth)
    assert abs(out["precision"] - 2 / 3) < 1e-9 and abs(out["recall"] - 2 / 3) < 1e-9
    members = [Hypothesis([], [], {}) for _ in range(3)]
    for m, keys in zip(members, ([k("a", "b"), k("b", "c")], [k("a", "b"), k("b", "c")], [k("a", "b")])):
        m.reactions = [R(*kk) for kk in keys]
    v = vote(members)
    assert v["medoid"] in (0, 1) and v["majority"] == [k("a", "b"), k("b", "c")] and v["n_distinct"] == 2
    assert jaccard_distance([1, 1, 2], [1, 2, 3]) == 0.5
    s = _system()
    traj = Trajectory(np.array([0.0, 1.0]), {sp: np.array([1.0, 1.0]) for sp in s.species})
    cands = candidates(s, traj)
    assert len(cands) == 2 * len(s.species) + sum(s.initial[sp] > 0 for sp in s.species)
    assert smape(traj, traj, s.species) == 0.0
