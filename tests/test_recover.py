from pathlib import Path

from committee.experiment import seeds_for
from committee.recover import Target, assign, seed_of
from committee.seeds import make_seeds
from committee.synth_devin import devin_prompt

URLS = {"transitions.md": "u1", "buffer.json": "u2", "check.py": "u3"}


def test_seed_reads_back_from_the_prompt_and_added_runs_continue_the_batch():
    assert seed_of(devin_prompt([], "Objects of type `a` show mixed outcomes.", URLS)) == "Objects of type `a` show mixed outcomes."
    assert seed_of(devin_prompt([], None, URLS)) is None
    assert seeds_for([], 1, 7) == [make_seeds([], 8)[7]] != [make_seeds([], 1)[0]]


def _target(condition, seeds, missing):
    return Target("g", 1, condition, "objects", [], [], Path("/nonexistent"), seeds, missing)


def test_orphans_land_on_the_run_their_seed_names():
    base = _target("baseline", [None, None, None], [0, 1])
    com = _target("committee", ["s0", "s1", "s2"], [0, 2])
    orphans = [{"seed": "s2", "source": ""}, {"seed": None, "source": ""}, {"seed": "s1", "source": ""}]
    got = [(t.condition, i) for _, t, i in assign(orphans, [base, com])]
    assert got == [("committee", 2), ("baseline", 0)]
    assert orphans[2]["unassigned"] is True
