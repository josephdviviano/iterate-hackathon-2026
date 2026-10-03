"""A task binds instances, walls, the exact certifier and the agent contract.

The climb, the committee runner and the synthesis workspace are task-agnostic.
Two tasks exist: Hoeffding's problem and the linear-inequality family.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

from . import linear, problem, tiers, verify

HERE = Path(__file__).parent


def _strip_relative_imports(src: str) -> str:
    """Drop relative imports and the __future__ line so that modules can be concatenated into one file."""
    return "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("from .") and not l.startswith("from __future__"))


@dataclass
class Task:
    name: str
    contract: str
    train: list
    test: list
    certify: Callable
    lower: Callable[[object], float]
    upper: Callable[[object], float | None]
    args_of: Callable[[object], list]
    verifier_source: str
    stub: str
    estimate: Callable = None

    def rows(self, instances) -> list[dict]:
        out = []
        for i in instances:
            row = dict(i.to_json(), key=i.key, args=self.args_of(i), lower=self.lower(i), upper=self.upper(i))
            out.append(row)
        return out


HOEFFDING_CONTRACT = """\
# Task: push a rigorous lower bound on an open extremal probability

Let X_1, ..., X_n be iid on [0, 1] with E X = m, and S_n their sum. For
0 <= t < n m, define p_n(m, t) = sup over such laws of P(S_n <= t). Only n = 1
and n = 2 are solved. Hoeffding's inequality gives the upper wall
exp(-n kl(t/n || m)); the Bernoulli(m) law gives a lower wall P(Bin(n, m) <= t).
Any discrete law you name gives a certified lower bound, computed in exact
rational arithmetic by `verify.py`.

Write `strategy(n, m, t) -> (atoms, weights)` in `strategy.py`:

- `m`, `t` are `fractions.Fraction`. Return two lists: atoms in [0, 1] and
  non-negative weights. Fractions, ints, floats or numeric strings are all
  accepted. Floats are rounded to rationals with denominator <= 10^12.
- The checker re-solves the last two weights so that sum w = 1 and
  sum w a = m hold exactly. If that needs a negative weight, the candidate is
  rejected. Keep atoms distinct.
- Only the Python standard library is available. No file or network access.
"""

LINEAR_CONTRACT = """\
# Task: push a rigorous lower bound on an open extremal probability

For a coefficient vector c = (c_1, ..., c_n) of integers, define
C(c) = sup over iid laws mu on [0, infinity) of P(c_1 X_1 + ... + c_n X_n < 0),
strict inequality. The event is scale-invariant, so atoms are taken in [0, 1].
Known: c = (2, -1, -1) has C = 2/3, proven, attained in the limit by the deck
1, 2, 4, ..., 2^k. c = (1, 1, 1, -2) has 0.400695 <= C <= 0.417 (Bellec and
Fritz 2024); their construction puts mass p at 0 and spreads 1 - p over atoms
1 - 2^-i, and reaches 0.400695 only in a limit that breaks ties. Other c have no
published value. Any discrete law you name gives a certified lower bound,
computed in exact rational arithmetic by `verify.py`. Ties (sum exactly 0)
do not count.

Write `strategy(c) -> (atoms, weights)` in `strategy.py`:

- `c` is a list of ints. Return two lists: distinct atoms in [0, 1] and
  non-negative weights. Fractions, ints, floats or numeric strings are all
  accepted. Floats are rounded to rationals with denominator <= 10^12. Weights
  are normalised to sum 1. At most 64 atoms.
- Certification convolves len(c) copies of the law; its cost grows with the
  number of distinct partial sums, so dyadic or small-denominator atoms are
  much cheaper than random floats. Keep each certification under 20 seconds.
- Only the Python standard library is available. No file or network access.
"""

COMMON_RULES = """
Rules:

1. The strategy must be a rule of its arguments, not a table of answers. It
   will be run on instances you have not seen.
2. Run `python3 check.py`. It prints, per train instance, your value, the
   lower wall (a naive law), the upper wall when one is known, and the
   structure of your law. The value is CERTIFIED in exact arithmetic when the
   law can beat your best so far on that instance; otherwise it is a float
   ESTIMATE, which saves the certification cost. Only certified values count.
   Iterate until you cannot improve. Numerical optimization inside `strategy`
   is allowed; keep each call under 5 seconds.
3. Define `confidence(...)` early, in your first edit, and refine it later. It
   has the same arguments as `strategy` and returns your
   probability, in [0, 1], that the certified value is within 1e-4 of the true
   supremum. It is called on unseen instances too, so make it a rule of the
   arguments and of what your search found. Be honest: it is scored for
   calibration, not for optimism. Before you stop, also write `report.json`
   mapping each train key (as printed by the checker) to that probability.
4. Put a 5-line header comment in `strategy.py` stating the structure you
   found and what you could not settle.

Files: `instances.json`, `verify.py`, `check.py`, `strategy.py`, `report.json`.
"""

HOEFFDING_STUB = '''"""Structure: (fill in)
"""
from fractions import Fraction


def strategy(n, m, t):
    # Bernoulli(m): atoms {0, 1}. Replace with a better law.
    return [0, 1], [1 - m, m]


def confidence(n, m, t):
    # Probability that strategy(n, m, t) is within 1e-4 of the supremum. Replace.
    return 0.0
'''

LINEAR_STUB = '''"""Structure: (fill in)
"""
from fractions import Fraction


def strategy(c):
    # Two atoms {0, 1}. Replace with a better law.
    return [0, 1], [Fraction(1, 2), Fraction(1, 2)]


def confidence(c):
    # Probability that strategy(c) is within 1e-4 of the supremum. Replace.
    return 0.0
'''


def _hoeffding_verifier() -> str:
    src = "from __future__ import annotations\nimport random\n" + \
        _strip_relative_imports((HERE / "verify.py").read_text()) + "\n\n" + \
        _strip_relative_imports((HERE / "tiers.py").read_text())
    return src + '''

def _inst(row):
    class _I:
        pass
    inst = _I(); inst.n, inst.m, inst.t = int(row["n"]), Fraction(row["m"]), Fraction(row["t"])
    return inst

def certify_raw(atoms, weights, row):
    c = certify(atoms, weights, _inst(row))
    return {"atoms": c.atoms, "weights": c.weights, "value": float(c.value), "repaired": c.repaired}

def estimate_raw(atoms, weights, row):
    e = estimate_hoeffding(atoms, weights, _inst(row))
    return {"atoms": [_rational(a) for a in atoms], "weights": [_rational(w) for w in weights],
            "value": e.value, "upper": e.upper, "tie_mass": e.tie_mass}
'''


def _linear_verifier() -> str:
    src = "from __future__ import annotations\nimport random\n" + \
        _strip_relative_imports((HERE / "verify.py").read_text()) + "\n\n" + \
        _strip_relative_imports((HERE / "linear.py").read_text()) + "\n\n" + \
        _strip_relative_imports((HERE / "tiers.py").read_text())
    return src + '''

def certify_raw(atoms, weights, row):
    c = certify_linear(atoms, weights, LinearInstance(tuple(int(x) for x in row["c"])))
    return {"atoms": c.atoms, "weights": c.weights, "value": float(c.value), "repaired": c.repaired}

def estimate_raw(atoms, weights, row):
    e = estimate_linear(atoms, weights, LinearInstance(tuple(int(x) for x in row["c"])))
    return {"atoms": [_rational(a) for a in atoms], "weights": [_rational(w) for w in weights],
            "value": e.value, "upper": e.upper, "tie_mass": e.tie_mass}
'''


def hoeffding_task() -> Task:
    return Task(
        name="hoeffding",
        contract=HOEFFDING_CONTRACT + COMMON_RULES,
        train=problem.TRAIN, test=problem.TEST,
        certify=verify.certify,
        lower=lambda i: float(problem.bernoulli_value(i)),
        upper=problem.hoeffding_bound,
        args_of=lambda i: [i.n, str(i.m), str(i.t)],
        verifier_source=_hoeffding_verifier(),
        stub=HOEFFDING_STUB,
        estimate=tiers.estimate_hoeffding,
    )


FOCUS_NOTE = """
# This run: one vector, c = (1, 1, 1, -2), and a higher atom cap

Only c = (1, 1, 1, -2) is scored. The atom cap is {cap}. Landmarks: naive agent 0.389;
the Bellec-Fritz construction reaches 0.400695 only as a limit, and its authors
conjecture that is the supremum; the proven ceiling is 0.417. A certified law
above 0.400695 would refute the conjecture and is the goal of the explore lane.
Finite refinements of the ladder approach the limit from below and are the goal
of the refine lane. Say in your header which lane your law belongs to.
"""


def linear_focus_task(cap: int = 256) -> Task:
    import functools
    t = linear_task()
    return Task(
        name="linear1",
        contract=LINEAR_CONTRACT.replace("At most 64 atoms.", f"At most {cap} atoms.") + FOCUS_NOTE.format(cap=cap) + COMMON_RULES,
        train=linear.FOCUS, test=linear.TEST,
        certify=functools.partial(linear.certify_linear, max_atoms=cap),
        lower=t.lower, upper=t.upper, args_of=t.args_of,
        verifier_source=t.verifier_source.replace("MAX_ATOMS = 64", f"MAX_ATOMS = {cap}"),
        stub=t.stub, estimate=t.estimate,
    )


def linear_task() -> Task:
    return Task(
        name="linear",
        contract=LINEAR_CONTRACT + COMMON_RULES,
        train=linear.TRAIN, test=linear.TEST,
        certify=linear.certify_linear,
        lower=linear.naive_value,
        upper=linear.upper_wall,
        args_of=lambda i: [list(i.c)],
        verifier_source=_linear_verifier(),
        stub=LINEAR_STUB,
        estimate=tiers.estimate_linear,
    )


TASKS = {"hoeffding": hoeffding_task, "linear": linear_task, "linear1": linear_focus_task}
