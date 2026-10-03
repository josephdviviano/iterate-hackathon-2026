"""A hypothesis is a list of reactions in text; the verifier fits its rate constants and replays the experiments.

Format, one reaction per line, modifiers in brackets, the rate law after a
semicolon, any identifier that is not a species or compartment is a parameter:

    R1: A + B -> C ; k1 * A * B
    R2: C -> ; k2 * C
    R3: -> A ; k3
    R4: S -> P [E] ; V4 * E * S / (K4 + S)
    params: k1 = 0.5, k2 = 0.1

The structure is the model's claim; the numbers are fitted to the observed
trajectories by least squares, so admission depends on the mechanism and
not on a guessed constant.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import libsbml
import numpy as np
import roadrunner
from scipy.optimize import least_squares

from .data import Reaction, System
from .env import Experiment, Trajectory, simulate, smape

EPS = 0.15          # admission: SMAPE at or below this on every observed experiment (set from a six-system pilot, B5)
FIT_STEPS = 200     # candidates are simulated on this grid during fitting; observations are interpolated onto it
N_FIT_POINTS = 40
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FUNCS = {"exp", "log", "ln", "pow", "sqrt", "abs", "sin", "cos", "tan", "floor", "ceil", "piecewise", "time", "pi"}


@dataclass
class Hypothesis:
    reactions: list[Reaction]
    laws: list[str]
    params: dict[str, float]
    text: str = ""
    fitted: dict[str, float] = field(default_factory=dict)
    sbml: str = ""
    errors: list[float] = field(default_factory=list)   # SMAPE per observed experiment
    fit_notes: list[str] = field(default_factory=list)

    @property
    def max_error(self) -> float:
        return max(self.errors) if self.errors else 1.0

    @property
    def admitted(self) -> bool:
        return bool(self.errors) and self.max_error <= EPS

    @property
    def keys(self) -> list:
        return [r.key for r in self.reactions]


class ParseError(ValueError):
    pass


def parse(text: str, system: System) -> Hypothesis:
    known = set(system.species) | set(system.compartments) | set(system.parameters)
    reactions, laws, params = [], [], {}
    body = text
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.S)
    if m:
        body = m.group(1)
    for raw in body.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if line.lower().startswith("params"):
            for k, v in re.findall(r"([A-Za-z_]\w*)\s*=\s*([-+0-9.eE]+)", line.split(":", 1)[-1]):
                params[k] = float(v)
            continue
        if ";" not in line or "->" not in line:
            raise ParseError(f"not a reaction line: {raw.strip()!r}")
        head, law = line.split(";", 1)
        head = head.split(":", 1)[-1] if re.match(r"^\s*\w+\s*:", head) else head
        mods = tuple(s for s in re.findall(r"\[([^\]]*)\]", head)[0].replace(",", " ").split()) if "[" in head else ()
        head = re.sub(r"\[[^\]]*\]", "", head)
        lhs, rhs = head.split("->", 1)
        side = lambda s: tuple(x for term in s.replace("+", " + ").split("+") for x in _expand(term))
        reactants, products = side(lhs), side(rhs)
        for sp in reactants + products + mods:
            if sp not in system.species:
                raise ParseError(f"unknown species {sp!r} in {raw.strip()!r}")
        law = law.strip()
        if not law:
            raise ParseError(f"missing rate law in {raw.strip()!r}")
        if libsbml.parseL3Formula(law) is None:
            raise ParseError(f"rate law does not parse: {law!r}")
        for ident in set(_IDENT.findall(law)):
            if ident not in known and ident not in _FUNCS:
                params.setdefault(ident, 1.0)
        reactions.append(Reaction(f"R{len(reactions) + 1}", reactants, products, mods))
        laws.append(law)
    if not reactions:
        raise ParseError("no reactions")
    return Hypothesis(reactions, laws, params, text=text)


def _expand(term: str) -> list[str]:
    term = term.strip()
    if not term:
        return []
    m = re.match(r"^(\d+)\s*\*?\s*([A-Za-z_]\w*)$", term)
    if m:
        return [m.group(2)] * int(m.group(1))
    return [term]


def build_sbml(h: Hypothesis, system: System, params: dict[str, float] | None = None) -> str:
    doc = libsbml.readSBMLFromString(system.partial)
    model = doc.getModel()
    for name, value in {**h.params, **(params or {})}.items():
        if model.getParameter(name) is None:
            p = model.createParameter()
            p.setId(name)
            p.setConstant(True)
        model.getParameter(name).setValue(float(value))
    for r, law in zip(h.reactions, h.laws):
        rx = model.createReaction()
        rx.setId(r.id)
        rx.setReversible(False)
        if doc.getLevel() >= 3:
            rx.setFast(False)
        for side, items in (("r", r.reactants), ("p", r.products)):
            counts: dict[str, int] = {}
            for s in items:
                counts[s] = counts.get(s, 0) + 1
            for s, n in counts.items():
                ref = rx.createReactant() if side == "r" else rx.createProduct()
                ref.setSpecies(s)
                ref.setStoichiometry(n)
                if doc.getLevel() >= 3:
                    ref.setConstant(True)
        for s in r.modifiers:
            mod = rx.createModifier()
            mod.setSpecies(s)
        kl = rx.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula(law))
    return libsbml.writeSBMLToString(doc)


def _subsample(traj: Trajectory, n: int) -> np.ndarray:
    return np.linspace(0, len(traj.time) - 1, min(n, len(traj.time))).astype(int)


def fit(h: Hypothesis, system: System, observed: list[tuple[Experiment, Trajectory]], max_nfev: int = 40,
        n_starts: int = 3) -> Hypothesis:
    """Fit the rate constants in log space to every observed experiment; attach the
    fitted values, the SBML and the per-experiment SMAPE."""
    names = sorted(h.params)
    base = build_sbml(h, system)
    try:
        rr = roadrunner.RoadRunner(base)
    except Exception:
        h.errors = [1.0] * len(observed)
        return h
    n_fit = min(FIT_STEPS, system.n_steps)
    grid = np.linspace(0, system.t_end, n_fit + 1)
    pick = np.linspace(0, n_fit, min(N_FIT_POINTS, n_fit + 1)).astype(int)
    target = [{s: np.interp(grid, t.time, t.values[s])[pick] for s in system.species} for _, t in observed]
    scale = {s: max(1e-9, float(np.mean([np.abs(t.values[s]).max() for _, t in observed if s in t.values] or [1.0])))
             for s in system.species}

    def residuals(logp):
        params = {n: 10.0 ** v for n, v in zip(names, logp)}
        out = []
        for i, (exp, _) in enumerate(observed):
            sim = simulate(base, exp, system.t_end, n_fit, params, rr=rr)
            for s in system.species:
                if sim is None or s not in sim.values:
                    out.append(np.full(len(pick), 10.0))
                else:
                    out.append((sim.values[s][pick] - target[i][s]) / scale[s])
        return np.concatenate(out)

    start_vals = h.fitted if h.fitted and set(h.fitted) == set(names) else h.params
    if h.fitted and set(h.fitted) == set(names):
        n_starts = 1  # a refit after a new experiment starts from the previous solution
    x0 = np.array([math.log10(min(max(abs(start_vals[n]), 1e-9), 1e9)) for n in names])
    best, best_cost = x0, float(np.sum(residuals(x0) ** 2)) if names else 0.0
    if names:
        # the proposer's values first, then starts spread around the rate scale 1 / t_end and the
        # concentration scale, since a generic start leaves the optimiser on a flat plateau
        rng = np.random.default_rng(len(names))
        conc = math.log10(max(1e-9, float(np.median([v for v in scale.values()]))))
        starts = [x0] + [rng.normal(-math.log10(system.t_end), 1.5, len(names)) for _ in range(n_starts // 2)] \
                 + [rng.normal(conc, 1.5, len(names)) for _ in range(n_starts - n_starts // 2)]
        for start in starts:
            try:
                sol = least_squares(residuals, np.clip(start, -8.9, 8.9), bounds=(-9, 9), max_nfev=max_nfev,
                                    xtol=1e-8, ftol=1e-8, x_scale="jac")
                cost = 2 * float(sol.cost)  # scipy's cost is half the sum of squares
                if np.isfinite(cost) and cost < best_cost:
                    best, best_cost = sol.x, cost
            except Exception as exc:
                h.fit_notes.append(f"{type(exc).__name__}: {str(exc)[:120]}")
                continue
    h.fitted = {n: 10.0 ** v for n, v in zip(names, best)}
    h.sbml = build_sbml(h, system, h.fitted)
    try:
        rr_fit = roadrunner.RoadRunner(h.sbml)  # a fresh instance: the fitted values are baked in
    except Exception:
        rr_fit = None
    h.errors = []
    for exp, obs in observed:
        sim = simulate(h.sbml, exp, system.t_end, system.n_steps, rr=rr_fit) if rr_fit else None
        h.errors.append(1.0 if sim is None else smape(obs, sim, system.species))
    return h


def report(h: Hypothesis, system: System, observed: list[tuple[Experiment, Trajectory]], worst: int = 2) -> str:
    """What the checker tells the synthesizer: the error per experiment and, for
    the worst species, observed against predicted values at five times."""
    lines = []
    rr = roadrunner.RoadRunner(h.sbml) if h.sbml else None
    for (exp, obs), err in zip(observed, h.errors):
        lines.append(f"- {exp.label()}: SMAPE {err:.3f} ({'ok' if err <= EPS else 'mismatch'})")
        if err <= EPS or rr is None:
            continue
        sim = simulate(h.sbml, exp, system.t_end, system.n_steps, rr=rr)
        if sim is None:
            lines.append("  the model could not be simulated under this experiment")
            continue
        per = sorted(((smape(Trajectory(obs.time, {s: obs.values[s]}), Trajectory(sim.time, {s: sim.values[s]}), [s]), s)
                      for s in system.species), reverse=True)[:worst]
        pts = _subsample(obs, 5)
        for e, s in per:
            lines.append(f"  {s}: observed {_fmt(obs.values[s][pts])} predicted {_fmt(sim.values[s][pts])} at t={_fmt(obs.time[pts])}")
    return "\n".join(lines)


def _fmt(a) -> str:
    return "[" + ", ".join(f"{x:.3g}" for x in a) + "]"


def truth_text(system: System) -> str:
    """The reference reactions in the hypothesis format, for checks only."""
    model = libsbml.readSBMLFromString(system.truth).getModel()
    lines, params = [], {}
    for i, r in enumerate(model.getListOfReactions(), 1):
        kl = r.getKineticLaw()
        formula = libsbml.formulaToL3String(kl.getMath()) if kl and kl.isSetMath() else "0"
        for lp in list(kl.getListOfParameters()) + list(kl.getListOfLocalParameters()) if kl else []:
            new = f"{lp.getId()}_r{i}"
            formula = re.sub(rf"\b{re.escape(lp.getId())}\b", new, formula)
            params[new] = lp.getValue()
        side = lambda refs: " + ".join(f"{int(s.getStoichiometry())} {s.getSpecies()}" if s.getStoichiometry() not in (1, 1.0) else s.getSpecies() for s in refs)
        mods = " [" + " ".join(m.getSpecies() for m in r.getListOfModifiers()) + "]" if r.getNumModifiers() else ""
        lines.append(f"R{i}: {side(r.getListOfReactants())} -> {side(r.getListOfProducts())}{mods} ; {formula}")
    for p in model.getListOfParameters():
        if p.getId() not in system.parameters:
            params[p.getId()] = p.getValue()
    lines.append("params: " + ", ".join(f"{k} = {v:.6g}" for k, v in params.items()))
    return "\n".join(lines)
