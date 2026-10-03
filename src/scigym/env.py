"""The dry lab: simulate a system under a perturbation and return every species' trajectory.

Experiments are the ones SciGym allows: observe, set a species' initial
concentration, knock a species out. The same simulator runs a candidate
model, which is how a hypothesis is checked against what was observed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import roadrunner

roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)  # a stiff candidate is a failed hypothesis, not something to print


@dataclass(frozen=True)
class Experiment:
    kind: str                 # observe | set | knockout
    species: str | None = None
    value: float | None = None

    def label(self) -> str:
        if self.kind == "observe":
            return "baseline, no perturbation"
        if self.kind == "knockout":
            return f"knockout of {self.species} (initial concentration 0, held at 0)"
        return f"initial concentration of {self.species} set to {self.value:.4g}"


@dataclass
class Trajectory:
    time: np.ndarray
    values: dict[str, np.ndarray] = field(default_factory=dict)


def simulate(sbml: str, exp: Experiment, t_end: float, n_steps: int, params: dict[str, float] | None = None,
             rr: roadrunner.RoadRunner | None = None) -> Trajectory | None:
    """None when the model cannot be loaded or integrated (a failed hypothesis, not an error)."""
    try:
        rr = rr or roadrunner.RoadRunner(sbml)
        rr.resetAll()
        for k, v in (params or {}).items():
            rr.setValue(k, v)
        rr.reset()
        if exp.kind in ("set", "knockout"):
            rr.setValue(f"init([{exp.species}])", 0.0 if exp.kind == "knockout" else float(exp.value))
            if exp.kind == "knockout":
                rr.setBoundary(exp.species, True) if hasattr(rr, "setBoundary") else None
            rr.reset()
        res = rr.simulate(0, t_end, n_steps + 1)
        ids = rr.model.getFloatingSpeciesIds() + rr.model.getBoundarySpeciesIds()
        out = Trajectory(np.array(res[:, 0]))
        for i, name in enumerate(res.colnames[1:], 1):
            sid = name.strip("[]")
            if sid in ids:
                out.values[sid] = np.array(res[:, i])
        for sid in ids:  # a boundary species (a knockout) is constant and absent from the output columns
            if sid not in out.values:
                out.values[sid] = np.full(len(out.time), float(rr.getValue(f"[{sid}]")))
        if any(not np.all(np.isfinite(v)) for v in out.values.values()):
            return None
        return out
    except Exception:
        return None


class Environment:
    """The reference model behind a budget of experiments."""

    def __init__(self, system) -> None:
        self.system = system
        self._rr = roadrunner.RoadRunner(system.truth)
        self.log: list[tuple[Experiment, Trajectory]] = []

    def run(self, exp: Experiment) -> Trajectory:
        traj = simulate(self.system.truth, exp, self.system.t_end, self.system.n_steps, rr=self._rr)
        if traj is None:
            raise RuntimeError(f"reference model failed under {exp}")
        self.log.append((exp, traj))
        return traj


def smape(a: Trajectory, b: Trajectory, species) -> float:
    """SciGym's trajectory error: symmetric mean absolute percentage error, mean over species and time."""
    errs = []
    for s in species:
        if s not in a.values or s not in b.values:
            return 1.0
        x, y = a.values[s], b.values[s]
        denom = np.abs(x) + np.abs(y)
        errs.append(np.mean(np.where(denom > 1e-12, np.abs(x - y) / np.maximum(denom, 1e-12), 0.0)))
    return float(np.mean(errs)) if errs else 1.0
