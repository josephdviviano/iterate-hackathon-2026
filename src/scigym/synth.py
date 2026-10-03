"""Members are reaction networks proposed by a chat model and repaired against the simulator's report.

Round 1 sends the species, the observed trajectories and a seed hint. Each
later round sends the verifier's report on the previous proposal: the error
per experiment and, where it mismatches, observed against predicted values.
The loop stops at admission or at the round budget and keeps the proposal
with the lowest error. This is the repair loop of `committee.synth_api` with
a tolerance verifier in place of exact replay.
"""

from __future__ import annotations

import time

import numpy as np

from .data import System
from .env import Experiment, Trajectory
from .model import EPS, Hypothesis, ParseError, fit, parse, report

SYSTEM_PROMPT = """\
You are a systems biologist working in a dry lab. A biochemical network has had
all of its reactions removed. You see the species (identifiers are anonymised),
the compartments, and time courses of every species from experiments on the
real system. Propose the reactions that explain every experiment.

Format, one reaction per line, modifiers (enzymes, regulators) in square
brackets, the rate law after a semicolon, parameter values on a last line:

R1: A + B -> C ; k1 * A * B
R2: C -> ; k2 * C
R3: -> A ; k3
R4: S -> P [E] ; V4 * E * S / (K4 + S)
params: k1 = 0.5, k2 = 0.1, k3 = 2, V4 = 1, K4 = 10

Rules:
1. Use only the listed species. Any other identifier in a rate law is a new
   parameter. Give values when you can; a fitter refines them.
2. A species that no reaction produces or consumes stays constant. A species
   that is constant in every experiment is likely an enzyme or an input.
3. Reply with one ```text block holding the reactions and the params line,
   and nothing else.
"""

SEEDS = [
    "Prefer the simplest mass-action network that explains the data.",
    "Consider that some species act as enzymes or regulators: put them in brackets with saturating kinetics.",
    "Consider synthesis from nothing and degradation to nothing for species whose level changes without a visible partner.",
    "Consider reversible pairs: a forward and a backward reaction between two species.",
    "Consider a conversion chain A -> B -> C that follows the order in which species rise and fall.",
    "Consider competition: two reactions that consume the same species.",
    "Consider inhibition: a species that slows a reaction, written as a divisor in its rate law.",
    "Consider the species that stay constant: they are catalysts or inputs, not reactants.",
]

N_POINTS = 11


def render_trajectory(exp: Experiment, traj: Trajectory, species) -> str:
    idx = np.linspace(0, len(traj.time) - 1, min(N_POINTS, len(traj.time))).astype(int)
    head = "time      " + "  ".join(f"{s:>10s}" for s in species)
    rows = ["  ".join([f"{traj.time[i]:<9.4g}"] + [f"{traj.values[s][i]:>10.4g}" for s in species]) for i in idx]
    return f"Experiment: {exp.label()}\n{head}\n" + "\n".join(rows)


def task_message(system: System, observed: list[tuple[Experiment, Trajectory]], seed: str,
                 counterexample: str | None = None) -> str:
    parts = [f"Compartments: {', '.join(system.compartments)}",
             "Species and initial concentrations: " + ", ".join(f"{s} = {system.initial[s]:.4g}" for s in system.species)]
    if system.parameters:
        parts.append("Known parameters: " + ", ".join(f"{k} = {v:.4g}" for k, v in system.parameters.items()))
    parts.append("\n\n".join(render_trajectory(e, t, system.species) for e, t in observed))
    parts.append(f"Hint: {seed}")
    if counterexample:
        parts.append(counterexample)
    parts.append("Propose the reactions.")
    return "\n\n".join(parts)


def synthesize(system: System, observed: list[tuple[Experiment, Trajectory]], backend, seed: str, *,
               rounds: int = 3, temperature: float = 0.7, max_tokens: int = 2000, sample_seed: int = 0,
               counterexample: str | None = None) -> tuple[Hypothesis | None, dict]:
    """The best hypothesis found and a record of the loop."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task_message(system, observed, seed, counterexample)}]
    best: Hypothesis | None = None
    record = {"seed": seed, "rounds": [], "wall_s": 0.0}
    t0 = time.time()
    for r in range(1, rounds + 1):
        try:
            reply = backend(messages, temperature=temperature, max_tokens=max_tokens, seed=sample_seed * 100 + r, logprobs=False)
        except Exception as exc:
            record["rounds"].append({"round": r, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            break
        text = reply.text
        messages.append({"role": "assistant", "content": text})
        try:
            h = parse(text, system)
        except ParseError as exc:
            record["rounds"].append({"round": r, "parse_error": str(exc)[:200]})
            messages.append({"role": "user", "content": f"That did not parse: {exc}. Reply again in the format."})
            continue
        h = fit(h, system, observed)
        rep = report(h, system, observed)
        record["rounds"].append({"round": r, "n_reactions": len(h.reactions), "max_error": h.max_error, "admitted": h.admitted})
        if best is None or h.max_error < best.max_error:
            best = h
        if h.admitted:
            break
        messages.append({"role": "user", "content": f"The simulator ran your network with fitted parameters:\n{rep}\n\n"
                         f"Admission needs SMAPE at or below {EPS} on every experiment. Revise the reactions and reply in the format."})
    record["wall_s"] = round(time.time() - t0, 1)
    return best, record
