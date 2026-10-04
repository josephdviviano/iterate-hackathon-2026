"""Three arms on the same systems, the same experiment budget and the same model.

    uv run python -m scigym.loop --arm committee_probe --model gptoss --n-systems 30
    uv run python -m scigym.loop --arm committee_fixed --model gptoss --n-systems 30
    uv run python -m scigym.loop --arm single_fixed --model gptoss --n-systems 30

committee_probe: K members; each experiment is the candidate on which admitted
members disagree most; refuted members are resynthesized on the counterexample.
committee_fixed: the same committee with experiments in a fixed order.
single_fixed: one member, the fixed order, repaired on each new experiment.
One JSON per system under artifacts/scigym/<arm>_<model>/; a re-run skips
systems already stored.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .committee import candidates, choose_probe, heldout, predictions, refute, rms, ste, vote
from .data import System, load_systems
from .env import Environment, Experiment
from .model import EPS, fit
from .synth import SEEDS, render_trajectory, synthesize

ART = Path(__file__).resolve().parents[2] / "artifacts" / "scigym"
ARMS = ("committee_probe", "committee_fixed", "single_fixed", "committee_probe_nocx")


def chooses_probe(arm: str) -> bool:
    """The committee arms that pick the experiment where admitted members' predictions diverge most."""
    return arm.startswith("committee_probe")


def uses_counterexample(arm: str) -> bool:
    """The `_nocx` arms resynthesize a refuted member on the data alone, with no counterexample text:
    the control that separates the observations from the statement."""
    return not arm.endswith("_nocx")
MAX_TOKENS = {"gptoss": 6000}


def counterexample_text(member, system: System, exp: Experiment, obs) -> str:
    pred = predictions([member], system, exp)[0]
    lines = ["A previous hypothesis was refuted by the last experiment:",
             "\n".join(f"  {r.id}: {' + '.join(r.reactants)} -> {' + '.join(r.products)}" for r in member.reactions),
             f"Under '{exp.label()}' it predicted:" if pred is not None else f"Under '{exp.label()}' it could not be simulated.",
             render_trajectory(exp, pred, system.species) if pred is not None else "",
             "The observation is the last experiment above. Propose a network that explains it as well."]
    return "\n".join(l for l in lines if l)


def run_system(system: System, arm: str, model: str, k: int, budget: int, rounds: int, max_calls: int) -> dict:
    from bioprot.backends import make_backend

    backend = make_backend(model)
    max_tokens = MAX_TOKENS.get(model, 2000)
    env = Environment(system)
    observed = [(Experiment("observe"), env.run(Experiment("observe")))]
    cands = candidates(system, observed[0][1])
    used: set = set()
    calls = 0
    members, records = [], []
    t0 = time.time()

    def synth(seed_i: int, counterexample=None):
        nonlocal calls
        h, rec = synthesize(system, observed, backend, SEEDS[seed_i % len(SEEDS)], rounds=rounds, max_tokens=max_tokens,
                            sample_seed=seed_i, counterexample=counterexample)
        calls += len(rec["rounds"])
        records.append(rec)
        return h

    n_members = 1 if arm == "single_fixed" else k
    for i in range(n_members):
        h = synth(i)
        if h is not None:
            members.append(h)
    steps = []
    infeasible: list[str] = []
    t = 0
    while t < budget:
        pool = [m for m in members if m.admitted] or members
        if chooses_probe(arm) and len(pool) > 1:
            exp, score = choose_probe(pool, system, cands, used)
        else:
            exp = next((c for c in cands if c not in used), None)
            score = None
        if exp is None:
            break
        used.add(exp)
        try:
            traj = env.run(exp)
        except RuntimeError:  # the reference model cannot be integrated there: an infeasible experiment, not a result
            infeasible.append(exp.label())
            continue
        observed.append((exp, traj))
        # refit keeps the structure and refreshes the constants; a member is refuted when its
        # mechanism cannot reproduce the new experiment even after that
        for m in members:
            fit(m, system, observed)
        ok = refute(members, system, exp, traj)
        refuted = [m for m, good in zip(members, ok) if not good]
        before = sum(ok)
        survivors = [m for m, good in zip(members, ok) if good]
        new = []
        for j, m in enumerate(refuted):
            if calls >= max_calls:
                break
            h = synth(len(members) * (t + 1) + j, counterexample_text(m, system, exp, traj) if uses_counterexample(arm) else None)
            if h is not None:
                new.append(h)
        members = survivors + new + [m for m in refuted if calls >= max_calls and m not in refuted[:len(new)]]
        v = vote([m for m in members if m.admitted] or members)
        plan = ([m for m in members if m.admitted] or members)[v["medoid"]] if v["medoid"] is not None else None
        steps.append({"t": t + 1, "experiment": exp.label(), "disagreement_at_choice": score, "survivors": before,
                      "refuted": len(refuted), "resynthesized": len(new), "n_admitted": sum(m.admitted for m in members),
                      "spread": v["spread"], "rms_medoid_f1": rms(plan.keys, system.truth_reactions)["f1"] if plan else None,
                      "rms_majority_f1": rms(v["majority"], system.truth_reactions)["f1"], "calls": calls})
        t += 1
    final_pool = [m for m in members if m.admitted] or members
    v = vote(final_pool)
    plan = final_pool[v["medoid"]] if v["medoid"] is not None else None
    held = heldout(system)
    return {"system": system.id, "arm": arm, "model": model, "k": n_members, "budget": budget, "rounds": rounds,
            "n_species": len(system.species), "n_truth_reactions": len(system.truth_reactions),
            "n_members": len(members), "n_admitted": sum(m.admitted for m in members), "any_admitted": any(m.admitted for m in members),
            "rms_medoid": rms(plan.keys, system.truth_reactions) if plan else None,
            "rms_majority": rms(v["majority"], system.truth_reactions),
            "rms_best_member": max((rms(m.keys, system.truth_reactions)["f1"] for m in members), default=None),
            "ste_medoid": ste(plan.sbml, system, held) if plan else 1.0,
            "ste_partial": ste(system.partial, system, held),
            "spread": v["spread"], "n_distinct": v["n_distinct"], "steps": steps, "calls": calls,
            "wall_s": round(time.time() - t0, 1), "experiments": [e.label() for e, _ in observed], "infeasible": infeasible,
            "members": [{"text": m.text[-1500:], "keys": [list(map(list, kk)) for kk in m.keys], "errors": m.errors, "admitted": m.admitted} for m in members],
            "synthesis": records}


def _job(args):
    system, arm, model, k, budget, rounds, max_calls = args
    try:
        return run_system(system, arm, model, k, budget, rounds, max_calls)
    except Exception as exc:
        import traceback
        return {"system": system.id, "arm": arm, "model": model, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--model", default="gptoss")
    ap.add_argument("--n-systems", type=int, default=30)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-calls", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="system ids (smoke test)")
    a = ap.parse_args()
    out = ART / f"{a.arm}_{a.model}"
    out.mkdir(parents=True, exist_ok=True)
    systems = load_systems(limit=a.n_systems)
    if a.only:
        systems = [s for s in load_systems() if s.id in a.only]
    todo = [s for s in systems if not (out / f"{s.id}.json").exists() or "error" in json.loads((out / f"{s.id}.json").read_text())]
    print(f"{a.arm} {a.model}: {len(systems) - len(todo)} stored, {len(todo)} to run", flush=True)
    jobs = [(s, a.arm, a.model, a.k, a.budget, a.rounds, a.max_calls) for s in todo]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for n, fut in enumerate(as_completed([ex.submit(_job, j) for j in jobs]), 1):
            r = fut.result()
            (out / f"{r['system']}.json").write_text(json.dumps(r, indent=1, default=float))
            if "error" in r:
                print(f"{r['system']}: ERROR {r['error'][:120]}", flush=True)
            else:
                print(f"{r['system']}: admitted {r['n_admitted']}/{r['n_members']} rms_f1 medoid {r['rms_medoid']['f1'] if r['rms_medoid'] else None:.2f} "
                      f"majority {r['rms_majority']['f1']:.2f} ste {r['ste_medoid']:.3f} (partial {r['ste_partial']:.3f}) calls {r['calls']} {r['wall_s']}s "
                      f"[{n}/{len(jobs)} in {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
