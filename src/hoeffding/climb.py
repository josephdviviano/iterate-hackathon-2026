"""Outer hill-climbing loop over a population of strategies.

Round 0 is the existing seed runs. Each later round evaluates the population
on the climb instances, keeps the top parents by fitness, and spawns children
whose workspace holds the parents' source, the per-instance best certified
value so far, and the instances on which the parents disagree. Fitness is the
mean fraction of the Bernoulli-to-Hoeffding gap closed, rejected instances
counting zero. The certified value is the only reward.

Layout: artifacts/hoeffding/climb/round<r>/child<k>/{strategy.py, meta.json}
        artifacts/hoeffding/climb/log.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import random
import time as _time

from .committee import HoeffdingCommittee, Member, run_candidates, run_strategy
from .prioritize import Prioritizer, Record
from .tiers import Counters, decide
from .verify import Rejected
from .problem import MS, TRAIN, Instance, grid
from .synth import synthesize_in
from .task import TASKS, Task
from .verify import Certificate

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "hoeffding"
CLIMB_TRAIN = TRAIN + grid([15, 20], MS, [Fraction(1, 2), Fraction(3, 4)])


def climb_dir(task: Task) -> Path:
    return ART / ("climb" if task.name == "hoeffding" else f"climb_{task.name}")


LANES = {
    "refine": "Lane: REFINE. Start from the best parent's law and reduce its finite-ladder loss: more levels, "
              "better level ratios, better weights per level, within the atom cap. Measured: the same structure at "
              "104 atoms certified 0.39919 against 0.39826 at 64, and the certifier handles 104 atoms in about 1 s "
              "and 128 in a few seconds. Use the atom budget. Every 1e-5 of certified gain counts.",
    "deck": "Lane: DECK. Search a different space from the parents: decks. A deck is n card values v_1 <= ... <= v_n, "
            "each with weight 1/n (scale so the largest is 1). Its strict value is the fraction of ordered 4-tuples "
            "(i, j, k, l) with v_i + v_j + v_k < 2 v_l; its tie mass is the fraction with equality. Bellec and Fritz's "
            "perturbation trick turns ties into wins in the limit, so search on strict / (1 - tie) but CERTIFY an explicit "
            "two-level law: atoms v_a + eta v_b with weights 1/n^2 and a small rational eta, which is the law you return. "
            "Search is combinatorial: start from the geometric deck 1, 2, 4, ..., 2^(n-1) (which gives 2/3 on the "
            "two-card game), then change one card at a time, with annealing, over integer values up to 2^24 and n up to 16. "
            "This is the space the proven upper bound lives in. Report the best deck, its strict and tie fractions, and "
            "the certified value of the two-level law.",
    "grid": "Lane: GRID. Search a different space from the parents: arbitrary weights on a dyadic grid. Atoms are "
            "k / 2^m for m in 6..8 (up to 256 atoms), weights are free. Optimize the exact objective, a degree-4 "
            "polynomial in the weights, directly: multiplicative weights or projected gradient from several random "
            "starts, then prune atoms with weight below 1e-6. Do not assume centres, ladders or self-similarity; if the "
            "optimizer rebuilds a ladder on its own, say so, since that is evidence about the problem. Ties at exactly "
            "zero are not counted; if tie mass is large, add a second level with a small rational eta before certifying. "
            "Report the structure the weights took and the certified value.",
    "explore": "Lane: EXPLORE. The parents' structure (centres plus lexicographic tie-breaking ladders, self-similar "
               "at 0) has a limit of 0.400695 and cannot pass it. Look for a law outside that structure: non-lexicographic "
               "tie-breaking, more than two clusters, a different fixed point, or atoms that are not perturbations of "
               "centres. A certified value above 0.400695 is the goal; a certified value below the parents is still "
               "informative if the structure is new. Say what you tried and what it certified.",
}


def climb_instances(task: Task) -> list:
    return CLIMB_TRAIN if task.name == "hoeffding" else task.train


@dataclass
class Scored:
    name: str
    source: str
    results: list          # Certificate, or a float estimate for a skipped candidate, or an error string
    confidences: list
    fitness: float
    gaps: list[float]
    decisions: list | None = None
    lane: str | None = None

    def value(self, i: int):
        """Certified value only. Skipped candidates have none."""
        r = self.results[i]
        return r.value if isinstance(r, Certificate) else None

    def score_value(self, i: int) -> float | None:
        """Certified value, or the float estimate of a skipped candidate (never reported as a bound)."""
        r = self.results[i]
        if isinstance(r, Certificate):
            return float(r.value)
        return r if isinstance(r, float) else None


def _gap(v: float | None, inst, task: Task) -> float:
    if v is None:
        return 0.0
    b, w = task.lower(inst), task.upper(inst)
    if w is None:
        w = 1.0
    return min(1.0, max(0.0, (v - b) / (w - b))) if w > b else 1.0


def score(name: str, source: str, instances: list, timeout_s: float, task: Task,
          incumbents: list | None = None, counters: Counters | None = None,
          audit_rate: float = 0.1, rng: random.Random | None = None, audited_once: set | None = None) -> Scored:
    """Score a member. With `incumbents` and `counters` the two-tier policy decides which candidates to certify.

    `audited_once` is shared across the children of one round: the first skip on each instance in a round is audited."""
    confs: list = []
    raw = run_candidates(source, instances, timeout_s=timeout_s, confidences=confs, task=task)
    results, decisions = [], []
    if audited_once is None:
        audited_once = set()
    rng = rng or random.Random(0)
    for i, (r, inst) in enumerate(zip(raw, instances)):
        if isinstance(r, str):
            results.append(r); decisions.append(None)
            continue
        if counters is None:
            try:
                results.append(task.certify(r["atoms"], r["weights"], inst))
            except (Rejected, ValueError, ZeroDivisionError, OverflowError) as e:
                results.append(f"rejected: {e}")
            decisions.append(None)
            continue
        counters.requested += 1
        t0 = _time.time()
        try:
            est = task.estimate(r["atoms"], r["weights"], inst)
        except (Rejected, ValueError, ZeroDivisionError, OverflowError) as e:
            results.append(f"rejected: {e}"); decisions.append(None)
            continue
        counters.estimate_seconds += _time.time() - t0
        d = decide(est, incumbents[i] if incumbents else None, audit_rate, rng)
        if d.action == "skip" and i not in audited_once:
            d.action = "audit"  # the first skip on each instance is always audited
            audited_once.add(i)
        decisions.append(d)
        if d.action == "skip":
            counters.skipped += 1
            results.append(est.value)
            continue
        t0 = _time.time()
        try:
            cert = task.certify(r["atoms"], r["weights"], inst)
        except (Rejected, ValueError, ZeroDivisionError, OverflowError) as e:
            results.append(f"rejected: {e}")
            continue
        finally:
            counters.exact_seconds += _time.time() - t0
        if d.action == "audit":
            counters.audited += 1
            if float(cert.value) > est.upper:
                counters.audit_disagreements += 1
        else:
            counters.certified += 1
        results.append(cert)
    gaps = [_gap(Scored(name, source, results, confs, 0.0, []).score_value(i), inst, task)
            for i, inst in enumerate(instances)]
    return Scored(name, source, results, confs, sum(gaps) / len(gaps), gaps, decisions)


def incumbents_of(pop: list[Scored], n: int) -> list[float | None]:
    out = []
    for i in range(n):
        vals = [s.value(i) for s in pop if s.value(i) is not None]
        out.append(float(max(vals)) if vals else None)
    return out


def best_table(pop: list[Scored], instances: list[Instance]) -> list[dict]:
    rows = []
    for i, inst in enumerate(instances):
        vals = [(s.value(i), s.name) for s in pop if s.value(i) is not None]
        best = max(vals, key=lambda v: v[0]) if vals else (None, None)
        distinct = sorted({float(v) for v, _ in vals})
        rows.append({"key": inst.key, "best": float(best[0]) if best[0] is not None else None, "by": best[1],
                     "n_distinct": len(distinct), "spread": (distinct[-1] - distinct[0]) if distinct else None,
                     "n_rejected": sum(s.value(i) is None for s in pop)})
    return rows


def population_block(parents: list[Scored], table: list[dict]) -> str:
    out = ["# Population so far\n",
           "You start from the parents below. Beat the best certified value on any instance, "
           "or match it on every instance with a simpler or faster rule. A strategy that is rejected "
           "or times out on an instance scores zero there.\n"]
    out.append("## Best certified value per train instance (which parent)\n")
    out.append("```")
    for r in table:
        out.append(f"{r['key']:24s} {r['best'] if r['best'] is not None else 'none':>10} by {r['by']}  "
                   f"distinct values {r['n_distinct']}  rejected {r['n_rejected']}")
    out.append("```\n")
    dis = [r for r in table if r["n_distinct"] > 1 or r["n_rejected"] > 0]
    if dis:
        out.append("## Instances where the parents disagree or fail. Start here.\n")
        out.append("```")
        for r in dis:
            out.append(f"{r['key']:24s} spread {r['spread']}  rejected {r['n_rejected']}")
        out.append("```\n")
    else:
        out.append("## The parents agree on every instance. A win needs a law outside their search space.\n")
    for p in parents:
        out.append(f"## Parent `{p.name}`, fitness {p.fitness:.4f}\n")
        out.append("```python\n" + p.source.strip() + "\n```\n")
    return "\n".join(out)


def load_seed_population(instances: list, timeout_s: float, task: Task) -> list[Scored]:
    pop = []
    seed_dir = ART / "synth" if task.name == "hoeffding" else ART / f"synth_{task.name}"
    for p in sorted(seed_dir.glob("*/run*/strategy.py")):
        name = f"seed:{p.parent.parent.name}/{p.parent.name}"
        pop.append(score(name, p.read_text(), instances, timeout_s, task))
        print(f"  {name}: fitness {pop[-1].fitness:.4f}")
    if not pop:
        pop.append(score("seed:stub", task.stub, instances, timeout_s, task))
        print(f"  seed:stub: fitness {pop[-1].fitness:.4f}")
    return pop


def climb(rounds: int, children: int, top: int, model: str, max_turns: int, timeout_s: float,
          eval_timeout_s: float, parallel: int, resume: bool, task: Task, tiers: bool = False,
          audit_rate: float = 0.1, checker_policy: bool = True, tag: str = "", lanes: list | None = None,
          stall_rounds: int = 0, stall_gain: float = 1e-5, prioritize: bool = False, interleave: bool = False) -> None:
    CLIMB = climb_dir(task)
    counters = Counters() if tiers else None
    prio_path = CLIMB / "prioritizer.json"
    prio = Prioritizer.from_json(json.loads(prio_path.read_text())) if (prioritize and resume and prio_path.exists()) else Prioritizer()
    CLIMB.mkdir(parents=True, exist_ok=True)
    instances = climb_instances(task)
    log_path = CLIMB / "log.json"
    log = json.loads(log_path.read_text()) if (resume and log_path.exists()) else {"rounds": []}
    print(f"round 0: scoring seed population on {len(instances)} instances")
    pop = load_seed_population(instances, eval_timeout_s, task)
    if resume:
        for p in sorted(CLIMB.glob("round*/child*/strategy.py")):
            name = f"{p.parent.parent.name}/{p.parent.name}"
            pop.append(score(name, p.read_text(), instances, eval_timeout_s, task))
            meta_p = p.parent / "meta.json"
            pop[-1].lane = json.loads(meta_p.read_text()).get("lane") if meta_p.exists() else None
            print(f"  {name}: fitness {pop[-1].fitness:.4f}")
    start_round = len(log["rounds"]) + 1 if resume else 1
    if not resume:
        log["rounds"].append(round_summary(0, pop, instances, task))
        log_path.write_text(json.dumps(log, indent=1))
    for r in range(start_round, start_round + rounds):
        ranked = sorted(pop, key=lambda s: -s.fitness)
        parents = ranked[:top]
        if lanes:
            # one parent per lane when available, so the explore lane is not crowded out by refine's higher fitness
            for lane in lanes:
                best_lane = next((s for s in ranked if getattr(s, "lane", None) == lane), None)
                if best_lane is not None and best_lane not in parents:
                    parents.append(best_lane)
        table = best_table(pop, instances)
        block = population_block(parents, table)
        round_audits: set = set()
        plan: list[dict] = []
        if prioritize and lanes:
            # distinct parents by certified value, top 3, scored against both lanes
            cands, seen_vals = [], set()
            for s_ in ranked:
                v = s_.value(0)
                key = round(float(v), 9) if v is not None else None
                if key in seen_vals:
                    continue
                seen_vals.add(key); cands.append(s_)
                if len(cands) == 3:
                    break
            p_of = lambda s_: (s_.confidences[0] if s_.confidences and s_.confidences[0] is not None else None)
            policy = "random" if (interleave and r % 2 == 0) else "prioritized"
            plan = (prio.propose_random if policy == "random" else prio.propose)(cands, lanes, children, random.Random(r), p_of)
            for d in plan:
                d["policy"] = policy
            print(f"round {r}: plan [{policy}] " + "; ".join(f"{d['parent'].name}+{d['lane']} p={d['p_success']:.2f} gain={d['predicted_gain']:.1e} room={d['room']:.2f}" for d in plan))
        print(f"round {r}: parents {[p.name for p in parents]} fitness {[round(p.fitness, 4) for p in parents]}")

        def job(k: int) -> Scored:
            if plan:
                lane, chosen_parent = plan[k]["lane"], plan[k]["parent"]
            else:
                lane, chosen_parent = (lanes[k % len(lanes)] if lanes else None), None
            out = CLIMB / f"round{r}" / f"child{tag}{k}"
            out.mkdir(parents=True, exist_ok=True)
            seed_text = block + ("\n\n# Your lane\n\n" + LANES[lane] + "\n" if lane else "")
            if chosen_parent is not None:
                seed_text += f"\nYour assigned parent is `{chosen_parent.name}`. Start from its law.\n"
                child_name = f"round{r}/child{tag}{k}"
                pv = chosen_parent.value(0)
                prio.record(Record(r, child_name, lane, chosen_parent.name, float(pv) if pv is not None else 0.0,
                                   plan[k]["p_success"], plan[k]["predicted_gain"], policy=plan[k]["policy"]))
            res = synthesize_in(instances, seed=seed_text, model=model, max_turns=max_turns, timeout_s=timeout_s,
                                task=task, checker_policy=checker_policy)
            (out / "strategy.py").write_text(res.source)
            s = score(f"round{r}/child{tag}{k}", res.source, instances, eval_timeout_s, task,
                      incumbents=incumbents_of(pop, len(instances)) if tiers else None,
                      counters=counters, audit_rate=audit_rate, rng=random.Random(1000 * r + k),
                      audited_once=round_audits)
            s.lane = lane
            (out / "meta.json").write_text(json.dumps(dict(res.meta, fitness=s.fitness, gaps=s.gaps, lane=lane,
                                                           parents=[p.name for p in parents],
                                                           check_output=res.check_output), indent=1))
            print(f"  round{r}/child{tag}{k}: fitness {s.fitness:.4f}, {res.meta.get('num_turns')} turns, "
                  f"{res.meta.get('wall_s')} s, ${res.meta.get('total_cost_usd')}, checker runs "
                  f"{res.meta.get('check_runs')}, totals {res.meta.get('check_totals')}")
            return s

        with ThreadPoolExecutor(max_workers=parallel) as ex:
            new = list(ex.map(job, range(children)))
        pop.extend(new)
        summary = round_summary(r, pop, instances, task)
        if counters is not None:
            summary["tiers"] = counters.to_json()
            print(f"  tiers: {counters.to_json()}")
        if plan:
            for s_ in new:
                v = s_.value(0)
                rec = prio.resolve(s_.name, float(v) if v is not None else None)
                if rec:
                    print(f"  {s_.name} [{rec.lane} from {rec.parent}]: gain {rec.gain:+.2e} success={rec.success} (predicted p={rec.p_success:.2f})")
            rep = prio.report()
            summary["prioritizer"] = {k: v for k, v in rep.items() if k != "records"}
            prio_path.write_text(json.dumps(rep, indent=1))
            print(f"  success rate {rep['success_rate']}, by lane " + ", ".join(f"{l}: {d['successes']}/{d['attempts']}" for l, d in rep["by_lane"].items()) + f", brier {rep['brier']}")
            arms = rep["arms"]
            print("  arms: " + "; ".join(f"{a}: {d['children']} children, {d['successes']} successes, mean gain {d['mean_gain']}" for a, d in arms.items()))
        log["rounds"].append(summary)
        log_path.write_text(json.dumps(log, indent=1))
        if stall_rounds:
            hist = [x["best_fitness"] for x in log["rounds"]]
            if len(hist) > stall_rounds and hist[-1] - hist[-1 - stall_rounds] < stall_gain:
                print(f"stop: best fitness gained less than {stall_gain} over {stall_rounds} rounds")
                break


def round_summary(r: int, pop: list[Scored], instances: list, task: Task) -> dict:
    table = best_table(pop, instances)
    members = [Member(s.name, s.source, s.results, s.confidences) for s in pop]
    com = HoeffdingCommittee(members, task=task)
    reps = com.reports(instances)
    best = max(pop, key=lambda s: s.fitness)
    return {
        "round": r,
        "population": len(pop),
        "best_member": best.name,
        "best_fitness": round(best.fitness, 5),
        "fitness": {s.name: round(s.fitness, 5) for s in pop},
        "mean_best_value": sum(x["best"] or 0.0 for x in table) / len(table),
        "instances_with_disagreement": sum(x["n_distinct"] > 1 for x in table),
        "instances_with_rejection": sum(x["n_rejected"] > 0 for x in table),
        "mean_disagreement": round(sum(x["disagreement"] for x in reps) / len(reps), 4),
        "best_by": {x["key"]: x["by"] for x in table},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--children", type=int, default=2)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--eval-timeout", type=float, default=600)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--task", choices=list(TASKS), default="hoeffding")
    ap.add_argument("--tiers", action="store_true", help="two-tier verification: certify only candidates that can win")
    ap.add_argument("--audit-rate", type=float, default=0.1)
    ap.add_argument("--no-checker-policy", action="store_true", help="control arm: the child's checker certifies every evaluation")
    ap.add_argument("--tag", default="", help="child name prefix, to run two arms into the same round directory")
    ap.add_argument("--lanes", nargs="*", default=None, choices=list(LANES), help="child k gets lane k mod len(lanes)")
    ap.add_argument("--stall-rounds", type=int, default=0, help="stop when best fitness gains < --stall-gain over this many rounds")
    ap.add_argument("--stall-gain", type=float, default=1e-5)
    ap.add_argument("--atom-cap", type=int, default=256, help="linear1 only")
    ap.add_argument("--prioritize", action="store_true", help="choose (parent, lane) per child by expected certified gain; track success rate")
    ap.add_argument("--interleave-random", action="store_true", help="control: even rounds assign (parent, lane) at random from the same proposal set")
    a = ap.parse_args()
    task = TASKS[a.task](a.atom_cap) if a.task == "linear1" else TASKS[a.task]()
    climb(a.rounds, a.children, a.top, a.model, a.max_turns, a.timeout, a.eval_timeout, a.parallel, a.resume,
          task, tiers=a.tiers, audit_rate=a.audit_rate, checker_policy=not a.no_checker_policy, tag=a.tag,
          lanes=a.lanes, stall_rounds=a.stall_rounds, stall_gain=a.stall_gain, prioritize=a.prioritize,
          interleave=a.interleave_random)


if __name__ == "__main__":
    main()
