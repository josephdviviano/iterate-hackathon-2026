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

from .committee import HoeffdingCommittee, Member, run_strategy
from .problem import MS, TRAIN, Instance, bernoulli_value, grid, hoeffding_bound
from .synth import SynthResult, _write_workspace, synthesize_in
from .verify import Certificate

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "hoeffding"
CLIMB = ART / "climb"
CLIMB_TRAIN = TRAIN + grid([15, 20], MS, [Fraction(1, 2), Fraction(3, 4)])


@dataclass
class Scored:
    name: str
    source: str
    results: list
    confidences: list
    fitness: float
    gaps: list[float]

    def value(self, i: int):
        r = self.results[i]
        return r.value if isinstance(r, Certificate) else None


def score(name: str, source: str, instances: list[Instance], timeout_s: float) -> Scored:
    confs: list = []
    results = run_strategy(source, instances, timeout_s=timeout_s, confidences=confs)
    gaps = []
    for r, inst in zip(results, instances):
        if isinstance(r, Certificate):
            b, w = float(bernoulli_value(inst)), hoeffding_bound(inst)
            gaps.append(min(1.0, max(0.0, (float(r.value) - b) / (w - b))) if w > b else 1.0)
        else:
            gaps.append(0.0)
    return Scored(name, source, results, confs, sum(gaps) / len(gaps), gaps)


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


def load_seed_population(instances: list[Instance], timeout_s: float) -> list[Scored]:
    pop = []
    for p in sorted((ART / "synth").glob("*/run*/strategy.py")):
        name = f"seed:{p.parent.parent.name}/{p.parent.name}"
        pop.append(score(name, p.read_text(), instances, timeout_s))
        print(f"  {name}: fitness {pop[-1].fitness:.4f}")
    return pop


def climb(rounds: int, children: int, top: int, model: str, max_turns: int, timeout_s: float,
          eval_timeout_s: float, parallel: int, resume: bool) -> None:
    CLIMB.mkdir(parents=True, exist_ok=True)
    instances = CLIMB_TRAIN
    log_path = CLIMB / "log.json"
    log = json.loads(log_path.read_text()) if (resume and log_path.exists()) else {"rounds": []}
    print(f"round 0: scoring seed population on {len(instances)} instances")
    pop = load_seed_population(instances, eval_timeout_s)
    if resume:
        for p in sorted(CLIMB.glob("round*/child*/strategy.py")):
            name = f"{p.parent.parent.name}/{p.parent.name}"
            pop.append(score(name, p.read_text(), instances, eval_timeout_s))
            print(f"  {name}: fitness {pop[-1].fitness:.4f}")
    start_round = len(log["rounds"]) + 1 if resume else 1
    if not resume:
        log["rounds"].append(round_summary(0, pop, instances))
        log_path.write_text(json.dumps(log, indent=1))
    for r in range(start_round, start_round + rounds):
        parents = sorted(pop, key=lambda s: -s.fitness)[:top]
        table = best_table(pop, instances)
        block = population_block(parents, table)
        print(f"round {r}: parents {[p.name for p in parents]} fitness {[round(p.fitness, 4) for p in parents]}")

        def job(k: int) -> Scored:
            out = CLIMB / f"round{r}" / f"child{k}"
            out.mkdir(parents=True, exist_ok=True)
            res = synthesize_in(instances, seed=block, model=model, max_turns=max_turns, timeout_s=timeout_s)
            (out / "strategy.py").write_text(res.source)
            s = score(f"round{r}/child{k}", res.source, instances, eval_timeout_s)
            (out / "meta.json").write_text(json.dumps(dict(res.meta, fitness=s.fitness, gaps=s.gaps,
                                                           parents=[p.name for p in parents],
                                                           check_output=res.check_output), indent=1))
            print(f"  round{r}/child{k}: fitness {s.fitness:.4f}, {res.meta.get('num_turns')} turns, "
                  f"{res.meta.get('wall_s')} s, ${res.meta.get('total_cost_usd')}")
            return s

        with ThreadPoolExecutor(max_workers=parallel) as ex:
            new = list(ex.map(job, range(children)))
        pop.extend(new)
        log["rounds"].append(round_summary(r, pop, instances))
        log_path.write_text(json.dumps(log, indent=1))


def round_summary(r: int, pop: list[Scored], instances: list[Instance]) -> dict:
    table = best_table(pop, instances)
    members = [Member(s.name, s.source, s.results, s.confidences) for s in pop]
    com = HoeffdingCommittee(members)
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
    a = ap.parse_args()
    climb(a.rounds, a.children, a.top, a.model, a.max_turns, a.timeout, a.eval_timeout, a.parallel, a.resume)


if __name__ == "__main__":
    main()
