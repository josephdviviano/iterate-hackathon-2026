"""Print the arena3 comparison from the pre-computed results in results/arena3/.

    python3 demo.py

Each arm is one autonomous agent on the CIFAR-100 speedrun task (objective: minimise training
time subject to mean accuracy >= 0.753). Arms differ only in the research framework.
"""

import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {
    "baseline-r1": "greedy loop (karpathy/autoresearch), run 1",
    "baseline-r2": "greedy loop (karpathy/autoresearch), run 2",
    "hypothesis": "reflective: hypotheses + world model + pre-registration",
    "hypothesis-lit": "reflective + literature feed",
    "hypothesis-dr": "reflective + forced deep research",
    "hypothesis-unc": "reflective + uncertainty committee",
}


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "results", "arena3", "*.tsv"))):
        arm = os.path.basename(path)[:-4]
        runs = load(path)
        feas = [r for r in runs if r["feasible"] == "yes" and r["status"] != "crash"]
        kept = [r for r in runs if r["status"] == "keep"]
        first = min(feas, key=lambda r: int(r["exp"][1:])) if feas else None
        best = min(feas, key=lambda r: float(r["time"])) if feas else None
        rows.append((arm, len(runs), len(kept), first, best))
    rows.sort(key=lambda r: float(r[4]["time"]) if r[4] else 1e9)

    print(f"{'arm':<15} {'exps':>4} {'kept':>4} {'first feasible':>16} {'fastest feasible':>18}  framework")
    for arm, n, k, first, best in rows:
        f = f"{float(first['time']):.2f}s ({first['exp']})" if first else "-"
        b = f"{float(best['time']):.2f}s ({best['exp']})" if best else "-"
        acc = f" acc {float(best['accuracy']):.4f}" if best else ""
        print(f"{arm:<15} {n:>4} {k:>4} {f:>16} {b:>18}  {ARMS.get(arm, '')}{acc}")


if __name__ == "__main__":
    main()
