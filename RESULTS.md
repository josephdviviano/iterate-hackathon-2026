# Results

## R1: arena3, CIFAR-100 speedrun, six framework arms

- **Metric:** mean training time (prepare + train, s) over 3 trials of `uv run python -m benchmark.run --submission arena --n 3`, among experiments with mean accuracy ≥ 0.753 and all trials complete ("feasible").
- **Runs:** one autonomous agent per arm, a single arena run (arena3). The number of experiments differs per arm (32 to 141).
- **Data split:** CIFAR-100 train (50k) for training. Accuracy comes from the benchmark's evaluation on the CIFAR-100 test set (10k), as the organizers' benchmark defines it.
- **Baseline:** `framework/baseline` (karpathy/autoresearch greedy loop), two runs (baseline-r1, baseline-r2).
- **Command:** `python3 demo.py` prints this table from `results/arena3/*.tsv`.
- **Source:** arena3 results at commit 050c5ea, harness at commit 2ef7748.

| Arm | Framework | Experiments | Kept | First feasible | Fastest feasible |
|---|---|---|---|---|---|
| hypothesis | reflective | 141 | 24 | 7.22 s (E001) | 4.19 s (E133), acc 0.7550 |
| hypothesis-dr | reflective + deep research | 84 | 29 | 9.28 s (E001) | 4.57 s (E083), acc 0.7556 |
| baseline-r2 | greedy | 32 | 24 | 23.38 s (E001) | 4.65 s (E030), acc 0.7568 |
| hypothesis-lit | reflective + literature | 72 | 26 | 8.71 s (E007) | 4.82 s (E068), acc 0.7547 |
| hypothesis-unc | reflective + uncertainty committee | 66 | 22 | 9.10 s (E005) | 4.99 s (E052), acc 0.7555 |
| baseline-r1 | greedy | 37 | 27 | 31.90 s (E001) | 5.05 s (E036), acc 0.7559 |

**Caveats.**
- This is one run per arm, so the ranking has no error bars.
- The reflective arms started from a published airbench96-style recipe and the greedy arms from their own first recipe. Compare the final times with that in mind.
- The kept results sit 0.0017 to 0.0038 above the 0.753 gate on 3 trials. The official evaluation uses 40 trials, so recipes at the margin can fall below the 0.75 target there.
