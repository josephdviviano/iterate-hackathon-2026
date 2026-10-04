# Reflexive Autoresearch

Track 1.1 (Autoresearch framework), shown on the Track 1.2 CIFAR-100 speedrun task.

**One sentence:** an autoresearch agent that acts like a scientist rather than a random searcher. It keeps a tree of hypotheses and a written world model, pre-registers a prediction before each experiment, and scores itself afterwards. A committee of independent world models measures where it is uncertain instead of asserting it.

## Problem

Greedy autoresearch loops (karpathy/autoresearch: try a change, keep it if the metric improves, else reset) have no memory of *why* something worked. They cannot tell a real effect from noise, and they do not know when they are wrong. We want an agent that reasons about its own experiments and flags what it does not know.

## Approach

The harness keeps three concerns apart, so a comparison between frameworks measures the framework and nothing else:

```
framework/
  core/ar.py              runs and measures experiments for any framework, from task.json
  baseline/program.md     upstream greedy loop (try, keep if better, else reset)
  hypothesis/             reflexive loop: hypothesis tree, idea queue, world model,
                          pre-registration, meta-steps (ideate / revise) while runs are in flight
  hypothesis-lit/         + asynchronous literature feed (arXiv, OpenAlex)
  hypothesis-dr/          + forced deep-research step
  hypothesis-unc/         + uncertainty committee (committee.py): independent world models
                          forecast an 80% range per metric before each run, are scored after,
                          flag counterexamples, and drive calibration reports
tasks/<name>/             the problem only: goal, editable paths, run command, metric regexes,
                          objective, constraints
arena/                    builds one workspace per arm, runs one autonomous Claude Code agent per
                          arm with an identical prompt, and compares arms with an isolation audit
```

`tests/` checks that no framework file contains task knowledge, and that the programs share every section except the research method.

## Demo

```bash
python3 demo.py            # arena3 comparison from the pre-computed results (no GPU needed)
```

The full run needs a GPU machine with the CIFAR-100 speedrun benchmark repository. Build an arm with `python3 arena/make_workspace.py <dest> --framework hypothesis --task tasks/cifar100 --repo <benchmark repo> --prepare tasks/cifar100/prepare_workspace.sh` (set `CIFAR100_DATA` first). Start it with `arena/run_arm.sh <arena dir> <arm>`.

## Results

Arena3: six arms, one autonomous agent each, on the CIFAR-100 speedrun task. The objective is to minimise mean training time subject to mean accuracy ≥ 0.753 over 3 trials. See [RESULTS.md](RESULTS.md) for the full table, method and caveats.

| Arm | Experiments | Fastest feasible |
|---|---|---|
| hypothesis (reflexive) | 141 | 4.19 s, acc 0.7550 |
| hypothesis-dr | 84 | 4.57 s, acc 0.7556 |
| baseline-r2 (greedy) | 32 | 4.65 s, acc 0.7568 |
| hypothesis-lit | 72 | 4.82 s, acc 0.7547 |
| hypothesis-unc | 66 | 4.99 s, acc 0.7555 |
| baseline-r1 (greedy) | 37 | 5.05 s, acc 0.7559 |

## Credits

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch): the greedy loop that `framework/baseline` follows, and the original idea.
- CIFAR-100 speedrun benchmark (MIT), provided by the organizers: the task, harness and scoring that the `tasks/cifar100` spec drives.
- [airbench](https://github.com/KellerJordan/cifar10-airbench) (MIT, Keller Jordan) and the futurebiohackers airbench96-style CIFAR-100 recipe: the agents' starting recipes on the speedrun task.
- Claude Code with Claude Opus 5.5 (Anthropic): the autonomous research agents and the meta-step LLM calls.
- arXiv API and OpenAlex API: the literature feed.
- PyTorch, torchvision and uv: the speedrun task's environment.
- [ONC-AGI](https://github.com/BradSegal/ONC-AGI) (BSD 3-Clause, Bradley Segal): ARC-style biomarker-discovery worlds with planted mechanisms, kept as a read-only reference in `external/onc-agi`.
