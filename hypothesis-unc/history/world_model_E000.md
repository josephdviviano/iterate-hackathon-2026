# World model

_Last updated after: E000_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E000: template (64 images, 3 SGD steps). accuracy 0.012, time 0.14 s. NOT feasible (needs accuracy >= 0.753).

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->
- time = mean(prepare+train) over seeds 0,1,2; harness syncs CUDA around them. Trivial recipe costs 0.14 s, so fixed overhead is negligible.
- Accuracy per-seed std for a ResNet9-ish recipe is ~0.25 pp (README L40 calibration, 50 trials); 3-seed mean std ~0.15 pp. Feasibility gate is 0.753 on the 3-seed mean.
- Reference point (README): ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe. Our A100 SXM should be similar or a bit faster.

## Where the cost goes
<!-- what dominates the objective, and why -->
- For a real recipe: GPU training steps (epochs x step cost). Data is 50k uint8 32x32 images (150 MB) -> GPU transfer is ~tens of ms. build() is untimed: compile/warmup there.

## Established facts
<!-- each with the experiments that support it, e.g. "(E003, E007)" -->

## Lessons from failures
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->

## Dead ends

## Open questions and surprises

## Changelog
<!-- one line per experiment: "E007 (I012, H2.1.3): <result> -> <what changed in this model>" -->
E000 (baseline): template, acc 0.012, time 0.14 s -> harness overhead negligible; need a real recipe first.
