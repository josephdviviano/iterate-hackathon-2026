# World model

_Last updated after: E002_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E002: airbench96-style port (whiten 2x2 stem, groups 128/384/576 x3 conv, BN-GELU, Nesterov SGD lr 9 wd 0.012 bs 1024, LS 0.3, eager fp16 channels_last) at 10 epochs. accuracy 0.7495, time 10.55 s. NOT feasible yet (needs >= 0.753).

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->
- time = mean(prepare+train) over seeds 0,1,2; harness syncs CUDA around them. Trivial recipe costs 0.14 s, so fixed overhead is negligible.
- Accuracy per-seed std for a ResNet9-ish recipe is ~0.25 pp (README L40 calibration, 50 trials); 3-seed mean std ~0.15 pp. Feasibility gate is 0.753 on the 3-seed mean.
- Reference point (README): ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe. Our A100 SXM should be similar or a bit faster.

## Where the cost goes
<!-- what dominates the objective, and why -->
- E001: train ~9.30 s for 9 epochs x 48 steps = 432 steps -> ~21.5 ms/step, ~1.03 s/epoch (eager). prepare 0.07-0.2 s (first trial slower). Eval 0.1 s.
- For a real recipe: GPU training steps (epochs x step cost). Data is 50k uint8 32x32 images (150 MB) -> GPU transfer is ~tens of ms. build() is untimed: compile/warmup there.

## Established facts
<!-- each with the experiments that support it, e.g. "(E003, E007)" -->
- The 9-epoch airbench96-style port lands ~1 pp below the bar on CIFAR-100 without TTA (E001: 0.743).
- Epochs dial near 9-10: +1 epoch = +0.66 pp, +1.1 s (+11.8%) (E001->E002). Extrapolated 0.753 needs ~10.5 epochs (~11.1 s eager) at this recipe.
- Per-seed accuracy spread can be ~1 pp (E001: 0.7355 to 0.7475), but E002 (same seeds) spread only 0.25 pp, so single-trial outliers come from GPU nondeterminism, not the seed. 3-seed mean noise is probably ~0.2-0.4 pp run to run (to be calibrated by I005).

## Lessons from failures
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->

## Dead ends

## Open questions and surprises

## Changelog
<!-- one line per experiment: "E007 (I012, H2.1.3): <result> -> <what changed in this model>" -->
E000 (baseline): template, acc 0.012, time 0.14 s -> harness overhead negligible; need a real recipe first.
E001 (I001, H1.1.1): port at 9 ep: acc 0.743, 9.43 s -> refutes H1.1.1 (accuracy); ~1.03 s/epoch eager; need +1 pp from epochs or recipe.
E002 (I002, H4.2.1): 10 ep: acc 0.7495, 10.55 s -> supports smooth epochs dial (+0.66 pp/epoch, +1.1 s/epoch); ~10.5 epochs would reach the bar.
