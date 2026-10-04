# World model

_Last updated after: E005_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E005 (FEASIBLE): E004 + airbench lookahead EMA (every 5 steps, alpha = 0.95^5 * (t/T)^3, final weights = EMA). accuracy 0.7583, time 9.10 s. Margin above 0.753: +0.53 pp.
E004 (was): airbench96-style port (whiten 2x2 stem, groups 128/384/576 x3 conv, BN-GELU, Nesterov SGD lr 9 wd 0.012 bs 1024, LS 0.3, fp16 channels_last) at 10 epochs, train step torch.compile'd (max-autotune-no-cudagraphs) + fused SGD, eager eval. accuracy 0.7486, time 9.04 s. NOT feasible yet (needs >= 0.753; ~+0.45 pp short).

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->
- Compiled training is bit-deterministic per seed (E003 == E004 per-seed accuracies). So a byte-identical re-run measures nothing; the relevant noise is how per-seed accuracy varies under any recipe perturbation (~0.2-0.6 pp per seed, i.e. ~0.15-0.35 pp on the 3-seed mean).
- Trial 0 prepare is ~0.15 s slower than later trials (first-use allocation), ~1.7% of 9 s.
- time = mean(prepare+train) over seeds 0,1,2; harness syncs CUDA around them. Trivial recipe costs 0.14 s, so fixed overhead is negligible.
- Accuracy per-seed std for a ResNet9-ish recipe is ~0.25 pp (README L40 calibration, 50 trials); 3-seed mean std ~0.15 pp. Feasibility gate is 0.753 on the 3-seed mean.
- Reference point (README): ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe. Our A100 SXM should be similar or a bit faster.

## Where the cost goes
<!-- what dominates the objective, and why -->
- Compiled (inductor max-autotune-no-cudagraphs) + fused SGD: steady-state train 8.92 s vs 10.39 s eager at 10 epochs (-14%, ~18.6 ms/step) (E003 trials 1-2).
- E001: train ~9.30 s for 9 epochs x 48 steps = 432 steps -> ~21.5 ms/step, ~1.03 s/epoch (eager). prepare 0.07-0.2 s (first trial slower). Eval 0.1 s.
- For a real recipe: GPU training steps (epochs x step cost). Data is 50k uint8 32x32 images (150 MB) -> GPU transfer is ~tens of ms. build() is untimed: compile/warmup there.

## Established facts
<!-- each with the experiments that support it, e.g. "(E003, E007)" -->
- Lookahead EMA is worth ~+1 pp at 10 epochs for +0.6% time (E005 vs E004; every seed +0.9-1.15 pp). End-of-training noise is a big lever on CIFAR-100 short runs: one epoch buys ~0.66 pp, so EMA is worth ~1.5 epochs (~1.3 s).
- The 9-epoch airbench96-style port lands ~1 pp below the bar on CIFAR-100 without TTA (E001: 0.743).
- Epochs dial near 9-10: +1 epoch = +0.66 pp, +1.1 s (+11.8%) (E001->E002). Extrapolated 0.753 needs ~10.5 epochs (~11.1 s eager) at this recipe.
- Per-seed accuracy spread can be ~1 pp (E001: 0.7355 to 0.7475), but E002 (same seeds) spread only 0.25 pp, so single-trial outliers come from GPU nondeterminism, not the seed. 3-seed mean noise is probably ~0.2-0.4 pp run to run (to be calibrated by I005).

## Lessons from failures
- torch.compile guards on strides/layout, not just shape: a warmup with contiguous NCHW input while training uses channels_last recompiled inside trial 0 (+9 s). Warm up via the same data path; verify with TORCH_LOGS=recompiles on random data first (E003).
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->

## Dead ends

## Open questions and surprises

## Changelog
<!-- one line per experiment: "E007 (I012, H2.1.3): <result> -> <what changed in this model>" -->
E000 (baseline): template, acc 0.012, time 0.14 s -> harness overhead negligible; need a real recipe first.
E001 (I001, H1.1.1): port at 9 ep: acc 0.743, 9.43 s -> refutes H1.1.1 (accuracy); ~1.03 s/epoch eager; need +1 pp from epochs or recipe.
E002 (I002, H4.2.1): 10 ep: acc 0.7495, 10.55 s -> supports smooth epochs dial (+0.66 pp/epoch, +1.1 s/epoch); ~10.5 epochs would reach the bar.
E003 (I003, H2.2.1): compile+fused SGD: steady 8.92 s/trial (-14%) but trial 0 recompiled (+9 s, stride guard) -> mean 12.07 s, discarded; compile works once warmup matches layout.
E004 (I003 re-run, H2.2.1): compile+fused SGD with channels_last warmup: 9.04 s (-14%), acc 0.7486 (same as E003, deterministic) -> new base; compile layout lesson confirmed.
E005 (I004, H1.3.2): lookahead EMA at 10 ep compiled: acc 0.7583 (+0.97 pp), 9.10 s -> FIRST FEASIBLE; late-training noise reduction is first-order on CIFAR-100; now trade margin for epochs.
