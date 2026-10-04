# World model

_Last updated after: E015_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E015 (BEST, FEASIBLE): E010 with compile mode max-autotune (cudagraphs on for model fwd/bwd; the Muon update is skipped by inductor because it mutates inputs). accuracy 0.7574 (lucky trajectory; treat the true mean as ~0.755 as for E010), time 6.58 s.
E010: E008 with last group 576 -> 512. accuracy 0.7548, time 6.64 s.
E008: E007 at 9.5 epochs (238 steps; resolution switches at 40%/70% of steps), error_on_recompile after build. accuracy 0.7572, time 6.98 s.
E007: E006 + progressive resizing (epochs 0-3 at 24 px, 4-6 at 28 px, 7-9 at 32 px via bilinear-antialias interpolate of the augmented epoch; compile dynamic=False, each res warmed in build, build ~120 s). accuracy 0.7590, time 7.33 s.
E006: E005 with conv filters on Muon (NS5 x3 bf16, compiled update, lr 0.24, mom 0.6, fixed-norm weights), SGD on BN biases/whiten bias/head, batch 2000 (25 steps/epoch). accuracy 0.7649, time 9.45 s.
E005: E004 + airbench lookahead EMA (every 5 steps, alpha = 0.95^5 * (t/T)^3, final weights = EMA). accuracy 0.7583, time 9.10 s. Margin above 0.753: +0.53 pp.
E004 (was): airbench96-style port (whiten 2x2 stem, groups 128/384/576 x3 conv, BN-GELU, Nesterov SGD lr 9 wd 0.012 bs 1024, LS 0.3, fp16 channels_last) at 10 epochs, train step torch.compile'd (max-autotune-no-cudagraphs) + fused SGD, eager eval. accuracy 0.7486, time 9.04 s. NOT feasible yet (needs >= 0.753; ~+0.45 pp short).

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->
- Held-out seeds 3-12 on E010: mean 0.7540 vs 0.7548 on seeds 0-2, per-seed std 0.24 pp (E012). 3-seed mean SE ~0.14 pp. The official 40-seed bar (0.75) has ~0.4 pp margin at the tip; the local 0.753 bar is binding.
- Compiled training is bit-deterministic per seed for a fixed compiled artifact (E003 == E004 == E012), but changing the compile mode re-autotunes kernels and changes trajectories (E015: +0.26 pp with identical math). Treat any kernel/compile change as a seed perturbation. So a byte-identical re-run measures nothing; the relevant noise is how per-seed accuracy varies under any recipe perturbation (~0.2-0.6 pp per seed, i.e. ~0.15-0.35 pp on the 3-seed mean).
- Trial 0 prepare is ~0.15 s slower than later trials (first-use allocation), ~1.7% of 9 s.
- time = mean(prepare+train) over seeds 0,1,2; harness syncs CUDA around them. Trivial recipe costs 0.14 s, so fixed overhead is negligible.
- Accuracy per-seed std for a ResNet9-ish recipe is ~0.25 pp (README L40 calibration, 50 trials); 3-seed mean std ~0.15 pp. Feasibility gate is 0.753 on the 3-seed mean.
- Reference point (README): ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe. Our A100 SXM should be similar or a bit faster.

## Where the cost goes
<!-- what dominates the objective, and why -->
- Per-epoch train cost on E010 (E012 CUDA events): 24 px 0.51 s, 28 px 0.71 s, 32 px 0.89 s; first epoch of trial 0 +60 ms; prepare 0.06 s (0.21 s in trial 0).
- CUDA graphs on fwd/bwd save only 0.9% (E015): launch overhead is hidden behind ~20-36 ms GPU steps. Launch-overhead work is exhausted.
- ~95% of train time is pixel-proportional conv compute: E011 cut pixel-weighted cost 15.4% and train time fell 14.6%. Fixed per-step overhead (Muon NS, SGD, EMA, Python launch) is only ~5% -> CUDA graphs/launch work can buy at most ~0.3 s.
- E007 per-epoch train cost (compiled, Muon, bs 2000): ~0.58 s at 24 px, ~0.75 s at 28 px, ~0.93 s at 32 px (estimates from pixel scaling; total 7.22 s for 10 epochs).
- Compiled (inductor max-autotune-no-cudagraphs) + fused SGD: steady-state train 8.92 s vs 10.39 s eager at 10 epochs (-14%, ~18.6 ms/step) (E003 trials 1-2).
- E001: train ~9.30 s for 9 epochs x 48 steps = 432 steps -> ~21.5 ms/step, ~1.03 s/epoch (eager). prepare 0.07-0.2 s (first trial slower). Eval 0.1 s.
- For a real recipe: GPU training steps (epochs x step cost). Data is 50k uint8 32x32 images (150 MB) -> GPU transfer is ~tens of ms. build() is untimed: compile/warmup there.

## Established facts
<!-- each with the experiments that support it, e.g. "(E003, E007)" -->
- Exchange rates (time saved per pp of accuracy lost): epoch trim ~1.9 s/pp (E008), last-group width 576->512 ~1.4 s/pp (E010), progressive resizing ~3.6 s/pp (E007). Prefer cheap-resolution and epoch trims over narrowing.
- Progressive resizing 24/28/32 (40/30/30%) cuts time 22% for -0.59 pp at 10 epochs (E007 vs E006). Compiled step time scales ~linearly with pixel count. Low-res epochs are a better deal than cutting epochs: -0.59 pp ~ 0.9 epoch (~0.7 s) bought 2.1 s.
- Muon on conv filters (bs 2000) is worth +0.66 pp at 10 epochs for +3.8% time (E006 vs E005; all seeds up). The committee predicted a loss; the airbench94_muon CIFAR-10 settings transfer to CIFAR-100 untuned.
- Lookahead EMA is worth ~+1 pp at 10 epochs for +0.6% time (E005 vs E004; every seed +0.9-1.15 pp). End-of-training noise is a big lever on CIFAR-100 short runs: one epoch buys ~0.66 pp, so EMA is worth ~1.5 epochs (~1.3 s).
- The 9-epoch airbench96-style port lands ~1 pp below the bar on CIFAR-100 without TTA (E001: 0.743).
- With EMA+Muon+resizing, the epochs dial is flatter: 10 -> 9.5 epochs = -0.18 pp, -0.35 s (E007->E008), i.e. ~0.36 pp/epoch and ~0.7 s/epoch.
- Epochs dial near 9-10 (eager SGD, no EMA): +1 epoch = +0.66 pp, +1.1 s (+11.8%) (E001->E002). Extrapolated 0.753 needs ~10.5 epochs (~11.1 s eager) at this recipe.
- Per-seed accuracy spread can be ~1 pp (E001: 0.7355 to 0.7475), but E002 (same seeds) spread only 0.25 pp, so single-trial outliers come from GPU nondeterminism, not the seed. 3-seed mean noise is probably ~0.2-0.4 pp run to run (to be calibrated by I005).

## Lessons from failures
- torch.compile guards on strides/layout, not just shape: a warmup with contiguous NCHW input while training uses channels_last recompiled inside trial 0 (+9 s). Warm up via the same data path; verify with TORCH_LOGS=recompiles on random data first (E003).
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->

## Dead ends
- Muon lr 0.24 -> 0.36: +0.09 pp (noise) (E014). Muon lr is flat near 0.24-0.36.
- Label smoothing 0.3 -> 0.2: +0.04 pp (noise) (E013). LS is flat in 0.2-0.3; not a lever.
- Aggressive resizing 20/24/28/32 at 30/30/20/20%: -0.97 pp for -14% time (~1.0 s/pp, worse than epochs) (E011). Confounded: 20 px phase vs shorter 32 px tail.
- Decaying LR to 0 instead of 0.07 of peak: -0.08 pp (noise) with EMA on (E009). EMA already supplies the end-of-run noise reduction; don't tune the LR floor further.

## Open questions and surprises
- Scalar hyperparameters (LS, Muon lr, LR floor) all came out flat (E009, E013, E014). The recipe sits on a plateau for scalars; remaining accuracy levers are likely structural (data aug like cutout/flip variants, architecture, BN/eval details, batch size/steps) or the time levers (resolution schedule shape, epochs).
- How does the Muon+EMA recipe's accuracy fall with epochs? If slope is ~0.66 pp/epoch, ~8.5 epochs at ~0.755 would cost ~8.1 s.
- Graph breaks in the train step (D001 C0003): not checked yet; torch._dynamo.explain on the step is a cheap check. Also set error_on_recompile after build to make leaks fail loudly.
- The committee has twice under-forecast accuracy gains (E005, E006): CIFAR-100 short runs seem more responsive to optimizer/averaging changes than their CIFAR-10 priors.

## Changelog
<!-- one line per experiment: "E007 (I012, H2.1.3): <result> -> <what changed in this model>" -->
E000 (baseline): template, acc 0.012, time 0.14 s -> harness overhead negligible; need a real recipe first.
E001 (I001, H1.1.1): port at 9 ep: acc 0.743, 9.43 s -> refutes H1.1.1 (accuracy); ~1.03 s/epoch eager; need +1 pp from epochs or recipe.
E002 (I002, H4.2.1): 10 ep: acc 0.7495, 10.55 s -> supports smooth epochs dial (+0.66 pp/epoch, +1.1 s/epoch); ~10.5 epochs would reach the bar.
E003 (I003, H2.2.1): compile+fused SGD: steady 8.92 s/trial (-14%) but trial 0 recompiled (+9 s, stride guard) -> mean 12.07 s, discarded; compile works once warmup matches layout.
E004 (I003 re-run, H2.2.1): compile+fused SGD with channels_last warmup: 9.04 s (-14%), acc 0.7486 (same as E003, deterministic) -> new base; compile layout lesson confirmed.
E005 (I004, H1.3.2): lookahead EMA at 10 ep compiled: acc 0.7583 (+0.97 pp), 9.10 s -> FIRST FEASIBLE; late-training noise reduction is first-order on CIFAR-100; now trade margin for epochs.
E006 (I007, H1.2.1): Muon filters + bs 2000 at 10 ep: acc 0.7649 (+0.66 pp), 9.45 s (+3.8%) -> supports; kept (--force) to trade the margin for epochs.
E007 (I008, H2.3.1): progressive resize 24/28/32: 7.33 s (-22%), acc 0.7590 (-0.59 pp) -> refutes H2.3.1's 0.2 pp bound but is the new best; resolution is a near-linear time dial.
E008 (I009, H4.2.2): 9.5 epochs: acc 0.7572 (-0.18 pp), 6.98 s (-4.8%) -> supports; slope with EMA ~0.36 pp/epoch; margin 0.42 pp.
E009 (I010, H1.3.4): LR floor 0.07 -> 0: acc 0.7564 (-0.08 pp), 6.99 s -> refutes; EMA and low final LR are redundant.
E010 (I012, H1.5.2): last group 512: acc 0.7548 (-0.25 pp), 6.64 s (-4.8%) -> supports; new best but margin only +0.18 pp; need accuracy boosters next.
E011 (I011, H2.3): 20/24/28/32 resizing: 5.70 s (-14%), acc 0.7451 (-0.97 pp) -> refutes; time ~proportional to pixels, but accuracy has a cliff with coarse/short-tail schedules.
E012 (I016, H4.1.2): held-out seeds 3-12 mean 0.7540 (std 0.24 pp), seeds 0-2 reproduce bit-identically -> supports; per-epoch 0.51/0.71/0.89 s at 24/28/32 px.
E013 (I017, H1.3.1): LS 0.2: acc 0.7551 (+0.04 pp, noise), 6.63 s -> refutes 'LS 0.3 >> 0.2'; LS flat, discarded.
E014 (I018, H1.5.1): Muon lr 0.36: acc 0.7557 (+0.09 pp, noise), 6.63 s -> inconclusive; scalars are on a plateau.
E015 (I014, H2.2.2): cudagraphs: 6.58 s (-0.9%), acc 0.7574 (trajectory noise) -> refutes (<1%); kept for the tiny real gain; per-step overhead exhausted.
