# World model

_Last updated after: E019_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E017 (BEST, FEASIBLE): E016 with a sync-free gather crop+flip (same distribution). seeds 0-2: accuracy 0.7537, time 6.55 s; held-out seeds 3-12: 0.7577 (std 0.23 pp). True mean ~0.757.
E016: E015 with weight decay 0.012 -> 0.006 (applies to SGD groups: BN biases, whitening bias, head; Muon filters have none). accuracy 0.7607 on seeds 0-2 (lucky; held-out 3-12: 0.7564), time 6.58 s.
E015: E010 with compile mode max-autotune (cudagraphs on for model fwd/bwd; the Muon update is skipped by inductor because it mutates inputs). accuracy 0.7574 (lucky trajectory; treat the true mean as ~0.755 as for E010), time 6.58 s.
E010: E008 with last group 576 -> 512. accuracy 0.7548, time 6.64 s.
E008: E007 at 9.5 epochs (238 steps; resolution switches at 40%/70% of steps), error_on_recompile after build. accuracy 0.7572, time 6.98 s.
E007: E006 + progressive resizing (epochs 0-3 at 24 px, 4-6 at 28 px, 7-9 at 32 px via bilinear-antialias interpolate of the augmented epoch; compile dynamic=False, each res warmed in build, build ~120 s). accuracy 0.7590, time 7.33 s.
E006: E005 with conv filters on Muon (NS5 x3 bf16, compiled update, lr 0.24, mom 0.6, fixed-norm weights), SGD on BN biases/whiten bias/head, batch 2000 (25 steps/epoch). accuracy 0.7649, time 9.45 s.
E005: E004 + airbench lookahead EMA (every 5 steps, alpha = 0.95^5 * (t/T)^3, final weights = EMA). accuracy 0.7583, time 9.10 s. Margin above 0.753: +0.53 pp.
E004 (was): airbench96-style port (whiten 2x2 stem, groups 128/384/576 x3 conv, BN-GELU, Nesterov SGD lr 9 wd 0.012 bs 1024, LS 0.3, fp16 channels_last) at 10 epochs, train step torch.compile'd (max-autotune-no-cudagraphs) + fused SGD, eager eval. accuracy 0.7486, time 9.04 s. NOT feasible yet (needs >= 0.753; ~+0.45 pp short).

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->
- Seeds 0-2 luck is a (seed x RNG-stream) effect: three builds of E017 give seeds 0-2 means 0.7537/0.7541/0.7536 (+-0.03 pp; per-seed values differ since cudagraph builds aren't bit-reproducible) vs held-out 0.7577. Changes that alter random-number consumption (augment, perm, init order) re-roll the ~0.35 pp luck; kernel/build changes barely move it. SEED-OVERFITTING RISK: don't credit gains that only re-roll seeds 0-2 luck; confirm with held-out seeds when a keep hinges on < 0.5 pp.
- KEY: the seeds 0-2 mean is much noisier than per-seed std suggests. Deviation of the seeds 0-2 mean from the 10-seed held-out mean: E010 +0.08, E016 +0.43, E017 -0.40 pp -> ~0.35 pp std. 3-seed differences < ~0.6 pp are not interpretable. Seed 1 is persistently the lowest. Settle accuracy claims with held-out seeds 3-12 offline (`benchmark.run --seed 3 --n 10`, ~2.5 min).
- Held-out 10-seed means: E010 0.7540, E016 0.7564, E017 0.7577, E018 0.7509.
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
- Halving weight decay (SGD groups) is worth ~+0.2 pp (held-out: E016 0.7564 vs E010 0.7540; confounded with E015's kernel change), not the +0.6 the seeds 0-2 suggested. The strong bias decay (bias lr = 64x) is over-regularizing in a 238-step run.
- Lookahead EMA is worth ~+1 pp at 10 epochs for +0.6% time (E005 vs E004; every seed +0.9-1.15 pp). End-of-training noise is a big lever on CIFAR-100 short runs: one epoch buys ~0.66 pp, so EMA is worth ~1.5 epochs (~1.3 s).
- The 9-epoch airbench96-style port lands ~1 pp below the bar on CIFAR-100 without TTA (E001: 0.743).
- With EMA+Muon+resizing, the epochs dial is flatter: 10 -> 9.5 epochs = -0.18 pp, -0.35 s (E007->E008), i.e. ~0.36 pp/epoch and ~0.7 s/epoch.
- Epochs dial near 9-10 (eager SGD, no EMA): +1 epoch = +0.66 pp, +1.1 s (+11.8%) (E001->E002). Extrapolated 0.753 needs ~10.5 epochs (~11.1 s eager) at this recipe.
- Per-seed accuracy spread can be ~1 pp (E001: 0.7355 to 0.7475), but E002 (same seeds) spread only 0.25 pp, so single-trial outliers come from GPU nondeterminism, not the seed. 3-seed mean noise is probably ~0.2-0.4 pp run to run (to be calibrated by I005).

## Lessons from failures
- torch.compile guards on strides/layout, not just shape: a warmup with contiguous NCHW input while training uses channels_last recompiled inside trial 0 (+9 s). Warm up via the same data path; verify with TORCH_LOGS=recompiles on random data first (E003).
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->

## Dead ends
- Dropping the 28 px phase (24/32 at 70/30%): -8.5% time but held-out -0.68 pp (E018). The middle-resolution phase matters; accuracy is not set only by the 32 px tail. Rate ~0.8 s/pp.
- Muon lr 0.24 -> 0.36: +0.09 pp (noise) (E014). Muon lr is flat near 0.24-0.36.
- Label smoothing 0.3 -> 0.2: +0.04 pp (noise) (E013). LS is flat in 0.2-0.3; not a lever.
- Aggressive resizing 20/24/28/32 at 30/30/20/20%: -0.97 pp for -14% time (~1.0 s/pp, worse than epochs) (E011). Confounded: 20 px phase vs shorter 32 px tail.
- Decaying LR to 0 instead of 0.07 of peak: -0.08 pp (noise) with EMA on (E009). EMA already supplies the end-of-run noise reduction; don't tune the LR floor further.

## Open questions and surprises
- Scalar hyperparameters LS, Muon lr, LR floor came out flat (E009, E013, E014), but weight decay was not (E016: +0.3-0.6 pp at x0.5). wd x0.25 / 0 is open. The recipe sits on a plateau for scalars; remaining accuracy levers are likely structural (data aug like cutout/flip variants, architecture, BN/eval details, batch size/steps) or the time levers (resolution schedule shape, epochs).
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
E016 (I019, H1.5.1): wd x0.5: acc 0.7607 (+0.33 pp vs E015), 6.58 s -> supports; margin now ~0.6+ pp to trade for time.
E017 (I021, H2.2.3): sync-free gather crop: 6.55 s (-0.5%), seeds 0-2 acc 0.7537 but held-out 0.7577 -> refutes (syncs off critical path); kept; 3-seed noise is ~0.35 pp, not 0.14.
E018 (I023, H2.3): 24/32 at 70/30%: 5.99 s (-8.5%), seeds 0-2 0.7502, held-out 0.7509 (-0.68 pp) -> refutes; the 28 px phase carries accuracy.
E019 (I030, H4.3.1): rerun + fresh-cache rebuild of E017: per-seed not bit-identical, 3-seed mean stable (+-0.03 pp) -> seeds 0-2 offset is RNG-stream luck, not build noise.
