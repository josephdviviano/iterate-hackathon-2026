# World model

_Last updated after: E008_

## Current best
E007 (FEASIBLE): airbench-style net (frozen whitened 2x2 conv, widths 128/384/576, 3 convs/group, GELU, BN w/ frozen weight), Muon (lr 0.24, mom 0.6, Nesterov, 3 NS steps, weight-normalized) on conv filters + Nesterov SGD (airbench lr 9 / wd 0.012 convention, 64x BN-bias lr) on biases/head, batch 2000, 9.0 epochs (225 steps), triangular LR, Lookahead, ls 0.3, alt-flip + 2px translate, uint8 data on GPU, fp16 channels_last, torch.compile max-autotune: accuracy 0.7544, time 8.71 s. Margin over 0.753 is thin (+0.14 pp).

## Measurement
- Score = mean (prepare + train) seconds over seeds 0,1,2 on an A100-SXM4-80GB (official: A100 PCIe, 40 seeds, mean acc >= 0.75; local bar 0.753).
- Harness overhead of a trivial recipe is ~0.1 s, so essentially all time is ours.
- Reference point (README): ResNet9-style, width 64, 40 epochs -> 75.36% in 59.3 s on A100 PCIe (SXM likely somewhat faster). So a feasible recipe exists at ~60 s; the target is to go well below that.
- Re-running the same code gives a bit-identical 3-seed accuracy (E002 = E003 = 0.74823) and time within ~0.5% (9.03 vs 9.00 s). So any accuracy change is a real effect of the code change on these 3 seeds, but it is still subject to seed-sampling error relative to the 40-seed official mean: per-seed std ~0.15-0.3 pp -> 3-seed mean std ~0.1-0.2 pp. Time deltas < ~1% are noise.
- Implication: tiny accuracy changes (e.g. +0.05 pp) are "real" on these seeds but may not generalize; judge sample-efficiency changes by >= ~0.2 pp effects.

## Where the cost goes
- E001: train ~7.93 s for 408 steps = ~19.4 ms/step (~0.93 s/epoch) at widths 128/384/576; prepare 0.06 s (0.20 s on the first trial: one-off first-use cost). So time is ~proportional to steps x per-step cost; prepare is <1%.
- Per-step time is NOT FLOP-proportional: widening group 3 from 576 to 768 (+19% MACs) costs only +6% per step (20.5 vs 19.4 ms) (E004). Low-resolution late layers are under-utilized; the cost sits mostly in the high-resolution early layers. Rough estimate: ~3.5 TFLOP/step at 19.4 ms = ~190 TFLOPS effective, already a large fraction of A100 fp16 peak overall.

## Established facts
- Harness and template run end-to-end; trivial recipe costs 0.10 s (E000).
- airbench port at 8.5 epochs reaches only 0.738 on CIFAR-100 without TTA (E001); per-seed std ~0.3 pp.
- Epoch slope at 8.5->9.5 epochs is ~1.0 pp/epoch for ~0.99 s/epoch (E001, E002): the recipe is under-trained; every saved second is worth ~1 pp, and every +1 pp of sample efficiency is worth ~1 s.
- torch.compile max-autotune vs default: only -1.9% per step, no recompiles (E006). Compiler flags are not a lever; convs (cuDNN) dominate.
- Muon on conv filters (batch 2000) beats Nesterov SGD (batch 1024) by ~0.7 pp at equal wall time (E006 vs E007); per-sample step cost is ~4% higher (NS iterations). The lit claim that SGD reaches modest targets faster (lit:C0007, ResNet-110 long schedules) did not transfer.
- Under Muon, widening group 3 to 768 at -0.5 epoch LOSES (0.7514 @ 8.90 s vs 0.7544 @ 8.71 s) (E007, E008): the SGD-era width bonus below does not transfer; 768 costs +7% per step at batch 2000.
- (SGD only) Wider-and-shorter beats narrower-and-longer: 128/384/768 @ 7.5 ep = 0.7389 @ 7.49 s vs 128/384/576 @ 8.5 ep = 0.7383 @ 8.04 s (E001, E004). Width in the deep group buys sample efficiency at little per-step cost.
- Per-seed accuracy spread can be up to ~1 pp (E004: 0.734-0.744), so the 3-seed mean's sampling std is ~0.2-0.3 pp: the 0.3 pp margin of the local bar is ~1-1.5 sigma, not more.
- The first trial of a run had ~0.18 s extra prepare (first-use kernel/library init) (E001, E002); warming the prepare path (eigh, reflect pad, gather) on synthetic data in build removes it, and prepare is now ~0.06 s on every trial (E005). Prepare is <1% of time; further prepare work has little left to gain.

## Lessons from failures
- Findings about architecture efficiency are optimizer-dependent: re-validate them after switching optimizers (E004 -> E008).
- The time-per-step for wider nets must be predicted from measured ms/step (E004 post-mortem).

## Dead ends

## Open questions and surprises
- How many epochs / what width is the minimum for 75.3% with a strong recipe (airbench-style: whitening init, label smoothing, lookahead/EMA, flip+translate aug, channels_last, bf16, torch.compile)?

## Changelog
- E000 (baseline, -): template -> acc 0.0122, time 0.10 s; harness works, infeasible.
- E001 (I001, H1.1.1): airbench port 8.5 ep -> acc 0.7383, 8.04 s; H1.1.1 refuted (1.5 pp short); time ~19.4 ms/step.
- E002 (I002, H1.1.2): 9.5 ep -> acc 0.7482 (+0.99 pp), 9.03 s (+12%); slope ~1 pp/epoch, still under-trained.
- E003 (I003, H1.1.3): replicate of E002 -> identical acc 0.74823, time 9.00 s; accuracy deterministic per seed set, time noise ~0.5%.
- E004 (I004, H1.5.1): 128/384/768 @ 7.5 ep -> acc 0.7389, 7.49 s; ~0.6 pp above the 576 epoch line; width in group 3 is nearly free (+6% ms/step); H1.5.1 refuted.
- E005 (I005, H3.1.1): uint8-on-GPU + fused normalization + warmed prepare -> acc 0.7476 (RNG-stream change), 8.93 s (-0.10 s); prepare 0.06 s flat. Kept (--force).
- E006 (I006, H2.2.1): max-autotune -> acc 0.7475, 8.76 s (-1.9%); kept (--force); compiler mode is a minor lever.
- E007 (I007, H1.4.1): Muon on filters, batch 2000, 9.0 ep -> acc 0.7544, 8.71 s. FIRST FEASIBLE. H1.4 refuted.
- E008 (I009 adapted, H1.5): Muon + 768 @ 8.5 ep -> 0.7514, 8.90 s; worse than 576 @ 9.0; width bonus does not transfer to Muon. Discarded.
