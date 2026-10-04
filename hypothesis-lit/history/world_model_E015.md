# World model

_Last updated after: E015_

## Current best
E014: airbench-style net (frozen whitened 2x2 conv; groups of 2/3/3 convs at widths 128/384/576; GELU; BN with frozen weight), Muon (lr 0.24, mom 0.6, Nesterov, 3 NS steps, weight-normalized) on conv filters + Nesterov SGD (airbench lr 9 / wd 0.012 convention, 64x BN-bias lr) on biases/head, batch 2000, 9.5 epochs (238 steps), triangular LR, Lookahead, ls 0.3, alt-flip + 2px translate, uint8 data on GPU, fp16 channels_last, torch.compile max-autotune: accuracy 0.7559, time 8.44 s (35.2 ms/step).

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
- Progressive resizing stacks with the group-1 depth cut: on the 2/3/3 net, 10.5 ep with 5 at 24 px = 0.7516 @ 7.62 s vs 9.5 full-res = 0.7559 @ 8.44 s (0.52 pp/s) (E015). Needs ~+0.4 pp more (≈ +0.5 epoch) to be feasible at ~7.9-8.1 s.
- Removing one group-1 conv (128->128 @16x16, 6% of MACs) cuts per-step time 9% (38.5 -> 35.0 ms @ batch 2000) and costs only 0.29 pp (E009 vs E007): 0.41 pp/s, a much better trade than epochs (~1 pp/s). High-resolution layers are the expensive ones; the next step is 2-conv group 1 + more epochs.
- Narrowing group 1 from 128 to 96 (3 convs) cuts per-step time 7% at -0.47 pp: 0.82 pp/s, worse than the depth cut (E010 vs E009). Group-1 width matters more for accuracy than its third conv.
- Progressive resizing (first 5/9 epochs at 24 px, downscaled) cuts time 21% (8.71 -> 6.84 s) for -0.84 pp: 0.45 pp/s (E011). Like the group-1 depth cut (0.41 pp/s), this is ~2x better than the epoch line, so the efficient frontier is: cheaper steps + more of them.
- Dropping the 25% lowest-loss examples in second-half epochs: -12% steps, -0.73 pp: 0.70 pp/s (E013); worse than resizing/depth cuts. Dead end.
- Label smoothing 0.3 beats 0.2 by 0.21 pp under Muon (E012 vs E007): weak knob, keep 0.3.
- Muon's optimizer overhead is ~4% of step time (19.3 vs 18.6 ms per 1000 images, E006 vs E007); NS-step cuts (lit:C0008) could recover at most that.
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
- Loss-based example filtering (E013).
- Widening group 3 under Muon (E008).
- Narrowing group 1 to 96 (E010; worse trade than the depth cut).

## Open questions and surprises
- Muon epoch slope measured on the 2/3/3 net: +0.44 pp for 9.0 -> 9.5 epochs (~0.9 pp/epoch, ~1 pp/s) (E009, E014). So the SGD-era '~1 pp/s epoch line' is roughly right under Muon too.
- (resolved partly by E014) MISSING CONTROL: the Muon epoch slope (accuracy per epoch near 8-9 epochs). All 'pp/s vs the epoch line' judgements assume SGD's ~1 pp/epoch; measure the incumbent at ~8 epochs.
- How do the two cheap-step levers (2-conv group 1, 24 px early epochs) combine, and how many extra epochs do they need to regain 0.755? What is the Muon epoch slope (unmeasured; assumed ~1 pp/epoch from SGD)?
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
- E009 (I011, H1.5): group 1 = 2 convs (Muon, 576, 9 ep) -> 0.7515, 8.01 s; 0.41 pp/s trade, above the epoch line; infeasible alone -> pair with +0.5 ep.
- E010 (I012, H1.5): group-1 width 96 -> 0.7497, 8.14 s; 0.82 pp/s; depth cut (E009) is the better group-1 saving. Discarded.
- E011 (I008, H2.3.1): 24 px for first 5/9 epochs -> 0.7460, 6.84 s (-21% time, -0.84 pp; 0.45 pp/s). Discarded alone; pair with +epochs.
- E012 (I013, H1.3.1): ls 0.2 -> 0.7523 (-0.21 pp), 8.77 s; keep 0.3. Discarded.
- E013 (I014, H1.6.1): loss-based filtering (25% easy dropped in 2nd half) -> 0.7471, 7.66 s; 0.70 pp/s; refuted, dead end.
- E014 (I015, H2.4.2): 2/3/3 convs, 9.5 ep -> 0.7559, 8.44 s. NEW BEST (-0.27 s, +0.15 pp vs E007). Muon slope ~0.9 pp/epoch.
- E015 (I016, H2.3): 2/3/3 + 5 low-res of 10.5 ep -> 0.7516, 7.62 s; 0.52 pp/s; infeasible by 0.14 pp; next +0.5 ep.
