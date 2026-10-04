# World model

_Last updated after: E004_

## Current best
E002: airbench port (whitened 2x2 conv, widths 128/384/576, 3 convs/group, SGD-Nesterov lr 9 wd 0.012, ls 0.3, Lookahead, alt-flip + 2px translate, fp16 channels_last, torch.compile, batch 1024 drop_last) at 9.5 epochs (456 steps): accuracy 0.7482, time 9.03 s. Still infeasible (needs >= 0.753); ~10.0-10.5 epochs should cross.

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
- Wider-and-shorter beats narrower-and-longer: 128/384/768 @ 7.5 ep = 0.7389 @ 7.49 s vs 128/384/576 @ 8.5 ep = 0.7383 @ 8.04 s (E001, E004). Width in the deep group buys sample efficiency at little per-step cost.
- Per-seed accuracy spread can be up to ~1 pp (E004: 0.734-0.744), so the 3-seed mean's sampling std is ~0.2-0.3 pp: the 0.3 pp margin of the local bar is ~1-1.5 sigma, not more.
- The first trial of a run has ~0.18 s extra prepare (first-use kernel/library init) (E001, E002); later trials prepare in ~0.06-0.10 s.

## Lessons from failures

## Dead ends

## Open questions and surprises
- How many epochs / what width is the minimum for 75.3% with a strong recipe (airbench-style: whitening init, label smoothing, lookahead/EMA, flip+translate aug, channels_last, bf16, torch.compile)?

## Changelog
- E000 (baseline, -): template -> acc 0.0122, time 0.10 s; harness works, infeasible.
- E001 (I001, H1.1.1): airbench port 8.5 ep -> acc 0.7383, 8.04 s; H1.1.1 refuted (1.5 pp short); time ~19.4 ms/step.
- E002 (I002, H1.1.2): 9.5 ep -> acc 0.7482 (+0.99 pp), 9.03 s (+12%); slope ~1 pp/epoch, still under-trained.
- E003 (I003, H1.1.3): replicate of E002 -> identical acc 0.74823, time 9.00 s; accuracy deterministic per seed set, time noise ~0.5%.
- E004 (I004, H1.5.1): 128/384/768 @ 7.5 ep -> acc 0.7389, 7.49 s; ~0.6 pp above the 576 epoch line; width in group 3 is nearly free (+6% ms/step); H1.5.1 refuted.
