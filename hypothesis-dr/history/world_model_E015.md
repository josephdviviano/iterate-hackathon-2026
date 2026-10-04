# World model

_Last updated after: E015_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E001 @ 866d2a2: airbench96-style (whitening 2x2 stem, groups 128/384/576 depth 3 w/ residual, GELU, BN
weight frozen, bias lr x64), Nesterov SGD lr 9 / wd 0.012 / mom 0.85 / bs 1024, LS 0.3, lookahead, alternating
flip + translate 2, bf16 autocast + channels_last + torch.compile (warmed up in build), 10 epochs.
E001 (10 epochs): accuracy 0.7584, time 9.28 s.
E002 @ 479b9a3 = same at 9 epochs: accuracy 0.7552, time 8.34 s.
E004 @ b4e9bbb = E002 with LS 0.2: accuracy 0.7560, time 8.35 s.
E007 @ 52204ad = E004 + torch.compile mode="max-autotune-no-cudagraphs": accuracy 0.7542, time 8.257 s.
E010 @ 5aa9aa5 = E007 with global pool F.max_pool2d(x, 4) instead of adaptive_max_pool2d: accuracy 0.7551, time 8.092 s.
  CURRENT TIP. Margin ~0.2 pp over 0.753 (re-draw); official 40-seed bar is 0.75.
Reference point from README: ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe (2 trials).

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->
- time = mean(prepare+train) over 3 fixed seeds; accuracy = mean over the same 3 seeds. Fixed per-trial overhead
  of a trivial recipe is ~0.1 s (E000), so the score is essentially all training compute.
- Feasibility gate is 0.753 on 3 seeds (official: 0.75 on 40). README L40 calibration: per-trial accuracy std
  ~0.25 pp, so a 3-seed mean has std ~0.15 pp; 0.753 is a ~2-sigma margin. Time std ~0.14 s at ~60 s.
- Dev GPU is A100-SXM4 (official: A100 PCIe, lower power cap -> expect official times somewhat slower).
- No TTA allowed: eval is a single view through model.forward in eval mode.

## Where the cost goes
<!-- what dominates the objective, and why -->
- (E001) prepare is 0.06-0.15 s (<2%); train is ~9.16 s for 480 steps = 19 ms/step at bs 1024.
  ~0.6 GFLOP fwd/img -> ~95 TFLOP/s achieved, ~30% of A100 bf16 peak. Time is ~linear in epochs x net FLOPs.
- Per-trial time std is tiny (9.24-9.30 s across seeds).
- (E003 profile) The step is GPU-bound, not launch-bound: 18.8 ms/step on real batches; per step conv bwd ~8.5 ms,
  conv fwd ~4.1 ms (cuDNN, ~75%), Triton elementwise (BN/GELU/pool) ~3.4 ms, adaptive_max_pool2d bwd 0.48 ms,
  SGD 0.44 ms. Per-run overheads (augment, permute, lookahead) ~0.16 s (~2%). Prepare ~0.06-0.16 s.
  => Time can only fall substantially by fewer steps (epochs), fewer FLOPs per step (narrower/shallower net,
  lower resolution), or faster convs. Optimizer/launch fusion is worth <=3%.

## Established facts
<!-- each with the experiments that support it, e.g. "(E003, E007)" -->
- **Muon on the 3x3 conv filters (airbench94_muon style: NS5 quintic bf16, Nesterov mom 0.6, lr 0.24 x the same
  warmup/decay schedule, per-step weight-norm projection; whitening frozen; BN biases/head on SGD; lookahead kept)
  gives +1.30 pp at 9 epochs (0.7681 vs 0.7551, every seed +1.2-1.4 pp) for +13.4% time (~2.5 ms/step) (E014).**
  This is the largest lever found. Next: cash it in at fewer epochs (~7) and cut NS overhead.
- The airbench96-style base without TTA clears 0.753 at 10 epochs with ~0.5 pp margin (E001).
- Per-seed accuracy spread at 10 epochs: 0.7568-0.7613 (range 0.45 pp) (E001); at 9 epochs 0.7546-0.7558 (E002).
- Time scales ~with FLOPs down to 24px at bs 1024: a 24px-crop epoch costs ~0.53 s vs ~0.90 s at 32px (E013).
- Block-1 depth 3 vs 2: +0.40 pp for +7.5% time (E015).
- Block-3 width 576 vs 512: +0.54 pp for +4% time (E011). Shrinking width is not a free saving.
- Lookahead (decay 0.95^5 x (t/T)^3 every 5 steps) is worth ~1.0 pp at 9 epochs (E008: 0.7443 without vs 0.7542
  with; all seeds -0.7 to -1.2 pp), i.e. >1 epoch. Weight averaging is a major lever; its schedule is untuned for C100.
- Run-to-run noise with fixed seeds: accuracy mean bit-identical, time +-1% (E002 vs E003).
- BUT a recipe change re-draws each seed's outcome: per-seed sd ~0.2 pp (E004 vs E002: -0.02/-0.16/+0.40 pp).
  So a 3-seed mean difference < ~0.2-0.25 pp is not evidence about the 40-seed mean.
- Timing jitter between runs is ~+-1% (E003: +0.9% on an identical commit; E008: +0.9% despite less work). Only time
  differences > ~1.5-2% from one run are real. E007's -1.1% keep is within that noise (kept as harmless).
- Time is linear in epochs: ~0.93 s/epoch + ~0.1 s fixed (E001, E002).
- Accuracy vs epochs near the knee: ~0.3 pp/epoch between 9 and 10 epochs (E001, E002), but ~1 pp/epoch between
  8.5 and 9 (E006: 0.7510 at 8.5 vs 0.7560 at 9, LS 0.2). Concave curve; the knee of the current recipe is ~8.8-9 epochs.
  Note the 9-vs-8.5 comparison mixes a real effect with per-seed re-draw (~0.12 pp on the mean). So 1 epoch (~10% time) buys only ~0.3 pp: any change that costs <10% time
  per 0.3 pp gained is worth it, and any change that loses >0.3 pp must save >1 epoch.

## Lessons from failures
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->
- (E009) torch 2.4 Inductor's compiled backward of x.amax(dim=(2,3)) yields NaN grads (eager is fine). After any change
  to the compiled graph, run a one-step finite-gradient check (drafts/debug_nan.py) before launching.
- (E007) Prediction ranges must contain the point estimate and the no-effect outcome; don't anchor on the supports case.
- (E005) Profile before betting on systems ideas: only kernels that are a visible share of GPU time can pay off.
- (E004) Bit-identical reruns hide seed sensitivity: any recipe change re-draws per-seed accuracy (sd ~0.2 pp).

## Dead ends
- LS 0.3 vs 0.2: no meaningful difference at 9 epochs (E004). Not a lever worth more runs.
- (E010) Removing adaptive_max_pool2d's atomic backward (fixed-kernel max_pool2d) saved 2.0% (0.17 s). Systems work so
  far totals ~3%; what is left outside cuDNN convs is memory-bound fused BN/GELU elementwise (~18% of step time).
- LR schedule: warmup 12% + linear decay to 0 is -0.23 pp vs airbench's 23% warmup / decay to 0.07 (E012). Weak
  evidence (near re-draw size, two knobs bundled); schedule shape is not an obvious lever.
- Kernel selection: Inductor max-autotune buys only ~1.1% (E007, kept). cuDNN conv choice is near-optimal; don't
  expect more from autotuning/compile modes.
- Fused/compiled optimizer + lookahead with tensor lr (I009): 0% time change (E005). Launch overhead is not a lever;
  CUDA graphs / fully compiled step are expected to be equally useless here.

## Open questions and surprises
- Exchange rates near the knee (accuracy per 1% of time; higher = component is more valuable per unit time):
  epochs 8.5->9 ~0.096 (E006: 0.5 pp / 5.2%), 9->10 ~0.03 (E001/E002: 0.31 pp / 11%); block-3 width 512->576 ~0.135
  (E011); block-1 depth 2->3 ~0.054 (E015); 24px->32px for 3 of 9 epochs ~0.058 (E013); lookahead ~1.4 (E008);
  Muon vs SGD ~0.10 at its current 13.4% overhead (E014) -- but Muon's gain is a level shift worth ~2 epochs.
  Components with a rate below the epoch rate (~0.1) are candidates to cut and re-spend on epochs. Averaging/regularisation tricks that
  cost little time are the best bargains; look for more of those (EMA schedule, BN tweaks).
- Cheap systems wins visible in the E003 profile: adaptive_max_pool2d backward uses a slow atomic kernel (0.48 ms of
  18.8 ms/step, ~2.5%); a plain amax/max_pool2d(4) head would likely remove most of it with identical math.
- Where are the FLOPs? fwd MFLOP/img by layer: g1.conv1 28, g1.conv2+3 75, g2.conv1 113, g2.conv2+3 170, g3.conv1 127,
  g3.conv2+3 96 (~610 total). Group 2 is the largest share; width trade-offs there matter most.

## Changelog
<!-- one line per experiment: "E007 (I012, H2.1.3): <result> -> <what changed in this model>" -->
E000 (baseline): acc 0.012, time 0.11 s -> harness works; per-trial overhead ~0.1 s; need a real recipe.
E001 (I001, H1.1.1): acc 0.7584, time 9.28 s, feasible -> base recipe established; prepare negligible; ~30% MFU.
E002 (I003, H1.1.3): 9 epochs acc 0.7552, time 8.34 s, feasible (inconclusive: 0.31 pp/epoch) -> slope flatter than assumed; knee ~8.3-8.8 epochs.
E003 (I004, H4.1.1): rerun of E002: acc identical, time +0.9% -> noise floor tiny; profile shows GPU-bound convs (~97% of time).
E004 (I005, H1.3.1): LS 0.2 acc 0.7560 vs 0.7552, time same -> refutes LS 0.3 > 0.2; LS ~flat; per-seed sd ~0.2 pp re-draw on any change.
E005 (I009, H2.2.1): compiled foreach SGD+lookahead: time 8.347 vs 8.350 s -> refutes; step is GPU-bound, discard.
E006 (I010, H1.1.2): 8.5 epochs acc 0.7510, 7.92 s, infeasible -> supports; knee ~8.8-9 ep; slope steepens below 9.
E007 (I011, H2.1): max-autotune-no-cudagraphs: 8.257 s (-1.1%), acc 0.7542 -> keep; refutes >=2%; kernel choice near-optimal.
E008 (I006, H4.2.1): no lookahead: acc 0.7443 (-1.0 pp), time +0.9% (jitter) -> supports; lookahead is a major lever.
E009 (I014, H2.2): amax global pool -> crash (Inductor amax backward NaN); relaunching with F.max_pool2d(x, 4).
E010 (I014, H2.2): F.max_pool2d(x,4) global pool: 8.092 s (-2.0%), acc 0.7551 -> keep; atomic pool backward removed.
E011 (I015, H1.5.1): block 3 512 wide: acc 0.7497 (-0.54 pp), 7.767 s (-4%) -> supports; infeasible; width worth its cost.
E012 (I016, H1.1): warmup 12% + decay to 0: acc 0.7528 (-0.23 pp), time same -> refutes; airbench schedule kept.
E013 (I008, H2.3.1): 3/9 epochs at 24px crops: acc 0.7471 (-0.8 pp), 6.97 s (-13.9%) -> inconclusive; poor exchange rate.
E014 (I012, H1.2.1): Muon on conv filters: acc 0.7681 (+1.30 pp), 9.18 s (+13.4%) -> inconclusive by rule; biggest lever; run at fewer epochs.
E015 (I017, H1.5): block-1 depth 2: acc 0.7511 (-0.40 pp), 7.48 s (-7.5%) -> refutes; infeasible; rate 0.054 pp/1%.
