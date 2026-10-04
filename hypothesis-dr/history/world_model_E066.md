# World model

_Last updated after: E066_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
**FULL CURRENT RECIPE (tip E058 @ fb3f626, 0.7537, 4.696 s) = E042 + head lr x2 + resolution schedule 20px (epochs
0-1.5), 28px (1.5-3.5), 32px (3.5-6.75) + no translate in epochs 0-2 + block 1 frozen (output detached) from 5.4 ep:** airbench-style net (2x2 whitening frozen weights, bias
trained in epoch 0 only then detached; groups 128/384/576, depth 3, SiLU; global max_pool2d head x1/9), bs 1536,
6.75 epochs total of which the FIRST 1.5 EPOCHS ARE 24px (whole images bilinear-antialias downsampled), rest 32px;
Muon (NS3 quintic bf16, batched + compiled, lr 0.16, Nesterov mom 0.8, weight-norm projection) on 3x3 filters;
Nesterov SGD (lr 9, wd 0.012, mom 0.85, BN-bias x64; loss is a SUM) on BN biases/head; schedule 23% warmup from 0.2x,
linear decay to 0.07x; lookahead every 5 steps, decay 0.97^5 x (t/T)^3; LS 0.2; flip (alternating) + translate 2 +
brightness/contrast jitter; bf16 autocast, channels_last, torch.compile max-autotune-no-cudagraphs warmed in build.
E001 @ 866d2a2: airbench96-style (whitening 2x2 stem, groups 128/384/576 depth 3 w/ residual, GELU, BN
weight frozen, bias lr x64), Nesterov SGD lr 9 / wd 0.012 / mom 0.85 / bs 1024, LS 0.3, lookahead, alternating
flip + translate 2, bf16 autocast + channels_last + torch.compile (warmed up in build), 10 epochs.
E001 (10 epochs): accuracy 0.7584, time 9.28 s.
E002 @ 479b9a3 = same at 9 epochs: accuracy 0.7552, time 8.34 s.
E004 @ b4e9bbb = E002 with LS 0.2: accuracy 0.7560, time 8.35 s.
E007 @ 52204ad = E004 + torch.compile mode="max-autotune-no-cudagraphs": accuracy 0.7542, time 8.257 s.
E010 @ 5aa9aa5 = E007 with global pool F.max_pool2d(x, 4) instead of adaptive_max_pool2d: accuracy 0.7551, time 8.092 s.
E016 @ 59dac7e = E010 + Muon on conv filters (NS5, mom 0.6, lr 0.24, weight-norm projection), 8 epochs:
  accuracy 0.7619, time 8.114 s (force-kept: same time as E010, +0.68 pp margin to spend).
E017 @ eff75ec = E016 with NS 3 steps: accuracy 0.7639, time 7.887 s.
E018 @ d634e3b = E017 at 7.5 epochs: accuracy 0.7596, time 7.390 s.
E021 @ c33cae8 = E018 with SiLU instead of GELU: accuracy 0.7620, time 7.231 s.
E022 @ 44f6372 = E021 at 7 epochs: accuracy 0.7582, time 6.740 s.
E023 @ 9c08ac0 = E022 with a fused compiled Muon step (foreach momentum, batched NS per shape pair, 0-d lr):
  accuracy 0.7568, time 6.559 s.
E024 @ a4bb8fb = E023 with Muon lr 0.16: accuracy 0.7585, time 6.531 s.
E026 @ 6b89d6f = E024 with Muon momentum 0.8: accuracy 0.7603, time 6.548 s.
E027 @ 3a8eb92 = E026 at 6.5 epochs: accuracy 0.7546, time 6.094 s.
E028 @ e95ff94 = E027 with lookahead base decay 0.97^5: accuracy 0.7546, time 6.077 s.
E034 @ f74c715 = E028 with whitening bias trained only in epoch 0, then detached in forward (no block-1 dgrad):
  accuracy 0.7551, time 5.894 s.
E040 @ 18ce8ac = E034 with the first 1.5 epochs on whole images bilinearly downsampled (antialias) to 24 px, 6.75 epochs
  total: accuracy 0.7559, time 5.718 s. Build 121 s.
E041 @ fbc845c = E040 with batch size 1536 (216 steps; loss is a sum so SGD scales automatically; Muon lr 0.16):
  accuracy 0.7533, time 5.422 s.
E042 @ 2b85e5e = E041 + per-image brightness (+-0.14) / contrast (x1+-0.13) jitter: accuracy 0.7549, time 5.449 s.
  (force-kept; margin 0.19 pp).
E045 @ bfcff33 = E042 with head lr x2 (head wd halved): accuracy 0.7568, time 5.464 s (force-kept).
E047 @ 3d6a396 = E045 with the first 1.5 epochs at 20px instead of 24px: accuracy 0.7550, time 5.273 s.
E048 @ dc07126 = E047 + a 28px stage for epochs 1.5-3.0: accuracy 0.7552, time 5.000 s. Build 228 s (6 graph variants).
E051 @ 9f17acf = E048 with the 28px stage extended to epoch 3.5: accuracy 0.7541, time 4.926 s.
E052 @ 447a7db = E051 without translate in epochs 0-2 (entirely low-res): accuracy 0.7546, time 4.928 s (force-kept).
E058 @ fb3f626 = E052 + block-1 output detached for the last 20% of training (Muon skips block-1 filters):
  accuracy 0.7537, time 4.696 s. CURRENT TIP (margin 0.07 pp). Cold build 375 s (warms only used variants).
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
- Muon at 8 epochs: 0.7619 (E016); 9 -> 8 epochs costs 0.62 pp with Muon.
- Muon epoch curve: 9 ep 0.7681 (NS5), 8 ep 0.7619 (NS5) / 0.7639 (NS3), 7.5 ep 0.7596 (NS3) (E014, E016-E018):
  ~0.6-0.9 pp/epoch; 7 ep 0.7582 with SiLU (E022; 7.5 -> 7 cost 0.38 pp); with lr 0.16 / mom 0.8: 7 ep 0.7603,
  6.5 ep 0.7546 (E026, E027: 1.14 pp/epoch). The knee is now ~6.4 epochs; slope steepening.
- BN recalibration on the final averaged weights: +0.03 pp (E035). Lookahead's blended BN buffers are already right.
- Lookahead base decay 0.95^5 vs 0.97^5: identical accuracy (E028). The averaging window length is flat here.
- Lookahead is still worth ~0.86 pp with Muon at 7.5 epochs (E020); Muon does not replace averaging.
- Muon-group warmup (0.2 -> 1.0 over 23%) vs none: -0.06 pp without it (E031); flat.
- Frequency-wise (DCT-II) Muon-C split, row-norm matched: -1.2 pp, +3.7% time (E030). Flattened [out, in*9] is better;
  equalising spatial frequencies over-weights high-frequency kernel components. Kernel-split branch closed.
- Whitening bias trained to 3.5 epochs (vs 1): -0.14 pp, +0.5% time (E060). Axis closed.
- Head on Muon (RMS-matched lr 0.25, decoupled decay): -0.61 pp (E059). Keep the SGD head.
- BN-bias lr x128 (vs x64, lr*wd fixed): -0.49 pp (E056). x64 is at/above the optimum.
- Head lr x2 (lr*wd fixed): +0.18 pp (E045, kept). Head lr x0.5: -0.41 pp (E039). x3 ~= x2 (E049). Axis closed. The head is not over-driven; a higher head lr
  (x1.5-2) is the untested side.
- Muon momentum 0.8 vs 0.6: +0.18 pp (E026), within re-draw; with lr 0.16 (E024) both point to smoother/smaller
  Muon steps being at least as good. Combined margin now ~0.7 pp at 7 epochs -> 6.5 epochs is the next cash-in.
- Muon lr 0.20 vs 0.16 at bs 1536: -0.10 pp (E046). Muon lr axis flat 0.16-0.24 at both batch sizes; closed.
- Muon lr 0.16 vs 0.24: +0.17 pp at 7 epochs (E024), within re-draw: lr axis flat in [0.16, 0.24], cliff above.
- Muon lr 0.36 (1.5x) costs 1.04 pp at 7.5 epochs (E019, every seed). The Muon lr is sensitive; with weight-norm
  projection it sets the relative per-step rotation of each filter. Lower (0.16) is untested.
- (E066) NorMuon (per-filter 2nd-moment normalisation after NS3, norm-preserving): -0.26 pp. Muon geometry saturated.
- (E065) LESS whitening (pre-NS scale 0.5 -> 0.08, magnitude-matched) costs 1.51 pp. Muon's value is the whitening;
  NS3 at scale 1 is at the low edge of useful strength (NS5 ~= NS3, E017). Only stronger whitening is worth testing.
- NS 3 iterations are as good as 5 (E017: 0.7639 vs 0.7619, within re-draw) and 2.8% faster.
- Muon cost (E016 diagnostic): GPU 2.73 ms/step (NS5) / 1.93 ms (NS3) out of ~20.5 ms. Fused compiled Muon (E023)
  brings NS3 to 1.43 ms/step (-2.7% total time). The remainder is NS matmuls on small matrices + norm/projection.
- The airbench96-style base without TTA clears 0.753 at 10 epochs with ~0.5 pp margin (E001).
- Per-seed accuracy spread at 10 epochs: 0.7568-0.7613 (range 0.45 pp) (E001); at 9 epochs 0.7546-0.7558 (E002).
- (E042) Mild per-image brightness/contrast jitter: +0.16 pp for +0.5% time (re-draw-sized, not harmful).
- (E055) bs 1536 -> 2048 everywhere: -0.45 pp for -3.5% (0.13 pp per 1%); infeasible. Batch size closed at 1536.
- (E041) bs 1024 -> 1536 at fixed epochs: -0.26 pp for -5.2% time (0.05 pp per 1%): a cheaper time lever than epochs.
- (I075 gate, no run) All hyperparameter edits reached the timed path: edit pairs E027/E028, E045/E046, E048/E049 differ
  per seed while identical commits rerun bit-identically (E003). The flat lookahead / Muon-lr / head axes are really
  flat (H1.6.6 refuted).
- (E064) Thinning block-1 updates (frozen every other step) in epochs 3.5-5.4: -0.54 pp for -4.3%. Mid-run block-1
  updates matter; only the tail is cheap to freeze.
- (E063) +0.25 epoch at 32px on the freeze-at-0.7 recipe: only +0.07 pp for +4% time (~0.02-0.04 pp per 1%). The marginal
  epoch is now cheap in accuracy terms too; time/accuracy trade-offs have converged (~0.02-0.05 pp per 1% everywhere):
  near a local optimum. Further gains likely need a new mechanism, not re-balancing.
- (E062) Also freezing block 2 for the last 10%: -0.49 pp for -4.4% (0.11 pp per 1%). Cascade closed at block 1.
- (E061) Block-1 freeze from 70% (vs 80%): -0.11 pp for -2.8% (0.04 pp per 1%); infeasible by 0.04 on the thin margin.
- (E058) FreezeOut-style late freeze of block 1 (last 20%): -0.09 pp for -4.7% (0.02 pp per 1%): the cheapest time
  lever found. Late training of early layers is nearly worthless.
- (E054) 28px stage to epoch 4.0 (from 3.5): -0.22 pp for -2.3% (infeasible by 0.06): boundary price rising, ~0.1 pp/1%.
- (E053) bs 2048 only in the low-res stages (sample-indexed schedules): -0.49 pp for -2.1%. The low-res phase is
  step-starved; stage-wise batch size closed.
- BUILD TIME (E053 diagnostic): a COLD-cache build with 4 max-autotune graph variants takes ~416 s (limit 600 s;
  official runs are cold). Harness builds here look faster (~120-230 s) because Inductor's cache is warm. Every new
  graph variant costs ~100 s cold. Budget graph variants carefully; dropping max-autotune (E007, ~1%) is an option.
  Measured: the tip E052 (6 variants: 20/28/32px x whitening-bias grad on/off) builds in 402 s cold (private cache dir).
  Headroom ~200 s, i.e. at most ~2 more variants. (Never delete /tmp/torchinductor_jovyan: it is shared with other jobs;
  use TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR for cold measurements.)
- (E052) No translate during the low-res epochs: +0.04 pp (neutral).
- (E051) 28px stage to epoch 3.5 (from 3.0): -0.10 pp for -1.5% time.
- (E050) Warmup stage at 16px (final map 1x1): -1.41 pp for only -0.8% time. 20px is the floor; keep final maps >= 2x2.
- (E048) Resolution curriculum 20px -> 28px -> 32px (boundaries 1.5 / 3.0 of 6.75 epochs): 5.000 s at tip accuracy.
  Only the last 3.75 epochs (~56% of steps) run at full resolution.
- (E047) Warmup-stage resolution 24 -> 20px: -0.18 pp for -3.5% time (0.05 pp per 1%). 20px sizes 19/9/4/2.
- (E044) A 28px stage for epochs 1.5-3.0 (after 24px 0-1.5): -0.23 pp for -4.4% time (0.052 pp per 1%); infeasible
  by 0.03 pp on the 0.7549 tip -> a good trade to stack once accuracy is banked elsewhere. Build 232 s (6 graphs).
- (E043) Extending 24px from 1.5 to 2.25 epochs: -0.38 pp for -5.0% time. Low-res is ~free during warmup (~first 23%
  of steps) and costly in the peak-lr phase; boundary ~1.5 epochs.
- (E040) Under Muon, early epochs on whole images DOWNSAMPLED to 24 px (not cropped) are nearly as useful as 32 px
  epochs at ~0.6x the cost: 1.5 low-res + 5.25 full epochs = tip accuracy at -3% time. Sizes at 24px: 23/11/5/2.
- Time scales ~with FLOPs down to 24px at bs 1024: a 24px-crop epoch costs ~0.53 s vs ~0.90 s at 32px (E013).
- Block-1 depth 3 vs 2: +0.40 pp for +7.5% time under SGD (E015); under Muon ~+0.8 pp (E037: depth 2 + 0.5 epoch
  = -0.24 pp at -1.5% time). Capacity cuts are worse trades under Muon.
- Block-3 width 640 vs 576 (Muon, 6.5 ep): +0.34 pp for +5.5% time (E029) = 0.062 pp per 1%, worse than epochs.
  Widths are near a local optimum in both directions.
- Block-3 width 576 vs 512: +0.54 pp for +4% time (E011). Shrinking width is not a free saving.
- Lookahead (decay 0.95^5 x (t/T)^3 every 5 steps) is worth ~1.0 pp at 9 epochs (E008: 0.7443 without vs 0.7542
  with; all seeds -0.7 to -1.2 pp), i.e. >1 epoch. Weight averaging is a major lever; its schedule is untuned for C100.
- Run-to-run noise with fixed seeds: accuracy mean bit-identical, time +-1% (E002 vs E003).
- (I085, offline, seeds 3-12) Tip E058: mean 0.7533 (seeds 0-2: 0.7537), per-seed sd 0.28 pp -> no ratchet inflation;
  13-seed mean ~0.7534, i.e. ~0.33 pp above the OFFICIAL 0.75 bar (40-seed mean sd ~0.045 pp): official risk is low.
  Paired E061-E058 on seeds 3-12: -0.11 pp (same as seeds 0-2), paired per-seed sd ~0.24 pp -> 3-seed paired mean sd
  ~0.14 pp. Small effects (< ~0.15 pp) on 3 seeds remain unreliable.
- (E057) On the Muon+curriculum base, a pure RNG shift (same recipe) moved the mean +0.04 pp, per-seed -0.07/+0.12/+0.09:
  paired per-seed sd ~0.1 pp, 3-seed-mean sd ~0.06 pp. Smaller than on the SGD base; effects >= ~0.12 pp are likely real.
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
- (E065) "Precision doesn't matter" (NS5 ~= NS3) does not imply "strength doesn't matter": weaker whitening fell off a
  cliff. Check both sides of a robust-looking knob before extrapolating.
- (E050) Resolution trends are not linear across structural thresholds (1x1 final map changes pooling semantics);
  tiny maps also stop saving time. Check the final map size before extrapolating.
- (E035) Warm up with the exact shape AND memory layout of the timed path (x.repeat drops channels_last -> cuDNN
  re-autotuned inside trial 0, +0.6 s). In diagnostics, compare trial-0 vs later trials for any new code path.
- (E034) Under torch.compile, freezing via requires_grad doesn't prune the backward graph; make it visible in the
  forward (detach behind a module flag dynamo guards on) and confirm with a per-step timing diagnostic.
- (E025) Separate "lr level" from "tail shape": E019's whole-run lr result did not imply that tail noise was harmful.
  With lookahead, decay-to-zero hurt (E012, E025).
- (E019) Don't assume an inherited hyperparameter sits at a flat optimum: Muon lr 1.5x cost ~1 pp. Use wide ranges for
  optimizer-scale knobs.
- (E009) torch 2.4 Inductor's compiled backward of x.amax(dim=(2,3)) yields NaN grads (eager is fine). After any change
  to the compiled graph, run a one-step finite-gradient check (drafts/debug_nan.py) before launching.
- (E007) Prediction ranges must contain the point estimate and the no-effect outcome; don't anchor on the supports case.
- (E005) Profile before betting on systems ideas: only kernels that are a visible share of GPU time can pay off.
- (E004) Bit-identical reruns hide seed sensitivity: any recipe change re-draws per-seed accuracy (sd ~0.2 pp).

## Dead ends
- LS 0.3 vs 0.2: no meaningful difference at 9 epochs (E004); LS 0.1 vs 0.2 under Muon at 6.5 ep: -0.02 pp (E036).
  LS axis closed (flat 0.1-0.3).
- (E034) Freezing the whitening bias after epoch 0 via a guarded detach flag: -3.0% time, accuracy unchanged.
  requires_grad toggling alone does NOT change the compiled graph (Inductor still computed the dgrad, E034 diagnostic).
- (E032) Slicing the 31x31 whitening output to 30x30: 0% time. Odd spatial sizes are not a cost; conv-shape work closed.
- (E021) SiLU instead of GELU: -2.2% time, accuracy +0.24 pp (not worse). Activation choice is a small systems win.
- (E010) Removing adaptive_max_pool2d's atomic backward (fixed-kernel max_pool2d) saved 2.0% (0.17 s). Systems work so
  far totals ~3%; what is left outside cuDNN convs is memory-bound fused BN/GELU elementwise (~18% of step time).
- Muon-group lr floor: 0 -> -0.51 pp (E025), 0.15x -> -0.17 pp (E033); 0.07x is near the optimum. Axis closed.
- Muon-group lr decay to 0 (instead of the 0.07x floor) costs 0.51 pp (E025). With lookahead, a non-zero tail lr
  helps; decay-to-zero is harmful for both groups. A higher Muon floor (0.15-0.2x) is the untested direction.
- LR schedule: warmup 12% + linear decay to 0 is -0.23 pp vs airbench's 23% warmup / decay to 0.07 (E012). Weak
  evidence (near re-draw size, two knobs bundled); schedule shape is not an obvious lever.
- Kernel selection: Inductor max-autotune buys only ~1.1% (E007, kept). cuDNN conv choice is near-optimal; don't
  expect more from autotuning/compile modes.
- CUDA-graph replay of the compiled Muon step: -0.15% (E038). Launch overhead is hidden; Muon's ~1.4 ms is GPU work.
- Fused/compiled optimizer + lookahead with tensor lr (I009): 0% time change (E005). Launch overhead is not a lever;
  CUDA graphs / fully compiled step are expected to be equally useless here.

## Open questions and surprises
- (after E066) Status: every re-balancing lever now prices at ~0.02-0.1 pp per 1% time (curriculum, freeze, epochs,
  bs), and Muon's geometry is saturated (E030 split, E065 weaker, E066 NorMuon all lose). Untested levers that are new
  mechanisms rather than re-balancing:
  (a) STRONGER whitening: NS4, or NS3 with more aggressive coefficients (E065 shows strength matters; only "weaker"
      and NS5 at scale 1 were tested);
  (b) BN momentum (0.6 inherited from airbench; never tested) and BN eps;
  (c) head output scale 1/9 (never tested) and global pool type (max only; max+avg concat untested);
  (d) whitening patch size / width (2x2, 24 channels; never tested); dirac init on/off;
  (e) Muon momentum 0.9 (0.6 -> 0.8 was +0.18 pp, E026; the upward direction is open);
  (f) per-stage lr schedule tied to resolution (warmup ends at the 20px -> 28px switch; untested coupling).
- (after E032) Where the remaining time goes at the tip (6.08 s, 312 steps, ~19.0 ms/step): fwd/bwd convs ~70%,
  elementwise BN/SiLU ~15%, Muon ~1.4 ms (~7%), lookahead/augment/prepare ~2-3%. Untested levers, by my estimate of value:
  (1) batch size under Muon (768 or 512 with lr ~/sqrt2): more steps per epoch may buy accuracy per epoch; NS cost per
      epoch rises with step count;
  (2) Muon on the 2D head (100 x 576) [LS re-check done: flat, E036];
  (3) pure-bf16 weights/activations for BN (memory-traffic cut on the ~15% elementwise share);
  (4) lookahead sync cadence (every 3 vs 5 steps) -- the window length was flat (E028), cadence is untested;
  (5) [tested E033: higher Muon floor 0.15x is -0.17 pp; closed]
  (6) cutout / stronger aug: likely bad at <7 epochs (more difficulty) -- low priority.
- Exchange rates near the knee (accuracy per 1% of time; higher = component is more valuable per unit time):
  epochs 8.5->9 ~0.096 (E006: 0.5 pp / 5.2%), 9->10 ~0.03 (E001/E002: 0.31 pp / 11%); block-3 width 512->576 ~0.135
  (E011); block-1 depth 2->3 ~0.054 (E015); 24px->32px for 3 of 9 epochs ~0.058 (E013); lookahead ~1.4 (E008);
  Muon vs SGD ~0.10 at its current 13.4% overhead (E014) -- but Muon's gain is a level shift worth ~2 epochs.
  Components with a rate below the epoch rate (~0.1) are candidates to cut and re-spend on epochs. Averaging/regularisation tricks that
  cost little time are the best bargains; look for more of those (EMA schedule, BN tweaks).
- Cheap systems wins visible in the E003 profile: adaptive_max_pool2d backward uses a slow atomic kernel (0.48 ms of
  18.8 ms/step, ~2.5%); a plain amax/max_pool2d(4) head would likely remove most of it with identical math.
- NOTE: the unpadded 2x2 whitening conv outputs 31x31, so the real spatial sizes are 31/15/7/3, not 32/16/8/4
  (FLOP counts below are slightly overstated; the final max_pool2d kernel is 3).
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
E016 (I018, H1.2.1): Muon at 8 epochs: acc 0.7619, 8.114 s -> supports; force-kept as base; Muon GPU cost 2.7 ms/step.
E017 (I019, H1.2.1): Muon NS3: acc 0.7639, 7.887 s (-2.8%) -> supports; new best.
E018 (I020, H1.2.1): Muon 7.5 epochs: acc 0.7596, 7.390 s -> supports; new best; knee likely ~7 epochs.
E019 (I021, H1.2.1): Muon lr 0.36: acc 0.7492 (-1.04 pp) -> refutes; lr sensitive, try lower next.
E020 (I022, H1.2.2): Muon without lookahead: acc 0.7510 (-0.86 pp) -> supports; averaging complementary to Muon.
E021 (I023, H2.4.1): SiLU: acc 0.7620, 7.231 s (-2.2%) -> inconclusive by rule; kept, new best.
E022 (I025, H1.2.1): Muon 7 epochs (SiLU): acc 0.7582, 6.740 s -> supports; new best; margin 0.5 pp.
E023 (I026, H2.4): fused compiled Muon step: acc 0.7568, 6.559 s (-2.7%) -> supports; new best.
E024 (I027, H1.2.1): Muon lr 0.16: acc 0.7585 (+0.17), 6.531 s -> inconclusive; kept (force); lr flat 0.16-0.24.
E025 (I028, H1.6.2): Muon lr decays to 0: acc 0.7535 (-0.51 pp) -> refutes; the 0.07 floor helps with lookahead.
E026 (I029, H1.2): Muon momentum 0.8: acc 0.7603 (+0.18), 6.548 s -> inconclusive; kept (force); margin 0.7 pp.
E027 (I031, H1.2.1): Muon 6.5 epochs: acc 0.7546, 6.094 s -> supports; new best; margin 0.16 pp.
E028 (I032, H1.6.1): lookahead decay 0.97^5: acc 0.7546 (=), 6.077 s -> inconclusive; kept (tie); window not a lever.
E029 (I033, H1.5.1): block 3 640 wide: acc 0.7580 (+0.34), 6.412 s (+5.5%) -> inconclusive; worse trade than epochs.
E030 (I030, H1.2): frequency-wise Muon-C: acc 0.7426 (-1.2 pp), +3.7% -> refutes; closed.
E031 (I034, H1.2): Muon group no warmup: acc 0.7541 (-0.06), same time -> inconclusive; discard; warmup axis flat.
E032 (I035, H2.5.2): whitening output sliced to 30x30: 0% time -> refutes; discard.
E033 (I036, H1.6.4): Muon floor 0.15x: acc 0.7529 (-0.17) -> refutes; 0.07x optimal; tail axis closed.
E034 (I037, H2.4.2): whiten bias 1 epoch then detached: acc 0.7551, 5.894 s (-3.0%) -> supports; new best.
E035 (I038, H1.6.5): BN recalibration: acc +0.03 pp, time +4.8% (warmup layout gap) -> refutes; discard.
E036 (I039, H1.7.2): LS 0.1: acc 0.7549 (-0.02) -> inconclusive; discard; LS axis closed.
E037 (I040, H5.1.1): block-1 depth 2 + 7 epochs: acc 0.7527 (-0.24), 5.808 s -> refutes; depth worth more under Muon.
E038 (I041, H2.4.4): CUDA-graph Muon step: -0.15% time -> refutes; discard; Muon cost is GPU work.
E039 (I044, H1.7.1): head lr x0.5: acc 0.7510 (-0.41) -> refutes; try higher head lr instead.
E040 (I046, H2.3.2): 1.5 epochs at 24px downsampled + 5.25 at 32px: acc 0.7559, 5.718 s (-3.0%) -> supports; new best.
E041 (I042, H1.2.3): bs 1536: acc 0.7533 (-0.26), 5.422 s (-5.2%) -> inconclusive; kept; new best; margin 0.03 pp.
E042 (I043, H1.8.1): brightness/contrast jitter: acc 0.7549 (+0.16), 5.449 s -> inconclusive; kept (force); margin 0.19.
E043 (I051, H2.3.2): 24px for 2.25 epochs: acc 0.7511 (-0.38), 5.178 s (-5%) -> inconclusive; infeasible; discard.
E044 (I052, H2.3.3): +28px stage: acc 0.7527 (-0.23), 5.209 s (-4.4%) -> inconclusive; infeasible by 0.03; pair later.
E045 (I053, H1.7.3): head lr x2: acc 0.7568 (+0.18), 5.464 s -> inconclusive; kept (force); margin 0.38 pp.
E046 (I056, H1.2.3): Muon lr 0.20 at bs 1536: acc 0.7558 (-0.10) -> inconclusive; discard; lr axis closed.
E047 (I060, H2.3): first stage 20px: acc 0.7550 (-0.18), 5.273 s (-3.5%) -> supports; new best.
E048 (I061, H2.3.3): +28px stage on the 20px tip: acc 0.7552, 5.000 s (-5.2%) -> supports; new best.
E049 (I066, H1.7.3): head lr x3: acc 0.7548 (-0.03) -> inconclusive; discard; head axis closed.
E050 (I067, H2.3): first stage 16px: acc 0.7411 (-1.41), -0.8% time -> refutes; 20px floor.
E051 (I069, H2.3.3): 28px stage to 3.5 ep: acc 0.7541 (-0.10), 4.926 s (-1.5%) -> inconclusive; kept (deterministic FLOP cut).
E052 (I070, H1.8.2): no translate in low-res epochs: acc 0.7546 (+0.04) -> inconclusive; kept (force).
E053 (I072, H1.2.3): bs 2048 in low-res stages: acc 0.7497 (-0.49), 4.827 s (-2.1%) -> refutes; discard.
E054 (I074, H6.3.1): 28px to 4.0 ep: acc 0.7524 (-0.22), 4.816 s (-2.3%) -> inconclusive; infeasible by 0.06.
E055 (I057, H1.2.3): bs 2048: acc 0.7501 (-0.45), 4.755 s (-3.5%) -> inconclusive; infeasible; bs axis closed.
E056 (I064, H1.7.1): BN-bias lr x128: acc 0.7497 (-0.49) -> refutes; discard.
E057 (I068, H4.1.4): RNG-shift noise probe: mean +0.04, per-seed <= 0.12 -> supports; re-draw noise small; discarded.
E058 (I076, H2.4.7): block 1 frozen for last 20%: acc 0.7537 (-0.09), 4.696 s (-4.7%) -> supports; new best.
E059 (I077, H1.10.1): head on Muon: acc 0.7476 (-0.61) -> refutes; discard.
E060 (I078, H6.1.1): whitening bias trained to 3.5 ep: acc 0.7523 (-0.14), +0.5% -> inconclusive; discard.
E061 (I079, H2.4.7): freeze block 1 from 70%: acc 0.7526 (-0.11), 4.565 s (-2.8%) -> inconclusive; infeasible by 0.04.
E062 (I080, H2.4.7): freeze block 2 last 10%: acc 0.7489 (-0.49), -4.4% -> refutes; cascade stops at block 1.
E063 (I081, H2.4.7): freeze@0.7 + 7.0 epochs: acc 0.7533, 4.751 s (slower) -> refutes; +0.25 ep worth only +0.07 pp.
E064 (I082, H2.4.7): block-1 update thinning 3.5-5.4 ep: acc 0.7483 (-0.54) -> refutes; discard.
E065 (I083, H1.9): scheduled weaker whitening (magnitude-matched): acc 0.7386 (-1.51) -> refutes; whitening essential.
E066 (I084, H1.2.6): NorMuon: acc 0.7511 (-0.26) -> refutes; discard; Muon geometry saturated.
