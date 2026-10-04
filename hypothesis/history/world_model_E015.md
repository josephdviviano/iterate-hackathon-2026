# World model

_Last updated after: E015_

## Current best
E001: futurebiohackers airbench96-style recipe as published (widths 128/384/576, depth 3, dirac init, 8.5 ep, bs 1024,
SGD-Nesterov lr 9 / kilostep scaling, LS 0.3, lookahead EMA, fp16 channels_last, compile max-autotune):
accuracy 0.7549, time 7.222 s. Margin above 0.753 only ~0.19 pp.

## Measurement
- Score = mean(prepare + train) over 3 seeds; feasibility = mean acc >= 0.753 (3 seeds), official 75% over 40 seeds.
- First trial pays lazy CUDA/cuDNN init (~0.33 s in E000) unless build warms up the exact same kernels (E000).
- Our A100-SXM4 runs the reference recipe in 7.22 s vs 8.11 s quoted for it (E001): anchor on own numbers.
- Accuracy is bit-deterministic given the seed (E001 == E002 per seed): any accuracy change is a real effect on
  these 3 seeds, but generalization to 40 seeds has across-seed std ~0.16-0.25 pp. Time noise ~0.1-0.5% (E002), so a
  >= 1% time change is real.
- Per-seed accuracy (E001): 0.7531, 0.7556, 0.7561 -> seed 0 is the weakest so far.
- Reference point from README: ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe.

## Where the cost goes
Epoch cost by training resolution (measured after E013, s/epoch): 24 px 0.485 | 26 px 0.632 | 28 px 0.671 |
30 px 0.808 | 32 px 0.848. Stepwise (post-pool map sizes), not quadratic: 24 px is the sweet spot (-43%).
Per-image fwd MACs (128/384/576, depth 3): group1 ~93M, group2 ~230M (48%), group3 ~151M; conv1 of groups 2 and 3
run at pre-pool resolution (15x15, 7x7) and are ~100M each. 1 epoch ~ 0.85 s ~ 11.8% of the 8.5-epoch run.
**Training is power-capped** (E004 + nvidia-smi): during training the GPU sits at its 400 W cap (throttle 0x4 SW power
cap, SM 1335-1380 MHz vs 1410 max). Time ~ GPU energy. Removing CPU-side gaps (prepare dirac loop, -45 ms) did not
lower the total: train grew by the same amount (E004). Only less GPU work (FLOPs, bytes moved) shortens a trial.
Official A100 PCIe is capped at 300 W -> even more energy-bound.
E001: prepare 0.06-0.07 s steady (0.146 s first trial), train ~7.13 s for 8.5 epochs (~0.84 s/epoch incl. aug).
Training compute dominates (~99%).
Prepare breakdown (profiled after E003, steady state ~63 ms): nn.init.dirac_ ~45 ms (Python loop, ~2700 tiny H2D
copies + syncs), uint8 pageable H2D 7.3 ms (pinned would be 5.8), normalize 5.5 ms, reflect pad 3 ms, whitening
patches+eigh ~2-6 ms. Trials 1-2 of a fresh process show ~130-145 ms prepare (extra first-use cost not covered by
build's synthetic warmup), trial 3+ ~63 ms. Reference write-up: ~95% of GPU time is conv/BN/GELU kernels.

## Established facts
- Prog. resize 26/28/32 px over 30/30/40% of 9.0 epochs: 6.665 s (-7.7%), 0.7506 (-0.44 pp) (E013): 0.057 pp per 1%
  time, worse than E006's 24/28/32 (0.037) because 26/28 px are not much cheaper than 32.
- Epoch slope near 8-8.5 ep: 8.0 ep -> 0.7502 (-0.48 pp), 6.804 s (-5.8%) (E007). ~0.08 pp accuracy per 1% time.
  Progressive resizing (E006) trades at ~0.037 pp per 1% -> 2x better than cutting epochs.
- Progressive resizing 24->28->32 px at 9.0 epochs (40%/30%/30% of steps): 5.880 s (-18.6%) at 0.7480 (-0.69 pp)
  (E006). Best exchange rate found: low-res epochs cost ~0.47 s (24 px) / ~0.65 s (28 px) vs 0.84 s (32 px).
- Shrinking prepare 63 -> 18 ms did not change total time (E004): per-trial totals conserved by the power cap.
- A pure perturbation (whitening subset 5000->1500, fp16 normalization) moved the 3-seed mean by -0.2 pp (E004):
  3-seed accuracy has ~0.15-0.2 pp of "perturbation noise"; target >= 0.755 for a safe keep.
- The reference recipe already includes dirac init on all 3x3 convs (Conv.reset_parameters), so "add dirac" is moot (code).
- Harness overhead floor is ~0.003 s per trial after warmup (E000).

## Lessons from failures
- Cutting full-res epochs is the most expensive saving (~0.95 pp/epoch, E007); ideas must beat 0.08 pp per 1% time.
- Short schedules under-fit: extra augmentation (E008) does not buy epochs; activation swaps cost accuracy (E003).
- Any numerically different change perturbs per-seed accuracy by up to ~0.75 pp and the 3-seed mean by ~0.3 pp
  (E004, E011). The E001 margin (0.19 pp) is too thin to keep neutral engineering changes; buy margin first.
- Do not predict time savings from CPU-side overhead cuts; the GPU is power-capped and energy-bound (E004).

## Dead ends
- LR decay to 0 instead of 0.07x at 8.0 ep: -0.16 pp vs control (E015). Reference schedule near-optimal.
- 26 px instead of 24 px in E006's first phase: 0.7470 (-0.1 pp vs E006) at +10% time (E014). 24 px dominates;
  the 2x2 final map at 24 px is fine.
- Selective backprop (top 614/1024 by loss from epoch 3): -0.9% time, -0.64 pp (E012). 9x worse than epoch cuts.
- Fused SGD: -0.4% time, accuracy perturbed -0.3 pp (E011). Overhead line (H3) is exhausted.
- Batch size 1024 -> 2000 (kilostep scaling, lookahead every 3): time unchanged (7.216 s) with half the steps;
  accuracy -0.8 pp (E010). Per-step overhead is invisible under the power cap.
- Label smoothing 0.3 -> 0.2 at 8.0 ep: -0.09 pp vs control, same time (E009). LS 0.3 stays.
- Brightness/contrast jitter (c 0.15, b 0.1) at 8.0 ep: -0.16 pp vs control, +0.3% time (E008). Short schedules
  under-fit; more augmentation does not help.
- Last group 576 -> 768 with 8.5 -> 7.5 epochs: -2.9% time but -0.36 pp (E005). Per-epoch cost +10%.
- GELU -> SiLU: -2.2% time but -0.6 pp accuracy (E003). Activation is not a free knob on CIFAR-100.

## Open questions and surprises
- Would a smaller batch (e.g. 768/512) buy accuracy at ~equal time, since step count is free under the power cap (E010)?
- Which progressive-resizing schedule recovers 0.753+ while keeping most of the -18%? (E006 follow-ups.)
- How few epochs / how small a net can reach 75.3% on CIFAR-100? (airbench-style tricks: whitening init, lookahead/EMA,
  label smoothing, flip+translate aug, bf16, channels_last, torch.compile.)

## Changelog
- E000 (baseline): template acc 0.0122, time 0.113 s -> first-trial lazy init cost noted; need a real recipe.
- E001 (I001, H1.1.1): reference recipe acc 0.7549, 7.222 s, feasible -> new best; machine is ~11% faster than reference's quote.
- E002 (I002, H4.1.1): identical rerun, same per-seed acc, time 7.214 s (-0.1%) -> noise floor tiny; acc deterministic.
- E003 (I004, H2.2.1): SiLU acc 0.7489 (-0.6 pp), 7.061 s (-2.2%) -> discard; GELU matters for accuracy.
- E004 (I005, H3.2.1): prepare -70% but total 7.219 s (unchanged), acc 0.7529 -> discard; GPU is power-capped, time ~ energy.
- E005 (I006, H1.3.1): 768/7.5ep acc 0.7513, 7.013 s (-2.9%) -> discard; widening group 3 does not buy an epoch.
- E006 (I007, H2.1.1): prog. resize 24/28/32 @9ep acc 0.7480, 5.880 s (-18.6%) -> discard (infeasible) but best lever.
- E007 (I009, H1.4.1 control): 8.0 ep acc 0.7502, 6.804 s -> discard; slope ~0.95 pp/epoch, steep.
- E008 (I010, H1.4.1): color jitter @8.0ep acc 0.7486 (-0.16 pp vs control), 6.826 s -> discard; more aug doesn't help.
- E009 (I011, H1): LS 0.2 @8.0ep acc 0.7492 (-0.09 pp vs control) -> discard; reference regularization near-optimal.
- E010 (I008, H3.3.1): bs 2000 acc 0.7468 (-0.8 pp), 7.216 s (unchanged) -> discard; steps are free, FLOPs are not.
- E011 (I012, H3.1.1): fused SGD acc 0.7520, 7.194 s (-0.4%) -> discard; overhead line exhausted.
- E012 (I013, H2.3.1): selective backprop acc 0.7485, 7.158 s (-0.9%) -> discard; dead end.
- E013 (I014, H2.1.2): 26/28/32 @9ep acc 0.7506, 6.665 s (-7.7%) -> discard; measured stepwise epoch-cost curve, 24 px is the sweet spot.
- E014 (I015, H2.5.1): E006 with 26 px acc 0.7470, 6.491 s -> discard; 24 px dominates 26 px.
- E015 (I016, H6.1.1): final_lr 0 @8.0ep acc 0.7486 (-0.16 pp) -> discard; third null hparam tweak.
