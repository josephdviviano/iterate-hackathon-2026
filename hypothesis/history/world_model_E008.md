# World model

_Last updated after: E008_

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
- Do not predict time savings from CPU-side overhead cuts; the GPU is power-capped and energy-bound (E004).

## Dead ends
- Brightness/contrast jitter (c 0.15, b 0.1) at 8.0 ep: -0.16 pp vs control, +0.3% time (E008). Short schedules
  under-fit; more augmentation does not help.
- Last group 576 -> 768 with 8.5 -> 7.5 epochs: -2.9% time but -0.36 pp (E005). Per-epoch cost +10%.
- GELU -> SiLU: -2.2% time but -0.6 pp accuracy (E003). Activation is not a free knob on CIFAR-100.

## Open questions and surprises
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
