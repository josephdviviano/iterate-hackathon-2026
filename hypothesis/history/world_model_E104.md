# World model

_Last updated after: E104_

## Current best
E100 (commit 176e5ec): E096 recipe + Muon LR x1.25 during the 32 px phase. Scored 0.7561 / 4.467 s and 0.7516 / 4.462 s
(distinct mean 0.7538); off-seed 22 seeds 0.7523 (E096 baseline 0.7514, E090 0.7525; one process each).
Recipe: airbench96-style net (whiten 2x2 stem, 128/384/576 x3 convs, dirac init, max-values global pool), Muon
(lr 0.24 (x1.25 at 32 px), mom 0.6, 3 NS, compiled + batched, sparse renorm, floor 0.05 at 96% of samples) on 9 convs +
head (0.5x lr), SGD-Nesterov (kilostep rule per phase batch, 64x BN-bias LR) for biases, LS 0.3, lookahead until 8
steps before the end, then a light final EMA (lerp 0.3 every 4 steps). 7.25 ep: 2@16 + 1.25@24 at bs 1536, 1.25@28 +
2.75@32 at bs 1024; samples-indexed schedules; static graphs; pinned upload; fp16 normalization.
History: E001 SGD 0.7549/7.222 -> E020 Muon 0.7583/6.663 -> E024 compiled Muon 0.7594/5.996 -> E026 16/24/32
0.7565/5.501 -> E038 bs 1536 0.7542/5.216 -> E046 3@16 0.7553/4.955 -> E057 static+maxpool 0.7534/4.717 -> E062
prepare+head LR 0.7533/4.680 -> E068 tail 0.7579/4.670 -> E070 7.25-ep schedule 0.7542/4.584 -> E086 pinned upload
0.7532/4.561 -> E090 28 px + bs 1024 tail 0.7531/4.459 -> E096 8-step window 0.7539/4.467 -> E100 Muon x1.25 at 32 px.

## Measurement
- Per-process cuDNN-pick clusters shift the whole run (3 seeds or 22 seeds) by up to ~0.25 pp (E094, E098): a single
  process per arm cannot resolve 0.1 pp. 22-seed off-seed: E096 0.7514, E090 0.7525; E090 scored distinct samples
  0.7531/0.7517/0.7496. Tip family true 3-seed mean ~0.752-0.753, official 40-seed ~0.751-0.752.
- Determinism is recoverable only via cuDNN determinism (E036: bit-identical, but +9.8% time). Default-mode Muon
  compile alone does not help (E042: seed-0 reruns 0.7522/0.7549/0.7530). Source = cuDNN algorithm picks per
  process. Accept the noise: single-seed sd ~0.15 pp, 3-seed mean sd ~0.1 pp.
- **Run-to-run accuracy noise exists for the compiled-Muon recipes** (E028): an identical rerun of E026 gave 0.7543 vs
  0.7565 (per-seed -0.27/+0.19/-0.59 pp). Likely max-autotune kernel choice for the Muon update varies by process.
  The SGD recipe (E001/E002) was bit-deterministic. Treat 3-seed differences < ~0.25 pp as noise; keep >= 0.2 pp margin.
- Tip (E026 code) over seeds 3-10 (untimed side check): mean 0.7553, sd 0.22 pp; 11-seed estimate ~0.7553.
  Seeds 0-2 are not systematically favorable once the rerun is included (E026+E028 average 0.7554).
- Score = mean(prepare + train) over 3 seeds; feasibility = mean acc >= 0.753 (3 seeds), official 75% over 40 seeds.
- First trial pays lazy CUDA/cuDNN init (~0.33 s in E000) unless build warms up the exact same kernels (E000).
- Our A100-SXM4 runs the reference recipe in 7.22 s vs 8.11 s quoted for it (E001): anchor on own numbers.
- Accuracy is bit-deterministic given the seed (E001 == E002 per seed): any accuracy change is a real effect on
  these 3 seeds, but generalization to 40 seeds has across-seed std ~0.16-0.25 pp. Time noise ~0.1-0.5% (E002), so a
  >= 1% time change is real.
- Per-seed accuracy (E001): 0.7531, 0.7556, 0.7561 -> seed 0 is the weakest so far.
- Reference point from README: ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe.

## Where the cost goes
Profile of the E046 tip (E052): GPU busy ~98-100% of wall in all phases (no launch gaps); fused Triton elementwise/BN
kernels ~30% of GPU time; cuDNN conv fprop/dgrad/wgrad most of the rest; AdaptiveMaxPool2d backward
(atomicadaptivemaxgradinput) 25-28 ms per epoch (~4% of the run!) -- a fixable pathology; aug+resize <= 10 ms/epoch.
Per-phase power on the E044 tip (E045): 16 px epochs 282 W, util 82%, max clock -> overhead-bound (0.265 s/epoch);
24 px at the 400 W cap 84% of the time (0.52 s); 32 px at the cap ~100% (0.885 s). Only the 16 px phase (~10%) has
launch-overhead headroom. Static per-resolution graphs (automatic_dynamic_shapes=False) are ~1.6% faster than the
default dynamic graphs (24 px 0.498 vs 0.511, 32 px 0.870 vs 0.885 s/epoch at bs 1536) -- untested in a scored run.
12 px epoch 0.186 s (0.21x). E044 and E045 were bit-identical (determinism varies with cuDNN picks).
Epoch cost incl. compiled Muon (measured before E026, s/epoch): 16 px 0.297 (0.32x) | 20 px 0.43 (0.47x) |
24 px 0.553 (0.60x) | 32 px 0.921.
Epoch cost by training resolution, SGD (measured after E013, s/epoch): 24 px 0.485 | 26 px 0.632 | 28 px 0.671 |
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
- Extra updates (smaller batches) buy ~+0.1 pp per +2.5-3% time regardless of phase (E084 tail, E104 early): ~0.04 pp
  per 1% -- roughly break-even against schedule cuts.
- 28 px substitution for the first 1.25 epochs of a bs-1024 32 px phase: equal accuracy to the old tip at -2.2% time
  (E090/E091); beats trimming the same phase (E088/E089) by ~+0.09 pp at equal time. 28 px costs 0.80x at bs 1024.
- 32 px phase at bs 1024 (low-res at 1536): pool 0.7555 (+0.18 pp), off-seed +0.11 pp, +2.5% time (E084/E085). A margin
  source at ~0.05-0.07 pp per 1%.
- Fable-style tail (early LR floor + final light EMA over last 8 steps): +0.25 pp at no time cost (E068/E069 pool
  0.7562 vs 0.7537). The final EMA itself carries ~0.15-0.2 pp (E079 ablation: -0.15 scored, -0.18 off-seed).
- 2@16 + 1.5@24 + 4@32 (7.5 ep) vs tip 3@16 + 1@24 + 4@32 (8 ep), same code otherwise: pool 0.7552 vs 0.7540 at +0.9%
  time (E064/E065) -> 24 px time buys accuracy at ~0.13 pp per 1%, 16 px epochs are worth less. Schedule shape matters.
- Prepare cut (vectorized dirac + fp16 norm) now saves -0.046 s (-0.9%) (E060): train starts in the non-capped
  16 px phase, so the E004 absorption no longer applies. Not banked (accuracy draw 0.7527).
- Global max via max(dim).values instead of AdaptiveMaxPool2d (atomic backward): -0.14 s (-2.9%) on top of static
  shapes (E056: 4.718 s). Static + maxpool together: -0.24 s vs tip at equal math. Not banked (accuracy draws <0.753).
- Static per-resolution graphs: -2.0% time at equal math (E053: 4.858 s); not banked only because that run's accuracy
  draw (0.7524) fell below 0.753. With the tip's ~0.1 pp margin, ~1 in 3 tip-equivalent runs lands below 0.753.
- Fractional trim (half a 24 px epoch, 7.5 ep, step-indexed switch) on E044: -0.16 pp, -0.274 s (E047) -- ties the
  24->16 swap (E046). Both trade at ~0.03-0.04 pp per 1%.
- Head (576->100) on Muon instead of SGD: +0.32 pp at equal time (E044). Muon on all weight matrices helps CIFAR-100.
- bs 1024 -> 1536 under Muon (E035 base): -0.30 pp, -0.257 s (-4.7%) (E038). Per-step costs in the low-res phase
  are ~2 ms/step (more than Muon's ~1 ms): fewer steps pay off there.
- Muon renorm every 2+int(15*progress) steps: -0.5..-0.9% time, accuracy 0.7572 (E035). Renorm is ~0.08 ms/step.
- 4@16+4@32: 0.7486, 4.994 s (E032) -> -0.68 pp vs tip. Value of 16 px epochs drops after the first two
  (~0.15 pp each for epochs 0-1, ~0.3-0.4 pp each for epochs 2-3).
- 3@16+1@24+4@32: 0.7523, 5.235 s (E029) and 0.7522, 5.216 s on the E035 base (E040) -- replicated: 3rd 16 px
  epoch costs ~0.3-0.5 pp; bs 1536 (E038) reaches the same time with +0.2 pp more -> dominates.
- 2@16+2@24+4@32 vs 4@24+4@32 (compiled Muon): -0.29 pp, -0.495 s (E026): 0.035 pp per 1%. Early epochs tolerate
  16 px; the 32 px tail carries accuracy (contrast E025: cutting the tail costs 0.078 pp per 1%).
- Split 4@24+4@32 -> 5@24+3@32 under Muon: -0.51 pp, -0.39 s (E025). A 24 px epoch is worth ~half a 32 px epoch.
- Compiled + batched Muon update: -0.335 s (-5.3%) at equal accuracy (E024). Remaining Muon overhead ~1 ms/step.
- Muon at bs 2000 (8.5 ep): -0.25 pp vs bs 1024, -0.40 s (half the eager Muon overhead) (E022). Muon tolerates large
  batches far better than SGD (-0.81 pp, E010). The time gain is purely the Muon step count.
- Muon + 24 px for epochs 0-3 + 32 px for 4-7 (8.0 ep): 0.7578, 6.331 s (E021). Muon and low-res stack; the cost
  model (per-res epoch cost + ~1.9 ms/step eager Muon) predicts time within 2%.
- Muon at 7.0 ep: 0.7583, 6.663 s (E020) -> new tip. Muon epoch slope ~0.54 pp/epoch (8.5 -> 7.0), flatter than
  SGD's ~0.95 pp/epoch. Eager Muon adds ~0.1 s/epoch (0.94 vs 0.84 s/epoch).
- **Muon** (lr 0.24, mom 0.6, 3 NS steps, airbench94_muon style) on the 9 group convs at 8.5 ep: accuracy 0.7663
  (+1.14 pp) but 8.041 s (+11%, eager NS ~2 ms/step) (E019). Optimizer quality is the binding constraint; Muon's
  gain is worth ~1.2 epochs. Next: fewer epochs with Muon, cheaper Muon step, Muon + 24 px progressive.
- Low-res operator: random crop (no rescale) vs antialiased downsample at 26/28 px: +0.08 pp, +0.3% time (E016) --
  no meaningful difference. Either works.
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
- torch 2.4 Inductor: compiled x.flatten(2).amax(dim=2) in this fp16 net produces NaN weights within 10 steps (E055);
  max(dim).values is fine. Smoke-test finiteness for op swaps inside compiled graphs.
- Cutting full-res epochs is the most expensive saving (~0.95 pp/epoch, E007); ideas must beat 0.08 pp per 1% time.
- Short schedules under-fit: extra augmentation (E008) does not buy epochs; activation swaps cost accuracy (E003).
- Any numerically different change perturbs per-seed accuracy by up to ~0.75 pp and the 3-seed mean by ~0.3 pp
  (E004, E011). The E001 margin (0.19 pp) is too thin to keep neutral engineering changes; buy margin first.
- External negative results (e.g. "Muon lost on CIFAR-100") are weak priors; E019 contradicted it by +1.14 pp.
- Do not predict time savings from CPU-side overhead cuts; the GPU is power-capped and energy-bound (E004).

## Dead ends
- cudnn.benchmark_limit = 0 (exhaustive algorithm search): no time gain, build 270 s (E099).
- Removing the lookahead EMA: -0.3 pp scored, -0.18 off-seed, only -0.4% time (E092). Lookahead stays.
- 32 px phase trim 4 -> 3.75 epochs (bs 1024 tail): pool 0.7522, -0.33 pp vs E084, -2.4% vs tip (E088/E089).
- Renorm target decaying to 0.157x with unchanged Muon LR: collapse to ~0.49-0.51 (E082). lr/||w|| is the real knob.
- Final EMA window 20 steps instead of 8: -0.3 pp scored, -0.09 off-seed (E081); uniform LAWA over 20 steps:
  -0.16 scored, -0.29 off-seed (E083); 20-step EMA incl. BN stats: off-seed +0.0 (E103). Keep the 8-step EMA.
- MaxSup instead of LS 0.3: -0.35 pp scored, -0.18 pp off-seed (E078). LS 0.3 stays.
- Build-side dress rehearsal for trial 1 (E076): trial-1 prepare still +0.08 s; the slow H2D window (time-based, ~0.4 s)
  starts after build returns. Negligible over 40 trials; closed.
- Head Muon weight decay (wd*lr 1.6e-3): pool 0.7544 vs 0.7537 (E074/E075) -- noise.
- Equal-time low-res reshuffle 1.5@16+1.5@24 vs 2@16+1.25@24 (+4@32 tail): pools 0.7532 vs 0.7537 (E072/E073) -- neutral.
- Inductor coordinate-descent + pointwise autotune: -0.1% time, build 554 s (E066). Closed.
- Whitening-bias training 37.5% -> 20% of steps: noise (E061); -> 0.2 epoch (~3%) with LR decay: -0.44 pp off-seed,
  -0.56 scored, for -0.7% time (E093); -> 70%: no gain, +1.2% time (E102). Closed at ~41%.
- LR warmup 0.115: -0.2 pp (E051); 0.46: -0.1 pp (E059). Warmup stays at 0.23.
- Group 2 384 -> 352 under Muon: -2.6% time, -0.22 pp (E050). Step-time profile (E080): 352 recovers ~33% and 320
  ~60% of the FLOP-proportional saving -> width cuts closed.
- Head Muon LR 0.5x: E049 0.7558, E062 0.7533, E063 0.7546 -> pooled +0.15 pp: small real gain, kept. 0.4x: -0.3 pp
  scored, -0.07 off-seed (E077) -> closed at 0.5x.
- bs 2048 in the low-res phases (tail 1536): -1.1% time, -0.49 pp (E043). Batch-size lever exhausted.
- 20 px bridge instead of 24 px (epochs 2-3): -0.38 pp, -0.247 s (E041). Worse trade than bs 1536.
- Muon lr 0.24 -> 0.20: -0.03 pp (noise) (E034). Muon hparams (lr, momentum, NS steps) are all at a flat optimum.
- Muon NS iterations 3 -> 2: -1.82 pp for -2.3% time (E031); 3 -> 4: +0 pp for +2.5% (E033). 3 is the sweet spot.
  Real Muon-input spectra are heavy-tailed (median normalized sv 0.02); fixed x3 puts 56% of sv in [0.7,1.3], x4 83%,
  Polar-Express 3-step (l=1e-3) only 21% -> coefficient tuning is closed (analysis in E033).
- Muon momentum 0.6 -> 0.85: -1.18 pp (E027); 0.6 -> 0.5: -0.04 pp (noise, E030). Flat at 0.5-0.6; keep 0.6.
- tanh-approximate GELU: same accuracy, +5.1% time (E023). Exact erf GELU is cheaper under Inductor.
- BN re-estimation at 32 px after E006's schedule: -0.34 pp, +1.1% time (E018). Running stats (fast EMA, momentum
  0.4/step) are already fresh; re-estimating hurts.
- Group 2 384 -> 320: -8.2% time, -0.94 pp (E017): 0.115 pp per 1%, worse than epoch cuts. Capacity is valuable.
- LR decay to 0 instead of 0.07x at 8.0 ep: -0.16 pp vs control (E015). Reference schedule near-optimal.
- 26 px instead of 24 px in E006's first phase: 0.7470 (-0.1 pp vs E006) at +10% time (E014). 24 px dominates;
  the 2x2 final map at 24 px is fine.
- Selective backprop (top 614/1024 by loss from epoch 3): -0.9% time, -0.64 pp (E012). 9x worse than epoch cuts.
- Fused SGD: -0.4% time, accuracy perturbed -0.3 pp (E011). Overhead line (H3) is exhausted.
- Batch size 1024 -> 2000 (kilostep scaling, lookahead every 3): time unchanged (7.216 s) with half the steps;
  accuracy -0.8 pp (E010). Per-step overhead is invisible under the power cap.
- Label smoothing 0.3 -> 0.2: -0.09 pp under SGD (E009), -0.1..-0.2 pp (noise) under Muon (E037). LS 0.3 stays.
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
- E016 (I018, H2.5.2): crop mode @26/28 acc 0.7513 (+0.08 vs E013), 6.687 s -> discard; operator is second-order.
- E017 (I019, H2.6.1): group2 320 acc 0.7455 (-0.94 pp), 6.631 s (-8.2%) -> discard; width cuts are a bad trade.
- E018 (I020, H6.2.1): BN re-estimation on E006 acc 0.7446 (-0.34 pp), 5.947 s -> discard.
- E019 (I021, H5.1.1): Muon acc 0.7663 (+1.14 pp), 8.041 s (+11%) -> discard on time; refutes H5; top lead.
- E020 (I022, H5): Muon @7.0ep acc 0.7583, 6.663 s (-7.7%) -> NEW TIP; Muon slope 0.54 pp/epoch.
- E021 (I023, H2.1): Muon + 24->32 px @8ep acc 0.7578, 6.331 s (-5% vs E020) -> NEW TIP.
- E022 (I024, H1.6): Muon bs 2000 @8.5ep acc 0.7638 (-0.25 pp vs E019), 7.637 s (-0.40 s) -> discard; Muon tolerates big batch.
- E023 (I025, H2.9.1): tanh-GELU on E019 acc 0.7663 (=), 8.452 s (+5.1%) -> discard; exact GELU is cheaper.
- E024 (I027, H5): compiled batched Muon acc 0.7594, 5.996 s (-5.3%) -> NEW TIP.
- E025 (I028, H2.7): 5@24+3@32 acc 0.7544 (-0.51 pp), 5.605 s (-6.5%) -> NEW TIP (thin margin 0.14 pp).
- E026 (I029, H2.8.1): 2@16+2@24+4@32 acc 0.7565, 5.501 s -> NEW TIP (dominates E025).
- E027 (I030, H5): Muon momentum 0.85 acc 0.7448 (-1.18 pp) -> discard; momentum is sensitive.
- E028 (I032, H4): identical rerun of E026 acc 0.7543 (not bit-identical), seeds 3-10 mean 0.7553 -> tip true acc ~0.7554; run-to-run noise ~0.1-0.2 pp.
- E029 (I034, H2.8.1): 3@16+1@24+4@32 acc 0.7523, 5.235 s -> discard (infeasible by 0.07 pp); need accuracy headroom.
- E030 (I035, H5): Muon momentum 0.5 acc 0.7550 (= tip within noise) -> discard; momentum flat at 0.5-0.6.
- E031 (I036, H5): NS 2 steps acc 0.7372 (-1.8 pp), 5.377 s (-2.3%) -> discard; NS count is an accuracy knob.
- E032 (I038, H2.8): 4@16+4@32 acc 0.7486, 4.994 s -> discard; 24 px bridge matters.
- E033 (I041, H7): NS 4 steps acc 0.7543 (noise), 5.638 s (+2.5%) -> discard; NS saturated at 3.
- E034 (I043, H7.1.2): Muon lr 0.20 acc 0.7551 (noise) -> discard; Muon hparams closed.
- E035 (I044, H7.2.1): sparse Muon renorm acc 0.7572, 5.472 s -> NEW TIP (small gain).
- E036 (I045, H4.3.1): deterministic settings acc 0.7558, 6.008 s (+9.8%), seed-0 rerun bit-identical -> discard.
- E037 (I046, H6.3.1): LS 0.2 under Muon acc 0.7550 (noise) -> discard; loss tuning closed.
- E038 (I047, H7.2.3): bs 1536 acc 0.7542, 5.216 s (-4.7%) -> NEW TIP (thin margin 0.11 pp).
- E039 (confirm E038): acc 0.7542, 5.213 s -> E038 confirmed as tip (two-run mean 0.7542).
- E040 (I048, H2.8): 3@16+1@24+4@32 on E035 acc 0.7522, 5.216 s -> discard; replicates E029.
- E041 (I049, H2.8): 2@16+2@20+4@32 on E035 acc 0.7534, 5.225 s -> discard; 24 px bridge stays.
- E042 (I050, H4.3.1): default-mode Muon compile, still nondeterministic, +0.9% time -> discard; source is cuDNN.
- E043 (I051, H7.2): low-res bs 2048 acc 0.7493, 5.159 s -> discard; batch lever exhausted.
- E044 (I052, H7): head on Muon acc 0.7574 (+0.32 pp), 5.237 s (=) -> NEW TIP (--force); margin to spend.
- E045 (I040, H2.4.1): power by phase -> 16 px overhead-bound (282 W), 24/32 px power-capped.
- E046 (I054, H2.8): 3@16+1@24+4@32 on E044 acc 0.7553, 4.955 s -> NEW TIP.
- E047 (I055, H2.10): 2@16+1.5@24+4@32 on E044 acc 0.7558, 4.963 s -> ties E046; discard.
- E048 (I056, H4.2.3): rerun E046 tip 0.7543, 4.971 s; seeds 3-10 0.7542 -> tip confirmed, margin ~0.15 pp.
- E049 (I057, H7): head Muon lr 0.5x acc 0.7558, 4.944 s -> noise; discard.
- E050 (I058, H2.13.1): group2 352 acc 0.7526, 4.827 s -> discard; width trades poorly.
- E051 (I059, H6.4.1): warmup 0.115 acc 0.7528 -> discard.
- E052 (I060, H2.4.2): profile -> no gaps, aug small; AdaptiveMaxPool bwd ~4%; tip rerun 0.7518 (3-run mean 0.7538).
- E053 (I062, H2.14.1): static shapes 4.858 s (-2%), acc draw 0.7524 -> discard (infeasible by noise); carry forward.
- E054 (I063, H2.10): 0.75@24 trim + static acc 0.7519, 4.733 s -> discard (infeasible).
- E055 (I064): amax pool crashed (Inductor NaN) -> fixed with max(dim).values, rerun.
- E056 (I064, H2.14.2): static + max().values pool 4.718 s (-4.8%), acc 0.7525 -> discard; tip family mean 0.7533 = at gate.
- E057 (I065, H6.5.1): Muon floor 0.05 on static+maxpool base acc 0.7534, 4.717 s -> provisional tip (floor itself = noise).
- E058 (confirm E057): 0.7523, 4.747 s -> two-run mean 0.7529; E057 kept (E046 family identical in accuracy). Need +0.2-0.3 pp.
- E059 (I066, H6.4.2): warmup 0.46 acc 0.7521 -> discard; warmup closed.
- E060 (I061, H2.4): prepare 64->19 ms, total 4.675 s (-0.9%), acc 0.7527 -> discard (infeasible by noise).
- E061 (I068, H8.1): whiten bias 20% acc 0.7523 -> discard; closed.
- E062 (I069, H7.3.2): head LR 0.5x + prepare cut acc 0.7533, 4.680 s -> NEW TIP (by objective); 2nd scoring pending.
- E063 (I069 2nd scoring): 0.7546, 4.669 s -> pool supports head LR 0.5x (+0.15 pp); tip confirmed.
- E064 (I071, H2.10.3): 7.5-ep reshuffle acc 0.7553, 4.713 s (+0.9%) -> discard on time; second scoring next.
- E065 (I071 2nd scoring): 0.7551, 4.709 s -> pool 0.7552; reshuffle +0.12 pp for +0.9% time -> discard (slower).
- E066 (I072, H2.16.2): Inductor tuning 4.664 s (-0.1%), build 554 s -> discard; closed.
- E067 (I073, H2.16.1): profile Triton ~30%, conv ~67%; tip rerun 0.7533 (bit-identical to E062).
- E068 (I074, H6.5.2): Fable tail acc 0.7579 (+0.42 pp vs pool), 4.670 s -> kept (--force); 2nd scoring next.
- E069 (I074 2nd scoring): 0.7544 -> pool 0.7562; tail confirmed (+0.25 pp).
- E070 (I076, H2.10.3): 2@16+1.25@24+4@32 on tail tip acc 0.7542, 4.584 s -> provisional tip; 2nd scoring next.
- E071 (I076 2nd scoring): 0.7532 -> pool 0.7537; tip kept at ~4.58 s; margin ~0.07 pp.
- E072 (I077, H2.10): 1.5@16+1.5@24+4@32 acc 0.7542, 4.594 s -> = tip; 2nd scoring next.
- E073 (I077 2nd scoring): 0.7523 -> pool 0.7532 = tip; discard; low-res reshuffles exhausted.
- E074 (I079, H7.3.3): head wd acc 0.7557, 4.607 s -> 2nd scoring next.
- E075 (I079 2nd scoring): 0.7532 -> pool 0.7544 (noise) -> discard.
- E076 (I085, H2.14.3): dress rehearsal, trial-1 prepare still 0.098 s -> discard; closed.
- E077 (I086, H7.3.4): head lr 0.4x scored 0.7506, off-seed -0.07 -> discard; head lr closed.
- E078 (I087, H6.6.1): MaxSup acc 0.7502, off-seed -0.18 -> discard.
- E079 (I088, H6.5.3): no final EMA -0.15/-0.18 pp -> final EMA carries the tail gain; discard.
- E080 (I081, H2.13.2): width profile refutes; tip rerun 0.7542 (bit-identical E070).
- E081 (I089, H6.5): EMA window 20 -> worse; discard.
- E082 (I091, H9.1.1): decaying renorm target -> collapse (0.487) -> discard.
- E083 (I090, H6.5): LAWA 20 steps scored 0.7521, off-seed -0.29 -> discard.
- E084 (I092, H2.11.3): 32 px at bs 1024 acc 0.7560, 4.699 s; off-seed +0.11 -> 2nd scoring next.
- E085 (I092 2nd scoring): 0.7550 -> pool 0.7555 (+0.18 pp, +2.5% time) -> discard (slower); margin source.
- E086 (I093, H2.14.4): pinned staging upload, trial-1 prepare 0.098->0.034 s, 4.561 s -> NEW TIP.
- E087 (I093 2nd scoring): 0.7532, 4.568 s -> confirmed.
- E088 (I095, H2.10.5): bs1024 32px 3.75 ep acc 0.7529 (infeasible by 0.0001), 4.457 s -> 2nd scoring next.
- E089 (I095 2nd scoring): 0.7515 -> pool 0.7522 -> refutes; 32 px phase binding.
- E090 (I096, H2.17.1): 28 px substitution acc 0.7531, 4.459 s (-2.2%) -> NEW TIP (provisional); 2nd scoring next.
- E091 (I096 2nd scoring): bit-identical; pool 0.7531 -> E090 stays tip (equal acc, -2.2%).
- E092 (I097, H6.8.1): no lookahead acc 0.7499 -> discard.
- E093 (I098, H2.15.2): whiten bias 0.2 ep acc 0.7475 -> discard.
- E094 (I099, H4.2.6): off-seed calibration (clusters by cuDNN picks), tip rerun 0.7517 -> tip ~0.7525 on average.
- E095 (I100, H2.11.4): bs1024 tail with 8-step window: off-seed 0.7541 = E084 -> gain is update count.
- E096 (I102): 8-step bs-1024 tail window on tip acc 0.7539, 4.467 s, off-seed +0.17 -> kept (--force); 2nd scoring next.
- E097 (I102 2nd scoring): bit-identical 0.7539 -> E096 confirmed tip.
- E098 (I109, H4.2.7): 22-seed off-seed E096 0.7514 / E090 0.7525; E090 rerun 0.7496 -> process offsets dominate.
- E099 (I110, H2.14.5): exhaustive cuDNN search, no gain -> discard.
- E100 (I111, H10.2.1): Muon LR x1.25 at 32 px acc 0.7561, 4.467 s -> kept (--force); 2nd scoring next.
- E101 (I111 2nd scoring): 0.7516 -> distinct mean 0.7538 -> E100 stays tip.
- E102 (I112, H11.1.1): whiten bias to 70% acc 0.7521, 4.519 s -> discard.
- E103 (I113, H6.5.4): 20-step EMA + BN stats, off-seed +0.0 -> discard (simplicity).
- E104 (I104, H10.1.1): bs 1024 everywhere off-seed +0.12, scored 0.7525, +2.8% time -> discard; placement-independent.
