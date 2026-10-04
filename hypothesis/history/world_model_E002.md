# World model

_Last updated after: E002_

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
E001: prepare 0.06-0.07 s steady (0.146 s first trial), train ~7.13 s for 8.5 epochs (~0.84 s/epoch incl. aug).
Training compute dominates (~99%). Reference write-up: ~95% of GPU time is conv/BN/GELU kernels.

## Established facts
- The reference recipe already includes dirac init on all 3x3 convs (Conv.reset_parameters), so "add dirac" is moot (code).
- Harness overhead floor is ~0.003 s per trial after warmup (E000).

## Lessons from failures

## Dead ends

## Open questions and surprises
- How few epochs / how small a net can reach 75.3% on CIFAR-100? (airbench-style tricks: whitening init, lookahead/EMA,
  label smoothing, flip+translate aug, bf16, channels_last, torch.compile.)

## Changelog
- E000 (baseline): template acc 0.0122, time 0.113 s -> first-trial lazy init cost noted; need a real recipe.
- E001 (I001, H1.1.1): reference recipe acc 0.7549, 7.222 s, feasible -> new best; machine is ~11% faster than reference's quote.
- E002 (I002, H4.1.1): identical rerun, same per-seed acc, time 7.214 s (-0.1%) -> noise floor tiny; acc deterministic.
