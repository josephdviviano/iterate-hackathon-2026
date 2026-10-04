# World model

_Last updated after: E000_

## Current best
E000 template (tiny conv, 64 images, 3 steps): accuracy 0.0122, time 0.113 s. Infeasible; any recipe reaching
accuracy >= 0.753 beats it.

## Measurement
- Score = mean(prepare + train) over 3 seeds; feasibility = mean acc >= 0.753 (3 seeds), official 75% over 40 seeds.
- First trial pays lazy CUDA/cuDNN init (~0.33 s in E000) unless build warms up the exact same kernels (E000).
- Reference point from README: ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe.

## Where the cost goes
Not yet measured for a real recipe. Expect: GPU compute of training epochs dominates; prepare (H2D of 150 MB uint8,
normalization) should be < 0.1 s.

## Established facts
- Harness overhead floor is ~0.003 s per trial after warmup (E000).

## Lessons from failures

## Dead ends

## Open questions and surprises
- How few epochs / how small a net can reach 75.3% on CIFAR-100? (airbench-style tricks: whitening init, lookahead/EMA,
  label smoothing, flip+translate aug, bf16, channels_last, torch.compile.)

## Changelog
- E000 (baseline): template acc 0.0122, time 0.113 s -> first-trial lazy init cost noted; need a real recipe.
