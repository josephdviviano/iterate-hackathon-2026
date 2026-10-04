# World model

_Last updated after: E000_

## Current best
Template baseline (E000): accuracy 0.0122, time 0.10 s — infeasible (needs accuracy >= 0.753). Until a run is feasible, higher accuracy is better.

## Measurement
- Score = mean (prepare + train) seconds over seeds 0,1,2 on an A100-SXM4-80GB (official: A100 PCIe, 40 seeds, mean acc >= 0.75; local bar 0.753).
- Harness overhead of a trivial recipe is ~0.1 s, so essentially all time is ours.
- Reference point (README): ResNet9-style, width 64, 40 epochs -> 75.36% in 59.3 s on A100 PCIe (SXM likely somewhat faster). So a feasible recipe exists at ~60 s; the target is to go well below that.
- Run-to-run noise in accuracy: unknown yet; README L40 calibration std 0.25 pp per trial -> std of a 3-seed mean ~0.15 pp. Margin above 0.753 should be ~0.3 pp to be safe.

## Where the cost goes
Unknown yet. Expect: GPU compute of the forward/backward passes (scales with epochs x FLOPs/epoch), plus data transfer/augmentation, plus possible Python/launch overhead at small batch sizes.

## Established facts
- Harness and template run end-to-end; trivial recipe costs 0.10 s (E000).

## Lessons from failures

## Dead ends

## Open questions and surprises
- How many epochs / what width is the minimum for 75.3% with a strong recipe (airbench-style: whitening init, label smoothing, lookahead/EMA, flip+translate aug, channels_last, bf16, torch.compile)?

## Changelog
- E000 (baseline, -): template -> acc 0.0122, time 0.10 s; harness works, infeasible.
