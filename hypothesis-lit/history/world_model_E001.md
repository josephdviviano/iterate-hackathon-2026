# World model

_Last updated after: E001_

## Current best
E001: airbench port (whitened 2x2 conv, widths 128/384/576, 3 convs/group, SGD-Nesterov lr 9 wd 0.012, ls 0.3, Lookahead, alt-flip + 2px translate, fp16 channels_last, torch.compile, batch 1024 drop_last, 8.5 epochs = 408 steps): accuracy 0.7383, time 8.04 s. Still infeasible (needs >= 0.753).

## Measurement
- Score = mean (prepare + train) seconds over seeds 0,1,2 on an A100-SXM4-80GB (official: A100 PCIe, 40 seeds, mean acc >= 0.75; local bar 0.753).
- Harness overhead of a trivial recipe is ~0.1 s, so essentially all time is ours.
- Reference point (README): ResNet9-style, width 64, 40 epochs -> 75.36% in 59.3 s on A100 PCIe (SXM likely somewhat faster). So a feasible recipe exists at ~60 s; the target is to go well below that.
- Run-to-run noise in accuracy: unknown yet; README L40 calibration std 0.25 pp per trial -> std of a 3-seed mean ~0.15 pp. Margin above 0.753 should be ~0.3 pp to be safe.

## Where the cost goes
- E001: train ~7.93 s for 408 steps = ~19.4 ms/step (~0.93 s/epoch) at widths 128/384/576; prepare 0.06 s (0.20 s on the first trial: one-off first-use cost). So time is ~proportional to steps x per-step cost; prepare is <1%.

## Established facts
- Harness and template run end-to-end; trivial recipe costs 0.10 s (E000).
- airbench port at 8.5 epochs reaches only 0.738 on CIFAR-100 without TTA (E001); per-seed std ~0.3 pp.

## Lessons from failures

## Dead ends

## Open questions and surprises
- How many epochs / what width is the minimum for 75.3% with a strong recipe (airbench-style: whitening init, label smoothing, lookahead/EMA, flip+translate aug, channels_last, bf16, torch.compile)?

## Changelog
- E000 (baseline, -): template -> acc 0.0122, time 0.10 s; harness works, infeasible.
- E001 (I001, H1.1.1): airbench port 8.5 ep -> acc 0.7383, 8.04 s; H1.1.1 refuted (1.5 pp short); time ~19.4 ms/step.
