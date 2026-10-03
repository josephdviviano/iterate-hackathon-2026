Source: https://github.com/MarioPaerle/Cifar100Speedrun (README.md at default branch, pushed 2026-07-03T12:13:19Z)
Retrieved: 2026-10-03 via gh api. Excerpt verbatim.

## Chosen constants

- Target: `k = 70%` plain validation accuracy.
- Official run count: 30 runs.
- Official epoch budget: 16 epochs.
- Baseline: `train_cifar100_resnet_muon.py` only.
- Compiled baseline: enabled by default with `torch.compile` / `C100_COMPILE=1`, default `C100_COMPILE_MODE=default`; warmup pays compile/cold-start cost before measured runs. `max-autotune` is intentionally not the default because it can spend minutes autotuning on Leonardo.
- Timed quantity: training time only; validation stays frozen and untimed.

## Record metric
## Feasibility note

The `k = 70%` target is mechanically configured and has one compiled 16-epoch seed clearing it at `70.58%`, but it is not yet validated over the official 30-run baseline. The smoke check only proves the code path executes. Run `slurm/official_baseline.sh` to measure whether the baseline clears 70% over 30 runs.

## Compiled one-seed probes

- Eager 12-epoch probe `48375215`: `69.70%` validation, `24.90s` timed training. Real plain validation, but not compiled and not enough margin for a 70% target.
- Compiled 14-epoch probe `48376210`: `70.11%` validation, `22.82s` timed training, warmup/compile row `44.99s`. Real but too close to 70 for a 30-run benchmark.
