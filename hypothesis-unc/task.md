# Task: CIFAR-100 training speedrun

Train a CIFAR-100 classifier from scratch as fast as possible while keeping its test accuracy at
75% or more. This repository is the benchmark: `README.md` explains it, `RULES.md` is the
competition rulebook, and `submission_template/README.md` is the submission contract. Read all
three before your first experiment.

## What you edit

Only `submissions/arena/` (created from `submission_template/` during setup). Everything in that
folder is fair game: model, optimizer, loss, schedule, augmentation, batch size, precision,
resolution, `torch.compile`, custom Triton/CUDA kernels. Do not modify `benchmark/`,
`pyproject.toml` or `uv.lock`, and do not install packages; the runtime is the pinned environment
(PyTorch 2.4.0, torchvision 0.19.0).

## How an experiment is measured

The run command (in `task.json`) trains your recipe from scratch on 3 seeds (0, 1, 2: the same
every time) and evaluates each on the full test set. It reports:

- `accuracy`: mean top-1 test accuracy over the 3 trials, as a fraction;
- `time`: mean `prepare + train` time per trial in seconds; this is the score;
- `complete`: whether all trials succeeded.

The official judgment uses 40 seeds and requires a mean accuracy of at least 75%. With only 3
seeds here, a run counts as meeting the target when `accuracy >= 0.753` (a safety margin) and the
run is complete. Among runs that meet it, lower `time` is better; until one does, higher accuracy
is better.

## Rules that are easy to break (the full list is RULES.md)

- No pretrained weights or external data. Train from scratch on the 50,000 training images.
- `prepare` must reset everything learned (parameters, buffers, optimizer, scheduler, EMA, RNGs);
  no information may flow between trials.
- No real data or trial seed in module import or `build`. `build` is untimed and may compile,
  autotune and warm up on synthetic inputs; anything touching real training data belongs in
  `prepare` or `train`, which are timed. Threads and subprocesses finish before `train` returns.
- Never touch the test set: no test files, no test-time augmentation, no state changes during
  evaluation. Evaluation must finish within 5 s and return finite `[B, 100]` logits for any batch
  size `B` from 1 to 1024.
- No sleeps, and no clock or GPU power changes.
- Limits: `build` 600 s (untimed), `prepare + train` 600 s per trial.

## Machine

Experiments run on one NVIDIA A100-SXM4-80GB (the official judging GPU is an A100 80GB PCIe) and
are pinned to 4 CPU cores, like the official 4-CPU quota. The data is already downloaded.
