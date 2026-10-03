# team_segal: CIFAR-100 speedrun recipe

An airbench-lineage convnet trained from scratch on GPU-resident data. Run it with its
defaults; no parameters are needed:

```bash
python -m benchmark.run --submission team_segal --n 40
```

## Recipe

| Part | Choice |
| --- | --- |
| Network | Frozen 2×2 patch-whitening conv (24 channels), then three stages of conv–max-pool–BN–GELU, conv–BN–GELU and a residual conv–BN–GELU branch at widths 128/384/640. Global max-pool (flattened max) and a bias-free linear head whose logits are scaled by 1/6. BatchNorm scales are frozen at 1. |
| Initialisation | Identity (dirac) init of every 3×3 conv; the remaining output channels of each stage's widening conv start as a DCT filter bank (fixed 2-D DCT patterns with per-trial random orthonormal channel mixing, not learned); whitening weights from the eigen-decomposition of 2×2 patches of 5,000 training images, computed in `prepare` |
| Data | Per-trial channel statistics; reflect-padded translation by up to 2 px; alternating flip (a fixed random flip per image, with all images flipped on odd epochs) |
| Resolution | Bilinear (antialiased) downsampling to 20 px for the first half of training, then the native 32 px |
| Optimiser | Nesterov SGD in airbench's decoupled parametrisation (lr 11.5 and wd 0.0153 per 1024 examples, momentum 0.85, BN-bias lr ×16), batch 1024, triangular schedule, lookahead, label smoothing 0.2; the whitening bias trains for 3 epochs, then leaves autograd (about 3% faster on the A100 PCIe, exact) |
| Budget | 7.75 epochs (372 steps of 1024 images) |
| Execution | fp16 channels_last with fp32 BatchNorm; `torch.compile(mode="max-autotune", dynamic=False)` (Inductor autotuning and CUDA graphs, built in `build`) and fused SGD on CUDA |

## Phase boundaries and rule compliance

- **Import and `build`** construct the model and warm it up on random synthetic tensors only.
  This runs the full schedule (both resolutions) and both evaluation batch shapes, so
  compilation and cuDNN autotuning finish before the first trial. No real data, seed or
  data-derived value is used.
- **`prepare`** (timed) resets every parameter and buffer in place: weights, BatchNorm
  statistics and the normalisation buffers. It also copies and scales the training images
  on the GPU, computes the channel statistics and whitening from them, and seeds the
  recipe's own generator from the trial seed.
- **`train`** (timed) builds a fresh optimiser and lookahead state, trains, and returns the
  eager module. Nothing persists between trials except allocated memory and compiled code.
- **Evaluation** is one forward pass per image of the eager module in eval mode. There is
  no test-time augmentation, the model is stateless at evaluation, and nothing is compiled.
- **Dependencies:** only the pinned PyTorch environment, with no custom kernels or extra
  packages.

## Parameters

`config.py` lists every parameter with its default and validation. Unknown parameters fail
the run. `epochs` is the main budget lever: 8.5 and 8.75 epochs are the conservative
fallbacks.
