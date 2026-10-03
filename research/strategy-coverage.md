# Strategy coverage matrix (T-018, X-003)

Every strategy known from the CIFAR-10 record lineage, the competitor (Codex, GPU 0) and novel
alternatives, with its status on single-view CIFAR-100. "Converged" means the 8.25-epoch
recipe in `submissions/team_segal`.

**Sources:**

- airbench: the arXiv 2404.00498 paper, and airbench94, airbench94_muon, airbench95/96 and
  airbench96_faster.
- hiverge: `cifar10_speedrun.py`.
- Fable/Fulcrum: the e1-e5 recipes, plus the README and the investigation notes.
- hlb-CIFAR10.
- Page: "How to Train Your ResNet".
- Competitor: `tickets/CODEX`, `research/codex-challenger/evidence`.

Every CIFAR-10 record time assumes test-time augmentation, worth about 0.85 pp at its budget.
That is banned here, so record epoch counts do not transfer.

**Status values:**

| Status | Meaning |
| --- | --- |
| adopted | in the converged recipe |
| rejected | tested here with evidence |
| N/A | cannot apply under the rules |
| pending | queued sweep |

## Architecture and representation

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| Frozen 2×2 patch-whitening stem, ± eigenvectors | airbench, hiverge | adopted | P1, F-003 |
| 3×3 whitening kernel | Page | pending | S21 |
| Dirac (identity) conv init | airbench | adopted | inherent |
| Frozen BN scale, BN bias lr ×64 | airbench, Page | adopted | inherent |
| Three convs per block with a residual | airbench96 | adopted | P1, S3 (F-003, F-006) |
| Width and allocation (last-stage width) | airbench95/96 | adopted 128/384/640 | S9, S12 (F-012, F-014) |
| Pool before the widening conv | novel | rejected | S11 (F-015) |
| Depth-2 first stage | novel | rejected (time-neutral) | S11 (F-015) |
| Pointwise residuals | competitor C12/C13 | rejected by competitor (−0.145 pp, 20 seeds) | codex F-014 |
| Stride-2 patch stem; preserved terminal grid | competitor C6/C8 | rejected by competitor (68.8-70.4%) | codex F-006, F-009 |
| Auxiliary stage-2 supervision | competitor C10 | rejected by competitor (−1.1 pp) | codex F-011 |
| ResNet-9 (Page / organisers' baseline) | Page | rejected as base on FLOPs | F-004 |
| Activation SiLU (hiverge) / CELU (Page) vs GELU | hiverge, Page | pending | S21 |
| Global max-pool | airbench | adopted | inherent |
| Max+mean global pool (×1.5 gain) | competitor C4/C7 | pending (compile-safe form) | S21 |
| Global pool via flatten-max (Fable GlobalAmaxPool) | Fable e1 | pending (time) | S23 |
| Reshape-amax 2×2 pooling | novel (from competitor profile) | rejected | S14 (F-018) |
| Logit scale (head gain) | competitor C7 | adopted 1/6 | S15 (F-019) |
| Logits / fan-in with unit-std head (head_norm) | airbench94_muon, hiverge | pending (with Muon) | S22 |

## Data and augmentation

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| GPU-resident data, per-trial statistics | all | adopted | T-002 |
| Translate radius | airbench (2), airbench96 (4) | adopted 2; 0, 1 and 4 rejected | S3, S4 (F-006, F-007) |
| Alternating flip vs random flip | airbench | adopted; random-flip ablation pending | S21 |
| Cutout | airbench96, Page | pending (8 px) | S21 |
| Colour jitter, multiplicative | novel variant | pending | S21 |
| Colour jitter, normalised-unit shift and scale | hiverge | pending (with Muon stack) | S22 |
| Mixup in the first half only | literature | pending | S21 |
| CutMix | hlb-CIFAR10 | not tested; dominated by mixup/cutout results | — |
| In-run hard-example selection | airbench96_faster | rejected | S7 (F-010); competitor C3 agrees |
| Random retention | competitor C3 | rejected by competitor | codex F-003 |
| Label smoothing 0.09-0.3 | all | adopted 0.2; 0.1, 0.3 equal | S4, S17 (F-007, F-022) |
| Progressive resizing (bilinear downsample) | Fable | adopted 20 → 32 at 50% | S4-S6, S17, S19 |
| Ending below 32 px (FixRes) | novel, competitor C2 | rejected (mismatched and matched eval) | S6 (F-009); codex F-002 |
| Window-crop low resolution | Fable e2/e4/e5 | not tested; Fulcrum shows it worse than downsampling | INV note |
| Whitening sample 960 vs 5000 images | hiverge | pending | S21 |
| Whitening-bias epochs 0.2 / 1 / 3 / 6 | hiverge, airbench | adopted 3; 1, 6 equal; 0.2 pending | S10 (F-016), S21 |
| Gather-based crop (faster, exact) | competitor | not adopted (saves ~2 ms per epoch) | codex F-008 |
| Precomputed augmentation in build | Fable e1 | N/A (gaming per Fulcrum and the organisers' red-team list) | F-024 |

## Optimisation

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| Nesterov SGD, decoupled lr and wd, triangular schedule | airbench | adopted | — |
| Learning rate ×0.8 / ×1.25 | — | rejected | S7 (F-011) |
| Warm-up peak, final lr | — | rejected (flat) | S10 (F-016) |
| Weight decay ×0.5 / ×2; momentum 0.8 / 0.9; BN momentum | hiverge values | pending | S21 |
| Lookahead (airbench) | airbench | adopted; off rejected | S4 (F-007); retest S21 |
| Lookahead flush of partial steps | competitor C9 | pending | S21 |
| Fable EMA tail with lr floor | Fable e1-e5 | not tested separately (Muon-schedule specific) | — |
| Optimiser restart at the resolution switch | competitor C9 | rejected by competitor | codex F-010 |
| Muon, airbench94_muon port | airbench94_muon | rejected (−0.8 to −1.4 pp; diverges without head_norm) | S4, S9 (F-007, F-012) |
| Muon, full hiverge stack | hiverge, Fable | pending | S22 |
| Batch 512 / 768 / 1536 / 2048 (SGD) | hiverge 1536, airbench 2000 | rejected | S4, S9 |
| Progressive freezing (FreezeOut) | novel | rejected | S6 (F-009) |
| BN statistics recalibration after lookahead | novel | pending | S21 |

## Execution

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| fp16 channels_last, fp32 BN | airbench | adopted | — |
| torch.compile, dynamic=False, warm-up of every phase in build | airbench, Fable | adopted | S8 (F-013) |
| Fused SGD | hiverge | adopted (speed-neutral) | S8 |
| CUDA graphs (reduce-overhead, max-autotune) | hiverge, Fable | pending (local); A100 m2 blocked on Modal billing | S23, m2 |
| Resize inside the compiled forward | Fable e1 | pending | S23 |
| Manual per-step or whole-run CUDA graphs | Fable e1 | not tested; Fulcrum measured about 5 ms and only with 24 px phases | README |
| Test-time augmentation (any form) | all records | N/A (banned) | RULES §3 |
| Thermal sleep, host lottery, untimed precompute | Fable e1 | N/A (gaming) | F-024 |
