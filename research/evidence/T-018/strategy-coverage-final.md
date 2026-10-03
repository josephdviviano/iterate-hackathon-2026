# Strategy coverage matrix (T-018; lineage portfolio X-005, representation portfolio X-003)

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
| 3×3 whitening kernel | Page | rejected (−0.28 pp, +11% time) | S21 (F-028) |
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
| Activation SiLU (hiverge) / CELU (Page) vs GELU | hiverge, Page | rejected (−0.17 / −0.43 pp) | S21 (F-028) |
| Global max-pool | airbench | adopted | inherent |
| Max+mean global pool (×1.5 gain) | competitor C4/C7 | rejected (−0.15 pp, +4% time) | S21 (F-028) |
| Global pool via flatten-max (Fable GlobalAmaxPool) | Fable e1 | **adopted** (−1.6% time, accuracy unchanged) | S23, S26 interleaved |
| Reshape-amax 2×2 pooling | novel (from competitor profile) | rejected | S14 (F-018) |
| Logit scale (head gain) | competitor C7 | adopted 1/6 | S15 (F-019) |
| Logits / fan-in with unit-std head (head_norm) | airbench94_muon, hiverge | rejected (part of the Muon stacks) | S4, S22 (F-025) |

## Data and augmentation

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| GPU-resident data, per-trial statistics | all | adopted | T-002 |
| Translate radius | airbench (2), airbench96 (4) | adopted 2; 0, 1 and 4 rejected | S3, S4 (F-006, F-007) |
| Alternating flip vs random flip | airbench | adopted; random flip −0.09 pp | S21 (F-028) |
| Cutout | airbench96, Page | rejected (−0.52 pp) | S21 (F-028) |
| Colour jitter, multiplicative | novel variant | rejected (+0.05 pp, +4.5% time) | S21 (F-028) |
| Colour jitter, normalised-unit shift and scale | hiverge | not run separately (multiplicative form neutral; hiverge stack rejected as a whole) | S21, S22 |
| Mixup in the first half only | literature | rejected (−0.11 pp) | S21 (F-028) |
| CutMix | hlb-CIFAR10 | not tested; dominated by mixup/cutout results | — |
| In-run hard-example selection | airbench96_faster | rejected | S7 (F-010); competitor C3 agrees |
| Random retention | competitor C3 | rejected by competitor | codex F-003 |
| Label smoothing 0.09-0.3 | all | adopted 0.2; 0.1, 0.3 equal | S4, S17 (F-007, F-022) |
| Progressive resizing (bilinear downsample) | Fable | adopted 20 → 32 at 50% | S4-S6, S17, S19 |
| Ending below 32 px (FixRes) | novel, competitor C2 | rejected (mismatched and matched eval) | S6 (F-009); codex F-002 |
| Window-crop low resolution | Fable e2/e4/e5 | not tested; Fulcrum shows it worse than downsampling | INV note |
| Whitening sample 960 vs 5000 images | hiverge | equal (±0.00 pp); not adopted | S21 (F-028) |
| Whitening-bias epochs 0.2 / 1 / 3 / 6 | hiverge, airbench | adopted 3; 1 and 6 equal; 0.2 −0.21 pp | S10 (F-016), S21 (F-028) |
| Gather-based crop (faster, exact) | competitor | not adopted (saves ~2 ms per epoch) | codex F-008 |
| Precomputed augmentation in build | Fable e1 | N/A (gaming per Fulcrum and the organisers' red-team list) | F-024 |

## Optimisation

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| Nesterov SGD, decoupled lr and wd, triangular schedule | airbench | adopted | — |
| Learning rate ×0.8 / ×1.25 | — | rejected | S7 (F-011) |
| Warm-up peak, final lr | — | rejected (flat) | S10 (F-016) |
| Weight decay ×0.5 / ×2; momentum 0.75 / 0.8 / 0.9; BN momentum 0.8 | hiverge values | rejected (wd −1.15 / −0.17; momentum 0.8 did not replicate on 20 fresh seeds; 0.9 −0.11; BN −0.04) | S21, S27 (F-028, F-029) |
| Lookahead (airbench) | airbench | adopted; off −0.77 pp | S4, S21 (F-007, F-028) |
| Lookahead flush of partial steps | competitor C9 | neutral (+0.03 pp) | S21 (F-028) |
| Fable EMA tail with lr floor | Fable e1-e5 | not tested separately (Muon-schedule specific) | — |
| Optimiser restart at the resolution switch | competitor C9 | rejected by competitor | codex F-010 |
| Muon, airbench94_muon port | airbench94_muon | rejected (−0.8 to −1.4 pp; diverges without head_norm) | S4, S9 (F-007, F-012) |
| Muon, full hiverge stack | hiverge, Fable | rejected (−1.5 to −3.3 pp) | S22 (F-025) |
| Batch 512 / 768 / 1536 / 2048 (SGD) | hiverge 1536, airbench 2000 | rejected | S4, S9 |
| Progressive freezing (FreezeOut) | novel | rejected | S6 (F-009) |
| BN statistics recalibration after lookahead | novel | neutral (+0.01 pp) | S21 (F-028) |

## Execution

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| fp16 channels_last, fp32 BN | airbench | adopted | — |
| torch.compile, dynamic=False, warm-up of every phase in build | airbench, Fable | adopted | S8 (F-013) |
| Fused SGD | hiverge | adopted (speed-neutral) | S8 |
| CUDA graphs (reduce-overhead, max-autotune) | hiverge, Fable | **adopted max-autotune** (−2.3% on A100 PCIe; 181 s untimed build) | S23, S26, m2 (F-030) |
| Resize inside the compiled forward | Fable e1 | rejected (+0.8% time) | S23 |
| Manual per-step or whole-run CUDA graphs | Fable e1 | not tested; Fulcrum measured about 5 ms and only with 24 px phases | README |
| Test-time augmentation (any form) | all records | N/A (banned) | RULES §3 |
| Thermal sleep, host lottery, untimed precompute | Fable e1 | N/A (gaming) | F-024 |

## Representation and learning alternatives (X-003)

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| Progressive deepening (residual branches ramped in at 30% / 50%) | novel | rejected (−0.43 / −0.92 pp for −4.1 / −6.9% time) | S25 (F-027) |
| Superclass auxiliary loss from the CIFAR-100 taxonomy | novel (rule-gated) | rejected (±0 at weight 0.3; unstable at 1.0) | S24 (F-026) |
| RepVGG-style train-time 1×1 branches | RepVGG | rejected (+0.05 pp at +21% time) | S24 (F-026) |
| PolyLoss-1 / squentropy | literature | rejected (−0.05 / −0.38 pp) | S24 (F-026) |
| ConvMixer-512/8 (k5, patch 2) | ConvMixer | rejected (25.6% in 19.9 s) | S24 (F-026) |
| Colour-space or DCT inputs | — | subsumed by 2×2 patch whitening | X-003 R6 |
| ViT / MLP-Mixer; in-run self-distillation | — | dormant (triggers not met) | X-003 R7, R8 |
