# Strategy coverage matrix (T-018, exploration portfolio X-003)

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

## Structural exploration and ideation round 1 (T-019, X-005)

Correction to F-012: the 2×2 whitening conv is unpadded, so the maps are 31→15→7→3 at 32 px
and 19→9→4→2 at 20 px. Stage 3's conv2 and residual conv run on a **3×3 grid** (2×2 at
20 px), not "4×4 to 8×8".

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| Snapshot ensemble (0.85; 0.7 + 0.85) in the untimed evaluation | novel | rejected (−0.68 / −1.24 pp) | S30 (F-033) |
| Jointly trained 2×(96/256/448) ensemble | novel | rejected (+0.17 pp at +26% time) | S30 (F-033) |
| Stage-2 multi-exit, averaged at evaluation | novel | rejected (−0.88 pp) | S30 (F-033) |
| Closed-form ridge head refit | novel | rejected (−1.0 pp) | S30 (F-033) |
| Warmup-stable-decay lr (decay from 60%) | literature | candidate (+0.24 pp) | S30 → S33 |
| Cosine lr decay | literature | rejected (−0.15 pp) | S30 (F-033) |
| Lookahead every 3 steps; lookahead power 2 | novel | rejected (−0.04 / −0.09 pp) | S30 (F-033) |
| Class-balanced batch order | novel | candidate (+0.15 pp) | S30 → S33 |
| Online label smoothing; PS-KD soft targets; LS anneal | agent R1/R4/R7, O3/O9 | pending | S31 |
| 1×1 head expansion; cosine head; class-mean head init; simplex-ETF head | agent R2/R5/R6, O7 | pending | S31 |
| Overlapping 3×3/2 pools on odd maps; ceil-mode stage 3 | agent R3 | pending | S31 |
| Annealed log-sum-exp global pool at 20 px | agent O8 | pending | S31 |
| fp32 master weights | agent O1 | pending | S31 |
| Clean (untranslated) tail epochs | agent O2 | pending | S31 |
| Per-group wd / head lr multipliers; bias scaler 32 / 128 | agent O4 | pending | S31 |
| Conv init gain 0.5 / 0.7 (rotational warm start) | agent O5 | pending | S31 |
| Momentum ×0.5 at the switch; stochastic 20/32 blend | agent O6 | pending | S31 |
| Skip pool-discarded conv rows (exact) | agent H1 | pending | S32, M5 |
| Whitening-bias autograd off after freeze (exact) | agent H2 | pending | S32, M5 |
| cuDNN benchmark limit 0; coordinate-descent tuning; loss in graph | agent H3-H5 | pending | M5 |
| 16-channel whitening pad; fused epoch data kernel; manual CUDA graph | agent H6-H8 | candidate (after M5 and profile) | — |
| fp16 BN / bf16 | agent H12 | rejected by analysis (eps underflow; no tensor-core gain) | X-005 |
| Evaluation-time input resize | agent R8 | N/A without organiser confirmation (close to TTA) | — |

### modded-nanogpt PR #380 (exact-match retrieval, n-gram count chain) — reviewed 2026-10-03

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| Non-parametric memory of the training set consulted at evaluation (kNN over train features, exact/near-duplicate retrieval, count-chain mixing) | PR #367/#380 | N/A: RULES §3 bans "training-data lookup" during evaluation; it would also exploit CIFAR-100 train/test near-duplicates (ciFAIR) | RULES.md §3 |
| Same memory distilled into parameters on the clock (closed-form head, class prototypes) | PR #380 analogue | rejected (ridge refit −1.0 pp; class-mean init −0.17 pp) | S30, S31 |
| Gate mixing two predictive distributions by the hidden state | PR #380 | covered by multi-exit averaging (−0.88 pp) | S30 |
| Smaller model, shorter run, retune hyperparameters left from the longer run | PR #380 | done (frontier, S10, S21, S27 at the converged budget) | F-014, F-028, F-029 |
| CPU-parallel index building during GPU training | PR #380 | N/A: no CPU-side work on the critical path (augmentation is on GPU); 4 CPUs | — |
| Reorder on-clock work, unmapped memory, false sharing | PR #380 | partially open: `prepare` is about 70 ms (H10, ≤0.5%) | ideation r1 systems |

### Rounds 2-3 and the converged entry (T-019, X-005), 2026-10-03

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| Whitening-bias autograd off after freeze | agent r1 H2 | **adopted** (-2.96% on A100 PCIe, exact) | M5, M6 (F-034, F-035) |
| BN-bias lr scaler 16 (from 64) | agent r1 O4 | **adopted** (+0.24 to +0.40 pp; buys 0.25 epoch) | S31, S37, S39, S40, S41 (F-039, F-043) |
| DCT filter-bank init of widening conv rows | lead (structured init) | **adopted** (+0.11 to +0.17 pp; buys 0.25 epoch) | S42, S43 (F-046) |
| Stage-1 lr cooldown 60-80% + exact stage-1 freeze; stage 2 without residual conv; 8.5 epochs | agent r2 H1-H3 | **adopted** (-10.7% local paired time at equal accuracy) | S44-S49 (F-048 to F-051) |
| WSD + class-balanced order | lead | not adopted (+0.15 pp alone; nothing on top of scaler 16) | S33, S34, S39 (F-037) |
| Wide-early (768 then 640) transplant | lead | candidate, not pursued (+0.15 pp for about +1.4% local time) | S36b (F-040) |
| Saliency channel narrowing at the switch | user idea | rejected (random selection equals Taylor; break-even) | S36b (F-040) |
| Score-bank data pruning (easy/hard/split/soft) | user idea | rejected (-0.05 to -1.83 pp at equal steps) | S35 (F-038) |
| Muon (teammates' working port) | hypothesis-branch | not better: +3.1% (I049) / +8.4% (I044) time vs ours on the same hosts | C2 (F-045) |
| Post-add residual activation | hypothesis-branch | rejected (-0.13 pp) | S40 (F-042) |
| Synthetic pre-training in build | user idea | N/A (RULES §3: no pretrained weights; build state must be reset) | — |
| Orthogonal / ZerO inits; SkipInit gates | lead | rejected (neutral; gates -0.6 to -1.0 pp) | S42 (F-046) |
| Stage-2 freeze; depth-2 stage 3; scaler/lr/peak/switch retune on the new base | agent r2, lead | rejected | S50 (F-052) |
| Bias decay multipliers; earlier freezes; square-conv perturbation; head centring; whitening eps | agent r3 (learning) | rejected (scaler-16 effect is bias size, not noise) | S51 (F-053) |
| Stage update thinning; weight-only freezes | agent r3 (compute) | rejected | S52 (F-054) |
| Native-scale window crops mid-run | agent r3 (compute) | rejected (time saving far below MAC saving) | S53 |
| Fused pool+BN Triton kernel | user idea | not built (Inductor kernels at 1.2-1.56 TB/s; ceiling 1-2%) | profiles |

### Round 4 (T-019), 2026-10-04

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| Vectorised identity init, gather crop without syncs, stop at the last lookahead update, full-size warm-up | agent r4 (audit) | **adopted** (exact; -1.7% local paired) | S57 (F-058) |
| Label smoothing 0.4 at 8.25 epochs | lead (re-tune on the new base) | **adopted** (matches the 8.5-epoch control; -2.5% steps) | S54, S55, S58, S61 (F-056, F-059, F-061) |
| Batch 768 / 896 | lead | rejected (null at matched time) | S55 (F-056) |
| Wide-early transplant on the new base | lead | rejected (within noise) | S54 (F-056) |
| 1x1 stage-3 convs, 4x4 terminal grid, width 768 via 1x1, centre-tap pruning | agent r4 (topology) | rejected | S56 (F-057) |
| Stage-3 residual 1x1; stage-1 width 96 | agent r4 (topology) | rejected (predicted savings not realised under max-autotune) | S59, S60 (F-060) |
| Pinned-memory staged H2D copy | agent r4 (audit) | not pursued (5-10 ms, host-dependent) | — |

### Round 5 (T-019), 2026-10-04

| Strategy | Source | Status | Evidence |
| --- | --- | --- | --- |
| Re-tune lr, wd, BN-bias scaler, freeze window after LS 0.4 | lead | converged (all within noise) | S63 (F-063) |
| Final lookahead blend with fast weights; fp32 evaluation copy | lead | rejected (null) | S63 (F-063) |
| SNOO (Nesterov momentum on the lookahead outer step) | literature agent (arXiv 2510.15830) | rejected (-0.18 to -2.96 pp) | S64 (F-065) |
| FrozenBN tail | literature agent (Wu & Johnson 2021) | rejected (non-finite logits with BN eps 1e-12) | S64 (F-065) |
| Cautious weight decay; batch ramp; schedule-free; NorMuon | literature agent | not run (weak transfer case; see the r5 report) | — |
| Teammates' latest tips (youssef-baseline / -hypothesis / -lit / -dr) | team branches | ours leads at matched accuracy (+16.6% / +4.8% / +76% time; dr fails to compile) | C3 (F-064) |
| Fable-style lr floor; lookahead base 0.93/0.97; 28 px phase | team branches | rejected | S65 (F-066) |
| uint8 data on GPU; pinned upload | team branches, audit agent | not pursued (not exact / 5-10 ms) | — |
