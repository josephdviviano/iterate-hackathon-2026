# CIFAR-100 speedrun: benchmark appraisal and entry proposal

3 October 2026. Target repository: `AIDDA-Institute/CIFAR-100-speedrun` at
`25237e3` (also reachable as `Daniel-T-S-Adams/CIFAR-100-speedrun`; identical
history). Method: writing-tools `scientific-exploration` (observation, design
space, mechanism-bearing hypotheses, discriminating probe, portfolio
disposition), with a methods-review pass over the benchmark itself.

Evidence labels used throughout: **[V]** verified here against code, a run, or
the primary source; **[R]** reported by a located source but not re-checked
here; **[I]** inference.

## 1. Governing claim

The 59.3 s baseline (ResNet-9, 40 epochs, n = 2 on the A100) is a weak anchor.
The fastest defensible entry is an airbench-lineage network retuned for
CIFAR-100 *without* test-time augmentation. It would train for roughly 15–25
epochs, compile during the untimed `build`, and be tuned to a development mean
of at least 75.2%. My projection is 10–20 s on the A100 PCIe, a 3–6× cut. That
figure is **[I]**: it is an extrapolation from CIFAR-10 records, not a
measurement.

One unknown governs nearly every downstream choice: where the width × epochs
frontier crosses 75% on CIFAR-100 when evaluation is single-view. Nobody has
published that curve. The single published CIFAR-100 anchor (airbench96,
79.27%) used TTA, which this competition bans. The first probe should measure
that curve before any optimiser or kernel work.

## 2. What the benchmark measures

**Estimand.** Score = mean over 40 organiser seeds of (`prepare` + `train`)
wall time on one A100 80GB PCIe. It counts only if the 40-trial mean top-1 is
at least 75% under single-view evaluation, in PyTorch 2.4.0 / CUDA 12.4.
Compilation, autotuning, and CUDA-graph capture on synthetic inputs during
`build` are free (600 s cap). Evaluation is free but capped at 5 s, and
test-time augmentation is banned (RULES §3) **[V]**.

**The accuracy threshold leaves little need for margin.** The threshold binds
the *mean* of 40 trials. The baseline's per-trial standard deviation was
0.25 pp over 50 L40 trials **[V]**, so the standard error of the official mean
is about 0.04–0.06 pp. Most of the risk therefore comes from your own
development estimate, not from the official draw **[V, computed]**:

| Per-trial sd | True mean 75.10 | 75.15 | 75.20 | 75.30 |
| --- | ---: | ---: | ---: | ---: |
| 0.30 pp, true mean known | 1.8% | 0.08% | <0.01% | <0.01% |
| 0.30 pp, mean estimated from 40 dev trials | 6.8% | 1.3% | 0.14% | <0.01% |
| 0.40 pp, mean estimated from 40 dev trials | 13% | 4.7% | 1.3% | 0.04% |

Each cell is the probability that the official 40-trial mean falls below 75%.
Aim for a development mean of **≥75.2% over ≥40 seeds**, or ≥75.3% if you have
fewer runs or sd > 0.35 pp. Near the threshold, each extra 0.1 pp should cost
well under one epoch **[I]**.

### Strengths

- **Labels never enter the worker.** Test labels stay in the supervisor
  process, and accuracy is computed there.
- **Evaluation cannot alter the model.** Parameters and buffers are snapshotted
  before `eval()` and compared after inference.
- **Results are reproducible.** Source is frozen and SHA-256-manifested
  alongside the harness hashes.
- **Seeds are fixed and shared.** One private seed file covers every team, and
  failed trials cannot be dropped. The process group is killed on timeout.
- **The rules anticipate known gaming.** They ban host selection, cooldowns,
  and changes to the timing boundary. Those are precisely the gaming modes an
  AI agent used on the CIFAR-10 speedrun, worth ~0.11 s of a claimed 0.26 s
  gain (fulcrum.inc, 9 Jul 2026) **[V]**.
- **The harness itself is healthy.** 40/40 CPU tests pass and the CPU smoke
  run completes **[V]**.

### Gaps

| # | Gap | Evidence | Cheap fix (for organisers) |
| --- | --- | --- | --- |
| G1 | Untimed `build()` can read the full training set and the test images. | Probe submission: `sys._getframe(1).f_locals` returned `train_data` (64×3×32×32 synthetic) and test `images` during `build`; the run completed normally **[V]**. `worker.py` loads both before calling `build`. | Load training data after `build` returns, and pass test images to the worker only at the eval phase. In-trial access remains review-only. |
| G2 | Timing is measured inside the submission's own process. `time.perf_counter`, `torch.cuda.synchronize`, and the `started` stamp the supervisor trusts are all monkeypatchable. | `timing.py`, `worker.py`, `harness.py` **[V]**. The rules ban this, but only review can detect it. | Supervisor-side cross-check: elapsed time between acknowledging `phase=train` and receiving `trained`. Linux `CLOCK_MONOTONIC` is system-wide. Flag gaps beyond a few ms. |
| G3 | Rule boundary for offline, data-derived constants that are not model tensors: example-index coresets, pruning scores, curricula, learned augmentation policies. | RULES §3 allows "architectures and scalar hyperparameters" and bans "constants encoding learned model tensors"; the space in between is unaddressed **[V]**. Static pruning is a large lever, which makes this consequential. | State explicitly whether such constants are permitted. Until then, contestants should not rely on them. |
| G4 | No tie or precision rule for close times. | Entries near 10 s could differ by less than A100-to-A100 or thermal variation. The ORGANIZERS guide asks only for the "same host/provider" **[V]**. | Run finalists on one physical GPU with interleaved order. Report CIs, or declare a tie margin. MarioPaerle's CIFAR-100 benchmark requires paired same-pod comparisons, a parallel design worth borrowing **[V]**. |
| G5 | The pinned stack cannot run on Blackwell GPUs, and fails cryptically. | The torch 2.4.0+cu124 arch list stops at `sm_90`. On this machine's RTX PRO 6000 (Blackwell), the five CUDA-parametrised tests fail with "no kernel image" instead of skipping **[V]**. | Add a startup check comparing `get_device_capability` against `get_arch_list`; skip the CUDA tests when no kernel image is available. |
| G6 | The baseline recipe is not public, and the A100 anchor is n = 2. | Calibration recipes live in the git-ignored `.local/` **[V]**. | Publish it, so contestants can calibrate their own A100 against 59.3 s. |

## 3. What earlier speedruns used

**There are no competing entries yet.** The repository has no PRs, forks, or
issues **[V]**. The relevant parallel work is the CIFAR-10 lineage, plus one
nascent CIFAR-100 effort.

**CIFAR-100 anchors:**

- **airbench96, untuned:** 79.27% (flip) and 79.76% (flip + Cutout) on
  CIFAR-100 at the 40-epoch configuration, both **with TTA**. The paper states
  the TTA caveat explicitly (arXiv 2404.00498v2, App. B, Table 5) **[V]**. The
  same configuration takes 46.3 s on CIFAR-10 on a 400 W A100.
- **MarioPaerle/Cifar100Speedrun:** a plain ResNet (64/128/256) with Muon
  reaches 70.6% at 16 epochs and 70.1% at 14 epochs. These are single seeds,
  about 23–26 s each, against a 70% target; the 30-seed baseline is still unrun
  **[V]**. The lesson: without the airbench tricks, a short-budget ResNet sits
  well below 75%.
- **Run-to-run variation:** about 0.24 pp for ResNet-18 on CIFAR-100 over 50
  runs (arXiv 2608.02705, Table 4) **[R]**, in line with the baseline's 0.25 pp
  **[V]**.

**Transferable techniques.** Effects are measured on CIFAR-10 unless stated.

| Technique | Evidence | Use here? |
| --- | --- | --- |
| Frozen 2×2 patch-whitening first conv (eigenvectors ± negations, learnable bias) | Epochs to 94%: 45→21; time 18.3→8.0 s (airbench §3) **[R]** | **Adopt.** The eigendecomposition runs in timed `prepare`, but costs only ms. |
| Dirac (identity) init | Epochs to 94%: 21→18; removing it costs 9.9→12.8 epochs (§3.3, §5.1) **[R]** | **Adopt.** |
| Frozen BN scale, BN-bias lr ×64 | Epochs to 94%: 18→13.5 **[R]** | **Adopt.** |
| Lookahead / EMA copied back into weights | Epochs to 94%: 13.5→12.0 **[R]**; Page: 36→33.5 s **[R]** | Adopt with SGD. The Muon variant drops it. |
| Alternating flip, plus reshuffling each epoch | +0.18 pp at 10 epochs, no TTA (93.10→93.28); no-flip 92.31 (Table 6) **[V]** | **Adopt.** It is free, and the gain is larger without TTA. |
| Label smoothing | 0.2 in all CIFAR-10 records **[R]**; +0.6 pp on CIFAR-100 for ResNet-56 at 0.1 (Müller et al. 2019, Table 1) **[R]** | **Adopt.** Sweep 0.1–0.3. |
| Cutout | Without TTA: −1.14 pp at 10 epochs, −0.38 at 20, +0.12 at 40, +0.36 at 80 (Table 6) **[V]** | **Off** below about 30 epochs. Re-test only if P1 lands at ≥30 epochs. |
| Mixup / CutMix | Needs longer schedules to pay off (arXiv 2101.04342) **[R]** | **Reject** for this budget. |
| Muon on conv filters (3 Newton–Schulz steps, bf16, Nesterov); fused SGD on biases and head | airbench94: 3.09→2.59 s, 9.9→8 epochs **[R]**. No timed CIFAR-100 result exists **[R]**. Not in torch 2.4 (`torch.optim.Muon` arrived in 2.9); a hand-written version is about 20 lines **[R]**. | Probe in P2. |
| `torch.compile(max-autotune)` in `build` | 3.83→3.29 s, −14% (airbench §3.7) **[R]** | **Adopt.** Compilation is untimed here. |
| GPU-resident data, batched augmentation, pure fp16 + channels_last | Page: 75→70 s; fp32-BN fix 256→186 s on V100 **[R]** | **Adopt.** This is required regardless of architecture. |
| Progressive resizing (bilinear downsample, 24→28→32 px) | −0.150 s, 1.978→1.828 s (−7.6%), at matched accuracy (Fulcrum) **[V]** | Probe in P2. CIFAR-100 may be more resolution-sensitive **[I]**. |
| In-run proxy net picks the top-50% loss half of each batch | airbench96: 34.7→27.3 s at 96%, confounded with 37→45 epochs **[R]** | Probe in P2; most promising at longer budgets. |
| Static data pruning | 30% removed at random: −4.4 pp on CIFAR-100 vs −1.0 on CIFAR-10 (InfoBatch, Table 1) **[R]** | **Reject.** It hurts on CIFAR-100, and offline index lists hit the G3 ambiguity. |
| Whole-run CUDA graph | Saves about 5 ms over compile; full stack 0.038 s *slower* (Fulcrum) **[R]** | Low priority. |
| TTA, including selective TTA (hiverge) and "max TTA since eval is untimed" | Behind every CIFAR-10 record; worth 0.70–0.79 pp there (Table 6) **[V]** | **Banned** (RULES §3). All CIFAR-10 epoch counts are therefore optimistic for this competition. |

## 4. Design space and live hypotheses

Decomposition: T ≈ T_fixed + E × t_epoch(architecture, precision, kernels,
resolution, examples per epoch), subject to mean accuracy a(architecture, E,
optimiser, regularisation) ≥ 75.2%.

**Material dimensions:**

- per-epoch throughput;
- sample efficiency (accuracy per epoch);
- capacity at a fixed number of epochs;
- data volume per epoch (resolution, example selection);
- the safety margin.

**Not material:**

- CPU data loading (untimed);
- inference speed (the baseline needs 0.12 s of a 5 s cap);
- multi-GPU training (not allowed).

| ID | Proposition and mechanism | Prediction | Disconfirming result | Decision it changes |
| --- | --- | --- | --- | --- |
| H1 | **Systems-only.** The baseline is mostly overhead, so the same ResNet-9 run with GPU-resident data, fp16/channels_last, and compilation is much faster. | t_epoch falls from 1.48 s to ≤0.7 s. | t_epoch ≥ 1.1 s. My estimate of ~114 TFLOP per epoch for ResNet-9 [I] already caps the gain near 1.5×. | Whether ResNet-9 is a viable base at all. I expect a ceiling of about 30–40 s, which would make H1 a calibration check rather than a strategy. |
| H2 | **airbench transfer.** Whitening, dirac init, BN-bias scaling, alternating flip, label smoothing, and lookahead cut epochs-to-75% on CIFAR-100 as they did on CIFAR-10. | An airbench-style net reaches 75.2% single-view in ≤20 epochs at some width. | No width reaches it below ~30 epochs. | Base architecture. |
| H3 | **Capacity-bound.** With 100 classes and no TTA, 75% is width-limited, so the minimum-time point sits at airbench95–96 width rather than airbench94. | On the width × epochs grid, the minimum-time iso-75% point sits at width ≥1.5× airbench94. | The airbench94 width reaches 75.2% at lower total time. | Width and depth (residual third conv per block). |
| H4 | **Muon.** Orthogonalised updates on conv filters cut steps, as on CIFAR-10. | ≥10% fewer epochs at matched accuracy, with small Newton–Schulz overhead. | Gain within noise at matched wall time. | Optimiser. |
| H5 | **Data volume.** Low-resolution early epochs or proxy-loss selection cut FLOPs per epoch faster than they cost accuracy. | ≥5% lower time at matched accuracy. | Accuracy loss on CIFAR-100 exceeds the epochs saved. | Schedule, and whether a second compiled shape is worth carrying. |
| H6 | **Variance.** Short-budget sd at 75% is ≤0.35 pp, so a 0.2 pp margin suffices. | 40-seed sd ≤ 0.35 pp. | sd ≥ 0.45 pp. | Target mean; whether EMA is needed for variance rather than accuracy. |

Accounts made redundant by evidence:

- **TTA-based accounts:** banned.
- **Mixup and static pruning:** the evidence is adverse on CIFAR-100.
- **Ensembles:** dormant. Two half-size models are rarely more
  compute-efficient than one model of matched FLOPs at this scale **[I]**.

## 5. Probes, in order

**P0. Environments (prerequisite).** Accuracy curves are hardware-independent
to first order. Run them on the two local Blackwell GPUs with a newer PyTorch
(≥2.7, cu128). That PyTorch install is **untested here** and is needed because
the pinned 2.4 stack cannot run on Blackwell **[V]**. Timing must come from a
rented A100 80GB PCIe running the repository's Dockerfile with
`--cpus 4 --network none`.

**P1. Width × epochs frontier, single view (selected first).**

- **Hypotheses separated:** H2 vs H3, and the scale of H1.
- **Method:**
  - One script with a width multiplier and an epoch count, trained with
    airbench94-style SGD + lookahead.
  - Fixed tricks: whitening, dirac init, BN-bias ×64, alternating flip, label
    smoothing 0.2, translate 2. No Cutout.
  - Arms: widths {1×, 1.5×, 2×} of 64/256/256, plus airbench96-shape
    (128/384/512, three convs per block with a residual).
  - Epochs {10, 14, 18, 24}, 5 seeds each: about 80 runs, roughly an hour on
    two GPUs [I].
  - Add one reimplemented ResNet-9 arm at 40 epochs, both to check
    accuracy-scale agreement with the 75.4% baseline and to time H1.
- **Decision rule:**
  - For each width, interpolate the epoch count E*(w) where the mean reaches
    75.3%.
  - Multiply by the A100-measured t_epoch(w) from P3 to get T*(w).
  - Select argmin T*.
- **Redirect:**
  - If nothing reaches 75.3% below 30 epochs, H3 dominates. Move to the Cutout
    regime and wider residual nets.
  - If the 1× width succeeds, the competition becomes systems-bound. Shift
    effort to P2's systems levers.

**P2. Add-ons at the selected point.** Muon versus SGD + lookahead; progressive
resizing; proxy-loss selection; label smoothing 0.1/0.2/0.3. Each is scored as
time to matched accuracy with 10 seeds and kept only if it beats the noise.

**P3. A100 PCIe timing calibration.**

- Measure t_epoch for each P1 width under compile, plus fixed costs (H2D copy
  of 153.6 MB uint8, whitening `eigh`, in-place resets).
- Record SM clocks and power to expose 300 W throttling.
- Re-time the ResNet-9 reimplementation against 59.3 s. This checks host
  equivalence with the organisers' machine.

**P4. Confirmation.** Run 40 seeds through the real harness in the official
container on the A100. Require a mean ≥75.2%, all trials complete, and the
eval pass under 5 s.

## 6. Starting recipe for P1

- **`build`** (untimed):
  - Construct the network.
  - Warm up compiled forward/backward and the optimiser step on random tensors
    at the training batch shape (and at each resolution, if P2 adopts
    resizing).
  - Warm up the eval path at batch sizes 1024 and 784. 10,000 = 9×1024 + 784,
    so an unwarmed compiled eval would recompile inside the 5 s cap.
  - Simpler still: return an uncompiled module that shares the weights.
- **`prepare`** (timed):
  - Copy uint8 data to the GPU.
  - Normalise to fp16 and pad once.
  - Compute the 2×2 whitening from about 5,000 training patches.
  - Reset parameters, BN running stats, optimiser state, and lookahead/EMA
    buffers **in place**, so captured graph pointers stay valid.
  - Seed private generators from `seed`.
- **`train`** (timed):
  - Fixed batch size with the last partial batch dropped (no recompilation).
  - Learning rate held in a 0-d CUDA tensor, because a float lr triggers
    recompilation under compile.
  - Warm-up then linear decay to zero; no `.item()` calls in the loop.
  - Return the module. The forward accepts float32 [0, 1] input, normalises
    internally, casts to fp16, and returns finite floating logits.

## 7. Compliance guardrails

- **Nothing data-derived outside the timer.** No real data, seeds, or
  data-derived constants in import or `build`. Synthetic warm-up only. Do not
  exploit G1, even though the harness would not catch it.
- **Nothing carried between trials.** Re-transfer data and recompute whitening
  in every `prepare`. Caching the GPU copy across trials is prohibited even
  though it is undetectable.
- **No offline index lists or pruning scores** until the organisers resolve
  G3.
- **Single-view evaluation.** No TTA and no state changes in `eval()`.
- **No measurement interference.** No clock patching, host selection, or
  cooldowns.

## 8. Portfolio disposition

- **Selected:** P1 (H2/H3).
- **Candidate:** H4 and H5 (P2), and H6 (P4).
- **Calibration only:** H1.
- **Dormant:**
  - Cutout (unless P1 lands at ≥30 epochs);
  - ensembles;
  - custom Triton kernels (only once profiling shows a launch-bound or
    memory-bound hotspot);
  - whole-run CUDA graph.
- **Rejected:**
  - TTA in any form;
  - mixup/CutMix;
  - static pruning;
  - pretrained or teacher distillation;
  - caching across trials.

## 9. Exact uncertainties

- **The TTA gap on CIFAR-100 is unmeasured.** It is 0.70–0.79 pp on CIFAR-10;
  I expect 1–1.5 pp on CIFAR-100 **[I]**. That would put single-view
  airbench96 at about 78%.
- **PCIe vs SXM slowdown is unmeasured.** The two cards share peak FLOPs; the
  PCIe card has a 300 W vs 400 W TDP and 5% lower bandwidth **[R]**. Every
  published airbench time comes from a 400 W card.
- **Accuracy may not transfer across PyTorch versions.** It is unverified that
  accuracy curves from PyTorch ≥2.7 on Blackwell match PyTorch 2.4 on the A100
  closely enough for selection. P4 is the guard.
- **Muon's benefit on CIFAR-100, and its interaction with the frozen-BN-scale
  parametrisation, is untested anywhere.**

## 10. The decision the next result changes

P1 decides the regime:

- **Capacity-bound** (wide nets at ≥20 epochs, about 15–25 s) means effort
  goes into sample efficiency: architecture, Muon, label smoothing, and
  possibly Cutout.
- **Throughput-bound** (narrow nets at ≤14 epochs, about 5–10 s) means effort
  goes into systems: compile, graphs, resizing, and fp16 kernels.

Building optimiser or kernel machinery before P1 risks optimising the wrong
regime.
