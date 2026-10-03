<!-- Frozen report of agent a82a6e89b72399c9c (ideation round 1, read-only agent; model output, not user input). -->

# Ideation report: representation, inductive bias, data and structural levers (read-only)

I read RULES.md, submission_template/README.md, submissions/team_segal/*.py, research/strategy-coverage.md and findings F-005, F-012, F-014, F-015, F-021 to F-023, F-026, F-030 and F-031. I changed no files and ran nothing.

**Exchange rate used throughout.** Below 8.5 epochs, accuracy rises about 1 pp per epoch (F-021), and one epoch costs about 0.73 s of the ~6.0 s trial. So 1 pp is worth about 12% of trial time. A lever costing x% of time must buy at least x/12 pp to break even.

**Detection power.** Per-seed SD is about 0.25 pp (F-005). With 10 paired seeds at 8.25 epochs, the SE of a difference is about 0.11 pp, so 0.25 pp is the smallest effect we can detect. Every probe below is 10 seeds against a same-host control unless stated.

**Structural fact behind items 2 and 3.** The 2×2 whitening conv has no padding, so the maps are 31→15→7→3 at 32 px and 19→9→4→2 at 20 px. Stage-3 conv2 and the residual conv run on a **3×3 grid** (2×2 in the low-res phase), not the "4×4 to 8×8" stated in F-012. Each floor-mode pool also throws away the last row and column. My MAC estimates at 32 px: stage 1 about 93M, stage 2 about 230M, stage 3 about 174M, of which conv1 at 7×7 is 108M. That is why widening stage 3 gets expensive quickly.

---

## Ranked hypotheses (expected value per unit cost)

### 1. Online label smoothing: soft targets built from the model's own class confusions
- **Mechanism:** Replace the uniform smoothing of 0.2 with a per-class distribution S[y]. S[y] is the average softmax over correctly classified training samples of class y in the previous epoch. The smoothing mass then goes to confusable classes (fine classes within a superclass), which gives the "dark knowledge" of distillation with no teacher and no extra forward pass.
- **Fit to evidence:** Smoothing helps and is flat from 0.1 to 0.3, so softer targets are useful but their *shape* has never been tuned. The hard taxonomy auxiliary loss gave ±0 (F-026), but it imposed fixed groups; this version adapts to the run.
- **Sketch:** Inside `fit`, under `no_grad`: `acc.index_add_(0, labels, p * correct[:, None])`, where `p = outputs.float().softmax(1)`. The multiply-by-mask form avoids a host sync. At each epoch end, set `S = acc / acc.sum(1, keepdim=True)`. Loss: `-(((1-α)·onehot + α·S[labels]) · log_softmax).sum()`. Use uniform smoothing in epoch 0.
- **Cost:** About 0.1% time (a few small eager kernels per step).
- **Probe:** α of 0.2 against smoothing of 0.2.
- **Expected:** +0.15 to 0.4 pp (the original paper reports about +1 pp on CIFAR-100 over long runs).
- **Reject if:** the gain is below 0.1 pp.
- **Compliance:** In-run statistics from training data only ("All work using real training data belongs in `prepare` or `train`"). Reset `acc` and `S` every trial ("reset … moving averages"). Clean.

### 2. A wide 1×1 expansion before the global max-pool
- **Mechanism:** Add `conv1x1(640→E)` with BN and GELU on the final 3×3 map, then the flatten-max, then `Linear(E, 100)`. Max-pooling over E position-wise detectors adds nonlinear capacity at the resolution where it is cheapest. This is the head conv in EfficientNet and MobileNetV3. It differs from the lead's MLP head, which sits *after* pooling.
- **Fit to evidence:** Short runs are capacity-bound, and last-stage capacity was the efficient lever. Widening past 640 failed only because conv1 at 7×7 grows with width. Here E=2048 adds about 11.8M MACs (about 2.4%) and leaves conv1 untouched.
- **Arm B, class-wise pooling (near-zero cost):** Apply the head as a 1×1 conv on the map, then max or logsumexp over positions, so each class picks its own location. Optionally add it to the current logits.
- **Sketch:** Add `expand` and `expand_norm` to `Net` (the default init is fine, since dirac does not apply to 1×1). `reset_model` handles them through `reset_parameters`.
- **Probe:** E ∈ {1280, 2048} against 640/672-wide controls at equal time.
- **Expected:** +0.2 to 0.5 pp for about +3% time.
- **Reject if:** the gain is below 0.25 pp, or it is beaten by adding epochs at equal time.
- **Compliance:** Architecture choice ("You may choose the architecture"). Clean.

### 3. Border-preserving pooling
- **Mechanism:** Floor-mode 2×2 pooling on odd maps drops roughly 1 row and column in 8 at every stage. At stage 3 it discards 13 of the 49 conv1 outputs. Because of the alternating flip, the dropped column swaps sides between epochs.
- **Arm A, zero cost:** Use overlapping 3×3, stride-2 pooling with no padding on odd maps. The shapes stay 31→15→7→3, so downstream cost is unchanged, and every position is covered (AlexNet's overlapping pooling gave about 0.3 to 0.4%). Keep k=2 where the map is even (4→2 at 20 px). The check is `k = 3 if x.size(-1) % 2 else 2`, which is static under `dynamic=False`.
- **Arm B:** `ceil_mode=True` in stage 3 only, so 7→4 at 32 px (unchanged at 20 px). This buys stage-3 capacity through spatial positions rather than width: about +10% MACs in the 32-px half, roughly +5% time. F-015 found that late-stage spatial resolution is productive.
- **Expected:** Arm A +0.05 to 0.2 pp at about 0% time. Arm B needs at least +0.4 pp.
- **Risk:** The backward of overlapping max-pool can be slower in channels_last fp16, so profile the pooling kernel.
- **Reject:** Arm A if it is at least 1% slower or shows no gain. Arm B if it is beaten by width 704 at equal time.
- **Compliance:** Architecture choice. Clean.

### 4. Per-sample temporal soft targets (progressive self-distillation, PS-KD style)
- **Mechanism:** Keep a 50k×100 fp16 bank (10 MB) of each image's softmax from its previous visit. The target becomes `(1-α_t)·smoothed_onehot + α_t·bank[idx]`, with α_t ramping from 0 to about 0.4. It is sample-specific: it picks up ambiguous or mislabelled images. Because the stored view and the next view differ in flip, it also pushes toward flip consistency for free.
- **Sketch:** Make `TrainingStream.epoch` also yield `idx`. After each step, write `bank[idx] = softmax(outputs.detach())`.
- **Cost:** About 0.3% time.
- **Risk:** Early predictions from the 20-px phase are weak and may anchor errors. Ramp α only after the resolution switch.
- **Expected:** +0.1 to 0.3 pp.
- **Reject if:** no gain, or below item 1. Run both, then test them stacked.
- **Compliance:** Same as item 1. The bank must be zeroed or reallocated in `prepare`. Clean.

### 5. Cosine head with sub-centre prototypes
- **Mechanism:** Logits = s·cos(f, w_k), with a fixed s of about 10 to 16 and K=2 prototypes per class (max over sub-centres, as in sub-centre ArcFace). This removes the feature-norm shortcut that max-pool features invite, evens out class-norm imbalance early in a 100-way head, and lets multimodal classes use two prototypes.
- **Fit to evidence:** The 1/6 logit scale mattered (F-019), so the head's geometry is a live lever. `head_norm` was rejected only as part of the Muon stacks, never tested on its own under SGD.
- **Sketch:** About 5 lines in `Net.forward`. Run a grid over s ∈ {8, 12, 16} × K ∈ {1, 2}.
- **Expected:** −0.2 to +0.3 pp (uncertain sign). Time cost about 0%.
- **Reject if:** every cell is at or below control.
- **Compliance:** Clean.

### 6. Class-mean head initialisation in `prepare`
- **Mechanism:** The dirac-initialised trunk starts close to an identity on whitened inputs. One forward pass of about 5k training images in `prepare` (about 10 ms) gives pooled features. Initialise head row c to the normalised mean feature of class c, scaled to match the default init norm. The head then starts well above chance and the first epoch is not spent aligning it.
- **Expected:** +0.05 to 0.15 pp. If positive, it could pair with trimming 0.1 epoch.
- **Reject if:** below 0.05 pp.
- **Compliance:** Explicitly allowed: "any data-derived initialization must happen inside `prepare`/`train`" (timed). Clean.

### 7. Anneal label smoothing to 0 over the 32-px phase
- **Mechanism:** Smoothing stabilises the early high-lr phase (lr 11.5, fp16 logits). In an underfit run, the late phase may gain from letting the head commit. A constant smoothing value is flat from 0.1 to 0.3, but a *schedule* has not been tested.
- **Sketch:** Make `label_smoothing` a per-step scalar. The cross-entropy sits outside the compiled module, so this needs no recompile.
- **Expected:** −0.1 to +0.2 pp.
- **Reject if:** no gain.
- **Note:** This conflicts with items 1 and 4. Test it only if they fail.
- **Compliance:** Clean.

### 8. Free probes at evaluation, run on saved trained models
- **(a) fp32 inference:** Upcast inside `forward` without caching, so no state changes. Expected about 0 pp; it only rules out fp16 near-ties in a 100-way argmax.
- **(b) Single-view input resize at eval (e.g. bilinear to 36 px):** The global max-pool accepts any resolution. Expected prior is ≤0, because training ends matched at 32 px.
- **Probe:** Sweep both on about 10 saved models for an evaluation-only cost.
- **Compliance:**
  - (a) is legal: it is computation, not "fitting, adaptation … or model state changes".
  - (b) is **ambiguous**. It is one view per image, but it is a test-time transform not used in training, close to the spirit of "no additional augmented views (test-time augmentation)". **It needs organiser confirmation.** The clean form is to train the final phase at the same resolution, which makes it the model's input convention ("use the same convention when training it"), but then it is charged as training.

---

## Considered and deprioritised (with reasons)

- **Data-driven hierarchical heads** (clustering head rows into about 20 groups, or a factorised softmax): the true taxonomy already gave ±0 and was unstable at weight 1.0 (F-026). A learned hierarchy is unlikely to beat it, and items 1 and 4 deliver the same structure softly.
- **2:4 structured sparsity:** In torch 2.4, semi-structured sparse tensor cores serve only linear layers and matmuls. cuDNN convs would need im2col or a custom implicit-GEMM kernel, and the head is negligible compute.
- **Low-rank or grouped convs in stage 2** to fund stage-3 width: largely covered by 128/320/768 (6.45 s against 6.29 s, F-014). Factorisation also cuts capacity in a capacity-bound regime.
- **Repeated augmentation or batches with both flips of an image:** this halves unique images per step when compute is the binding constraint, so it is expected negative.
- **Weight-tied recurrent block run for more iterations at eval** (to exploit the untimed evaluation): extrapolation gains appear on algorithmic tasks, not on classification.
- **Nearest-class-mean logits accumulated during the last epoch:** the features come from fast weights, while the final lookahead copy installs slow weights. That mismatch makes it inferior to the lead's head refit.
- **Self-supervised label augmentation** (class × rotation): it needs 4× the views per step and is unaffordable at about 8 epochs.

## Suggested order

1. **One sweep (about 0% time each):** items 1, 3A, 5 and 6, each against a shared 10-seed control.
2. **Second sweep:** item 2 (E=2048) and item 3B against width-704 and +epoch controls at equal time.
3. **Then:** stack the winners, and convert the accuracy gain into an epoch cut at the 1 pp ≈ 0.73 s rate.
4. **Before relying on item 8(b):** ask the organisers to confirm it.

Main source files: submissions/team_segal/model.py (`Net.forward`, `ConvGroup.forward`), train.py (`fit` loss), data.py (`TrainingStream.epoch` for idx) and submission.py (`prepare` for items 6 and 1/4 resets).
