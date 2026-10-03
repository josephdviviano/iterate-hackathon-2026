<!-- Frozen report of a read-only ideation agent (round 4, topology); model output, not user input. -->

# Topology hypotheses for the CIFAR-100 speedrun (read-only report, ranked)

I only read files. Nothing was modified and no GPU jobs were run.

**Long odds.** About 30 structural changes have already been tried on this recipe, and almost all of them landed at or below the exchange rate. My base rate for any single topology change beating it is about 15–25%. The best bet below has maybe a 40% chance.

**One principle from the evidence.** Taking compute away from the *resolution* that the dense widening convs see has always cost a lot of accuracy:

| Change | Accuracy cost | Source |
|---|---|---|
| Pool before the widening conv | −2.2 to −4.6 pp | F-015 |
| Stride-2 patch stem, 4×4 terminal grid either way | about −6 pp | codex F-006, F-009 |
| Polyphase (learned stride-2 conv instead of conv + pool) | −4.9 pp | codex F-016 |
| Max-first aggregation | −5.2 pp | codex F-022 |

Capacity, on the other hand, is cheapest at the coarse end (F-012, F-014). So I propose no stem, stride or 4-stage changes; that family is dead. Every hypothesis below removes compute where it is spatially redundant: the post-pool convs on the 3×3 (2×2 at 20 px) terminal grid, the kernel support of the widening conv, or stage-1 width.

## Cost model

All figures are forward MACs per image, from the layer shapes.

| Layer | 32 px map | MAC (M) | 20 px map | MAC (M) |
|---|---|---|---|---|
| whiten 3→24, k2 | 31² | 0.28 | 19² | 0.10 |
| s1c1 24→128 | 31² | 26.6 | 19² | 10.0 |
| s1c2 / s1res 128² | 15² | 33.2 each | 9² | 11.9 each |
| s2c1 128→384 | 15² | 99.5 | 9² | 35.8 |
| s2c2 384² | 7² | 65.0 | 4² | 21.2 |
| s3c1 384→640 | 7² | 108.4 | 4² | 35.4 |
| s3c2 / s3res 640² | 3² | 33.2 each | 2² | 14.7 each |
| **Total** | | **432.5** | | **155.9** |

**Training cost in "units".** One unit is 1M MACs per image, counting forward, weight-gradient and input-gradient passes for every trainable layer. The whitening layer is frozen, so s1c1 needs no input gradient.

| Step type | Units per step | Steps | Unit total |
|---|---|---|---|
| 20 px | 457.7 | 204 | 93.4k |
| 32 px, full | 1270.3 | 123 | 156.2k |
| 32 px, stage 1 frozen | 1011.6 | 81 | 81.9k |
| **Run total** | | | **331.6k** |

**Converting units to time.** In the A100 profile, conv kernels take about 75% of GPU time and Triton elementwise about 22%. S50's depth-2 stage 3 removed 8.8% of units and measured about 9.6% less time per epoch (two GPUs, indicative). So stage-3 convs realise roughly their MAC share. I use 0.75–1.0 × the unit saving.

**Wasted taps.** On the terminal grid, 40% of the 3×3 taps fall on zero padding at 32 px and 56% at 20 px. At 7×7 it is 18%.

**Receptive field.** After stage 3's pool, every position already has a receptive field of about 39 px, which is more than the whole image. Stage 3's spatial mixing therefore combines features that each already see the full image.

## Ranking (expected net time saved per unit of test cost)

| # | Hypothesis | Δunits | Predicted Δtime | Matched-time epochs | Expected Δacc at equal epochs | P(net win) | Code |
|---|---|---|---|---|---|---|---|
| 1 | H1: 1×1 square convs in stage 3 (both / residual only) | −15.7% / −7.9% | −12% / −6% | 9.5 / 9.0 | −0.1 to −0.4 / 0 to −0.15 | 0.40 | ~15 lines |
| 2 | H2: H1 plus a 4×4 terminal grid (ceil pool), or no stage-3 pool (7×7) | −14.7% / −7.9% | −10% / about −2% | 9.5 / 8.75 | +0.1 to −0.3 | 0.35 | ~5 lines (ceil flag exists) |
| 3 | H3: thinner stage 1 (96/384/640) | −12.5% | −9 to −10% | 9.5 | −0.25 to −0.5 | 0.30 | config only |
| 4 | H4: 2×2 widening convs, unpadded so pooling tiles exactly (stage 3; stages 2+3) | −17.1% / −31.2% | −12% / −22% | 9.75 / 11.0 | −0.2 to −1.0 / −0.6 to −2 | 0.20 / 0.10 | ~25 lines + timing gate |
| 5 | H5: centre tap only for stage-3 square convs in the 20 px phase | −4.8% | −3 to −4% | 8.75 | 0 to −0.1 | 0.45 | ~10 lines |
| 6 | H6: H1 with the savings put into stage-3 width 768 | −9.6% | −6 to −7% | 9.0 | +0.1 to −0.2 vs control | conditional on H1 | config only |
| 7 | H7: 1×1 conv2 in stage 2 (7×7 grid) | −14.2% | −10% | 9.5 | −0.3 to −1.0 | 0.12 | shares H1 code |
| 8 | H8: identity-concat widening in stage 3 (DenseNet-style) | −15.9% | −11% | 9.5 | −0.4 to −1.5 | 0.08 | ~30 lines |

Matched-time epochs are 8.5 / (1 − predicted saving), rounded to 0.25. Replace the prediction with the measured saving once the timing gate has run.

---

### H1. A pointwise terminal stage: stage-3 conv2 and residual as 1×1

**Mechanism.** On a 3×3 grid (2×2 at 20 px), whose positions already see the whole image, the 3×3 kernels mostly multiply zero padding or mix redundant global views. A 1×1 keeps channel mixing, depth and the residual nonlinearity. The terminal stage becomes a per-position MLP followed by a global max.

There is also a speculative extra benefit. A 3×3 kernel behaves differently on a 2×2 grid (every position is a corner and uses 4 taps) than at the 3×3 centre (9 taps). That inconsistency may add to the shock at the 20→32 px switch, and a 1×1 removes it.

**Cost.**
- 32 px: each conv drops from 33.2M to 3.7M, saving 88.5 units per step per conv.
- 20 px: each conv drops from 14.7M to 1.6M, saving 39.3 units per step per conv.
- Converting both convs saves 15.7% of units, about 12% of time (range 9–15%). The residual alone saves 7.9%, about 6%.

**Evidence.**
- Strategy-coverage records "pointwise residuals rejected (−0.145 pp)". That 20-seed result was for **stage 1 only** (codex F-014).
- The stage-3 pointwise residual was run once, on 3 seeds: 75.010 vs 75.013% (codex F-013). Stage-3 conv2 as 1×1 has never been run.
- S50's depth-2 stage 3 lost 0.66 pp even with 0.25 more epochs, so the residual branch carries capacity. This probe tells us whether that capacity sits in the 3×3 spatial weights (9× the parameters) or in depth and nonlinearity.

**Interactions.** The DCT init touches no square conv, because they have no non-identity rows. A 1×1 dirac init is the identity. The stage-1 freeze is unaffected; stage 3 trains in every step, so the saving applies in the frozen tail too.

**Implementation** (`research/lab_recipe/`):
- `config.py`: add `square_kernels: tuple[int,int,int] = (3,3,3)` and `residual_kernels: tuple[int,int,int] = (3,3,3)`, with values in {1, 3}. Convert lists to tuples in `from_parameters`. Add both fields to the `_validate_narrowing` clash list.
- `model.py`: `Conv` already accepts `kernel_size`. Have `ConvGroup` take `k2` and `kres` and build `Conv(cout, cout, k2)` and the residual conv the same way. `AirbenchNet` passes the per-stage values.

**Arms.** Residual only (1, 1, 3 → `residual_kernels=[3,3,1]`) at 9.0 epochs. Both convs at 9.5 epochs. Both convs at 8.5 epochs, as a mechanism check.

**Kill rule.**
- Kill if the measured saving is below 6% for both convs.
- Kill if the matched-time Δacc is ≤ 0 for both arms.
- Advance any arm with matched-time Δacc ≥ +0.08 pp.
- Repair arm if both convs lose more than 0.15 pp: keep a narrow spatial path via a bottleneck residual (1×1 640→160, 3×3 160→160, 1×1 160→640), which costs about 3.9M MACs.

**Rules.** This is an architecture choice. The inits are unlearned constructions. Compliant.

### H2. Pointwise stage 3 that keeps more spatial positions

**Mechanism.** The ceil-mode stage 3 (7→4 terminal grid) gained +0.23 pp in S31 (20 seeds), but it cost +5.5% time because the 3×3 square convs ran on a 4×4 grid. With 1×1 convs, a 4×4 grid costs 6.55M per conv instead of 33.2M.

**Arms.**
- **(a) 1×1 + ceil.** Saves 14.7% of units: 159.8 per 32 px step and 78.7 per 20 px step. Predicted −10% time. The extra elementwise work on the larger grid is about +0.6%.
- **(b) 1×1, no stage-3 pool (7×7 terminal).** The global max subsumes the local 2×2 max. This removes 7.9% of conv units but adds about 4–5% of elementwise traffic (eight tensors per image grow from 11.8 to 64 MB per batch). Net about −2% time, so this is an accuracy bet.

**Interactions.** The same as H1.

**Implementation.** `stage3_ceil=True` already exists. For (b), add `stage3_pool: "floor" | "ceil" | "none"`, with a `pool_none` branch in `ConvGroup.forward`.

**Probe.** (a) at 9.5 epochs, (b) at 8.75.

**Kill rule.** Kill if the matched-time Δacc is ≤ H1's best arm, since the extra positions would then buy nothing.

**Rules.** Compliant.

### H3. Thinner stage 1 (96/384/640)

**Mechanism.** Early width does not buy accuracy (F-012). Stage-1 width also sets the input width, and so the cost, of s2c1, the largest conv in the net.

**Cost.**
- 32 px full step: −175 units. Frozen step: −85.5. 20 px step: −63.2.
- Total −12.5% of units. Stage 1 has the largest maps, so its elementwise work also falls. Predicted −9 to −10% time.

**Evidence.** The S9 5-seed result was −0.26 / −0.49 pp for −9.4 / −8.5% time, i.e. about 0.041 pp per 1% time. That is slightly better than the 0.054 exchange rate, but within noise. It has never been retested on the current base (freeze, DCT, scaler 16).

**Interactions.** The freeze shrinks the saving in the last 20% of steps (already counted above). s2c1's DCT rows mix over 96 input channels instead of 128.

**Probe.** Run at 9.5 epochs, and at 8.5 epochs for the mechanism. This is a config-only change.

**Kill rule.** Kill if the matched-time Δacc is ≤ 0.

**Candour.** This is a width tweak rather than a bold topology change. It ranks high only because it costs nothing to run.

### H4. Even 2×2 widening convs with exact pool tiling

**Mechanism.** An unpadded 2×2 conv on an odd map gives an even map that a floor-mode pool tiles exactly:

| Stage | 32 px | 20 px |
|---|---|---|
| Stage 2 | 15→14→7 | 9→8→4 |
| Stage 3 | 7→6→3 | 4→3→1, so pad (0,1,0,1) to get 5→4→2 |

The joint channel and spatial response before the max is kept, along with full resolution. That separates it from every failed family. Each pooled output still covers a 3×3 input neighbourhood, and the receptive field stays ≥ 31 px. As a bonus, the rows the pool would discard are never computed, with no explicit pad at 32 px. H1 of round 1 lost +1.19% to exactly that pad copy.

**Literature.** Wu et al., NeurIPS 2019, on even-sized kernels: 2×2 kernels were competitive on CIFAR in deep nets. That transfer to a shallow net is unproven.

**Cost.**
- Stage 3: s3c1 drops from 108.4M to 35.4M at 32 px and from 35.4M to 15.7M at 20 px. That is −17.1% of units, about −12% time.
- Stage 2: s2c1 drops from 99.5M to 38.5M at 32 px and from 35.8M to 12.6M at 20 px. Its input gradient is already skipped in frozen steps. −14.1% of units.
- Both stages: −31.2% of units, about −22% time.

**Interactions.** DCT init uses the 4-pattern 2×2 basis automatically, since `dct_basis(kh)` is called with the kernel size. Part of the +0.11–0.17 pp DCT gain may be lost. dirac_ places the identity at tap (1,1), a harmless one-pixel shift.

**Implementation.**
- Add a `padding` argument to `Conv` (it currently hardcodes `"same"`).
- Add `widen_kernels: tuple[int,int,int] = (3,3,3)`, allowing 2.
- In `ConvGroup.forward`: when the kernel is 2, pad with `if x.size(-1) % 2 == 0: x = F.pad(x, (0,1,0,1))`, then run the valid conv. This is static under `dynamic=False`.
- Bypass `skip_discarded`.

**Probe.**
1. A timing gate first, because shape effects reversed H1 of round 1 and the window crops (F-034, F-055). Use `research/profile_step.py` plus a 5-seed local paired run. Proceed only if stage 3 alone saves ≥ 8%.
2. Then stage 3 at 9.75 epochs and stages 2+3 at 11.0.

**Kill rule.** Kill if the equal-epoch Δacc is ≤ −0.6 pp (stage 3) or the matched-time Δacc is ≤ 0.

**Rules.** Compliant.

### H5. Kernel growth: centre tap only in the 20 px phase

**Mechanism.** Stage-3 square convs are dirac-initialised, so every off-centre tap is exactly 0. Using only the centre tap in the 20 px phase (`F.conv2d(x, w[:, :, 1:2, 1:2])`) is exact. The off-centre taps get no gradient, and weight decay and lookahead keep them at 0. They start learning from 0 at the switch.

This is the fallback if H1 fails at 32 px. At 20 px, 56% of the taps are wasted.

**Cost.** −4.8% of units, predicted −3 to −4% time.

**Implementation.** Add `centre_tap_below: int = 0`. In `ConvGroup`, slice the weight when `x.size(-1) < centre_tap_below` (set it to 3 for stage 3). The graph count does not change.

**Probe.** One arm at 8.75 epochs.

**Kill rule.** Kill if the matched-time Δacc is ≤ 0.

**Rules.** Compliant.

### H6. Put H1's savings into stage-3 width 768

**Mechanism.** H1 makes stage-3 width cheap. With 1×1 square convs, width 768 costs less than today's 640:
- 32 px: s3c1 is 130.0M (+21.7M), the two squares are 5.3M each, a net −102 units per step.
- 20 px: net −53 units per step.
- Total −9.6% of units.

S12 found 768 worth about +0.2 pp over 640 at matched epochs (5 seeds). This interacts with S54's wide-early arm, which is in flight.

**Probe.** Run only if H1 or H2 survives: 768 with 1×1 square convs at 9.0 epochs, and with ceil added.

**Kill rule.** Kill if it does not beat the surviving H1/H2 arm at matched time.

### H7. Pointwise conv2 in stage 2

**Mechanism.** The same lever on the 7×7 grid. That grid's receptive field is only about 19 px, so the spatial mixing is real, and the prior is much weaker than H1's.

**Cost.**
- 32 px: 65.0M → 7.2M, −173 units per step.
- 20 px: 21.2M → 2.4M, −56.6 units per step.
- Total −14.2% of units.

**Implementation.** `square_kernels=[3,1,3]`, on H1's code.

**Probe.** One arm at 9.5 epochs.

**Kill rule.** Kill if the matched-time Δacc is ≤ 0. If it fails, do not repeat it.

### H8. Identity-concat widening in stage 3

**Mechanism.** The 384 dirac rows of s3c1 become a literal copy of the input. The conv computes only the 256 new rows, and the copy and new rows are concatenated before pooling. This is exactly "freeze the identity rows at the identity from step 0, and stop computing them".

**Cost.** s3c1 drops from 108.4M to 43.4M at 32 px and from 35.4M to 14.2M at 20 px. −15.9% of units.

**Evidence against.** Freezing widening-conv weights late cost about 1 pp (F-054), and learned widening capacity is the most efficient lever (F-014).

**Implementation.**
- Add `widen_concat` per stage.
- `conv1 = Conv(cin, cout - cin, dirac=False)`, with every row initialised from the DCT bank (this needs a flag in `structured_init_`).
- Forward: `torch.cat((x, conv1(x)), 1)`, then pool, BN and GELU.

**Probe.** One arm at 9.5 epochs. Optionally a variant with 384 copied + 384 learned rows (width 768) on top of H1.

**Kill rule.** Kill if the equal-epoch Δacc is ≤ −0.5.

## Common probe protocol (local GPUs only)

1. **Structure check (CPU).**
   - Terminal grids are as expected: 3×3 / 2×2 (H1, H5), 4×4 (H2a), 7×7 / 4×4 (H2b), and unchanged for H4.
   - A fresh reset is an identity map where expected.
   - `research/flops.py` reproduces the MAC table.
   - A short GPU smoke run with `TORCH_LOGS=recompiles` shows no recompiles in timed trials. `cache_size_limit` is already 32.
2. **Timing gate.** Local paired timing as in S46/S48: 5 seeds per arm per GPU at 8.5 epochs, GPU 0 in arm order and GPU 1 reversed. Check `nvidia-smi` for other projects' jobs first. Drop any time-saving arm that realises less than half its predicted saving.
3. **Screen.** 20 seeds from the next unused block (for example 5800–5819, after S54's 5700s), both GPUs. Arms run at matched time using the measured saving.
   - **Sweep A (H1, H2, H3, H5): no new graphs beyond the variants, so it can run first.** Arms: control; H1 residual @9.0; H1 both @9.5; H1 both @8.5; H2a @9.5; H2b @8.75; H3 @9.5; H5 @8.75.
   - **Sweep B (H4, H6, H7, H8)** follows. H6 runs only if H1 or H2 survives.
4. **Decision.** At matched time, Δacc is the frontier shift (1 pp ≈ 18.5% time). Advance arms at ≥ +0.08 pp and kill arms at ≤ 0. Treat anything up to about +0.15 pp as possibly best-of-8 noise (20-seed difference SE is about 0.07).
5. **Confirm.** 40 fresh seeds (for example 5900–5939): control plus the survivor at two budgets bracketing control accuracy. Then local paired timing, 10 seeds per GPU in opposite order, on the chosen budget. Adopt if pooled accuracy ≥ control − 0.03 pp and paired time ≤ −3%.

These changes swap conv shapes (1×1, 2×2, ceil grids), so a local Blackwell saving is necessary but not sufficient. Before anything enters `submissions/team_segal`, the A100 PCIe realisation still needs a check, which is the user's call on Modal.

## Considered and rejected on the evidence

- **4-stage net or earlier downsample; stride-2 or 4×4 patch whitening; pool-first in stage 1.** Codex F-006/F-009 and F-015 (adding stage-1 pool-first cost a further −2.9 pp).
- **Padding the whitening conv for a 4×4 terminal at the current widths.** +30% MACs (561.5M vs 432.5M); H2a gets the 4×4 grid cheaply instead.
- **Grouped or depthwise-separable convs.** Codex F-031 measured fixed grouped filtering plus pointwise mixing at no time gain, or 63–78% slower when actually lowered. ConvMixer was poor and slow, and the depthwise-context arm still lost 3.9 pp (codex F-022).
- **3×1 then 1×3 factorised widening.** It saves less than H4's 2×2 kernel (50.6M vs 73.0M in stage 3) and adds an extra, non-linear-free layer.
- **1×1 head expansion before the pool.** −1.19 pp (S31).

The main files to change are `research/lab_recipe/model.py` (the `Conv`, `ConvGroup` and `AirbenchNet` constructors) and `research/lab_recipe/config.py`. New sweeps can be modelled on `research/sweeps/s53-window-crops.toml`.
