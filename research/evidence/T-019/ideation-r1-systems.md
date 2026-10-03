<!-- Frozen report of agent ae85d4360eb0535fd (ideation round 1, read-only agent; model output, not user input). -->

## Hypotheses for cutting per-step time on an A100 PCIe (team_segal recipe)

I read the recipe, the coverage matrix, the competitor profile and the installed PyTorch 2.4.0 source. Two things change how the brief should be read:

- **The competitor profile was taken in eager mode, so its pooling and BN shares don't carry over to our compiled run.** PyTorch 2.4's Inductor already rewrites `max_pool2d` as `_low_memory_max_pool2d_with_offsets`, which stores int8 offsets instead of int64 indices (`torch/_inductor/decomposition.py:708`). It also breaks BN and GELU into fused Triton kernels. A custom pool+BN+GELU Triton kernel therefore has little left to win (H11).
- **The 32 px phase dominates.** Going by FLOP counts, one 32 px step is about 2.8x a 20 px step. That puts a 32 px step near 21 ms and a 20 px step near 8.5 ms, with about 4.3 of the 6.0 s spent at 32 px. Convolution FLOPs are where the leverage is.

### Ranking by expected saving per unit of effort

| # | Hypothesis | Exact? | Expected saving | Effort |
|---|---|---|---|---|
| H1 | Don't compute the conv rows and columns that max-pool throws away | mathematically exact | **4-5% (~0.25-0.3 s)** | ~15 lines |
| H2 | Switch off the whitening-bias gradient once it is frozen | exact (lr is already 0) | 1.5-3% | ~10 lines |
| H3 | `cudnn.benchmark_limit = 0` (try every cuDNN algorithm) | yes | 0-3% | 1 line |
| H4 | `coordinate_descent_tuning=True` | yes | 0.5-2% | 1 line |
| H5 | Compute the loss inside the compiled graph | ~yes | ~1% | small |
| H6 | Pad the 12-channel whitening output to 16 channels | exact | 0-2% | small, microbench first |
| H7 | Per-epoch fused permute+crop+flip (+resize) | identical batches | 0.3-0.6% | small |
| H8 | Manual CUDA graph of the whole step with tensor lr | ~yes | 1-2% | high |
| H9 | Power-aware tuning (autotune at steady-state clocks) | yes | 0-2%, uncertain | medium |
| H10 | Break down the 70 ms `prepare` | yes | ≤0.5% | low |
| H11 | Custom Triton pool+BN-stats+GELU | ~yes | ≤1.5% | high |
| H12 | fp16 BN / bf16 (expected to fail; listed so it isn't retried) | — | — | — |

### H1. Skip the conv outputs that max-pool discards
- **Mechanism:** `max_pool2d(·, 2)` on an odd-sized map drops the last row and column. At 32 px the three pool-feeding convs (`conv1` in each group) output 31², 15² and 7² but only 30², 14² and 6² survive the pool. That wastes 6%, 13% and **27%** of those convs, and their backward passes (gradients there are zero) do the same wasted work. At 20 px the waste is 19² vs 18² and 9² vs 8².
- **Fix:** when the input size H is odd, apply `F.pad(x, (1,0,1,0))` and then the conv with `padding=0`. Output row i still sees input rows i-1 to i+1, so the result is exactly the original outputs 0..2k-1. BN runs after the pool, so its statistics are untouched. Inductor should fuse the pad into the producing pointwise kernel (GELU, or the residual add).
- **Where:** `ConvGroup.forward` / `Conv` in `submissions/team_segal/model.py`. H is static per graph, so the branch costs nothing.
- **Saving:** 32 px forward goes from 991 to 904 GFLOP (−8.8%, and the same fraction of backward). 20 px goes from 352 to 336 GFLOP (−4.7%). If conv time is 70-80% of the step and scales with FLOPs, that is about 1.3 ms per 32 px step and 0.25 ms per 20 px step.
- **Risks:** cuDNN sums in a different order (fp16 rounding only). Check in Inductor's generated code that the padded tensor stays channels_last with no extra copy.

### H2. Stop computing the whitening-bias gradient after it is frozen
- **Mechanism:** after step 144 (3 epochs) the whitening bias is frozen by lr=0, but autograd still runs three things for it: (a) `conv1`'s data-gradient (dgrad) in group 1, (b) the GELU backward on the 1024×31×31×12 whitening output, and (c) the bias reduction. (a) is a GEMM with only 12 output channels (27 GFLOP at 32 px), which tiles badly and is likely 0.4-0.8 ms. The whole 32 px phase (steps 198+) never needs it.
- **Fix:** set `model.whiten.bias.requires_grad_(False)` at the resolution switch, or at step `whiten_steps` (a third graph). Restore it to True at the end of `fit`, or in `reset_model`. Dynamo guards on `requires_grad`, so this triggers recompiles. The build warm-up already exercises the switch (17 steps, freeze at step 6, resize at step 9), so compiles stay untimed. The optimizer skips parameters whose grad is None. Lookahead still lerps the bias as before.
- **Risk:** the new graph must be exercised during warm-up, and `cache_size_limit` (8) must have room.

### H3. `torch.backends.cudnn.benchmark_limit = 0` in `build`
- **Mechanism:** cuDNN v8 benchmarking only times 10 algorithm candidates by default. Odd and tiny spatial maps (3², 2², 7²) and low-channel layers are where non-default kernels, including Winograd variants, win.
- **Cost:** benchmarking is untimed during build. Backward convs never use Inductor templates; they always go through cuDNN, so this is the only lever on them.

### H4. `torch._inductor.config.coordinate_descent_tuning = True`
- **Mechanism:** this is not part of `max-autotune` in 2.4 (`_inductor/__init__.py:139`). It tunes block sizes for the BN-statistics, BN-backward, pool and GELU kernels. Channel-wise reductions over N·H·W in channels_last layout are often badly tiled by default.
- **Cost:** only untimed compile time.

### H5. Loss inside the compiled region
- **Mechanism:** `cross_entropy` with label smoothing runs eagerly: about 6-8 kernels forward and 6-8 backward, plus a gap between graphs. Compiling a `model+loss` wrapper fuses these into 1-2 kernels inside the CUDA graph. Estimate 0.1-0.2 ms per step, which matters most in the 20 px phase.
- **Where:** `fit` in `submissions/team_segal/train.py`. Labels become a small cudagraph input.

### H6. Align the 12-channel whitening output for tensor cores
- **Mechanism:** fp16 NHWC kernels prefer C%8==0, and C=12 forces alignment-4 loads, which likely explains the `indexed_wo_smem` kernels in the profile. Zero-pad the GELU(whiten) output to 16 channels (fused) and apply `F.pad` to group 1's `conv1` weight, from 128×12 to 128×16, inside forward. The parameter shape and initialisation stay unchanged, so this is exact.
- **Test first:** microbenchmark the single conv's forward and backward with CUDA events. If H2 removes the dgrad, only the forward and weight-gradient remain to gain.

### H7. Data pipeline: one fused kernel per epoch
- **Mechanism:** each epoch currently runs a mask crop over 50k images (about 3 ms at 32 px; the gather version takes 1.1 ms, per `crop-benchmark.txt`). Odd epochs add a full 307 MB `flip` copy, and every step adds an `images[idx]` gather plus an eager antialiased resize.
- **Fix:** draw the shifts and order with the same RNG calls as now, then write `out[k] = flip?(crop(base[order[k]], shift[order[k]]))` once per epoch. In the 20 px phase, resize the whole epoch in one call. Per-step batches become contiguous slices, and the batches are bit-identical to today's.
- **Saving:** about 20-35 ms per run.

### H8. Whole-step CUDA graph
- **Mechanism:** compile with `max-autotune-no-cudagraphs` and capture forward, loss, backward and `_fused_sgd_` together. PyTorch 2.4 has a `lr: Tensor` overload (`_VariableFunctions.pyi:779`), so the lr updates by `copy_` into a static tensor. Capture a second graph with the lookahead update, used every 5th step. A zeroed momentum buffer reproduces `is_first_step` exactly, so a persistent optimizer reset in `prepare` is equivalent.
- **Saving:** removes all eager kernels and inter-graph gaps (0.2-0.4 ms per step).
- **Risks:** static gradient buffers (needs `zero_grad(set_to_none=False)`), one graph per resolution, effort. The earlier note says Fulcrum measured only ~5 ms from manual graphs, so do this last.

### H9. Power-cap effects
- **Mechanism:** at 300 W the SM clock sits near 1275 MHz. H1 and H2 cut joules as well as time, which may raise the clock and give a saving larger than the FLOP estimate. Separately, cuDNN and Inductor autotune during build, when the GPU is cool and at boost clock. At throttled steady state, a different algorithm may be faster.
- **Test:** run 20-30 s of sustained real-shape steps before the benchmark/autotune passes in build. That changes only which algorithm is chosen, not the timer. Clear it with the organisers first, because "thermal" tricks are on their red-team list.
- **Log during every A/B:** `nvidia-smi --query-gpu=clocks.sm,power.draw --loop-ms=100`, and report joules per step.

### H10. `prepare` (70 ms)
- Time each piece with CUDA events: the H2D copy of 153 MB from pageable memory (probably dominant), `std_mean`, `eigh`, the flip and reflect-pad of 307 MB. If the reflect-pad and flip are significant, fold them into the H7 epoch kernel.

### H11. Custom fused pool, BN statistics and GELU
- Only worth doing if the generated code shows the pooled activations stored and then re-read twice: once for statistics, once to normalise. A one-pass Triton kernel inside `torch.library.custom_op` would cut one read of 30-250 MB per group.
- **Risk:** the coverage matrix records that the earlier reshape-amax pooling diverged under compile.

### H12. Expected to fail (listed so they aren't retried)
- **fp16 BN:** eps=1e-12 underflows to 0 in fp16, so a dead channel divides by zero. Inductor already computes BN in fp32 with fp16 outputs anyway.
- **bf16:** same tensor-core rate as fp16 on A100, with worse precision.

### Measuring against 1.5-9% host variance
1. **Mechanism check (cheap, low noise):** build A and B in one process (two models, each compiled separately) and alternate trials ABAB… for at least 10 pairs. Time each step with CUDA events, split by phase (20 px and 32 px), and report the paired mean difference with a bootstrap CI. Within-host noise in per-step time should be well under 1%, so a 1% effect will show. Run kernel microbenchmarks for at least 2 s so clocks settle at the throttled state.
2. **Exact hypotheses (H1, H2, H6, H7):** check equivalence on a fixed seed: identical batches, logits within fp16 rounding, matching loss curve. That skips a 40-seed accuracy run.
3. **End-to-end:** interleaved same-host harness runs, 40 trials per arm, comparing paired means, with the power/clock logs.

Before anything else, profile the compiled step (`max-autotune-no-cudagraphs` with `torch.profiler`, plus `TORCH_LOGS=output_code`). That gives the real kernel shares to replace the eager competitor profile, and it confirms the int8-offset pooling and the H1/H2 targets.
