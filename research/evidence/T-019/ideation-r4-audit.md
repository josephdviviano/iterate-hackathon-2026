<!-- Frozen report of a read-only audit agent (round 4, timed path); model output, not user input. -->

I found three exact savings in the timed path that previous sweeps missed or set aside. Together with the pinned-copy change they come to about 75–120 ms per trial, roughly 1.5–2.4% of ~4.9 s. Every fix in the top four leaves the trained model unchanged, so none of them needs a new accuracy run.

Everything here comes from reading the code, the torch 2.4.0 source in `.venv`, the local `trials.jsonl` files, and two short CPU-only checks. I modified no files and launched no GPU work.

## Ranked findings

| # | Finding | Est. saving per trial | Exact? | Where |
|---|---|---|---|---|
| 1 | `nn.init.dirac_` runs a Python loop of 2,456 single-element assignments to CUDA tensors | **25–50 ms** | yes | `model.py` `Conv.reset_parameters` (called from `prepare`) |
| 2 | The last 3 optimiser steps (406–408) are thrown away by the final lookahead copy | **30–36 ms** | yes (output identical) | `train.py` `fit` |
| 3 | Per-epoch data work: mask crop with 20 host syncs, a 307 MB flip copy on odd epochs, a per-step gather, and a full crop for the half final epoch | **15–25 ms** | yes (identical batches) | `data.py` |
| 4 | 153.6 MB copy to the GPU from pageable memory, which blocks the CPU before `reset_model` can start | **5–10 ms** | yes | `submission.py` `prepare` |
| 5 | Warm-up uses 2,048 images, so trial 1 pays ~1.5 GB of first-time GPU allocation and loads a few kernels | ~0.2–0.5 ms on the 40-trial mean | yes | `submission.py` `_warm_up` |
| 6 | Small items: `eigh` via cuSOLVER (forces a sync), QR on the GPU, `dct_basis` rebuilt on the CPU each trial, a weight-0 lerp in the final lookahead | ≤1–3 ms | mostly yes (CPU `eigh` would not be) | `model.py`, `train.py` |

### 1. Element-by-element `dirac_` in `prepare` (largest single item)
- In torch 2.4, `nn.init.dirac_` loops over channels and sets one element at a time (`tensor[d, d, 1, 1] = 1`). Across your 8 `Conv` layers that is 24+128+128+128+384+384+640+640 = **2,456** assignments per trial.
- A CPU profile of `reset_model` shows 2,456 `aten::copy_` calls fed by `lift_fresh`. Each one builds a CPU scalar and copies it into the CUDA weight.
- In 2.4 that copy very likely goes through the blocking path (a 2-byte copy plus a stream sync). I couldn't confirm the sync without a GPU.
- Measured: the Python loop alone takes 12.3 ms on CPU tensors. On CUDA each iteration also launches a copy and probably syncs, so roughly 10–20 µs each, about 25–50 ms in total.
- This fits the measured local `prepare` time of **64–78 ms per trial** (from `results/sweeps/s54*/…/trials.jsonl`). The rest of `prepare` should only be about 20 ms.
- **Fix (exact):** `v = w[:cin]; v.zero_(); v.diagonal(0, 0, 1)[kh // 2, kw // 2].fill_(1)`. That is one kernel per conv with no sync. It uses no random numbers, so the random-number stream is unchanged. I confirmed a vectorised version matches `dirac_` exactly on CPU.
- **Verify:** wrap `reset_model` in `torch.cuda.set_sync_debug_mode("warn")` before and after. Check `torch.equal` on all weights after reset with the same seed. Time `prepare` segments with synchronize + `perf_counter`.

### 2. The last 3 steps have no effect on the returned model
- `total_steps = 408`, and lookahead updates after steps 5, 10, …, 405. Steps 406–408 then run.
- At the end, `lookahead.update(decay=1.0)` does a lerp with weight 0, which leaves the slow copy unchanged, then copies it into the model. The slow copy covers every float `state_dict` entry, including BN running stats.
- So the returned model is exactly the state right after step 405, and the last three 32 px steps (~11–12 ms each) are wasted.
- **Checked on CPU:** in a 17-step `fit` run, the final model matched a snapshot taken right after the last periodic lookahead tensor-for-tensor (`torch.equal` True for all).
- **Fix:** stop at `total_steps - total_steps % 5` when lookahead is on. Keep `total_steps = 408` for every schedule (lr, freeze, resolution switch, lookahead decay) so steps 0–404 are unchanged.
- **Caveat:** apply the trim only to the real run, or enlarge the warm-up. With trimming, the 17-step warm-up would run the frozen 32 px phase only once. CUDA graph trees warm up on the first call and record on the second, so recording would land in trial 1.
- **Verify on GPU:** snapshot after step 405 within one run and compare with the returned model. GPU runs aren't bit-reproducible across processes, so compare within a run.
- **README note:** the run would do 405 optimiser steps on a 408-step schedule.

### 3. Epoch data pipeline
- `batch_crop` uses boolean-mask indexing twice per shift for 5 shifts on each of 2 axes. That is **20 host syncs per epoch (180 per trial)**.
- On top of that: a full 307 MB flip copy on odd epochs, a 1,024-image gather every step, and a full 50k crop in epoch 8, which only uses 24 batches.
- Your own measurement (`research/codex-challenger/evidence/crop-benchmark.txt`) puts the 50k mask crop at 2.98 ms against 1.11 ms for a gather.
- **Fix (identical batches):** keep the same random draws in the same order (`randint` shifts, then `randperm`). Then do one gather per epoch: `out[k] = crop(base[order[k]])`, with the odd-epoch flip folded into the column index. Each step then takes a contiguous slice, and the final epoch crops only the rows it uses. `crop_prototype.py` already has the index arithmetic.
- About 2.2–2.6 ms saved per epoch. Some of the sync-bubble cost may be won back by clock recovery under the 300 W cap, so take 15 ms as the low end.
- **Verify:** `torch.equal` on every batch, old generator against new, same seed; this is deterministic on GPU. Then time paired runs.
- Optional, about 1 ms more: fold the flip, `where` and reflect-pad in `TrainingStream.__init__` into the same index arithmetic.

### 4. Copy to the GPU from pageable memory
- `data.images.to(device, non_blocking=True)` on pageable memory still blocks the CPU. Expect about 13–18 ms on a PCIe Gen4 host.
- **Fix (exact):** allocate a pinned uint8 staging buffer once in `build`; it contains no data, so this counts as "reusing allocated memory". In `prepare`, copy into it in about 5 chunks, each followed by an async copy to the GPU on a side stream. Run `reset_model` on the main stream while the copies are in flight, then wait on an event.
- Don't pin the harness tensor in place (`cudaHostRegister`), and don't cache the GPU copy between trials. The API says transfers belong in `prepare`/`train` and are charged to the submission.
- The gain depends on the host, so measure it on the A100.

### 5. First-trial costs
- Warm-up with 50,000 synthetic images (about 5 s more untimed build) instead of 2,048. Trial 1 would then reuse cached GPU memory and already-loaded kernels at the real sizes.
- This also gives every graph phase plenty of warm-up steps, which makes the trim in item 2 safe.

### 6. Minor (≤1–3 ms together)
- The DCT basis could be built once in `build`.
- After item 2, the final lookahead update is a no-op and can be skipped.
- `eigh` on the CPU would avoid cuSOLVER latency, but it is **not exact** (eigenvector signs can flip, which swaps whitening channel pairs).
- The QR inside `orthogonal_` could also move to the CPU, but that changes the random-number stream, so it is not exact either.

## Checked and fine
- **Recompiles and graph re-recording:** the build warm-up hits the same four graph variants in the same order (bias-grad off at step 6 vs 144, resolution switch at 9 vs 204, freeze at 14 vs 327). That is 4 cache entries against `cache_size_limit = 8`. Torch 2.4 has `guard_nn_modules=True` and `inline_inbuilt_nn_modules=False`, so changing `stage1_frozen` switches cached entries rather than recompiling. There is no partial batch (`50000 // 1024` drops the remainder). The trial-to-trial time SD of 12–15 ms rules out recompiles inside timed trials. To confirm cheaply, run 3 trials with `TORCH_LOGS=recompiles,perf_hints` and check that no graph skips CUDA graphs.
- **Syncs in the step loop:** none. `randperm`, `randint` and `rand` all use the GPU generator; indexing uses integer tensors; the loss and fused SGD use plain Python lr values. The only syncs in the whole loop are the ones in item 3.
- **Loop is GPU-bound:** Python per step (schedules, `zero_grad`, the 5 parameter groups) costs about 1.5–2 ms against 4–5 ms or more of GPU time. That is why moving the loss into the graph showed no significant gain on the PCIe card (F-035).
- **Lookahead foreach** (~5 ms per trial), **resize** (inside the compiled graph it measured slower), **dtype conversion and `std_mean`** (<1 ms), and **end of `train`** (nothing runs after the last step besides the final copy) are all fine.
- **Python GC:** no sign of it in the trial timings, so `gc.freeze` isn't worth adding.

## Verification plan
1. Bit-identity checks first: items 1 and 3 by `torch.equal`, item 2 by the within-run snapshot. No 40-seed accuracy run is needed.
2. Then paired harness runs on both local GPUs, with arm order reversed on the second GPU (`research/paired_timing.py`), recording `prepare_time` and `train_time` separately.
3. Finish with one re-check on an A100 PCIe at 300 W.

## Compliance
I found no risk in the current code; build touches only synthetic data. The only lines to stay behind are the two noted in item 4.
