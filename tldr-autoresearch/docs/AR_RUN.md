# autoresearch run wrapper: setup, ledger and noise calibration

Karpathy's `autoresearch` is the rollout environment for the RLTL;DR loop. The pi agent edits `train.py` and runs
each experiment with `./run.sh`. Every experiment ends up in `tools/ar_run.py`, which pins the run to the agent GPU
(physical GPU 2 on the source machine), enforces the time limit, and appends one JSON line per experiment to a
ledger. **One ledger line with a status other than `refused` is one rollout.**

All paths below are relative to the project root (`$RLTLDR_ROOT`, by default the directory that holds `tools/` and
`rltldr/`). `<agent_gpu_uuid>` is the `agent_gpu_uuid` from `config.json` (see `nvidia-smi -L`).

> **History.** Sections 2, 3, 6 and 7 record the first day of setup (2026-10-03). Since then the agent's path to the
> GPU has changed:
> - `run.sh` no longer calls `ar_run.py` itself. It calls `tools/run_client.py`, which POSTs `train.py` to the runner
>   daemon (`rltldr/runner.py`, or `rltldr/h2h_runner.py` for the head-to-head arms). The runner calls `ar_run.py`
>   in its own detached-HEAD workspace, with the attempt id set by the driver.
> - The agent runs in `tools/sandbox.sh` (or `tools/h2h_sandbox.sh`) with every GPU masked. Training runs in
>   `tools/train_sandbox.sh` with only the agent GPU.
> - The metric comes from the trusted evaluation bootstrap `tools/ar_bootstrap.py`, not from what `train.py`
>   prints.
>
> The docstring of `tools/ar_run.py` describes the current behaviour. Sections 1, 4 and 5 below are up to date.

## 1. Layout

| Path | What |
|---|---|
| `autoresearch/` | Clone of `karpathy/autoresearch` (upstream `228791f`) plus the sm_120 attention patch. |
| `autoresearch/.venv` | uv venv: Python 3.10, torch 2.9.1+cu128, 6.9 GB. It contains the GPU pin (§3). |
| `autoresearch/run.sh` | The agent's run command. Untracked (listed in `.git/info/exclude`). It is a copy of the template `tools/run.sh`, with `@RLTLDR_ROOT@` replaced by the real root. |
| `~/.cache/autoresearch/{data,tokenizer}` | 24 training shards plus the pinned validation shard 06542 (25 files, 2.2 GB), and the BPE tokenizer (vocab 8192). |
| `tools/ar_run.py` | Trusted run wrapper and ledger writer. Standard library only. |
| `tools/ar_bootstrap.py` | Trusted evaluation: a judge process computes val_bpb from the model's logits, and a child process runs `train.py`. |
| `tools/train_sandbox.sh`, `tools/sandbox_lib.sh` | The training sandbox: one GPU, no network, read-only filesystem, the caller's uid with `no_new_privs`. |
| `tools/run.sh`, `tools/run_client.py`, `tools/uds_forward.py` | The agent side: the `./run.sh` template, its client, and the loopback-to-socket forwarder inside the agent sandbox. |
| `tools/ar_parse.py` | Log parser: summary block, status classification, sanitized error tail. |
| `tools/ar_gpu_pin.py` | Installs and checks the venv GPU pin (`sitecustomize.py` + `.pth`). |
| `tools/gpu_watchdog.py` | Kills agent-marked processes found on any GPU other than the agent GPU. |
| `tools/ar_stats.py` | Ledger summary: mean/std by `train.py` hash, status and flag counts. |
| `tools/ar_calibrate.sh` | The noise-calibration driver used in §6. |
| `tools/test_ar_run.py` | Wrapper tests (§7). |
| `data/ledger.jsonl` | **Main ledger.** |
| `data/calib_ledger.jsonl` | Noise-calibration ledger. |
| `data/runs/<run_id>/` | Per run: `train.py` copy and `diff.patch` (both written **before** the run), the raw `run.log`, and `trusted/record.jsonl` (the judge's signed records). |
| `data/locks/<gpu-uuid>.lock` | flock that allows one run per GPU at a time. |
| `data/cache/{inductor,triton,nv}` | Compile caches for these runs only, not shared with the trainer. |

## 2. Setup performed

```bash
git clone https://github.com/karpathy/autoresearch autoresearch
cd autoresearch
# apply the sm_120 attention patch: FlexAttention for "S" layers, SDPA for "L" layers
# (the baseline commit of the autoresearch bundle already contains it)
# program.md: one line added under "Experimentation":
#   **GPU note**: This GPU is sm_120 (RTX PRO 6000 Blackwell, 96 GB). FA3 is unavailable; attention uses
#   FlexAttention (sliding-window layers) / SDPA (full-causal layers) via `attention()` in `train.py`.
uv sync                                            # 42 s
uv run prepare.py --num-shards 24                  # 49 s: 25 shards (2.2 GB) + tokenizer (32.9 s)
chmod a-w prepare.py                               # sha256 4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150
printf 'run.log\nresults.tsv\nrun.sh\n' >> .git/info/exclude
git config user.name "rltldr autoresearch"; git config user.email rltldr@localhost   # repo-local identity
git commit -m "sm120 attention patch (baseline)"   # c7666de on master
git checkout -b autoresearch/rltldr                # the experiment branch, also at c7666de
cd .. && python3 tools/ar_gpu_pin.py --repo autoresearch --gpu <agent_gpu_uuid>
```

- Disk before setup: 239 GB free on the filesystem that holds the home directory.
- Baseline `train.py` sha256 is `45696ba1f163c0173862c52d7ce07aa154a186803d50d84f35acff689481c954`. This is the **root parent** for the RL loop.
- Upstream `train.py` (the baseline's `parent_train_sha256`) is `2954175f4ac4...`.

## 3. GPU pinning (defence in depth)

These layers were built before the sandboxes, when the agent ran with no namespaces. They are all still active.
The sandboxes now add a stronger layer: the agent sees no GPU device at all, and training sees only
`/dev/nvidia<agent_gpu_minor>`.

| Layer | Mechanism | Verified |
|---|---|---|
| 1. venv pin | `.venv/lib/python3.10/site-packages/sitecustomize.py` **and** `00_ar_gpu_pin.pth` force `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<agent_gpu_uuid>` at interpreter start-up. A `sitecustomize.py` earlier on `PYTHONPATH` cannot shadow the `.pth` file. | `CUDA_VISIBLE_DEVICES=0,1,3 uv run gpucheck.py` reports 1 device: the agent GPU, by PCI bus id and uuid. The same result holds with `CUDA_VISIBLE_DEVICES=0` plus `CUDA_DEVICE_ORDER=FASTEST_FIRST`, and with a shadowing `sitecustomize.py` on `PYTHONPATH` plus `CUDA_VISIBLE_DEVICES=3`. |
| 2. wrapper env | `ar_run.py` drops every `PYTHON*`, `CUDA_*`, `NVIDIA_*`, `LD_*`, `UV_*`, `CONDA*`, `TORCHINDUCTOR_*` and `TRITON_*` variable from the caller. It then sets the GPU UUID, `PATH`, `HOME` (from passwd), `UV_FROZEN=1`, `UV_OFFLINE=1` and its own compile caches. | All 6 calibration and baseline runs were launched with `env -i PATH=/usr/bin:/bin`. |
| 3. self-healing pin | Before every run, `ar_run.py` rewrites the pin files if they are missing or modified, and records the flag `gpu_pin_repaired`. | Deleted and tampered pin files were restored, and the run carried the flag. |
| 4. tamper refusal | If `train.py` mentions `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, `NVIDIA_VISIBLE_DEVICES` or `putenv`, the run is `refused` (flag `gpu_tamper`) and the GPU is not touched. Code inside `train.py` runs after layer 1, so this is the one way `train.py` itself could move training to another GPU. | `os.environ["CUDA_VISIBLE_DEVICES"]="3"` in `train.py` was refused, exit 3. |
| 5. in-run poll | Every 5 s, `nvidia-smi` lists compute apps. A descendant of the run on another GPU kills the whole run (status `crash`, flag `gpu_violation`). Foreign processes on the agent GPU are recorded (`gpu_shared`, `gpu_shared_at_start`). | Tested with mocked `nvidia-smi` output; the run was killed in under 30 s. |
| 6. GPU lock | `data/locks/<uuid>.lock` is passed to the training process tree. A second concurrent run is `refused` (`gpu_locked`). If the wrapper is SIGKILLed, `timeout(1)` gets SIGTERM through `PR_SET_PDEATHSIG` and takes the training down within seconds, which releases the lock. | The fd is held by the whole training process tree (checked in `/proc/<pid>/fd`). |
| 7. watchdog | `gpu_watchdog.py --allowed <agent_gpu_uuid> --root-pid <pi pid> --incidents data/incidents.jsonl` SIGKILLs only processes that carry `AR_AGENT=1` in their environment or descend from pi. | With a bogus allowed UUID, the marked dummy on the agent GPU was killed. The unmarked dummy, vLLM and the trainer were untouched. `--dry-run` reports without killing. |

**Holes that existed before the sandboxes** (both are closed now, because the agent sees no GPU device):
- An agent shell command outside the venv (for example `python3 -c 'import torch...'` with a system interpreter) could open any GPU. Layer 7 and the pi `tool_call` guard covered this.
- Obfuscated code inside `train.py` can change GPU visibility (for example by building the variable name from strings). Layer 5 catches it within one poll interval. Inside `train_sandbox.sh` only the agent GPU's device node is usable anyway.

## 4. `ar_run.py` interface

```
ar_run.py --repo REPO --ledger LEDGER --gpu GPU_UUID --prepare-sha SHA256 --desc DESC
          [--timeout 660] [--budget 300] [--min-steps 20] [--max-overhead 330]
          [--runs-dir <ledger dir>/runs] [--venv <repo>/.venv] [--lock-dir $RLTLDR_ROOT/data/locks]
          [--cache-dir $RLTLDR_ROOT/data/cache] [--poll 5] [--gpu-minor N] [--no-autotune]
          [--full-summary] [--isolate] [--unsandboxed (tests only)]
env: AR_ATTEMPT_ID (default unset -> attempt_id null, no cap), AR_MAX_RUNS (default 3; <= 0 disables the cap),
     AR_AGENT_HEAD (the agent's own git HEAD, recorded as agent_head), RLTLDR_ROOT (default lock/cache dirs)
```

Run it with a trusted, standard-library-only interpreter in isolated mode, e.g.
`"${RLTLDR_TRUSTED_PY:-/usr/bin/python3}" -I tools/ar_run.py ...`, so that it does not depend on the caller's `PATH`,
`PYTHONPATH` or active venv. The runner daemons do exactly this.

The agent only ever calls `./run.sh "<description>"` in its clone. That is the template `tools/run.sh`, with the root
filled in when the clone is made:

```bash
exec /usr/bin/python3 -I "@RLTLDR_ROOT@/tools/run_client.py" "$@"
```

**Order of operations in one invocation:**
1. **Refusal checks**, which never use the GPU:
   - the GPU lock is busy;
   - the GPU is not visible to `nvidia-smi`;
   - the per-attempt cap: at least `AR_MAX_RUNS` non-refused entries with this `AR_ATTEMPT_ID` already exist in *this* ledger;
   - this attempt already has an `ok` run (one measured result per attempt);
   - GPU tampering in `train.py`.
2. **Snapshot before the run.** `train.py` is copied to `runs/<id>/train.py`, and `git diff <parent_rev> -- train.py` (parent commit vs. working tree) is written to `runs/<id>/diff.patch`.
   - The parent is the nearest first-parent ancestor whose `train.py` already has an `ok` entry in this ledger. The search starts at `HEAD` if `train.py` is dirty, if HEAD is detached (the runner workspace), or at `HEAD^` if the experiment was committed (the `program.md` flow).
   - If no ancestor qualifies, the parent is that immediate commit.
   - Effect: after a crash-fix commit, the parent is still the version the idea branched from.
3. **The run.** The pin is checked and repaired. Then `train.py` runs under the trusted bootstrap (`ar_bootstrap.py`), inside `train_sandbox.sh`, under `timeout --signal=TERM --kill-after=5 <timeout>`, in its own session. The environment is described in §3, and `nvidia-smi` is polled every 5 s.
   - The judge gets a one-time HMAC key and signs its records (start, eval, val_leak, end). `ar_run.py` keeps only correctly signed lines.
   - `--isolate` (h2h runner) also hides `$HOME` except the repo (read-only, at `~/autoresearch`), the venv interpreter and the data. It masks the validation shard (only the judge reads it, through an fd) and makes the compile cache an ephemeral copy-on-write layer.
   - SIGTERM, SIGINT or SIGHUP to the wrapper kills the sandbox tree at once (with `sudo kill`). The run is recorded as `crash` with flag `interrupted`.
4. **Parse and record.** The wrapper parses the log, takes val_bpb and the training time from the judge's records, sets flags, and appends **exactly one** JSON line under `flock` + `fsync`. `seq` (the 0-based line number) is computed under the lock.

**stdout, which is all the agent sees:**
- `ok`: exactly two lines, `val_bpb:          1.080073` and `peak_vram_mb:     45012.5` (with `--full-summary`: upstream's whole summary block, with the trusted val_bpb and training time).
- Invalid (the run finished, but a trusted check failed): `status: invalid`, then the reasons.
- Failure: `status: <status>`, then a sanitized error tail.
  - The tail is at most 4 KB and at most 40 lines; each line is cut at 400 characters.
  - Progress records split on `\r` are removed. Text glued after the last progress record, such as `Traceback ...` or `FAIL`, is kept.
  - Runs of site-packages and stdlib traceback frames collapse to `[... N library frame(s) omitted ...]`.
  - For timeouts, fail_loss and OOMs, a `last progress: step N (...)` line is added.
- Refused: `status: refused`, then the reason.
- The raw log is never printed. It stays in `runs/<id>/run.log`.

**Exit codes:** 0 = ok, 1 = failed or invalid run, 3 = refused.

**Status values:**

| Status | Meaning |
|---|---|
| `ok` | The summary block parsed and val_bpb is finite and > 0. |
| `crash` | Any other failure, including `gpu_violation` and `interrupted`. |
| `oom` | CUDA out of memory. |
| `fail_loss` | train.py's NaN or loss > 100 fast-fail. |
| `no_kernel` | "no kernel image is available". |
| `timeout` | The wall-clock limit was reached (`timeout` exited 124 or 137 near the limit, or the wrapper's own deadline fired). Metrics printed late are discarded. |
| `refused` | One of the step-1 refusal checks failed. |

**Ledger line fields.**

Required fields:
- `run_id` (`YYYYmmdd-HHMMSS-xxxxxx`, UTC), `attempt_id`, `ts` (ISO start time), `wall_s`.
- `head`, `train_sha256`, `parent_train_sha256`, `prepare_sha256`.
- `desc`, `status`, `returncode` (from `timeout`), `val_bpb` (the judge's value; null unless ok).
- `metrics`: train.py's own summary keys (`training_seconds`, `total_seconds`, `peak_vram_mb`, `mfu_percent`, `total_tokens_M`, `num_steps`, `num_params_M`, `depth`), `val_bpb` (trusted) and `val_bpb_stdout` (printed), plus the trusted timing fields (`t_train_trusted`, `t_train_source`, `t_train_limit`, `t_to_eval`, warm-up counts); `{}` unless ok.
- `error_tail` (null when ok; the refusal reason when refused), `flags`, `diff_path`, `log_path`, `agent_head`.

Extra fields:
- `seq`, `branch`, `mode` (`committed` or `dirty`), `parent_rev`, `train_copy_path`, `diff_lines`, `train_ast_sha256`, `parent_ast_sha256`.
- `deps_git_hashes` (pyproject.toml, uv.lock), `last_step`, `ts_end`, `gpu`, `max_runs`, `timeout_s`, `trusted` (the verified judge records).
- `parent_val_bpb` (mean of the parent's ok entries), `parent_n`, `delta_vs_parent` (= parent − new; positive is better).
- `gpu_foreign_at_start`, `gpu_foreign_max_mib`, `gpu_violations` (only when relevant).

**Flags.** Flags marked *invalid* make the run `status: invalid` for the agent and void it for the harness
(`rltldr/verdict.py`):

| Flag | Meaning |
|---|---|
| `no_trusted_eval` (invalid) | No signed evaluation record: `prepare.evaluate_bpb` was not called. |
| `multiple_evals` (invalid) | `evaluate_bpb` was called more than once. |
| `val_leak` (invalid) | Validation data was accessed outside the final evaluation. |
| `non_causal` (invalid) | The causality probe saw logits at positions <= t change when later tokens changed. |
| `eval_unverifiable` (invalid) | `model(idx)` did not return `[B, T, vocab]` logits. |
| `over_time_budget` (invalid) | The trusted training time exceeded the budget, or the warm-up did not look like a warm-up (`metrics.timing_problems`). |
| `prepare_py_modified` (invalid) | sha256 differs from `--prepare-sha`. |
| `too_few_steps` | `num_steps` < `--min-steps` (20). |
| `wall_clock_excessive` | wall > budget + `--max-overhead` (330 s). |
| `nonzero_exit` | ok summary, but non-zero exit. |
| `empty_diff` | No change vs. the parent (also when only comments or formatting changed: same AST). |
| `reeval` | This `train.py` (up to comments and formatting) already has an ok entry. |
| `startup_long`, `stdout_mismatch` | Informational: slow start-up or compile; printed val_bpb differs from the trusted one. |
| `gpu_pin_repaired`, `gpu_pin_missing_venv` | Layer 3 had to restore the pin; the venv was missing. |
| `gpu_shared_at_start`, `gpu_shared` | Another process was on the agent GPU. |
| `gpu_violation` | A run process appeared on another GPU. |
| `interrupted` | The wrapper received a signal. |
| `run_cap`, `attempt_has_ok_run`, `gpu_tamper`, `gpu_locked`, `gpu_missing` | Refusal reasons. |

**Keep inference:** entry *k* was kept iff some later entry has `parent_train_sha256 == entry_k.train_sha256`.

**Grouping:** filter `status != "refused"`, then use `group = rank // 8`.

## 5. Calibration and baseline commands

```bash
tools/ar_calibrate.sh 5 --baseline > logs/ar_calib.log 2>&1
python3 tools/ar_stats.py data/calib_ledger.jsonl
```

The script refuses to run on a dirty repo. It calls `ar_run.py` directly under `env -i`, not through `run.sh`
(which submits to the runner daemon). It runs N times with `AR_ATTEMPT_ID=calib-<i>` into
`data/calib_ledger.jsonl`, then once with `AR_ATTEMPT_ID=baseline` into the main ledger.

## 6. Noise calibration results (2026-10-03, agent GPU, unmodified baseline `c7666de`)

These runs used the first version of the wrapper (no sandbox, no trusted evaluation, Inductor autotuning on).

| attempt | val_bpb | steps | training_s | total_s | wall_s | peak VRAM MB | step-0 loss |
|---|---|---|---|---|---|---|---|
| calib-1 (cold compile cache) | 1.078444 | 510 | 300.5 | 347.2 | 351.3 | 45012.5 | 9.011380 |
| calib-2 | **1.070581** | 508 | 300.2 | 335.3 | 337.8 | 45012.5 | 9.011037 |
| calib-3 | 1.077206 | 511 | 300.6 | 331.5 | 334.0 | 45012.5 | 9.011410 |
| calib-4 | 1.080074 | 512 | 300.6 | 331.7 | 334.1 | 45012.5 | 9.011410 |
| calib-5 | 1.079887 | 512 | 300.6 | 331.5 | 334.0 | 45012.5 | 9.011410 |
| baseline (main ledger) | 1.080073 | 511 | 300.0 | 331.1 | 333.6 | 45012.5 | 9.011410 |

**calib-1..5:**
- val_bpb mean **1.077238**, sample std **0.00390**, range 0.00949.
- steps 510.6 ± 1.7.
- peak VRAM 45012.5 MB in every run.
- wall time 338.2 ± 7.5 s (the cold first run took 351 s; warm runs take 334 s).
- tokens 267.7 ± 0.9 M; about 874K tok/s at 600 ms/step.
- MFU 21.1% (against the H100 peak hard-coded in train.py).

All six runs (calib-1..5 and baseline) give mean 1.077711 and std 0.00368. The main-ledger baseline is 1.080073.

**Why the noise is larger than the earlier n=2 estimate (0.001):** the seed is fixed, but different compiled-kernel selections produce different numerics.
- Runs 3, 4, 5 and the baseline all have bitwise-identical step-0 and step-1 losses (9.011410 and 8.854344). Their spread comes only from GPU nondeterminism and step-count jitter: **std 0.00141, n=4**.
- calib-1 (cold Inductor cache) and calib-2 compiled different kernels, visible in their different step-0 losses. They landed 0.0009 and 0.0087 away from that cluster (mean 1.079310).
- Every agent edit to `train.py` triggers a recompile, so the per-evaluation σ with autotuning on is **about 0.004 bpb**.

**Implications for the RL reward** (as concluded on that day):
- Use `r = clip(delta_vs_parent / 0.004, -3, 3)`, or a binary "improved by more than ~2σ ≈ 0.008". Do not use raw deltas. Typical real wins are 0.0005–0.007, which is at or below 1σ.
- Re-evaluate any apparent win before trusting it. `reeval` entries average into `parent_val_bpb` automatically.
- Follow-up (done later): disabling Inductor runtime autotuning (`--no-autotune`, config `no_autotune`) brought the measured noise down to about 0.0001 bpb (see `reward_sigma` in `rltldr/config.py`).

## 7. Tests

```bash
python3 tools/test_ar_run.py --unit   # pure unit tests (timing rule, summary block): no sudo, GPU or venv
python3 tools/test_ar_run.py          # all GPU-free cases: needs passwordless sudo, the autoresearch venv and a
                                      # real GPU UUID (AR_TEST_GPU or agent_gpu_uuid); no CUDA context is created
python3 tools/test_ar_run.py --gpu    # + the CUDA path of the judge on that GPU
```

Each case uses a throwaway git repo with a fake `train.py` and a CPU-only fake `prepare.py`, the real venv, and a
private ledger and lock dir. The scratch dir (`AR_TEST_TMP`, default `scratch/`) must lie under `$HOME` and not
under `/tmp`. Coverage (the first version, on 2026-10-03, had 28 checks; the suite has since grown with the sandbox,
trusted-evaluation and `--isolate` cases):
- **ok path:** stdout is exactly the two lines; every required field is present; the snapshot is written.
- **Integrity flags:** a prepare-sha mismatch sets `prepare_py_modified` together with `reeval`.
- **Failure statuses:**
  - `crash` (SyntaxError), `fail_loss` (FAIL glued to the progress line), `oom`;
  - a 100 KB line plus 400 noise lines, which produce an error tail of at most 4096 B with no `\r` and no progress records.
- **Refusals:** the cap (AR_MAX_RUNS=2, third run refused with exit 3, and still exactly one line per invocation); `AR_MAX_RUNS=0` disables the cap; tamper refusal.
- **Process control:**
  - `timeout` (`--timeout 4`, killed in under 15 s);
  - a concurrent run refused with `gpu_locked`;
  - SIGTERM to the wrapper gives `crash` + `interrupted` with no leftover process;
  - mocked GPU violation gives `crash` + `gpu_violation`;
  - runner kill pattern (TERM, default 5 s poll): the ledger line is written about 0.06 s after TERM;
  - wrapper SIGKILLed mid-run: no orphaned training process, and the next run is not lock-refused.
- **Parent detection:** in a detached-HEAD checkout (the runner workspace), a `train.py` identical to HEAD gets `parent_rev=HEAD`, `diff_lines=0`, `empty_diff` + `reeval`, and `delta_vs_parent=0`.
- **Trusted evaluation and `--isolate`:** forged records and a walk to the judge's key find nothing; hidden run data, repos and `~/.claude`; the masked validation shard is reported as `val_leak`; arm-neutral paths in tracebacks; nothing carried between runs through the compile cache; warm-up exploits flagged `over_time_budget`.

**Real runs on 2026-10-03** (a separate test ledger, scratch clone of the repo; the main branch was never modified):
- Deliberate syntax error, `AR_ATTEMPT_ID=capdemo AR_MAX_RUNS=2`: runs 1 and 2 gave `status: crash` with tail `File ".../train.py", line 17 / import torch.nn as nn( / SyntaxError: invalid syntax`, exit 1. Run 3 gave `status: refused` with "Run cap reached: this attempt (capdemo) has already used 2/2 ...", exit 3, and no GPU use.
  - The same refusal was reproduced through the then-current `autoresearch/run.sh` under `env -i`.
- Real OOM (`DEVICE_BATCH_SIZE=256`) on the agent GPU: `status: oom` after 21.5 s wall. The tail is 1.6 KB: the train.py frames, `[... 6/12 library frame(s) omitted ...]`, the `torch.OutOfMemoryError` line, and `last progress: step 0 ...`.
- Pin self-heal: after deleting `sitecustomize.py` and corrupting the `.pth`, `ar_gpu_pin.py --check` reported "pin broken" (exit 1). The next `ar_run` restored both files (`gpu_pin_repaired`), and `--check` then reported "pin intact".
- `gpu_watchdog.py`: see §3, layer 7.

## 8. Notes for consumers

- **`program.md`:** the harness version (experiments through `./run.sh`) was committed in the autoresearch repo as `5ded6b9`.
- **Meaning of `flags`.** `flags` mixes integrity problems with informational markers:
  - Informational: `reeval` (this `train.py` was already measured `ok`), `startup_long`, `stdout_mismatch`, `gpu_pin_repaired`, `gpu_pin_missing_venv`, `gpu_shared`, `gpu_shared_at_start`.
  - A consumer that rejects every run with a non-empty `flags` will:
    - reject every confirmation re-run, since it is always `reeval`;
    - reject the first run in a fresh venv, which gets `gpu_pin_repaired`;
    - drop the `reeval` calibration runs from noise and baseline means.
  - `rltldr/verdict.py` holds the harness's decision: which flags void a measurement, and why.
- **`AR_ATTEMPT_ID` comes from the environment.** The runner daemon outside the agent sandbox sets it, so the agent cannot override it (or the ledger path).
- **If the wrapper is SIGKILLed** (not TERM/INT/HUP), no ledger line is written for that run. The training is stopped within seconds by `PR_SET_PDEATHSIG`, so the GPU lock is not held. TERM/INT/HUP stop the training at once, so even callers that SIGKILL a few seconds after TERM (`rltldr/runner.py`) get their ledger line.
- **`gpu_watchdog.py` is not started by the harness.** With the agent sandboxed (no GPU device nodes) it is a spare layer. Start it with pi's PID if you run the agent without the sandbox.
