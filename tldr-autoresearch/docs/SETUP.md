# Setup on a new machine

This file covers installing and configuring the harness. The design is in `docs/DESIGN.md`; results are in
`RESULTS.md`. All paths below are relative to the project root (`$RLTLDR_ROOT`, by default the directory that holds
`ctl.sh`). The project root must be this checkout (the directory that holds `tools/` and `rltldr/`): the scripts and the
driver load `tools/`, `rltldr/` and `serve.sh` from the root as well as `data/`, `logs/` and the repositories.

## Requirements

- Linux with 4 GPUs of at least 80 GB each:
  - 2 for serving (TP=2),
  - 1 for the nanochat experiments,
  - 1 for the policy trainer.

  We used 4× RTX PRO 6000 Blackwell (96 GB, sm_120).
- About 250 GB of disk.
- **Passwordless sudo.** The agent and training sandboxes use `sudo unshare` and `setpriv`. They drop to the invoking
  user's uid/gid, which is read before sudo. Do not run the loop without the sandboxes, because the reward would
  then be hackable.
- Internet access once, for models, packages and the autoresearch data.

## Install

```bash
# tools: uv, Node 24, pi
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install Node 24 LTS (e.g. the official tarball under ~/.local), then pi:
npm install -g --ignore-scripts --prefix ~/.local @earendil-works/pi-coding-agent@1.0.0

# serving env (vLLM 0.30.0, torch 2.13 cu130); default location used by the scripts: ~/envs/serve
uv venv ~/envs/serve --python 3.12 && uv pip install --python ~/envs/serve/bin/python "vllm==0.30.0"

# trainer env; default location: ~/envs/train
uv venv ~/envs/train --python 3.12 && uv pip install --python ~/envs/train/bin/python \
  "torch==2.13.0" "transformers==5.18.0" "peft==0.21.2" "accelerate==1.15.0" "liger-kernel==0.8.4" \
  "flash-linear-attention==0.5.2" "trl==1.14.1" aiohttp requests safetensors numpy

# model, plus the bf16 dequantisation the trainer uses as its base (= the served weights)
hf download Qwen/Qwen3.8-27B-FP8 --local-dir ~/models/Qwen3.8-27B-FP8
~/envs/train/bin/python tools/dequant_fp8.py      # -> ~/models/Qwen3.8-27B-FP8-dequant-bf16
```

FlashInfer's JIT needs a consistent CUDA toolkit of version 12.9 or later. If the system `nvcc` is older:
1. Pin the serve venv's CUDA wheels to 13.0 (`nvidia-cuda-nvcc`, `crt`, `cccl` and `nvvm`, all `13.0.*`).
2. Build a symlink overlay with `bin`, `include`, `nvvm`, and a `lib64` that holds `libcudart.so` and
   `stubs/libcuda.so`.
3. `serve.sh` points `CUDA_HOME` at `$RLTLDR_CUDA_HOME`, which defaults to `~/envs/cuda13`. See the comments in
   `serve.sh`.

### Environment variables (all optional)

| variable | default | used for |
|---|---|---|
| `RLTLDR_ROOT` | directory of `ctl.sh` / the `rltldr/` package | project root: the checkout (code under `tools/`, `rltldr/`) plus data, logs, sockets, repos. Do not point it at a data-only directory. |
| `RLTLDR_CONFIG` | `$RLTLDR_ROOT/config.json` | machine-specific settings (see below) |
| `RLTLDR_SERVE_PY` | `~/envs/serve/bin/python` | vLLM, gateway, runner, driver, the h2h/frz components, `python -m rltldr.config` in the scripts |
| `RLTLDR_TRAIN_PY` | `~/envs/train/bin/python` | trainer |
| `RLTLDR_TRUSTED_PY` | `/usr/bin/python3` | the stdlib-only trusted wrapper `tools/ar_run.py` (started by the runner, the h2h runner, `tools/ar_calibrate.sh` and `tools/test_ar_run.py`), and the fallback for bringing up loopback in the sandboxes. The agent's `run.sh` always uses `/usr/bin/python3`, because the sandbox hides `$HOME`. |
| `RLTLDR_CUDA_HOME` | `~/envs/cuda13` | CUDA toolkit for vLLM / FlashInfer JIT |
| `SERVE_GPUS` | `0,1` | the serving GPUs (`serve.sh`) |
| `UV_BIN` | `uv` | uv executable |
| `H2H_CONFIG` | `$RLTLDR_ROOT/h2h_config.json` | head-to-head arms (GPUs, ports); template `h2h_config.example.json` |
| `FRZ_GPU_x1`, `FRZ_GPU_x2` | none (required by `ctl_frz.sh init`) | the frz arms' GPUs, as `<uuid>:<minor>` |
| `PI_BIN` | `pi` | the pi binary for `pi/tests/e2e_run.sh` (the harness uses config `pi_bin`) |

## Configure

```bash
cp config.example.json config.json      # gitignored; then edit it
nvidia-smi -L                           # GPU UUIDs
```

Set these fields in `config.json`:
- `agent_gpu_uuid` and `agent_gpu_minor` (the `/dev/nvidia<N>` of that GPU) for the experiment GPU.
- `trainer_gpu_uuid` for the policy-update GPU.
- `model_dir`, `tokenizer_dir` and `trainer_base_dir` if the models live elsewhere.
- `pi_bin`: the pi executable. The code default is `pi`, looked up on `PATH`. The example sets `~/.local/bin/pi`,
  where the install step above puts it. `h2h_config.json` has its own `pi_bin`, default `pi`.

Paths may start with `~`. The GPU UUIDs have no default: anything that needs one stops with a clear error. Check a
value with `python -m rltldr.config get <key>`. `serve.sh` takes the serving GPUs from `SERVE_GPUS` (default `0,1`).

## The autoresearch repository

The research task is Karpathy's autoresearch. The harness needs a checkout at `autoresearch/` whose branch
`autoresearch/oct3` (config `branch`) is the baseline. On first start the driver imports it into the canonical bare
repository `canon.git`.

```bash
git clone https://github.com/karpathy/autoresearch autoresearch     # we used upstream commit 228791f
cd autoresearch && git checkout -b autoresearch/oct3
uv sync && uv run prepare.py --num-shards 24                        # 24 training shards + validation shard + tokenizer
```

Commit two changes on that branch before the first start. Both are in [`patches/`](../patches/README.md):

1. **GPU patch (only on GPUs without FA3 kernels, such as sm_120).** Upstream's FA3 kernels abort on sm_120. Our
   patch uses FlexAttention for the sliding-window layers and SDPA for the full-causal layers, so
   `WINDOW_PATTERN` keeps its meaning. Hopper keeps FA3.
2. **Harness `program.md`.** One experiment per session, run with `./run.sh "<description>"`. Training happens on
   the runner's GPU, not in the agent's sandbox. The keep/discard decision belongs to the harness.

```bash
cd autoresearch
git apply ../patches/sm120_attention.patch && git commit -qam "sm120 attention patch (baseline)"
cp ../patches/harness_program.md program.md && git commit -qam "harness: program.md for one-experiment sessions"
```

The first commit is the h2h `baseline_commit` (set it in `h2h_config.json`). The second is the `--seed` for
`ctl_frz.sh init`.

`prepare.py` must stay byte-identical to upstream: the runner checks its sha256 (`prepare_sha256` in
`rltldr/config.py`).

## Baseline and noise calibration

Before the driver starts, the ledger needs an `attempt_id: "baseline"` entry. Measure it with the final pipeline:
sandboxed, trusted evaluation, `--no-autotune`, and a fresh compile cache for each run.

```bash
tools/ar_calibrate.sh 3 --baseline     # 3 noise runs -> data/calib_ledger.jsonl, then 1 baseline -> data/ledger.jsonl
python3 tools/ar_stats.py data/calib_ledger.jsonl
```

`tools/ar_calibrate.sh` reads the GPU (`agent_gpu_uuid`, `agent_gpu_minor`), `no_autotune` and `prepare_sha256`
from the config. It runs `tools/ar_run.py` under `$RLTLDR_TRUSTED_PY` with a fresh compile cache under `data/` for
each run, and it refuses to run on an uncommitted repo. Stop the runner first, because both use the agent GPU. The
task prompt's starting best is the mean of these runs. Check their spread: on our machine the sd was 0.0001 with
autotuning off.

`val_bpb` after a fixed 5-minute budget depends on the GPU, because a faster GPU trains more steps. **Numbers from
different hardware are not comparable.**

## Run

```bash
./ctl.sh start      # vllm -> gateway -> runner -> trainer -> driver, each supervised
./ctl.sh status     # processes, GPUs, current policy, group progress
./ctl.sh report     # best val_bpb trajectory, group metrics, insights, updates
python3 tools/dashboard/build.py    # static dashboard
```

**Continuing from trained adapters.** To start from a trained policy instead of the base model:
1. Copy a trainer checkpoint to `data/trainer_ckpt/` and its adapters to `data/adapters/`.
2. Write `data/trainer_state.json` with the checkpoint's version.

On start, the trainer re-publishes its latest version to the gateway.

**Head-to-head and frozen-policy ablation.** See `README.md` ("Running").

## Using more compute

The loop is sequential by design: one experiment at a time, and the agent GPU is idle while the agent thinks. With more GPUs
the natural extensions are:
- **Parallel lanes.** Several independent runner/agent lanes, each with its own branch and best, give more rollouts
  per hour.
- **A bigger update.** A larger LoRA rank, or full fine-tuning sharded over several trainer GPUs.
- **A bigger task.** The higher-compute nanochat settings: a longer budget and deeper models.

`docs/RETROSPECTIVE.md` §3(iii) lists the code changes the lanes need.
