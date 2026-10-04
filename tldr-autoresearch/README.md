# TL;DR autoresearch: a local model that gets better at research by doing research

A 27B open-weight model runs Karpathy's **autoresearch** loop on 4 GPUs that we own. It edits a nanochat `train.py`,
trains for 5 minutes, and is judged on validation bits per byte. While it works, the model is trained **online** on
its own experiments with **RLTL;DR** (arXiv 2609.37633). Each experiment is a rollout. After a failed experiment, the
model writes a one-sentence TL;DR of what went wrong, and later attempts see these notes. Every 8 experiments a LoRA
update is trained and hot-swapped into the serving engine, and research does not stop for it.

## Problem

Autonomous ML-research agents such as autoresearch already find real improvements. They do not learn from their own
results: run 100 is made by the same model as run 1. They also usually depend on a frontier API. We ask two
questions:
- Can a **local** model, on hardware you own, become a better researcher **by doing research**?
- Can this be measured honestly, when the agent itself writes the code that produces the metric?

## Approach

The RLTL;DR paper maps onto autoresearch as follows:

| RLTL;DR | here |
|---|---|
| rollout | one autoresearch experiment: a fresh pi session edits `train.py`, commits, and calls `./run.sh` |
| verifier | the harness's trusted `val_bpb` of that run. Success means it beats the current best by 0.0005 on a fresh-compile confirmation re-run. |
| TL;DR insight | after a failure, the policy writes a one-sentence hint in a separate judge chat. Later attempts in the same group see these hints. |
| group of K | 8 sequential experiments |
| update | LoRA (r = 32) GRPO plus 0.5 × SFT on the insight tokens, trained on the exact token ids that vLLM served. The new adapter is loaded under a new name and becomes active at the next attempt. |

```
 GPU 0,1  vLLM, Qwen3.8-27B-FP8 TP=2, base + LoRA policy-v1, v2, ...      serve.sh
            ▲ chat (token ids + logprobs recorded)
 gateway  routes "policy" -> current version, injects TL;DR insights      rltldr/gateway.py
            ▲
 agent    pi coding agent, one sandboxed session per experiment           rltldr/driver.py, tools/sandbox.sh
            │ ./run.sh (unix socket)
 GPU 2    runner: pristine checkout + submitted train.py -> 5-min run,    rltldr/runner.py, tools/ar_run.py
          trusted eval -> ledger
 driver   keep/discard + confirmation re-runs, rewards, insights, groups of 8
 GPU 3    trainer: LoRA Dr.GRPO + insight-SFT -> policy-v{N} -> vLLM      rltldr/trainer.py
```

The paper-to-code mapping, its deviations and the data layout are in [`docs/DESIGN.md`](docs/DESIGN.md).

**Two follow-up experiments test where the gains come from:**
- **h2h.** The base model and the final policy v5 run as two continuous upstream-style sessions, side by side.
- **frz.** v5 is frozen and runs in the RL harness twice, once with TL;DR insights and once without.

## Integrity: the reward must not be hackable

The policy is optimised against this harness, so every path from agent-written code to the reward is closed or
checked.

- **Trusted evaluation** (`tools/ar_bootstrap.py`):
  - `val_bpb` is computed by harness code from the model's logits, using autoresearch's pristine
    `prepare.evaluate_bpb`. It is never read from `train.py`'s output.
  - A causality probe and an audit hook detect any reading of the validation data, as well as double evaluation.
  - A trusted step-based timer enforces the 5-minute budget.
  - The results are HMAC-signed with a one-time key.
- **Sandboxes.** `sudo unshare` gives private namespaces, `no_new_privs`, and the invoking user's uid.
  - The agent sees no GPU and has a read-only filesystem except its throwaway clone.
  - Its network is loopback only, with forwards to the gateway (chat only) and the runner (`/run` only).
  - Training sees one GPU and has no network (`tools/sandbox.sh`, `tools/train_sandbox.sh`, `tools/h2h_sandbox.sh`).
- **Canonical repo.** The branch lives in a driver-only `canon.git`. The agent gets a fresh clone each attempt, and
  `results.tsv` is written by the harness.
- **Confirmation re-runs.** An apparent win is re-run with a fresh compile cache. The keep decision and the new best
  use only that unselected re-run. The runner allows one result per attempt, so there is no re-run fishing.

The integrity checks have fired on policy-written code. For example, a prefetch thread that trained past the budget
was flagged and its re-run refused (RESULTS.md, R3).

## Results

All numbers are from [`RESULTS.md`](RESULTS.md), which lists the metric, runs, split, baseline, command and commit
for each. The tables are in [`results/`](results/). The metric is val_bpb on autoresearch's pinned validation shard
(lower is better). The baseline is 1.079941 (unmodified `train.py`, mean of 4 runs).

| | result |
|---|---|
| **R1. Online RLTL;DR run** (1 run, 40 experiments, policy v0 → v5) | **1.007697** (−6.7%) at experiment 35, 6.54 h. 9 keeps. |
| **R2. h2h: base vs v5**, continuous sessions (1 session each, 68 and 74 runs) | best single run 1.012084 vs 1.012210: a **tie**, within noise (σ ≈ 0.00038) |
| **R3. frz: frozen v5, insights on vs off** (snapshot, 8 experiments each, still running) | 1.048194 vs 1.021257 after one group: too early to conclude |

Caveats (detailed in RESULTS.md):
- 81% of R1's gain came from two batch halvings. This is a hardware adaptation that every system found.
- The RL harness beat continuous sessions (1.0077 in 35 experiments vs about 1.012 in about 70 runs). That gap is not
  shown to come from the RL updates, which were tiny (16 optimizer steps in total).
- The success rate does not differ by policy version (p = 0.39).
- Every result is from a single run.

## Running

**Prerequisites:**
- Linux with 4 GPUs of at least 80 GB each.
- Passwordless sudo, which the sandboxes need.
- uv, Node 24 and the pi coding agent.
- Two Python environments: serving (vLLM 0.30) and training (torch, transformers, PEFT).
- Qwen3.8-27B-FP8 weights.
- An autoresearch checkout with its data, plus our two changes in [`patches/`](patches/README.md).

The step-by-step setup and the baseline measurement are in [`docs/SETUP.md`](docs/SETUP.md).

```bash
cp config.example.json config.json   # set agent_gpu_uuid / trainer_gpu_uuid (nvidia-smi -L), model paths
./ctl.sh start                       # vllm -> gateway -> runner -> trainer -> driver, each supervised
./ctl.sh status                      # processes, GPUs, current policy, group progress
./ctl.sh report                      # best val_bpb trajectory, insights, updates
python3 tools/dashboard/build.py     # static dashboard -> dashboard/index.html
```

**Head-to-head (R2).** Set the two arms' GPUs in `h2h_config.json` (template `h2h_config.example.json`). vLLM keeps
running under `./ctl.sh`.

```bash
python3 tools/h2h/make_adapters.py && python3 tools/h2h/setup.py --create-start
./ctl_h2h.sh start && ./ctl_h2h.sh status
```

**Frozen-v5 ablation (R3).**

```bash
FRZ_GPU_x1=<uuid>:<minor> FRZ_GPU_x2=<uuid>:<minor> ./ctl_frz.sh init --seed <start commit>
./ctl_frz.sh start gateway-x1 runner-x1 gateway-x2 runner-x2
./ctl_frz.sh baseline x1 && ./ctl_frz.sh baseline x2 && ./ctl_frz.sh start driver-x1 driver-x2
```

**Environment variables.** The main ones are below. The full list, including `RLTLDR_CUDA_HOME`, `SERVE_GPUS`,
`H2H_CONFIG`, `FRZ_GPU_x1`/`FRZ_GPU_x2` and `PI_BIN`, is in [`docs/SETUP.md`](docs/SETUP.md).

| variable | default |
|---|---|
| `RLTLDR_ROOT`: the project root. It must be this checkout (the directory that holds `tools/` and `rltldr/`), because code is loaded from it as well as data, logs and repos. | the directory of `ctl.sh` |
| `RLTLDR_CONFIG` | `$RLTLDR_ROOT/config.json` |
| `RLTLDR_SERVE_PY` | `~/envs/serve/bin/python` |
| `RLTLDR_TRAIN_PY` | `~/envs/train/bin/python` |
| `RLTLDR_TRUSTED_PY`: runs the stdlib-only trusted wrapper `tools/ar_run.py` | `/usr/bin/python3` |
| `UV_BIN` | `uv` |

**Reproducing the result tables:**

```bash
python3 tools/export_results.py --until 2026-10-04T12:00:00Z
```

It reads `$RLTLDR_ROOT/data` and writes `results/`.

**Tests that need no GPU, sudo or running services:**
- `node pi/tests/guard.test.mjs` (the guard extension)
- `python3 tools/test_ar_run.py --unit`
- `python3 tests/test_h2h_dashboard.py`
- `python3 tests/test_h2h_setup.py --create-start-only`
- `bash tests/test_ctl_h2h.sh` (fake components on free ports)
- `$RLTLDR_SERVE_PY tests/test_h2h_gateway.py --part c` (a fake vLLM on ports 18999 and 18111)

**Tests that need sudo and machine setup** (`canon.git`, uv, the autoresearch venv, configured GPUs):
- the full `tools/test_ar_run.py` (a GPU UUID, sudo, the venv)
- the full `tests/test_h2h_setup.py` (the real `canon.git` and uv)
- `tests/test_h2h_runner.py` (trains a fake `train.py` on the arms' GPUs; refuses while production runners run)
- `tests/test_h2h_supervisor.py` and `tests/test_h2h_sandbox.sh` (pi and sudo)
- `tests/trainer/*` (the trainer GPU and the real weights)

## Repository layout

```
ctl.sh, serve.sh            RL run control (vLLM, gateway, runner, trainer, driver) and the vLLM launcher
ctl_h2h.sh, ctl_frz.sh      head-to-head and frozen-v5 ablation control
config.example.json         machine-specific settings template (copy to config.json)
rltldr/                     config, gateway, driver, runner, trainer (+ model_utils), insight generation, verdicts;
                            h2h_{config,gateway,runner,supervisor}.py for the head-to-head
tools/                      trusted run wrapper (ar_run.py), eval bootstrap (ar_bootstrap.py), sandboxes, run.sh
                            template + client, dashboards, export_results.py, FP8 -> bf16 dequantisation, h2h/ setup
pi/agent/                   pi configuration and the guard extension that blocks off-limits tool calls
examples/frz/               templates for the frz arms' configs
patches/                    our sm_120 attention patch and harness program.md for autoresearch
tests/                      h2h, sandbox and trainer tests
docs/                       DESIGN (method + deviations), SETUP, RETROSPECTIVE (mid-run analysis), H2H_SPEC, AR_RUN
results/                    derived result tables (see RESULTS.md)
```

## Credits

- **Qwen3.8-27B-FP8** (Qwen team): the policy model.
- **RLTL;DR**, "Self-improvement by Internalizing Self-generated Feedback" (arXiv 2609.37633): the method. Our
  implementation was written from the paper.
- **autoresearch** and **nanochat** by Andrej Karpathy: the research task. Our harness
  `patches/harness_program.md` and the h2h `tools/h2h/program.md` are adapted from autoresearch's `program.md`.
  `patches/sm120_attention.patch` is our diff against its `train.py` (see [`patches/README.md`](patches/README.md)).
- **pi coding agent** (`@earendil-works/pi-coding-agent`, earendil-works): the agent harness.
- **vLLM**: serving with runtime LoRA loading.
- **PyTorch**, Hugging Face **transformers**, **PEFT**, **accelerate**, **safetensors** and **tokenizers**, and
  **TRL** (activation offloading): training.
- **flash-linear-attention** and **Liger Kernel**: trainer kernels.
- **FlashInfer**: vLLM attention.
- **aiohttp** and **requests**: gateway, runner and supervisor.
- **pyarrow**: autoresearch data loading.
- **uv**, **Node.js**, **NumPy** and the Hugging Face **hf** CLI (`huggingface_hub`, model download).
- **IBM Plex** fonts (Google Fonts) in the dashboards.
- **Dr. GRPO** (Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective", 2025): the advantage
  `r - mean(r)` without division by the standard deviation.
- **DAPO** (Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", 2025): token-level loss
  averaging.
