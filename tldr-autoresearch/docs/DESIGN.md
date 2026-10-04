# RLTL;DR × autoresearch — online self-improvement of Qwen3.8-27B on Karpathy's autoresearch

The research agent is the **pi** coding-agent harness driven by **Qwen3.8-27B-FP8**. It runs Karpathy's
**autoresearch** loop: it edits a nanochat-style `train.py`, trains for 5 minutes, and is judged on `val_bpb`.
The policy is improved online with **RLTL;DR** ("Self-improvement by Internalizing Self-generated Feedback",
arXiv 2609.37633). The scheme:

- Every 8 sequential experiments form one rollout group.
- After each failed experiment, the policy writes a one-sentence TL;DR insight, which conditions later attempts.
- After each group, the policy is updated on GPU 3.
- The new policy is hot-swapped into the serving engine while autoresearch keeps running.

```
            ┌───────────────── GPU 0,1 ─────────────────┐
            │ vLLM 0.30  Qwen3.8-27B-FP8  TP=2           │   serve.sh
            │ base + LoRA adapters policy-v1, -v2, ...   │   :8000 (localhost only, dev endpoints)
            └────────────────────▲──────────────────────┘
                                 │ chat completions (+ return_token_ids, logprobs)
 ┌──────── sandbox (no GPU, loopback-only net, ro $HOME) ────────┐   ┌─────────── gateway :8100 ───────────┐
 │ pi 1.0 (one fresh session per experiment)                     │──▶│ routes "policy" -> current version   │
 │   read/edit train.py, git commit, ./run.sh "<idea>"           │   │ injects TL;DR insight messages       │
 └──────────────────────────────┬────────────────────────────────┘   │ records exact token ids + logprobs   │
                                │ ./run.sh (unix socket)              └──────────────────▲────────────────────┘
                                ▼                                                        │ new adapter
 ┌────────── runner (GPU 2) ──────────┐   ┌────────── driver ──────────┐   ┌──────── trainer (GPU 3) ────────┐
 │ pristine checkout + submitted      │   │ attempts, keep/discard,    │──▶│ LoRA r=32 on bf16(dequant FP8)  │
 │ train.py -> 5 min nanochat run ->  │──▶│ confirm re-runs, rewards,  │   │ Dr.GRPO + 0.5 * insight-SFT     │
 │ ledger.jsonl (val_bpb, diff, ...)  │   │ insight generation, groups │   │ publishes policy-v{N}           │
 └────────────────────────────────────┘   └────────────────────────────┘   └─────────────────────────────────┘
```

## Quick start

Setup (environments, models, the autoresearch repo, `config.json`) is in `docs/SETUP.md`.

```bash
cd tldr-autoresearch      # or set RLTLDR_ROOT to the project root
./ctl.sh start            # vllm -> gateway -> runner -> trainer -> driver (each supervised, auto-restart)
./ctl.sh status           # processes, GPUs, current policy, group progress, latest metrics
./ctl.sh logs driver      # or: vllm | gateway | runner | trainer
./ctl.sh stop             # driver first (finishes the current attempt), then the rest
./ctl.sh stop driver      # pause research only; serving/trainer keep running
```

Settings live in `rltldr/config.py`. Override any of them with `$RLTLDR_ROOT/config.json` (same keys; start from
`config.example.json`), then run `./ctl.sh restart driver trainer`. Shell scripts read resolved values with
`python -m rltldr.config get <key>`.

## Components

| | file | env | role |
|---|---|---|---|
| vLLM | `serve.sh` | serve env, `$RLTLDR_SERVE_PY` (vllm 0.30.0, torch 2.13 cu130) | Qwen3.8-27B-FP8, TP=2, 262k ctx, qwen3 reasoning/tool parsers, LoRA (versioned names, never in-place), `--logprobs-mode processed_logprobs` |
| gateway | `rltldr/gateway.py` | serve | OpenAI proxy. Routes `policy` → `qwen3.8-27b-fp8` (v0) or `policy-vN`; new versions take effect at the next attempt. Injects insights. Writes one record per LLM call to `data/attempts/<id>/calls.jsonl`. Re-registers adapters after a vLLM restart. Trusted TCP :8100 plus a sandbox socket that exposes chat only. |
| runner | `rltldr/runner.py` + `tools/ar_run.py` | serve + autoresearch venv | The only process that trains on GPU 2. Uses a pristine checkout `runner_ws/` (its own venv and prepare.py) plus the agent's submitted train.py. Writes one ledger line per run (`data/ledger.jsonl`). Caps runs per attempt. |
| driver | `rltldr/driver.py` | serve | One pi session per attempt, launched through `tools/sandbox.sh`. Applies the keep/discard rule with confirmation re-runs. Computes the reward, generates insights (`rltldr/insight.py`), and closes groups of 8 (`data/groups/gNNNN/READY`). |
| trainer | `rltldr/trainer.py` + `rltldr/model_utils.py` | train env, `$RLTLDR_TRAIN_PY` | Per group: one update phase, then `export_adapter`, then `/control/policy`. Checkpoint in `data/trainer_ckpt`. |
| pi | `pi/agent/{models.json,settings.json,extensions/guard.ts}` | node 24 | `@earendil-works/pi-coding-agent@1.0.0`, provider `rl` → gateway. A guard extension blocks off-limits tool calls. |

## The method as implemented (paper → this setup)

- **Task g and rollout.** g is the one-experiment autoresearch prompt: rules, current best, `results.tsv` history.
  One rollout is one pi session: edit → commit → `./run.sh` → summary, with at most 3 runs for crash fixes.
  The verifier is the runner ledger. Success means the run is ok, has no integrity flags, and has
  `val_bpb < best`.
- **Noise handling (deviation).** Run-to-run std was 0.0039 bpb with Inductor autotuning (as large as typical
  gains) and about 0.0001 without it. An apparent win is re-run outside the rollout with a fresh compile; the keep
  decision uses only the re-run(s), and the new best is their mean. This removes most of the winner's-curse bias;
  a residual of about −2e-4 remains for keeps near the margin, and agent code can re-enable autotuning, which
  raises the noise (see `docs/RETROSPECTIVE.md` and `RESULTS.md`). The baseline is the mean of its calibration
  runs.
- **Group (§3.2).** K = 8 sequential attempts form one GRPO group. Groups follow each other continuously, so
  autoresearch never stops.
- **Insights (A.1 / A.2).** After a failed attempt, the current policy writes JSON in a fresh chat using the
  judge prompt adapted to autoresearch. It sees the rollout without think traces, the diff and the harness
  output, with a 4096-token thinking budget. The judge returns summary / feedback / wrong_step_id /
  corrected_step / hint; only the hint is kept. Attempt k sees all insights of its group as separate user
  messages after the task, but only while the running success rate is ≤ 50% (at most 16 insights). The
  format is `==> A previous Attempt {i} on this same task FAILED the verifier. <==\nHint on what went wrong: {hint}`.
  Insights reset every group.
- **Update (§3.3 / A.3):** L = L_GRPO + 0.5 · L_SFT.
  - Advantages are Dr.GRPO: r − mean(r), with no std division.
  - Losses are averaged per token, as in DAPO. There is no KL term.
  - Zero-advantage groups are skipped.
  - Positive-ratio filtering keeps at least 75% positive rollouts.
  - PPO clip ε = 0.2, 2 epochs × 2 minibatches, gradient clip 1.0.
  - L_SFT is the NLL of the insight-sentence tokens given the task and earlier insights, i.e. the flipped mask.
    It is applied even when the GRPO term is empty, so all-fail groups do online SFTL;DR.
  - **Asynchrony (deviation).** Rollouts can come from an older version on an FP8 server, so the update uses
    decoupled PPO. The ratio is taken against the pre-update weights, times a truncated IS weight
    min(π_prox/π_behaviour, 2). The behaviour log-probs are vLLM's processed log-probs.
  - **LoRA (deviation).** The paper trains all 27B parameters on 8×B200. Here LoRA r=32 on all text linears,
    lr 2e-5, on top of the bf16 dequantisation of the served FP8 weights, fits one 96 GB GPU.
  - **Training data.** The exact token ids served by vLLM. Consecutive calls of an attempt merge into one
    sequence because each prompt extends the previous prompt plus completion exactly (verified).
- **Weight sync.** `policy-v{N}` is loaded into vLLM under a new name in about 0.2 s, without pausing. The
  gateway switches at the next attempt boundary. A full FP8 re-quantised weight sync would erase small RL
  updates; this was measured.
- **Deconfounded metrics (App. D).** `data/metrics_driver.jsonl` records success with and without insights, the
  first-attempt success per group (a Pass@1 proxy) and the insight advantage. `data/metrics_trainer.jsonl`
  records clip fraction, first-minibatch clip fraction, behaviour mismatch, losses and gradient norms.

## Isolation and integrity (the reward must not be hackable)

The policy is optimized against this harness, so every path from agent code to the reward is closed or checked:

- **Agent sandbox** (`tools/sandbox.sh`, sudo unshare): private mount/PID/network namespaces. The whole filesystem is
  read-only (incl. system Python installs, `$HOME`, harness code, ledger) except the agent's throwaway repo clone and its pi
  session dir; private `/tmp`, `/var/tmp`, `/dev/shm`; all GPUs masked; network = loopback with forwards to the
  gateway (chat only, sampling forced to the policy's settings) and the runner (`/run` only); `no_new_privs`.
- **Training sandbox** (`tools/train_sandbox.sh`): the agent's `train.py` runs with only `/dev/nvidia2`, no network,
  read-only filesystem except its compile cache and trusted-record dir.
- **Trusted evaluation** (`tools/ar_bootstrap.py`, run before `train.py`): val_bpb is computed by harness code from
  the model's logits (prepare's exact bpb accounting), not from train.py's stdout or loss code; a causality probe
  (perturb future tokens → earlier logits must not change), detection of validation-data access during training
  (loader, batch generator, file opens via an audit hook), single-evaluation check, and time-to-eval measured from
  process start. Results are HMAC-signed with a one-time key deleted before train.py starts. Violations →
  `status: invalid` (verified against fake-print, val-split training, non-causal model, loss scaling, double-eval
  and fake-summary variants).
- **Canonical repo**: the branch lives in driver-only `canon.git`; the agent gets a fresh clone every attempt and the
  harness never runs git inside the agent's tree (agent-written git config/hooks never execute). Kept commits are
  made in `canon_ws` from the runner's verified train.py snapshot. `results.tsv` is harness-owned
  (`data/results.tsv`).
- **One result per attempt**: the runner refuses further runs once an attempt has an ok run (no re-run fishing);
  AST-identical resubmissions of measured code are flagged `reeval`/`empty_diff` and cannot win.
- **Noise-robust keep rule**: an apparent win is re-run `confirm_runs` times with a fresh compile cache; keep iff
  the re-run mean beats the best by `keep_margin`; the new best is estimated from those unselected re-runs only.
  Inductor runtime autotuning is disabled in training runs (`no_autotune`) to cut kernel-selection noise.
- **Infrastructure failures** (LLM server down/stalled, runner unreachable) void and retry the attempt instead of
  scoring 0; an orphaned agent sandbox from a killed driver is reaped at restart.

Residual risk: arbitrary code inside the training process could still deliberately subvert in-process checks
(e.g. by introspecting the bootstrap); the sandbox bounds what it can reach. Large jumps in val_bpb are worth a
human look (`./ctl.sh status`, `data/results.tsv`).

## Data layout (`data/`)

```
ledger.jsonl                 one line per training run (runner)        calib_ledger.jsonl  baseline noise runs
results.tsv                  authoritative experiment history (copied into each agent clone)
runs/<run_id>/               train.py snapshot, diff.patch, run.log
attempts/<attempt_id>/       task.md, calls.jsonl (gateway), rollout.json, events.jsonl, session/, guard_blocks.jsonl
groups/gNNNN/                group.json, READY (driver), DONE (trainer result)
adapters/policy-vN/          exported LoRA adapters        trainer_ckpt/  LoRA fp32 + AdamW state
driver_state.json  trainer_state.json  gateway_state.json  metrics_driver.jsonl  metrics_trainer.jsonl
```

## Follow-up experiments on the same harness

- **Head-to-head (h2h):** base model vs policy v5 as two continuous upstream-style sessions that share vLLM, one
  nanochat GPU each. Spec: `docs/H2H_SPEC.md`. Code: `rltldr/h2h_*.py`, `tools/h2h/`, `tools/h2h_sandbox.sh`.
  Control: `ctl_h2h.sh`.
- **Frozen-v5 ablation (frz):** two independent instances of this harness. Policy v5 is frozen (no trainer), with
  insights on (arm x1) and off (arm x2). Each instance gets its own config file through `RLTLDR_CONFIG` (fields
  `sandbox_arm`, `runner_isolate`, `insights_enabled`). Control: `ctl_frz.sh`.

Results of all three: `RESULTS.md`.
