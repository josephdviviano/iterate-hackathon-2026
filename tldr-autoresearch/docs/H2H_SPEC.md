# Head-to-head (h2h): base vs RLTL;DR-v5, continuous autoresearch sessions — build spec

This is the build spec the h2h components were written against (kept for reference). It was written for our
machine: "GPU 2/3" are the two nanochat GPUs, `$ROOT` is the project root (`$RLTLDR_ROOT`), and machine-specific
values (GPU UUIDs, ports) now come from `h2h_config.json` (template `h2h_config.example.json`, see
`rltldr/h2h_config.py`). Results are in `RESULTS.md` (R2).

Goal: two **continuous** autoresearch sessions (one long pi session each, upstream-Karpathy style: the agent edits
train.py, commits, runs, logs results.tsv, keeps/resets with git, NEVER STOPS), running at the same time:

| arm  | policy (served by the existing vLLM on GPUs 0,1)                    | nanochat GPU |
|------|---------------------------------------------------------------------|--------------|
| base | Qwen3.8-27B-FP8 + all-zero LoRA `h2h-base0` (== base model exactly)  | GPU 2 (minor 2) |
| v5   | Qwen3.8-27B-FP8 + RLTL;DR v5 LoRA `h2h-v5` (unmerged)                | GPU 3 (minor 3) |

Why adapters for both: merging v5 into bf16 keeps only ~38% of the update (85% of delta entries are below bf16
half-ULP), so v5 is served unmerged; the base arm gets a zero adapter so both arms run the identical LoRA kernel
path and decode at the same speed (fairness), while its outputs are exactly the base model's.

Shared config: `rltldr/h2h_config.py` (`load_h2h_config()`, `Arm` with all paths/ports). Use it; do not hardcode
paths that it defines. Program text the agents follow: `tools/h2h/program.md` (final; do not edit).
Shell entry point: `ctl_h2h.sh`; one-time setup: `tools/h2h/make_adapters.py`, then `tools/h2h/setup.py --create-start`.

## Environment facts
- Root `$ROOT` (`$RLTLDR_ROOT`). Envs: serve python `$RLTLDR_SERVE_PY` (default `~/envs/serve/bin/python`: aiohttp,
  requests, vllm client side), train python `$RLTLDR_TRAIN_PY` (default `~/envs/train/bin/python`: torch,
  safetensors, peft). Trusted stdlib-only python `$RLTLDR_TRUSTED_PY` (default `/usr/bin/python3`). Passwordless
  `sudo -n` works.
- vLLM 0.30.0 is RUNNING on 127.0.0.1:8000 (GPUs 0,1), FP8 base, served model name = see `GET /v1/models`
  (base id) with `--enable-lora --max-lora-rank 64 --max-loras 2`, runtime LoRA loading enabled
  (`POST /v1/load_lora_adapter {"lora_name","lora_path"}`, `POST /v1/unload_lora_adapter`),
  `--logprobs-mode processed_logprobs`, reasoning parser qwen3, tool parser qwen3_coder. **Never restart or kill
  vLLM.** Light test requests are fine.
- GPUs 2 and 3 are idle and belong to the h2h runners. Never touch GPUs 0,1.
- The old RLTL;DR system (rltldr/gateway.py, runner.py, driver.py, trainer.py, ctl.sh) is STOPPED and frozen. Its
  code is the reference implementation to copy from. Shared files (`tools/ar_run.py`, `tools/ar_bootstrap.py`,
  `tools/sandbox_lib.sh`, `pi/agent/extensions/guard.ts`) may be extended only backward-compatibly.
- pi 1.0.0: config `pi_bin` (default `pi` on PATH); docs in the installed package
  `@earendil-works/pi-coding-agent/docs/` (rpc.md, rpc-commands.md, json.md, compaction.md, sessions.md,
  settings.md). pi config template: `pi/agent/` (models.json provider `rl` ->
  `http://127.0.0.1:8100/v1`, model id `policy`, contextWindow 131072; settings.json with compaction on).
- Canon repo `canon.git` (bare). Baseline commit `c7666de` = upstream autoresearch + sm120 attention patch, with
  upstream's program.md. **The arms must never see anything from branch `autoresearch/oct3` or any other run
  data** (it contains the old run's best solutions).

## Layout (from h2h_config)
`data/h2h/<arm>/{repo,session,runner_ws,ledger.jsonl,runs/,cache/,calls.jsonl,events.jsonl,status.json}`,
sockets `run/h2h/<arm>/{gateway,runner}.sock` (only that dir is bound into the arm's sandbox), gateway trusted TCP
127.0.0.1:8110, runner trusted TCP 127.0.0.1:8210 (base) / 8220 (v5), adapters `data/h2h/adapters/{base0,v5}`.

## Components (one owner each; do not edit files owned by another component)

### C1 adapters + gateway — `tools/h2h/make_adapters.py`, `rltldr/h2h_gateway.py`
- make_adapters: copy `data/adapters/policy-v5` -> `data/h2h/adapters/v5` (frozen copy); write
  `data/h2h/adapters/base0`: same `adapter_config.json`, safetensors with exactly the same keys/shapes/dtypes as v5,
  all zeros. Idempotent.
- h2h_gateway (aiohttp, serve env), modelled on `rltldr/gateway.py` (copy what you need; do not import it — it
  loads the old config/state at import):
  - per arm an **untrusted** UnixSite on `arm.gateway_sock` (dir created 0755, socket chmod 0666): only
    `POST /v1/chat/completions` and `GET /v1/models` (returns one model `policy`). Whitelist request fields as in
    gateway.py (`SANDBOX_FIELDS`), force sampling (`SANDBOX_SAMPLING`), cap `max_tokens` (`SANDBOX_MAX_TOKENS`),
    rewrite `model` to `arm.served_model` regardless of what the client sends. Identical treatment for both arms.
  - trusted TCP 127.0.0.1:`gateway_port`: `GET /health`, `GET /control/state` (per arm: in-flight, calls,
    prompt/completion tokens, mean decode tok/s, last call ts, adapter loaded), `POST /control/pause|resume`
    (per-arm or all: paused arms queue requests).
  - streaming relay like gateway.py (drain upstream after client hang-up, stall detection, wait for a restarting
    vLLM up to 30 min), and on a 404/"adapter not found" re-register the adapter and retry once. At startup:
    load both adapters (`load_lora_adapter`; treat "already loaded" as success); unload stale `oct3-*` adapters.
  - record one JSON line per call to `arm.calls`: ts_start, ts_end, arm, served_model, status, finish_reason,
    n_prompt, n_completion, ttft_s, decode_tok_s, plus prompt/completion token ids and completion logprobs exactly
    as gateway.py records them (keep the same request extras gateway.py adds to get them). Append + flush.
  - no insights, no attempts, no policy versions: the arm->model mapping is fixed.
- Tests (write `tests/test_h2h_gateway.py` or a script; run them against the live vLLM):
  (a) through the base0 adapter vs plain base model, greedy (temperature 0 sent directly to vLLM, not via the
  gateway), same prompt: identical tokens and logprobs (|diff| < 1e-6); v5 differs from base on at least one of a
  few prompts. (b) via the gateway sockets, both arms concurrently: both succeed, sampling forced, model rewritten,
  calls recorded; decode tok/s of the two arms within 10% when run concurrently.

### C2 runner + trusted timing — `rltldr/h2h_runner.py`, changes in `tools/ar_run.py`, `tools/ar_bootstrap.py`
- h2h_runner `--arm NAME` (serve env), modelled on `rltldr/runner.py`, differences:
  - workspace `arm.runner_ws`: clone of canon `h2h/start` (create via `git clone --single-branch --branch h2h/start
    --no-tags`), detached at that commit, own `.venv` (`uv sync --frozen`); every run: `git checkout -f` start
    commit, `git clean -ffd`, write submitted train.py. ar_run pins the venv to `arm.gpu_uuid` (ensure_pin).
  - `tools/run_client.py` (C2 owns it): additionally send `head` = `git rev-parse HEAD` of the cwd repo (run
    git with `-c core.hooksPath=/dev/null -c core.fsmonitor=false`; empty on failure). Keep it stdlib-only.
  - no attempt gating: runs are accepted whenever no run is in flight for this arm (one at a time, queue with a
    lock); `AR_ATTEMPT_ID` unset, `AR_MAX_RUNS=0`. Run id label in the ledger: `desc` from the client; also pass
    the client-reported git head (`head`, untrusted, max 40 hex chars) into the ledger entry (`agent_head`) via an
    env var that ar_run reads (add backward-compatibly).
  - ar_run args: `--ledger arm.ledger --runs-dir arm.runs_dir --cache-dir arm.cache_dir --gpu arm.gpu_uuid
    --gpu-minor arm.gpu_minor --prepare-sha cfg.prepare_sha256 [--no-autotune] --full-summary`.
  - UnixSite `arm.runner_sock` (untrusted: `POST /run` only) + TCP `arm.runner_port` (trusted: `/control/state`,
    `/control/kill` to kill the in-flight run). Reap leftover processes **only on this arm's GPU** at start;
    `kill_job` must only kill this runner's own process tree plus processes on this arm's GPU (never the other
    arm's). State in `arm.dir/runner_state.json`.
- ar_run `--full-summary` (new flag, default off = old behaviour): on success print the upstream block
  (`---`, `val_bpb`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `mfu_percent`, `total_tokens_M`,
  `num_steps`, `num_params_M`, `depth`, aligned like upstream) using the trusted val_bpb, trusted training time when
  available and the parsed train.py summary for the rest (omit lines that are unavailable). Failures unchanged.
- Trusted training-time anchor (closes the time-budget hole): in ar_bootstrap register a global optimizer step
  post-hook (`torch.optim.optimizer.register_optimizer_step_post_hook`) and record, per optimizer object, the time
  its step count reaches the warm-up boundary, so that the anchor matches train.py's own exclusion of the first
  warm-up steps (read train.py at c7666de: it only counts time for `step > 10`; pick the step index so an honest
  baseline gives ~300 s); record it in the signed record (`t_step_anchor`, plus the existing `t_train2`). In ar_run:
  when the anchor exists, t_train = t_to_eval - t_step_anchor and `over_time_budget` iff t_train > budget + 10 s;
  else fall back to the existing t_train2 rule with grace 20 s (was 60); if neither exists keep the existing
  EVAL_TIME_ALLOWANCE fallback. Verify the hook fires for train.py's MuonAdamW (torch.optim.Optimizer subclass).
  Keep `tools/test_ar_run.py` passing; add tests.
- GPU validation (allowed: GPUs 2 and 3 only, sequentially or concurrently): one real baseline run of
  `h2h/start`'s train.py on each GPU through h2h_runner's code path (concurrently on both GPUs), proving: summary
  printed, ledger line written with agent_head, anchor-based t_train ≈ 300-305 s and not flagged, both GPUs used
  only by their own arm. Report the val_bpb/steps per GPU (this is also GPU calibration data).

### C3 sandbox — `tools/h2h_sandbox.sh`
`h2h_sandbox.sh <arm> <cmd> [args...]` like `tools/sandbox.sh` (private mount+pid+net ns, lo up, all GPUs masked,
read-only fs, private tmpfs /tmp /var/tmp /dev/shm, the invoking user's uid/gid (read before sudo), no_new_privs, uds_forward 8100->arm.gateway_sock,
8200->arm.runner_sock) plus **information hiding**: after lockdown, cover the invoking user's `$HOME` with a tmpfs and re-expose
only: `~/.local` (ro; pi, node, uv), `~/.cache/autoresearch` (ro; data/tokenizer), `~/.cache/uv` (ro, if present),
`$ROOT/tools` (ro), `arm.sock_dir` (ro; the sockets), `arm.repo` (rw), `arm.session_dir` (rw). Everything else
under `$HOME` (rltldr/data, canon.git, the other arm, logs, ~/.claude, ~/.claude.json, ~/.pi, ~/models,
~/envs, scratch, RETROSPECTIVE.md, ...) must be invisible. Technique: rbind `$HOME` to a staging dir under the
new /tmp before mounting the tmpfs, bind the needed paths back from the staging dir, then lazily unmount the
staging dir. The repo `.venv` stays read-only. Write `tests/test_h2h_sandbox.sh` proving: hidden paths absent;
other arm's sockets/repo absent; own repo writable, rest read-only; /dev/nvidia* unusable; 8100/8200 reachable and
forwarded to the right arm (use tiny test UDS servers on temporary socket paths via env overrides); no external
network; `pi --version` and `.venv/bin/python -c "import torch"` work; cannot regain root.

### C4 setup + supervisor — `tools/h2h/setup.py`, `rltldr/h2h_supervisor.py`
- setup.py (idempotent, refuses to overwrite an arm that already has runs unless `--force`):
  1. canon.git branch `h2h/start` ALREADY EXISTS (commit 8721165 on top of c7666de: program.md replaced by
     `tools/h2h/program.md`, executable `run.sh` calling `tools/run_client.py`). setup.py only verifies it
     (program.md identical to tools/h2h/program.md, parent c7666de); never rewrite it. (On a new machine,
     `setup.py --create-start` creates the branch once if it is missing.)
  2. Per arm: `git clone --single-branch --branch h2h/start --no-tags canon.git arm.repo`, rename the branch to
     `autoresearch/h2h`, remove the `origin` remote, `git gc --prune=now` so only objects reachable from the start
     commit exist; local git identity `autoresearch <autoresearch@localhost>`; `uv sync --frozen` (the `.venv`).
     Assert `git log --all` shows nothing beyond the start commit's history.
  3. Per arm: `arm.session_dir/agent` = copy of `pi/agent` without auth/models-store/sessions; empty
     `arm.session_dir/session/`.
  4. Runner workspaces are created by h2h_runner itself.
- h2h_supervisor `--arm NAME` (serve env): runs ONE continuous pi session for the arm in RPC mode inside
  `tools/h2h_sandbox.sh`:
  `pi --mode rpc --no-approve --no-context-files --provider rl --model policy --thinking <cfg.agent_thinking>
  --session-dir <session>/session` (+ `--continue` when a session file already exists), env as in
  `rltldr/driver.py:run_pi` (`env -i`, PI_CODING_AGENT_DIR=<session>/agent, PI_OFFLINE=1, ...,
  AR_GUARD_LOG=<session>/guard_blocks.jsonl), cwd = arm.repo, `AGENT_REPO` not needed (h2h_sandbox takes the arm).
  - First prompt (fresh session): `Hi! Have a look at program.md and let's kick off a new experiment. The setup
    steps that need the human are already done (see program.md). Do the rest of the setup and start the experiment
    loop.`
  - Whenever the agent settles (`agent_settled`): send `Continue the experiment loop. NEVER STOP: do not ask for
    confirmation, keep running experiments until you are interrupted.` (the upstream "human says continue").
    Back-off: if 3 consecutive settles happened without any `./run.sh` call in between, wait 5 min before the next
    nudge (log it); reset when a run happens.
  - pi exits or crashes: restart with `--continue` after 30 s (exponential back-off up to 10 min on repeated
    failures) and send `You were interrupted by a restart. Continue the experiment loop.`.
  - Watchdog: no pi event for 45 min while no run is in flight for the arm (runner `/control/state`) -> restart.
  - Write filtered events to `arm.events` (drop streaming deltas such as message_update text/thinking deltas; keep
    message_end with usage, tool_execution_start/end with tool name + truncated args/result (<= 2 KB),
    compaction_*, agent_start/end/settled, errors, retries) with ts; `arm.status` json (started_at, pid, n_prompts,
    n_nudges, n_restarts, n_compactions, n_runsh_calls, last_event_ts, context tokens if available, state).
  - SIGTERM: close pi stdin (orderly shutdown), wait 30 s, then kill the sandbox tree with sudo; exit 0.
  - Strict JSONL framing per rpc.md (split on LF only, binary reads).
- Tests: a fake OpenAI-compatible server (stream chunks + one tool call) on a temporary UDS, used through an
  env override of the gateway socket, proving: kickoff prompt sent, nudge after settle, back-off, restart with
  --continue, events/status written, clean SIGTERM. (Run without the real vLLM to avoid load.)

### C5 control + dashboard — `ctl_h2h.sh`, `tools/h2h_dashboard/{build.py,template.html,watch.sh}`
- `ctl_h2h.sh start|stop|restart|status|logs [component...]` in the style of `ctl.sh` (supervised restart loop,
  stop files, pid files under `run/h2h/`, logs under `logs/h2h/`). Components in start order: `gateway`,
  `runner-base`, `runner-v5`, `agent-base`, `agent-v5` (agents started together, back to back). vLLM stays under
  `./ctl.sh` (check it is up before starting; never start/stop it here). `status`: processes, GPUs, gateway state,
  per arm: runs, last run, best kept, nudges/restarts/compactions.
- Dashboard `build.py` (stdlib only, `/usr/bin/python3`): reads per arm `ledger.jsonl` (trusted), the agent's
  `repo/results.tsv` (untrusted, informational: keep/discard decisions), `status.json`, `calls.jsonl`
  aggregates; writes `dashboard_h2h/index.html` from `template.html` with inlined JSON. Never run git in the agent
  repos (untrusted config); only read files. Content: header with elapsed time and per-arm cards (runs, valid
  runs, crashes, runs/hour, current best = trusted val_bpb of the latest run the agent logged as `keep` (match by
  `agent_head` == results.tsv commit, fallback train_sha), best valid single run, tokens generated, decode tok/s,
  nudges, restarts, compactions); chart: val_bpb of every valid run vs hours since start (dots) + the agent's kept
  frontier as a step line, both arms (two fixed colours, legend + direct labels, light/dark tokens, hover tooltip);
  second chart: the same vs experiment index; per-arm tables (newest first: time, desc, trusted val_bpb, status,
  flags, agent's keep/discard); a GPU calibration note (the first baseline run of each arm). Use the look of
  `tools/dashboard/template.html` (IBM Plex fonts, tokens, light/dark). Works at 400 px width.
- `watch.sh`: like `tools/dashboard/watch.sh`, tails `logs/h2h/*.log`, events: runner run finished, supervisor
  nudges/restarts/compactions/errors, gateway errors, Traceback; rebuilds and prints one line per event.
- Test with synthetic data in a temp dir (env override of the data root).

## Global rules for all builders
- Do not start long-lived daemons and leave them running; kill anything you start. Do not touch vLLM, GPUs 0/1,
  `data/` of the old run, `canon.git` branches other than creating `h2h/start` (C4 only), or another component's
  files. Do not run `./ctl.sh` / `./ctl_h2h.sh start` against production paths.
- Code style: match the existing modules (docstring at top explaining the design, logging via `logging`, small
  functions, comments only where non-obvious).
- Report: files changed, tests run with results, anything left unverified, and open risks.
