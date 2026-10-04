# pi harness for the RLTL;DR autoresearch loop

pi = `@earendil-works/pi-coding-agent@1.0.0` on Node 24 (we installed both under `~/.local/bin`; the harness
runs the binary named by the config key `pi_bin`, default `pi` from `PATH`).
It drives Qwen3.8-27B-FP8 served by vLLM. Everything pi needs lives in the isolated config dir
`pi/agent` of this project; nothing is written to `~/.pi`. The harness copies `pi/agent` into each session dir
and starts pi with `PI_CODING_AGENT_DIR` pointing at the copy.

Paths below are relative to the project root (`$RLTLDR_ROOT`, the directory that contains `pi/` and `rltldr/`).

## Files

| Path | What |
|---|---|
| `agent/models.json` | Providers `rl` (gateway `http://127.0.0.1:8100/v1`, model `policy`) and `vllm` (direct `http://127.0.0.1:8000/v1`, model `qwen3.8-27b-fp8`). Both: `openai-completions`, Qwen-safe compat, thinking map, 131072 ctx / 32768 max out, Qwen thinking-mode sampling. No `return_token_ids` (the gateway adds it). |
| `agent/settings.json` | Defaults `rl/policy`, thinking `medium`, `defaultProjectTrust: never` (project `.pi/` resources are agent-writable and never wanted), compaction (reserve 32768, keep 20000), retry 10x from 2 s, `httpIdleTimeoutMs` 1800000, provider timeout 1800000, `cacheWarming: off`, telemetry off. |
| `agent/extensions/guard.ts` | Auto-loaded `tool_call` guard (see below). Also answers `project_trust` with "no". |
| `tools/capture_proxy.py` | Logging reverse proxy (test tool): records request bodies, streamed text/reasoning/tool calls, timings, and with `--token-ids` the prompt/completion token ids. |
| `tools/check_prefix.py` | Analyses a capture: per-call tokens/TTFT/decode speed, token-id completeness, reasoning replay, and exact prefix extension in token space between consecutive calls. |
| `tests/guard.test.mjs` | Guard unit tests (`node pi/tests/guard.test.mjs`; no LLM, no network). |
| `tests/e2e_run.sh` | Runs pi headlessly on a fresh copy of a toy repo against a live server. |
| `tests/toy_repo/` | Tiny "autoresearch-like" repo: `train.py` (with a gradient-averaging bug) and a fixed `prepare.py`. |
| `tests/prompts/` | Prompts of the validation runs: `task.txt` (bug fix), `task2.txt` (small optimization study), `guard.txt` (5 forbidden + 1 allowed command). |

## Environment

```bash
export PATH=$HOME/.local/bin:$PATH                         # where pi and node are installed
export RLTLDR_ROOT=/path/to/project                       # project root: the guard's default protected dir
export PI_CODING_AGENT_DIR=$RLTLDR_ROOT/pi/agent          # isolated config (models, settings, extensions, auth)
export PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0 # no network on startup, no update check, no telemetry
# The bash tool inherits pi's environment, so pin the GPU on the pi process itself. The RL harness gives the
# agent no GPU at all (CUDA_VISIBLE_DEVICES=""); experiments run through the trusted run wrapper instead.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=<GPU UUID>                    # UUID, not index (nvidia-smi -L)
# Optional guard settings (defaults shown):
export AR_GUARD_LOG=/path/to/run/guard_blocks.jsonl   # one JSON line per blocked call (default: none)
# AR_GUARD_WRITABLE=train.py  AR_GUARD_SCRATCH_DIRS=/tmp  AR_GUARD_PROTECTED_DIRS=$RLTLDR_ROOT (~/rltldr if unset)
# AR_GUARD_REPO=<git toplevel of cwd>  AR_GUARD_ALLOW_VENV_EXEC=0
```

## One autonomous attempt, headless

```bash
cd /path/to/autoresearch-repo          # pi's cwd = the repo (guard, system prompt <cwd>, session grouping)
pi --mode json --no-approve --no-context-files \
   --provider rl --model policy --thinking medium \
   --session-dir /path/to/run/sessions \
   -- "$(cat /path/to/run/prompt.md)" \
   < /dev/null > /path/to/run/events.jsonl 2> /path/to/run/pi.stderr
```

- The process exits by itself when the agent settles (exit 0; non-zero if the final stop reason is error/aborted).
- `--provider vllm --model qwen3.8-27b-fp8` talks to vLLM directly (bypasses the gateway); used for all tests below.
- `--no-approve` + `--no-context-files`: never load agent-writable `.pi/` resources or `AGENTS.md`/`CLAUDE.md` from the repo (the guard also denies project trust).
- `events.jsonl` has one line per stream delta (about 1 MB for 15 LLM calls). The session file (`--session-dir`, about 37 KB for the same run) holds the full transcript. Redirect events to `/dev/null` if the gateway captures calls.
- For a long-lived loop with steering (insight injection between groups) use `--mode rpc` with the same flags; see pi `docs/rpc.md`.

## Guard (`agent/extensions/guard.ts`)

Blocked calls become error tool results with the text `BLOCKED by experiment guard [<rule>]: <reason>`, so the model sees why and can adapt. Every block is also appended to the session as a custom entry `ar-guard` (not sent to the model) and, if `AR_GUARD_LOG` is set, to that file.

**bash, raw text** (quoted strings, heredocs and `python -c` included):
- any mention of `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES` or `CUDA_DEVICE_ORDER`;
- `nvidia-smi` setting changes (reset, power limit, compute mode, clocks, MIG, persistence, ECC, ...); plain queries are allowed;
- `kill`/`pkill`/`killall`/`fuser`/`os.kill`/`psutil`;
- pip/uv/conda/npm `install|add|remove|sync|lock|upgrade|venv`, `uv pip`, `uv run --with`, `uvx`/`npx`, `python -m pip`;
- `sudo`/`su`/`doas`/`pkexec`/`unshare`/`nsenter`/`chroot`;
- `sitecustomize`, `usercustomize`, `.pth`, `PYTHONPATH`, `PYTHONSTARTUP`, `LD_PRELOAD`, `LD_LIBRARY_PATH`;
- `.venv`, `site-packages`, `VIRTUAL_ENV`;
- writing, moving or running `prepare.py` (redirects, sed/cp/mv/rm/tee/..., `git checkout/restore ... prepare.py`, `open(...,'w')`); reading it is allowed;
- `vllm`, `sglang`, `rltldr`, `trainer`, policy/LoRA adapter names, `load_lora`, `update_weights`, `$HOME/models|envs` (the user's home spelled out as an absolute path), `PI_CODING_AGENT_DIR`, `.pi/`;
- any protected dir by path (`AR_GUARD_PROTECTED_DIRS`, default `$RLTLDR_ROOT`), so the project root is covered even when its name does not contain `rltldr`;
- local network (`localhost`, `127.x`, `0.0.0.0`, `/dev/tcp`, `:8000`, `:8100`, `:8199`, `port 8000`) and network clients (curl, wget, nc, ssh, `requests.get`, `urllib.request`, ...);
- background or detached execution (`&`, `nohup`, `setsid`, `disown`, `tmux`, `screen`, `at`, `crontab`);
- `AGENTS.md`, `CLAUDE.md`, `SYSTEM.md`;
- `git config --global/--system`, `--git-dir`, `--work-tree`.

**bash, parsed git**:
- Blocked: `push`, `pull`, `fetch`, `rebase`, `merge`, `cherry-pick`, `switch`, `worktree`, `remote`, `filter-branch`, `update-ref`, `symbolic-ref`, `submodule`, `clone`.
- Blocked: `branch -d/-D/-f/-m/-c`, `checkout -b/-B/--orphan/--detach/-`, checkout of another branch or a bare commit, `clean -x`, `reflog expire/delete`.
- Blocked: `reset` to another branch, a remote or tag, or a commit contained in another branch.
- Allowed: `reset --hard HEAD~N` / `HEAD^` / ancestor SHAs / SHAs reset away on this branch, `add`, `commit`, `diff`, `log`, `status`, `show`, `stash`, `checkout -- <file>`, `checkout <sha> -- <file>`.
- Nested `bash -c '...'` and `eval` are parsed recursively.

**edit / write**:
- Inside the repo, only `train.py` (`AR_GUARD_WRITABLE`) may be touched; outside the repo only `/tmp/**` (`AR_GUARD_SCRATCH_DIRS`).
- Paths are resolved through symlinks, so a `/tmp` symlink to `prepare.py` is caught.
- The written text is scanned for the GPU-env, startup-hook, venv, network, kill, infrastructure and install patterns, so `os.environ['CUDA_VISIBLE_DEVICES']` cannot be planted in `train.py`.
- Ordinary numbers such as `WARMUP = 8000` or `x[:8000]` are not blocked.

**read / grep / find / ls**: anything under a protected dir is blocked (`AR_GUARD_PROTECTED_DIRS`, default `$RLTLDR_ROOT`, which the harness passes to pi; `~/rltldr` if neither is set), except the experiment repo itself when it lies strictly inside a protected dir (production layout: repo `$RLTLDR_ROOT/autoresearch`; its siblings `tools/`, `data/`, `pi/` ... stay blocked). Paths are normalized like pi's tools (leading `@`, `~`, `file://`, unicode spaces) and resolved through symlinks.

**Absolute repo paths in bash**: before the keyword and protected-path rules run, absolute paths into the repo are rewritten to `./...`, so `cat ~/rltldr/autoresearch/train.py` (spelled out absolutely) is judged like `cat train.py` (the repo path contains the blocked keyword `rltldr`). A path that climbs out with `..` after the repo root is not rewritten and stays blocked. Other mentions of `rltldr` (e.g. the branch name `autoresearch/rltldr` in a git command) are still blocked.

This is one layer of defence in depth, not a sandbox: same-uid code launched by the agent (e.g. inside `train.py`) can still do anything. Keep the GPU watchdog, venv `sitecustomize`, read-only `prepare.py` and the trusted run wrapper.

## Validation results (2026-10-03, live vLLM, TP=2, single stream)

| Test | Result |
|---|---|
| Guard unit tests | all pass (`node pi/tests/guard.test.mjs`; block and allow cases, including the nested production layout -- repo inside a protected dir whose name is a blocked keyword --, a protected root with a neutral name, and `AR_GUARD_ALLOW_VENV_EXEC=1`) |
| Live guard run (`--provider vllm`, medium; `tests/prompts/guard.txt`: 5 forbidden + 1 allowed command) | 5/5 blocked with reasons shown to the model (`gpu-env`, `kill` on `pkill -f vllm`, `package-install`, `write-path` on prepare.py, `git-push`), 1/1 allowed. The model reported each message verbatim. 13.0 s, exit 0. |
| Project trust | repo with a planted `.pi/extensions/planted.ts`: not loaded with the production settings (`--no-approve`, guard answers "no", `defaultProjectTrust: never`). Control: with `defaultProjectTrust: always`, no guard and no `--no-approve`, the planted code ran; with `never` it did not |
| E2E bug-fix task (`tests/prompts/task.txt`: read, edit x2, read, bash run, bash git commit), medium | 11.8 s wall, exit 0, 7 LLM calls, 764 completion tokens (61–242 per call), prompt 1747 → 3023 tokens, TTFT 0.07–0.21 s, decode 71.3–71.5 tok/s, commit created |
| Same task, `--thinking high` (`reasoning_effort: xhigh`) | 9.4 s, 4 calls, 445 completion tokens |
| Optimization study (`tests/prompts/task2.txt`: fix + Adam + 4 LR runs + write notes + commit), high | 61.7 s wall, 15 calls, 4251 completion tokens (78–1229 per call), prompt 1955 → 7074, decode 70.9–71.7 tok/s; tools: bash 8, edit 4, read 2, write 2 (bash `timeout: 60` numeric argument used) |
| Reasoning replay | every request carried `reasoning` on all earlier assistant messages (14/14 in the last call) |
| Token-space prefix extension (`prompt_{k+1}` starts with `prompt_k + completion_k`, `return_token_ids` via proxy) | **exact in 33/33 consecutive pairs** across 5 sessions (6/6, 3/3, 9/9, 14/14, 1/1). Includes parallel tool calls, content text before tool calls, JSON-array (`edits`) and numeric (`timeout`) parameters. |
| Token ids complete (`sum(len(chunk.token_ids)) == usage.completion_tokens`) | 38/38 calls |
| 20 KB first prompt as positional arg (20344 bytes, 200 lines, unicode, quotes, `$VARS`, backticks) | one user message (`content: [{type:text}]`), byte-identical to the file in both the request and the session JSONL; 9211 prompt tokens; end sentinel echoed correctly |
| Long tool call: `sleep 420 && echo slept-ok` | 421.5 s, tool result `slept-ok`, run finished normally (no tool timeout exists) |
| Slow stream: mock server stalls 330 s mid-stream | completes with production settings (331 s, exit 0). Control with `httpIdleTimeoutMs=20000`: fails after 21 s with `terminated`. So the setting is what governs a stalled stream; the 300 s default would have killed this. |

## Gotchas

1. **stdin must be `/dev/null` (or closed)** in `-p`/`--mode json`; an open pipe makes pi wait for EOF.
   Piped stdin is prepended to the prompt with **no separator**: `printf 'A\nB\n' | pi -p "Go"` sends `"A\nBGo"`.
2. **One argv string is limited to 128 KiB** (Linux `MAX_ARG_STRLEN`): a larger positional prompt fails with `Argument list too long`. For bigger prompts use RPC `prompt` or `@file`.
   `@file` is wrapped as `<file name="/abs/path">\n...\n</file>\n<message>`.
   Several positional messages become several sequential user turns.
   Put `--` before a prompt that may start with `-`.
3. **Qwen template constraints**: `supportsDeveloperRole:false` is mandatory (otherwise the template raises "Unexpected message role"), and `reasoning_effort` must be xhigh/medium/low/none. `medium` adds no instruction text; `xhigh`/`low` add one sentence to the system block.
4. **Timeouts**:
   - The bash tool has no default timeout.
   - `httpIdleTimeoutMs` sets undici's headers and body idle timeouts.
   - `retry.provider.timeoutMs` is the OpenAI SDK request timeout, which only runs until response headers arrive.
   - All three are 30 min here, so a gateway may hold a request (e.g. during a weight swap) for up to 30 min before pi errors and retries.
   - Retries: 10, from 2 s, capped at 60 s per wait. That is about 6 min of waiting if the gateway is down; computed, not tested.
5. **Run-wrapper paths**: the guard blocks any bash text containing `rltldr`, `trainer`, `vllm`, `policy_adapter`, etc. (absolute paths into the repo excepted, see above). A trusted run wrapper called by the agent must therefore be invoked through a neutral path, e.g. `./run.sh` in the repo or a symlink `~/.local/bin/ar-run`, or be exposed as a custom pi tool. Branch names containing a keyword (`autoresearch/rltldr`) are blocked when the agent types them; `HEAD` works.
6. **Setup steps of `program.md` are blocked by design**: `git checkout -b autoresearch/<tag>` and `uv run prepare.py`. The harness must create the branch, prepare data and run the baseline before starting pi.
   `results.tsv` can only be appended through bash (`echo ... >> results.tsv`), or set `AR_GUARD_WRITABLE=train.py,results.tsv`.
7. **The system prompt contains `<cwd>`** (the repo path) and pi's `<docs>` paths. Use the same repo path for every rollout so the system prompt, and with it the prefix cache and prompt distribution, stays identical. Replace or extend it with `--system-prompt` or `--append-system-prompt` if needed.
8. **Prefix extension is empirical, not guaranteed.** It holds because Qwen3.8's template keeps `<think>` for all history (`preserve_thinking` default), pi replays `reasoning`, and the model's own formatting so far always matched the template's re-rendering. These cases would break it:
   - compaction, by design;
   - a truncated (`length`) completion;
   - reasoning or content with leading/trailing whitespace that `|trim` removes;
   - a single `\n` instead of `\n\n` before `<tool_call>`;
   - tool-argument JSON spaced differently from `tojson`.

   The gateway should check `prompt_{k+1}[:n] == prompt_k + completion_k` per call and start a new training segment when it fails.
9. **Clients hang up early**: pi closes the socket right after the final SSE event, often before the chunked terminator. A proxy or gateway must ignore a reset at `write_eof` and still record the call. `capture_proxy.py` hit this and was fixed; 5 records were lost before the fix.
10. **Usage**: `cacheRead` is always 0 because vLLM runs without `--enable-prompt-tokens-details`. vLLM's own prefix-cache hit rate was about 79% during these runs.
11. **Compaction calls bypass extension provider hooks** (see research): the gateway sees them as ordinary requests with the summarizer system prompt and no tools. Tag or exclude them from training.
12. `cacheWarming: "off"` and no `promptCache` in models.json, so pi never sends cache-refresh replay requests to the policy server.
13. Never `pkill -f <pattern>` from a shell whose own command line contains the pattern: it kills that shell. This happened during testing.

## Re-running the tests

From the project root:

```bash
export PATH=$HOME/.local/bin:$PATH
node pi/tests/guard.test.mjs                     # unit tests, no server needed
# Live tests need vLLM serving the model (serve.sh). Put the capture proxy in front of it (port 8199), make a copy
# of pi/agent whose `vllm` provider baseUrl is http://127.0.0.1:8199/v1, then:
PY=${RLTLDR_SERVE_PY:-$HOME/envs/serve/bin/python}   # any python with aiohttp (proxy) and tokenizers (analysis)
"$PY" pi/tools/capture_proxy.py --listen 127.0.0.1:8199 --upstream http://127.0.0.1:8000 --out calls.jsonl --token-ids &
pi/tests/e2e_run.sh pi/tests/toy_repo /tmp/pi-e2e/run1 <agent_dir_copy> vllm qwen3.8-27b-fp8 medium pi/tests/prompts/task.txt
"$PY" pi/tools/check_prefix.py calls.jsonl --tokenizer ~/models/Qwen3.8-27B-FP8
```
