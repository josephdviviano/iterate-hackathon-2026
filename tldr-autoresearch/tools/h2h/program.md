# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

The parts of the setup that need the human are already done:

- **Branch**: you are on the dedicated branch `autoresearch/h2h` (a fresh run, created from master). Stay on it.
- **Data**: `~/.cache/autoresearch/` contains the data shards and the tokenizer.

Do the rest yourself:

1. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
2. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
3. **Go**: kick off the experimentation right away; there is nobody to confirm with.

## Experimentation

Each experiment runs on a single dedicated GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it ONLY through the provided wrapper: `./run.sh "<short description>"` — it submits your current `train.py` to the experiment runner, blocks until the run is done (~6 minutes) and prints the output.

**GPU note**: experiments run on an RTX PRO 6000 Blackwell (sm_120, 96 GB). FA3 is unavailable; attention uses FlexAttention (sliding-window layers) / SDPA (full-causal layers) via `attention()` in `train.py`. Your own shell has no GPU: training only happens through `./run.sh`. `.venv/bin/python` has torch (CPU only in your shell), which is enough for quick checks such as parameter counts or tensor shapes.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric. Call `evaluate_bpb(model, tokenizer, batch_size)` exactly once, at the end of training; `model(idx)` without targets must return logits `[B, T, vocab]` and the model must be causal (the runner verifies this and computes the loss itself). Train on the "train" split only.
- Exceed the time budget: the runner measures the training time itself (from the end of the 10 warm-up steps to the start of the evaluation); a run that trains longer than the 5-minute budget is invalid.
- train.py runs in a sandbox on the experiment GPU: it cannot write files or use the network.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the run finishes, `./run.sh` prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. If the run fails, it prints `status: <crash|oom|fail_loss|timeout|invalid|refused>` followed by the reason and the end of the Python stack trace instead. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on the dedicated branch `autoresearch/h2h`.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `./run.sh "<short description>" > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context; do NOT run it in the background)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run failed. Run `tail -n 50 run.log` to read the status and the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). The runner kills a run after 11 minutes and reports `status: timeout`; treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~6 minutes then you can run approx 10/hour, for a total of about 80 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
