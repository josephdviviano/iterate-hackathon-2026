# autoresearch (harness mode)

This is an experiment to have the LLM do its own research. In this setup an automated harness drives the
research loop: **every session runs exactly ONE experiment**, as described in the session prompt, and the
harness starts a fresh session for the next one.

## The setup (already done by the harness)

- You are on the experiment branch; the current best version of `train.py` is checked out.
- The data and tokenizer in `~/.cache/autoresearch/` are prepared; the baseline has been measured.
- `results.tsv` contains the history of all experiments so far. The harness maintains it — read it, don't edit it.

Files in scope: `README.md` (context), `prepare.py` (fixed constants, data prep, tokenizer, dataloader,
evaluation — do not modify), `train.py` (the file you modify: model, optimizer, training loop).

**GPU note**: experiments run on an RTX PRO 6000 Blackwell (sm_120, 96 GB). FA3 is unavailable; attention uses
FlexAttention (sliding-window layers) / SDPA (full-causal layers) via `attention()` in `train.py`. Your own
shell has no GPU: training only happens through `./run.sh`.

## Experimentation

Each experiment trains for a **fixed time budget of 5 minutes** (wall clock training time, excluding
startup/compilation) on a dedicated GPU. Launch it ONLY through the provided wrapper:
`./run.sh "<short description of the change>"` — it blocks for ~6 minutes and prints the result.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer,
  hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`, install packages or add dependencies.
- Modify the evaluation. At the end of training, call `evaluate_bpb(model, tokenizer, batch_size)` from
  `prepare.py` exactly once. `model(idx)` (no targets) must return logits `[B, T, vocab]` and the model must be
  causal: the harness verifies this and computes the loss itself. Train on the "train" split only.
- train.py runs in a sandbox: it cannot write files or use the network.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about
training time — it's always 5 minutes. The only constraint is that the code runs without crashing and finishes
within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow
up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity
is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's
a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement
magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb
improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**Noise**: run-to-run noise of val_bpb is a few thousandths. An apparent win is automatically re-run by the
harness, which keeps it only if the re-run confirms it.

## Output format

`./run.sh` prints, on success:

```
val_bpb:          0.997900
peak_vram_mb:     45060.2
```

or, if the run failed, `status: <crash|oom|fail_loss|timeout|invalid|refused>` followed by the reason / the end
of the Python stack trace.

## One experiment (one session)

1. Look at the git state, `train.py` and the history in `results.tsv`.
2. Pick ONE experimental idea and implement it by editing `train.py`.
3. `git commit -am "<short description>"`
4. Run it: `./run.sh "<short description>"` (do NOT run it in the background).
5. If it crashed because of something dumb and easy to fix (a typo, a missing import), fix it, commit and
   re-run (the session allows a few runs; only one successful run is recorded). If the idea itself is
   broken, stop.
6. End the session with a short final message: what you tried, the result, and what you learned.

Do not revert, reset, or log results yourself: the harness records the result, decides keep/discard (keep iff
val_bpb improves on the current best, confirmed by a re-run) and advances the branch. If you run out of ideas,
think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining
previous near-misses, try more radical architectural changes.
