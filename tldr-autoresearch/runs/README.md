# Raw run records

The experiment records behind [RESULTS.md](../RESULTS.md), exported with `tools/export_runs.py` (the frozen-v5
ablation `frz/` is a snapshot of a run that was still going). Paths are rewritten: `$RLTLDR_ROOT` is the data root,
`~` the home directory, `GPU<n>-UUID` a GPU.

| folder | run | RESULTS.md |
|---|---|---|
| `rl_oct3/` | online RLTL;DR run: 40 experiments in 5 groups of 8, policy v0 → v5 | R1 |
| `h2h/base`, `h2h/v5` | head-to-head: base model vs v5 in two continuous pi sessions | R2 |
| `frz/x1`, `frz/x2` | frozen v5 in the training harness, with (x1) and without (x2) TL;DR insights | R3 |

What each folder holds:

- `ledger.jsonl`: one line per training run, written by the trusted runner (`tools/ar_run.py`): the trusted
  `val_bpb`, status, integrity flags, training metrics and timing. The only source of scores.
- `results.tsv`: the experiment history the agent saw (RL/frz harness: written by the harness). For h2h,
  `agent_results.tsv` is the agent's own log (untrusted).
- `metrics_driver.jsonl` / `metrics_trainer.jsonl`: per group: successes, insight counts; per policy update:
  GRPO/SFT tokens, optimizer steps, losses.
- `calib_ledger.jsonl`: baseline calibration runs. `insight_blocklist.json`: an insight removed from training
  because it blamed a harness bug (see docs/RETROSPECTIVE.md).
- `attempts/<id>/`: one experiment (one fresh pi session): `task.md` (the prompt), `rollout.json` (policy version,
  idea, result, confirmation, verdict, the TL;DR insight it produced and the insights that were in its context) and
  `session/*.jsonl` (the full pi transcript: messages, reasoning, tool calls and outputs).
- `runs/<run_id>/`: one 5-minute training run: the agent's `train.py`, its `diff.patch` against the parent,
  `run.log` (progress redraws collapsed) and `trusted_record.jsonl` (the signed evaluation record).
- h2h `session/*.jsonl`: the single continuous pi session of that arm.

Not included: LLM call records with token ids and logprobs (training data, tens of MB per run), pi event streams,
LoRA adapters and trainer checkpoints, repositories, virtualenvs and compile caches.
