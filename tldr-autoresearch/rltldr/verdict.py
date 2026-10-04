"""Shared semantics of ar_run.py ledger flags: which flags invalidate a measurement, and why (plain language
for the judge prompt and the experiment history)."""

# Integrity failures: the measurement is not trustworthy (any run, including confirmation re-runs).
FATAL_FLAGS = {
    "prepare_py_modified", "over_time_budget", "too_few_steps", "wall_clock_excessive", "nonzero_exit",
    "interrupted", "gpu_violation",
    # trusted evaluation (tools/ar_bootstrap.py)
    "no_trusted_eval", "multiple_evals", "val_leak", "non_causal", "eval_unverifiable",
}
# For the agent's own run: re-measuring already-known code is not a new experiment.
AGENT_REJECT_FLAGS = FATAL_FLAGS | {"empty_diff", "reeval"}

FLAG_REASONS = {
    "prepare_py_modified": "prepare.py (the fixed evaluation) was modified",
    "over_time_budget": "training ran longer than the fixed 5-minute budget",
    "too_few_steps": "the run made too few optimizer steps to be a valid training run",
    "wall_clock_excessive": "the run took far longer than allowed",
    "nonzero_exit": "train.py exited with an error after printing results",
    "interrupted": "the run was interrupted",
    "gpu_violation": "the run tried to use a GPU other than the one assigned to it",
    "no_trusted_eval": "train.py did not evaluate the model with prepare.evaluate_bpb exactly once at the end, "
                       "so no valid val_bpb was measured",
    "multiple_evals": "prepare.evaluate_bpb was called more than once (selecting on validation data)",
    "val_leak": "validation data was accessed during training (data leak)",
    "non_causal": "the model is not causal: outputs depend on future tokens (information leak)",
    "eval_unverifiable": "model(idx) did not return logits [B, T, vocab], so the result could not be verified",
    "empty_diff": "train.py was not meaningfully changed relative to the current best (only comments or "
                  "formatting differ), so it is not a new experiment",
    "reeval": "this train.py (up to comments/formatting) had already been measured before, so it is not a new "
              "experiment",
}


def rejecting_flags(entry, reject=AGENT_REJECT_FLAGS) -> list:
    return sorted(set((entry or {}).get("flags") or []) & reject)


def valid_run(entry, reject=FATAL_FLAGS) -> bool:
    return bool(entry) and entry.get("status") == "ok" and entry.get("val_bpb") is not None \
        and not rejecting_flags(entry, reject)
