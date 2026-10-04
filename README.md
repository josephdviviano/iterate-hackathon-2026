# arena3: full run snapshots

Each folder is the complete working directory of one arm of the arena3 CIFAR-100 autoresearch run, as it stood on disk at the time of the latest commit on this branch. Only `.git`, `.venv`, `data/` and caches are left out. Each commit on this branch is one snapshot, so earlier states are in the branch history.

Per arm:
- `submissions/arena/submission.py`: the arm's current model (its last kept experiment, or an in-flight candidate if one was running)
- `program.md`, `research.py`, `ar.py`, `lit.py`, `task.json`, `task.md`: the copy of the harness this arm ran with (source: branch `youssef-harness`)
- `results.tsv`: every experiment, kept and discarded
- `hypothesis.tsv`, `ideas.tsv`, `predictions.tsv`, `meta.tsv`, `world_model.md`, `history/`, `notebook/`, `drafts/`: hypothesis-loop state (hypothesis arms only)
- `runs/`: per-experiment logs, `results/arena/`: benchmark outputs, `.ar/`: runner state

`logs/` holds each arm's agent session logs. The kept-commit history of each arm is on branch `youssef-<arm>`, whose tip is the arm's last kept experiment.

## Status at this snapshot

| Arm | Experiments | Last kept (branch tip) | Fastest feasible experiment |
|---|---|---|---|
| hypothesis | 141 | E139 `f3d0daa` 4.27 s, acc 0.7539 | E133 `ee8a1f7` 4.19 s, acc 0.7550 (discard) |
| hypothesis-dr | 84 | E083 `69a0351` 4.57 s, acc 0.7556 | E083 `69a0351` 4.57 s, acc 0.7556 (keep) |
| baseline-r2 | 32 | E030 `f55a3f5` 4.65 s, acc 0.7568 | E030 `f55a3f5` 4.65 s, acc 0.7568 (keep) |
| hypothesis-unc | 66 | E061 `376f231` 5.02 s, acc 0.7534 | E052 `ce17ae7` 4.99 s, acc 0.7555 (discard) |
| baseline-r1 | 37 | E036 `2979b7a` 5.05 s, acc 0.7559 | E036 `2979b7a` 5.05 s, acc 0.7559 (keep) |
| hypothesis-lit | 72 | E067 `515a931` 4.85 s, acc 0.7544 | E068 `d6c1ddf` 4.82 s, acc 0.7547 (discard) |
