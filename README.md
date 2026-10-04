# arena3: full run snapshots

Each folder is the complete working directory of one arm of the arena3 CIFAR-100 autoresearch run, as it stood on disk at push time. Only `.git`, `.venv`, `data/` and caches are left out.

Per arm:
- `submissions/arena/submission.py`: the current best model (the tip of the run's kept history)
- `program.md`, `research.py`, `ar.py`, `lit.py`, `task.json`, `task.md`: the copy of the harness this arm ran with (source: branch `youssef-harness`)
- `results.tsv`: every experiment, kept and discarded
- `hypothesis.tsv`, `ideas.tsv`, `predictions.tsv`, `meta.tsv`, `world_model.md`, `history/`, `notebook/`, `drafts/`: hypothesis-loop state (hypothesis arms only)
- `runs/`: per-experiment logs, `results/arena/`: benchmark outputs, `.ar/`: runner state

`logs/` holds each arm's agent session logs. The commit history of each arm's submission is on branches `youssef-<arm>`.
