# autoresearch

This is an experiment to have the LLM do its own research.

## The task

The problem you are working on is defined by two files in this directory:

- `task.md`: what the problem is, which files you may edit, how experiments are measured, and the
  rules. Read it, and the files it points to, before anything else.
- `task.json`: the machine-readable version, used by `ar.py`: the paths you may edit, the
  command that runs one experiment, how the metrics are read from its output, and the objective.

You change only the editable paths listed there. You never change how experiments are run or
measured: not the run command, not the metrics, not the objective, not the files the task says
are off limits. `ar.py`, `task.json`, `task.md` and this file are not yours to edit.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `oct3`). The branch
   `autoresearch/<tag>` must not already exist. This is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
3. **Read the in-scope files**: `task.md` and everything it points to.
4. **Initialize**: `python ar.py init`. It runs the task's setup (if any) and creates an empty
   `results.tsv`.
5. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Running experiments

Every experiment goes through `ar.py`, which runs the task's command, reads the task's metrics
from the output, and judges the result against the current best by the task's objective:

- `git commit` your change (only the editable paths), then `python ar.py run --description "..."`.
  It starts the run and waits up to 9 minutes. If it says the run is still going, call
  `python ar.py wait` again. If your shell tool has a shorter timeout than that (Claude Code's
  Bash tool defaults to 2 minutes), raise it to 10 minutes for these commands.
- When the run is done, `python ar.py log --status keep|discard|crash` records it in
  `results.tsv` and tells you whether it improved on the current best.
- `python ar.py status` shows the current best and recent results. `python ar.py tail` shows the
  end of the run log.

`results.tsv` and `runs/` (the logs) are not committed; leave them untracked.

**The first run**: your very first run should always be to establish the baseline, so you will
run the editable files as they are.

**Simplicity criterion**: all else being equal, simpler is better. A small improvement that adds
ugly complexity is not worth it. Conversely, removing something and getting equal or better
results is a great outcome: that's a simplification win. When evaluating whether to keep a
change, weigh the complexity cost against the improvement magnitude. To keep an equal-result
simplification, pass `--force` to `log`.

**Crashes**: if a run crashes (out of memory, a bug, etc.), use your judgment: if it's something
dumb and easy to fix (e.g. a typo, a missing import), fix it, commit, and re-run. If the idea
itself is fundamentally broken, just skip it, log it as a crash, and move on. `ar.py tail` shows
the error.

**Timeout**: `ar.py` kills a run that exceeds the task's time limit; log it as a crash.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/oct3`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Change the editable files with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python ar.py run --description "<what you tried>"`
5. Read the result it prints. If the run crashed, `python ar.py tail` shows the error; attempt a
   fix if it's easy. If you can't get things to work after more than a few attempts, give up.
6. Record it: `python ar.py log --status keep|discard|crash`
7. If the result improved, you "advance" the branch, keeping the git commit
8. If it is equal or worse, you git reset back to where you started (`log` prints the command)

The idea is that you are a completely autonomous researcher trying things out. If they work,
keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel
like you're getting stuck in some way, you can rewind but you should probably do this very very
sparingly (if ever).

**NEVER STOP**: once the experiment loop has begun (after the initial setup), do NOT pause to ask
the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping
point?". The human might be asleep, or gone from a computer and expects you to continue working
*indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think
harder: read papers relevant to the problem, re-read the in-scope files for new angles, try
combining previous near-misses, try more radical changes. The loop runs until the human
interrupts you, period.
