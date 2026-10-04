# autoresearch: hypothesis-driven

This is an experiment to have the LLM do its own research, as a scientist rather than a
random search.

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
   `results.tsv`. Then `python research.py init`, which creates the research state below.
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

## The research method

You don't just try tweaks. The research keeps a **hierarchy of hypotheses** about what drives the
task's objective, a **queue of experiment ideas** that test them, and a **world model** that every
experiment reads and then updates. You use the time while an experiment runs to think: pre-register
what you expect, learn from earlier results, and prepare the next experiment.

```
hypothesis.tsv ──(Lvl1 ⊃ Lvl2 ⊃ Lvl3)──┐
                                       ▼
ideas.tsv (queue) ──pop──▶ [E000] ──▶ [E001] ──▶ [E002] ──▶ …
                              │          │          │
                              ▼          ▼          ▼
                                    results.tsv

∀ Exp:  in:  current world model, experiment idea
        out: result, updated world model

M(hypotheses, results)                   → new ideas        ("ideate")
M(hypotheses, results, search, review)   → new hypotheses   ("revise")
```

**Who does what.** You are the experimenter: you run the experiments, one after another, in this
session. The two meta-steps are not yours. `research.py` runs each as a **separate call** with
exactly the inputs above, in the background, whenever it is due: it builds the first hypothesis
tree at setup, refills the queue when it runs low (ideate), and rewrites the tree every 10 experiments,
after 5 without an improvement, or when no hypothesis is open (revise, which also gets a literature
search and a skeptical review, each from its own call). What those calls see as "results" is
`results.tsv`, your verdicts and prediction checks, and the *Outcome* and *Post-mortem* sections
of your notebook pages, so those are how you inform them. You do not add or edit hypotheses or
ideas yourself; you may only drop a queued idea that the results have already settled
(`python research.py idea drop I0xx --reason "..."`).

`research.py` keeps the state (all untracked by git, so a `git reset` never touches it):

- `hypothesis.tsv`: **Lvl1** is a broad claim about what drives the objective, **Lvl2** a mechanism
  under it, **Lvl3** a falsifiable prediction under that. Ids encode the tree (`H2`, `H2.1`,
  `H2.1.3`); evidence tokens (`E007+`, `E012-`) come from your verdicts.
- `ideas.tsv`: the queue. Each idea is one change to the editable files that tests one hypothesis.
- `world_model.md`: **yours**: your current understanding. Read it before every experiment; update
  it after.
- `notebook/E007.md`: one page per experiment: hypothesis chain, pre-registration, contingencies,
  outcome, post-mortem.
- `predictions.tsv`, `meta.tsv` (the meta-steps and their cost), `history/` (a world-model
  snapshot per experiment), `drafts/` (the next experiment).

`python research.py status` shows the current best, open hypotheses, the queue, the meta-steps,
and what to do next. `python research.py tree` shows the whole tree.

After setup, launch the baseline while the first tree is being built:
`python research.py launch --idea - --description baseline`, then `python ar.py wait`, then
`python research.py log --status keep --verdict -`. If the queue is still empty afterwards,
`python research.py meta wait` waits for the first tree and ideas.

### The uncertainty committee

An uncertainty committee runs beside you (`committee.py`; it never runs experiments or touches your
files). It is several independent world models, each a separate LLM call with its own lens, that
never see each other. Before an idea runs, every member forecasts what it will measure (an 80%
range per metric) and votes on its hypothesis; after the run, every forecast is scored. So the
committee's uncertainty is measured, not asserted:

- **Disagreement** on an idea (members' ranges that do not overlap, split votes) marks where the
  world is not understood yet: running that idea decides between world models.
- A **counterexample** is a result outside every member's range: something no world model
  predicted, so a mechanism is missing. Counterexamples trigger a literature request and a revise
  of the hypothesis tree.
- **Calibration** (`research.py unc`): how often the ranges hit, yours included, and whether
  disagreement actually predicts misses on this task.

It acts on its own: it drops ideas that all members forecast as clearly harmful (keeping one veto
in four as an audit), raises the most disputed idea, replaces members that keep missing, and feeds
its forecasts, cruxes and counterexamples to the ideate and revise calls. Literature digests it
requests appear in `status` (read them with `python research.py lit read D003`).

- Pre-register first: the committee's forecast for your experiment is shown only after you seal
  (`prereg`), so your prediction stays your own. Then compare: where you disagree with the
  members, say why in the in-flight notes.
- `python research.py unc` shows the full report. Counterexamples and disputed points belong in
  your world model under "Open questions and surprises" until an experiment settles them.

### One experiment

**Before** (keep this short; nothing is running):

1. `python research.py status`: do whatever it lists as due before the next launch.
2. `python research.py idea next` pops the top idea and prints its hypothesis chain. If the queue is
   empty, `python research.py meta wait`.
3. Re-read `world_model.md`. If the idea is already settled, drop it and pop again.
4. Make the change (apply the draft if you prepared one), and `git commit` it.
5. `python research.py launch --idea I0xx` (add `--prereg drafts/I0xx.prereg.md` if you drafted the
   pre-registration). This starts the experiment in the background through `ar.py`.

**While it runs** (the experiment is measured by `ar.py` exactly as in any other framework; you do
not touch the editable files or git):

1. **Pre-register** (unless sealed at launch), before looking at the run's output: in
   `notebook/E0xx.md`, one `- Prediction: <metric> [lo, hi]` line per metric you predict (metric
   names from `task.json`), what would support or refute the hypothesis, and the next step for
   each outcome. Then `python research.py prereg`.
2. **Learn from the last result**: finish updating `world_model.md` for the previous experiment,
   and write the *Post-mortem* of every failure or missed prediction `status` lists: what you
   predicted, what happened, the root cause, what the world model got wrong, the lesson.
3. **Prepare the next experiment**: draft the most likely next change and its pre-registration in
   `drafts/` (`drafts/I0xx.prereg.md`, with the `## Pre-registration` and `## Contingencies`
   sections exactly as in a notebook page), so the next launch is quick.
4. **Wait**: `python ar.py wait` (repeat while it says the run is still going).

**After** (keep this short too):

1. `python research.py log --status keep|discard|crash --verdict supports|refutes|inconclusive|-`.
   It records the run through `ar.py log` (which says whether it improved on the current best),
   then links it to the idea and hypothesis, checks your predictions, and starts a meta-step if
   one is due. Judge the verdict by the rules you pre-registered.
2. If not kept, run the `git reset` it prints.
3. Fill *Outcome* in the notebook page: the numbers, which contingency fired, and why.
4. Add the changelog line and current best to `world_model.md`, run
   `python research.py snapshot`, and go to the next experiment.

**NEVER STOP**: once the experiment loop has begun (after the initial setup), do NOT pause to ask
the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping
point?". The human might be asleep, or gone from a computer and expects you to continue working
*indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think
harder: read papers relevant to the problem, re-read the in-scope files for new angles, try
combining previous near-misses, try more radical changes. The loop runs until the human
interrupts you, period.
