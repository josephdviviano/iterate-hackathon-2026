# Exploration runbook

Research tooling for the `c100-speedrun` programme. The entry itself lives in
`submissions/team_segal/`; everything under `research/` and `tickets/` stays out of the
upstream pull request.

| Path | Purpose |
| --- | --- |
| `research/PROPOSAL.md` | Appraisal, prior-art mining, hypotheses H1–H6, probes P0–P4 |
| `research/DIRECTIVES.md` | Verbatim team-lead directives |
| `research/evidence/` | Exact evidence cited by the programme design and tasks |
| `research/sweep.py` | Run, inspect and collate declarative sweeps through the harness |
| `research/sweeps/*.toml` | Sweep files, one per probe |
| `research/env/` | Dev-stack setup script and frozen package list |
| `tickets/PROGRAMME/` | writing-tools programme: design, tasks, explorations, findings, tracker |

## 1. Two stacks, two kinds of evidence

| Stack | Where | Use |
| --- | --- | --- |
| Pinned: Python 3.12, torch 2.4.0+cu124 (`.venv`, `uv sync --frozen`) | A100 80GB PCIe, CPU tests | Timing, qualification, CPU contract tests |
| Dev: torch 2.7.1+cu128 (`.venv-blackwell`) | Local RTX PRO 6000 Blackwell GPUs | Accuracy probes only |

The pinned wheels ship no `sm_120` kernels, so they cannot run on the local GPUs
(`research/evidence/pinned-stack-blackwell.txt`). Local wall times are never timing
evidence: the GPUs are shared with other projects and run power-capped at 300 W.
Accuracy transfer between stacks is unverified until task T-004 (requirement R-007).

Set up both stacks from the repository root:

```bash
uv sync --frozen                         # pinned stack in .venv
research/env/setup-blackwell.sh          # dev stack in .venv-blackwell, plus CIFAR-100 in data/
```

Check the recipe on CPU with the pinned stack, then on real data with the dev stack:

```bash
CUDA_VISIBLE_DEVICES= uv run pytest -q research/tests
CUDA_VISIBLE_DEVICES=0 .venv-blackwell/bin/python -m benchmark.run --submission team_segal \
  --n 1 --no-accuracy-target
```

The default recipe is airbench94 adapted to 100 classes (64/256/256, 10 epochs, batch 1024).
It reaches about 69.5% single-view on the dev stack. Recipe parameters, their defaults and
their validation live in `submissions/team_segal/config.py`; unknown keys fail the run.

## 2. Run a sweep

A sweep file names seeds, devices and configurations. `[base]` applies to every
configuration; each `[[grid]]` block expands the cartesian product of its list values; each
`[[configs]]` entry adds one configuration. `widths = [128, 384, 512]` is a single value.
Write `widths = [[64, 256, 256], [128, 384, 512]]` to make it a grid axis.

```bash
PY=.venv-blackwell/bin/python
$PY research/sweep.py run research/sweeps/p1-frontier.toml --dry-run   # list pending configs
$PY research/sweep.py run research/sweeps/p1-frontier.toml             # run them
$PY research/sweep.py status research/sweeps/p1-frontier.toml
$PY research/sweep.py collate research/sweeps/p1-frontier.toml --frontier epochs=0.753
```

Each configuration runs once through the official harness over all seeds, with
`--no-accuracy-target`, on one device slot. Results go to
`results/sweeps/<name>/runs/<config_id>/`:

- `spec.json`: the configuration, seeds and submission source hash.
- `harness/`: the harness's own results directory.
- `harness.log`: the harness's output.
- `record.json`: written only when the run ends. Its status is `complete` or `failed`.

`collate` writes `table.csv` with one row per configuration: mean, sd and standard error
of single-view accuracy over successful trials, plus failures with their reason. A blank
parameter cell means the recipe default. `--frontier X=TARGET` interpolates, per group of
the other parameters, the value of `X` where the mean first reaches `TARGET`.

A configuration's identity hashes its parameters, its seeds and every file in the
submission folder. Editing the recipe therefore never reuses results produced by older
code. Collation reads only the configurations the sweep file currently expands to.

## 3. Interruptions and failures

- **Ctrl-C or SIGTERM** stops scheduling and interrupts running trials. The harness keeps
  its partial results, and no `record.json` is written for those runs. Rerun the same
  command to resume. Completed configurations are skipped; interrupted ones restart from
  scratch.
- **A failed configuration** (exception, out of memory, timeout) is recorded as `failed`
  and is skipped on resume. Inspect its `harness.log` or the harness `error.txt`, fix the
  cause, then rerun with `--retry-failed`.
- **Never drop failures.** Collation lists them, and a finding must interpret them.

## 4. Scaling across GPUs and agents

- **Slots.** Each `devices` entry gets `slots_per_device` slots. Slot locks live in
  `/tmp/c100-speedrun-slots` (override with `C100_SLOT_DIR`), so several sweeps, from one
  agent or many, share the GPUs on this host without oversubscribing them. Keep one slot
  per device while other projects use the GPUs. Two slots raise throughput for narrow nets
  only when the GPUs are otherwise idle.
- **Configuration locks.** A per-configuration lock stops two sweep processes running the
  same configuration.
- **Parallel code variants.** Use one git worktree per variant
  (`git worktree add ../c100-<variant> research`). Each worktree's submission source hash
  differs, so its results never mix with another's. Set `results_root` to an absolute
  shared path if one table should compare variants.
- **One task per agent.** An agent executing a programme task should:
  1. Run `writing-tools programme start --workspace tickets/PROGRAMME --task T-00x`.
  2. Write one sweep file per probe under `research/sweeps/`.
  3. Run the sweep, then collate it.
  4. Copy the decision-bearing table into `research/evidence/T-00x/`.
  5. Record a finding (`writing-tools programme finding`) and reconcile the task's
     acceptance with that evidence.

## 5. A100 timing (calibration and assurance)

Timing evidence must come from an A100 80GB PCIe running the repository's Dockerfile.
Rental requires team-lead approval (programme constraint). On that host:

```bash
docker build --platform linux/amd64 -t cifar100-speedrun .
docker run --rm -v "$PWD/data:/data" cifar100-speedrun python -m benchmark.data --root /data
docker run --rm --gpus '"device=0"' --cpus 4 --network none --ipc=host \
  -v "$PWD/data:/data:ro" -v "$PWD/results:/results" \
  cifar100-speedrun python -m benchmark.run --submission team_segal --n 5 \
  --no-accuracy-target --params '{"compile": true}' --data-root /data --results-root /results
```

Record SM clocks, power and temperature alongside the runs:

```bash
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv -lms 500
```

For timing runs, enable `compile`. It compiles and autotunes during the untimed `build`.
Evaluation always uses the eager module, so the 784-image final eval batch never compiles
inside the 5 s limit.

## 6. Programme handoff

```bash
writing-tools programme resume --workspace tickets/PROGRAMME --format prompt
writing-tools programme status --workspace tickets/PROGRAMME
```

Read the open feedback in `tickets/PROGRAMME/TRACKER.md` before selecting work, and select
eligible tasks by consequence, not by identifier. The design source is
`tickets/programme-design.yaml`. The exploration portfolio for the frontier is
`tickets/PROGRAMME/explorations/X-001-*.json`.
