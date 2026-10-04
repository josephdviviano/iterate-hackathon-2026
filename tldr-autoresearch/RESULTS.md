# Results

Every number in README.md comes from this file.

All numbers here are derived from the run data with one command:

```bash
RLTLDR_ROOT=<project root> python3 tools/export_results.py --until 2026-10-04T12:00:00Z
```

The script uses only the standard library and reads `$RLTLDR_ROOT/data` without changing it. It writes the tables in
[`results/`](results/). `results/summary.tsv` holds each headline number as a `section`/`metric` row; the key shown
next to each number below is that row. `--until` fixes the snapshot of R3, which was still running.

**Common to all results**

| | |
|---|---|
| Metric | `val_bpb`: validation bits per byte after a fixed 5-minute training budget. **Lower is better.** |
| How it is measured | Harness code (`tools/ar_bootstrap.py`) runs autoresearch's own pristine `prepare.evaluate_bpb` on the logits of the agent's model. The number that `train.py` prints is never used. |
| Data split | Training uses autoresearch's training shards (24 shards were downloaded). Evaluation uses the pinned validation shard `shard_06542`, with `EVAL_TOKENS` = 40 × 524,288 tokens. This is upstream's split. There is no separate test split. If training reads the validation data, the run is detected and voided. |
| Hardware | 4× RTX PRO 6000 Blackwell (96 GB, sm_120). GPUs 0–1 serve the policy with vLLM. The 5-minute nanochat runs train on GPU 2, and the second h2h/frz arm trains on GPU 3. |
| Baseline | The unmodified autoresearch `train.py`, plus an sm_120 attention patch (FA3 → FlexAttention/SDPA). Result: **1.079941**, the mean of 4 runs (sd 0.000102). This baseline gets only 505–506 optimizer steps in 5 minutes on this GPU. Keys: `rl_run/baseline_*`. |
| Commit | `25a8965` (branch `tldr-autoresearch`): the commit that added `results/` and `tools/export_results.py`. Re-running the command above from that tree on our run data reproduces `results/` byte for byte. |

---

## R1. Online RLTL;DR run: Qwen3.8-27B-FP8 policy, updated every 8 experiments

- **Command:** `./ctl.sh start` (with `config.json` from `config.example.json`, and the baseline from
  `docs/SETUP.md`).
- **When:** 2026-10-03, 16:13 to 23:38 UTC.
- **Number of runs:** 1 run of the whole system. It made 40 experiments in 5 groups of 8, which took 51 training
  runs including the confirmation re-runs.
- **Policy versions:** v0 (the base model) → v5.
- **Tables:** `results/rl_run_experiments.tsv` and `results/rl_run_insights.tsv`.

| result | value | key (`rl_run/…`) |
|---|---|---|
| Best val_bpb (a confirmation re-run, so it is not selected) | **1.007697** | `final_best_val_bpb` |
| Change vs the baseline 1.079941 | −0.072245 (−6.7%) | `gain_total` |
| When the best was reached | Experiment 35 of 40. This was 46 training runs and 6.54 h after the first agent session started. | `experiments_to_final_best`, `gpu_training_runs_to_final_best`, `hours_to_final_best` |
| Kept experiments | 9 of 40 | `n_keeps` |
| Apparent wins reversed by the confirmation re-run | 2 of 11 | `apparent_wins_reversed_by_confirmation` |
| Gain from the two "halve batch size" keeps | 0.058305, which is 81% of the total gain | `gain_from_batch_halving` |
| Keeps by policy version | v0 3/9, v1 2/8, v2 1/8, v3 1/8, v4 2/7. Fisher test, v0 vs later versions: p = 0.39. | `keeps_by_policy_version`, `fisher_p_v0_vs_later` |
| Keeps with insights in context vs without | 8/33 vs 1/7 (p = 1.00). This is not randomised: insights are shown only while a group's success rate is ≤ 50%. | `keeps_with_insights`, `keeps_without_insights` |
| Policy updates | 5 (v1–v5), with 16 optimizer steps in total. They trained on 44,134 GRPO tokens and 3,672 insight-SFT tokens, and took 110 s each on average. | `policy_updates`, `optimizer_steps_total`, `grpo_tokens_total`, `sft_tokens_total`, `update_seconds_mean` |
| TL;DR insights written | 31. They average 27.9 words, and 15 contain numbers. | `insights_written`, `insight_mean_words`, `insights_with_digits` |
| Agent runs that crashed or were invalid | 0 of 40 | `agent_runs_crashed_or_invalid` |

The kept changes, in order (see `rl_run_experiments.tsv`):
1. Batch 2^19 → 2^18.
2. Batch 2^18 → 2^17.
3. LR warmdown 0.5 → 0.7.
4. Muon LR 0.04 → 0.03.
5. Window pattern SSSL → SSLL.
6. `max-autotune` compile.
7. Depth 9 at width 512.
8. Sliding window 1024 → 512.
9. Sliding window 512 → 256.

The run was stopped after group 4 to free the GPUs for R2. A 41st attempt was still in progress at that point; its one
run is excluded (`unfinished_attempt_runs`).

## R2. Head-to-head ("h2h"): base vs RLTL;DR-v5 as continuous upstream-style sessions

**Setup.**
- Two continuous pi sessions run at the same time, using upstream's "never stop" loop and upstream's keep rule.
- The harness adds trusted evaluation and sandboxes.
- Both arms share one vLLM server.
- Both arms start from the same baseline `train.py` and the upstream-style `program.md` (`tools/h2h/program.md`).
  Neither arm saw any data from R1.

**Arms.**
- **base** runs the base model, on GPU 2. It is served through an all-zero LoRA, so that both arms use the same
  kernels and decode at the same speed.
- **v5** runs the final RL policy, served as an unmerged LoRA, on GPU 3.

**Run details.**
- **Command:** `python3 tools/h2h/make_adapters.py && python3 tools/h2h/setup.py --create-start && ./ctl_h2h.sh start`
- **When:** 2026-10-04, 02:01 to 10:24 UTC. The run was stopped by the operator.
- **Number of runs:** 1 session per arm.
- **Table:** `results/h2h_runs.tsv`. Run 1 of each arm is the unmodified baseline. The keep/discard labels come from
  the agent's own `results.tsv`, which the harness does not trust.

| | base | v5 | key (`h2h_<arm>/…`) |
|---|---|---|---|
| Training runs, not counting the baseline run | 68 | 74 | `runs` |
| Valid runs / crashes / interrupted at stop | 65 / 2 / 1 | 68 / 5 / 1 | `runs` (note) |
| Runs per hour | 8.11 | 8.83 | `runs_per_hour` |
| Keeps logged by the agent | 15 | 9 | `agent_keeps` |
| Best single valid run | **1.012084** | **1.012210** | `best_single_run_val_bpb` |
| The agent's last kept run | 1.012371 | 1.012210 | `final_kept_val_bpb` |
| Best single run within R1's first 46 runs | 1.012144 | 1.013216 | `best_single_run_within_rl_runs_to_best` |
| Best single run within 6.54 h, R1's time to its best | 1.012144 | 1.013216 | `best_single_run_within_rl_time_to_best` |
| Unmodified baseline on the arm's GPU before the start (2 runs) | 1.080038, 1.079978 | 1.080244, 1.080157 | `gpu_calibration_baselines` |

**Result: a tie.** The two best single runs differ by 0.000126. That is well below the per-run noise σ ≈ 0.00038
(see Noise below). The base arm re-ran its best commit and got 1.013000 (`h2h_runs.tsv`, base run 61).

## R3. Frozen-v5 ablation ("frz"): insights on vs off in the RL harness — snapshot of a running experiment

**Setup.**
- Policy v5 is frozen: there are no updates.
- Each arm is an independent instance of the R1 harness. It has a fresh session per experiment, a history table, the
  keep rule with a confirmation re-run, and groups of 8.
- Both arms start from R1's start commit and baseline.

**Arms.**
- **x1:** insights on, GPU 2.
- **x2:** insights off, GPU 3.

**Run details.**
- **Commands:**
  ```bash
  FRZ_GPU_x1=<uuid>:<minor> FRZ_GPU_x2=<uuid>:<minor> ./ctl_frz.sh init --seed <start commit>
  ./ctl_frz.sh start gateway-x1 runner-x1 gateway-x2 runner-x2
  ./ctl_frz.sh baseline x1 && ./ctl_frz.sh baseline x2
  ./ctl_frz.sh start driver-x1 driver-x2
  ```
- **When:** started 2026-10-04 at 10:28 UTC. The experiment is **still running**.
- **Snapshot:** experiments finished by 2026-10-04 12:00 UTC (`frz_<arm>/snapshot_utc`).
- **Number of runs:** 1 group of 8 experiments per arm so far.
- **Baseline:** 1.079925. This is the mean of 6 runs: R1's 4 runs plus one fresh run on each arm's GPU.
- **Table:** `results/frz_runs.tsv`.

| after 8 experiments (group 0) | x1: v5 + insights | x2: v5, no insights | reference: R1 group 0 (v0 + insights) | key |
|---|---|---|---|---|
| Keeps | 3 | 2 | 3 | `frz_<arm>/n_keeps` |
| Best val_bpb | 1.048194 | 1.021257 | 1.019860 | `frz_matched/*_best_after_n` |
| Runs flagged by the integrity checks | 0 | 1 (`over_time_budget`), plus 1 refused re-run request | 0 | `frz_<arm>/integrity_flagged_runs`, `refused_rerun_requests` |

**No conclusion yet.** With one group per arm, a single early choice decides the result:
- x1 kept "depth 8 → 5" at experiment 2 and then tuned around the smaller model.
- x2 found the batch halvings.

**The integrity checks worked on a policy-written change.** In x2, a background-prefetch thread ran past the
5-minute training budget, and the run was flagged invalid. The agent then asked to run the experiment again, and the
runner refused.

---

## Noise and caveats (read these before quoting any number)

**Noise: per-run σ**

| source | σ | n | key |
|---|---|---|---|
| Baseline runs with Inductor autotuning off | 0.000102 (sd) | 4 runs | `rl_run/baseline_sd` |
| R1 screen-vs-confirmation pairs, before the agent turned on `max-autotune` | 0.000204 | 7 pairs | `rl_run/noise_sigma_before_autotune` |
| R1 screen-vs-confirmation pairs, after it (experiment 22 onwards) | 0.000504 | 4 pairs | `rl_run/noise_sigma_with_autotune` |
| h2h: the same code run twice | 0.000378 | 4 pairs | `h2h/noise_sigma_repeat_pairs` |

- The keep margin is 0.0005. With autotuning on, that is about 1σ of a two-run difference, so keeps and discards
  near the margin are close to coin flips.
- The screen runs in these pairs were selected as apparent wins. The pair estimates are therefore slightly
  inflated.

**Caveats**

- **Single-run bests are biased downward by selection.**
  - "Best single valid run" (R2) is the minimum of about 70 noisy runs, so it is optimistic by roughly 1–2σ.
  - R1's best, 1.007697, is an unselected confirmation re-run. The comparison below therefore favours h2h.
- **81% of R1's gain came from halving the batch size.**
  - This adapts the recipe to the hardware: the baseline gets only about 506 optimizer steps in 5 minutes on this
    GPU.
  - R1, both R2 arms and both R3 arms all found it.
  - After the halvings, R1 gained another 0.013939 (1.021636 → 1.007697).
- **The R1 harness did better than continuous sessions, but the RL updates are not shown to be the cause.**
  - R1 reached 1.007697 in 35 experiments (46 runs). The continuous sessions reached about 1.012 in 68–74 runs
    (R2), with either policy.
  - R1 differs from R2 in many ways: a fresh session per experiment with a history table, insights, confirmation
    re-runs, and the online updates.
  - In the continuous setting, R2 found no detectable difference between the v5 and base policies.
  - R3 is the controlled test of the insights.
- **There is no evidence that the RL updates helped.**
  - The success rates by policy version overlap (p = 0.39).
  - The updates were tiny: 16 optimizer steps and 44k GRPO tokens across all 5 updates.
  - The task gets harder as the best value improves, which confounds any trend over versions.
  - The full analysis is in `docs/RETROSPECTIVE.md`, written from a mid-run snapshot at about 23:05 UTC on
    2026-10-03.
- **n = 1 everywhere.** Each system ran once, and R3 has one group per arm. The difference in speed between the two
  nanochat GPUs is within the noise (see the calibration rows in R2).
