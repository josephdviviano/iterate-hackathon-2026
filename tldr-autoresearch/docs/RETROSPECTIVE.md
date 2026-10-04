# RLTL;DR × autoresearch: retrospective (snapshot about 23:05 UTC, 2026-10-03)

The four lens reports have been merged and corrected against their fact-checks. Refuted claims have been dropped or fixed. All data was read only.

This is a mid-run snapshot (37 of the final 40 experiments, 4 of 5 updates). Final numbers are in `RESULTS.md`. File:line references point at the code as it was at the time. "The README" means the design doc, now `docs/DESIGN.md`.

---

## 1. What we built and what the run shows so far

- **The pipeline is sound.** Qwen3.8-27B-FP8 with hot-swapped LoRA drives pi, one sandboxed session per experiment. Evaluation is trusted, with a fresh-compile confirmation. RLTL;DR updates run on GPU 3.
  - Exact served token ids are used for training. The vLLM vs trainer mismatch is 0.025–0.047 nats/token, and the IS weight is clamped on ≤0.015% of tokens.
  - All 224 agent calls finished cleanly. There have been 0 crashes and 0 voids, and every attempt used exactly 1 run.
- **Research progress is strong but mostly cheap.** val_bpb went from 1.079941 to 1.007697 (−6.7%) over 37 attempts in about 6.9 h, with 9 keeps.
  - 81% of the gain (0.0583 of 0.0722) came from two batch halvings. That is a hardware adaptation: the baseline gets 506 steps in 5 min here versus 953 on H100.
  - Since then the gain is −1.36% over about 33 attempts. Recent keeps mostly buy more steps in 5 minutes: throughput-motivated ideas kept 5/12 versus 4/24 for others (p=0.13).
  - Per experiment we are ahead of upstream; per hour we match the only same-GPU reference (about 5/h).
  - About 5 of our 9 keeps resemble upstream's annotated keeps, so we cannot rule out recall.
- **The RL signal is tiny.** There have been 4 updates and 12 Adam steps, totalling 35.7k GRPO tokens and 2.9k SFT tokens.
  - Per update that is 26–200× fewer backward tokens than the paper (0.8M, which counts both epochs). Rollouts per update are 21× fewer and distinct tasks 8× fewer.
  - In g1–g3 the positive-ratio filter dropped every negative, so GRPO was reward-weighted behaviour cloning on 1–2 trajectories.
  - Adam state carries over between updates (`trainer.py:226/230`). The current group supplied only 15–53% of the first moment, so changes cannot be attributed to individual versions.
  - The adapter does drift coherently: global ‖ΔW‖ went 0.45 → 0.62 → 0.88 → 0.99, and the cosine between each adapter and its next increment is 0.73–0.86.
- **There is no evidence either way that RL helps.**
  - Success by version: v0 3/9, v1 2/8, v2 1/8, v3 1/8, v4 2/4. v0 vs v≥1: Fisher p≈0.66.
  - With vs without insights: 8/29 vs 1/7 (p=0.65).
  - First-attempt success per group: n=5, 1 success.
  - Difficulty rises as the best ratchets down, which confounds any trend.
  - The only measurable change is the falling NLL of each group's insights before training on it (5.79 → 4.72 → 3.40 → 2.65 nats/token). This is circular: the judge is the policy, so part of it is self-distillation, and it is measured on the first minibatch only.
- **Measurement noise grew about 2.5×, and the agent caused it.**
  - Per-run σ was about 2.0e-4 before g0002-a6 (7 screen/confirm pairs) and about 5.0e-4 after (4 pairs, 95% CI about 3–14e-4, F-test p≈0.02).
  - Cause: `train.py:538` uses `mode="max-autotune-no-cudagraphs"`, which overrides the harness `no_autotune` (`ar_bootstrap.py:51-58`). The ledger still records `no_autotune=true`.
  - At σ=5e-4 the 0.0005 margin is about 1σ of a two-run difference:
    - A change with no real effect is kept 4–11% of the time (4% if the incumbent is itself a selected run, 11% if it is unselected).
    - The chance of keeping a true gain equal to the margin is exactly 1/3.
  - Confirmation reversed 2 of 11 apparent wins, both in the 0.0005–0.001 band.
  - The "no winner's-curse ratchet" claim in the README and `config.py:51` is false. The bias is about −2e-4 for keeps near the margin (g0002-a1, g0003-a3) and about 0 for the current incumbent (Δ 2.5e-3).
- **Insights are mostly hindsight about which idea to pick, not mistakes.**
  - 28 written (1 retracted). In 26/28 the judge's feedback says "not a code bug, experiment choice". They average 27.2 words (paper: 17), and 11/27 contain numbers.
  - About 7 restate the margin rule. 3 are wrong or misleading.
  - 43/88 injected copies were written against an older best.
  - The agent follows them: it explicitly follows the latest hint in about 24/29 conditioned attempts, and 4 keeps followed a directional hint (causation unproven).
  - A bad hint (g0002-a4, "confirm with a re-run") caused a refused self re-run in g0002-a6. That rollout was kept, so GRPO credited those tokens.
- **The machine runs as a strictly serial pipeline.**
  - Throughput is 4.85/h in the current regime, down from 6.2/h in g0.
  - Wall-time split: training run 52.8%, agent LLM 18.7%, confirmations 17.4%, insights 10.9%.
  - Only about 33% of 26 GPU-hours were used: vLLM 29%, GPU 2 70%, GPU 3 2%. vLLM and GPU 2 were busy at the same time for a total of 0.3 min.
  - LoRA decode is 29% slower (47.5 vs 67.2 tok/s). Insight calls are 37% of all generated tokens.
- **Integrity and robustness holes, none exploited yet.**
  - The time budget can be gamed (see 3.i-1). g0003-a7's prefetch thread got 333.5 s of trusted training time unflagged.
  - A repeatable trainer OOM restarts forever.
  - A wedged vLLM can block the driver for up to about 3 h.

---

## 2. What we missed or changed from the original approaches

### (a) RLTL;DR paper

| # | Paper | Ours (evidence) | Matters? | Fix |
|---|---|---|---|---|
| 1 | Built for Pass@128=0 tasks. On unfiltered data RLTL;DR "neither helps nor hurts" (paper, results on unfiltered data). | 25% success; every group had a success, so GRPO already has a signal. | **High, for expectations.** The method's main lever (breaking a zero-reward barrier) is not needed here. | Judge RL with a control arm (§4); add hard variants (iii-2). |
| 2 | 8 distinct tasks per update, about 170 rollouts, from 386–458 tasks each seen about once. | One task whose state keeps changing; 8 attempts per update. | **High.** π(f\|g) on one task is memorisation, and nothing is held out. | Lanes plus task variants (iii-1, iii-2); probes now (ii-2). |
| 3 | A group is K attempts at the **same** task. | The parent changed mid-group in **every** group (g0 after a3, a4; g1 after a4; g2 after a1, a6; g3 after a3, retroactively; g4 after a2, a3). The header still says "on this same task" (`gateway.py:47`). | **Medium.** The GRPO baseline mixes states of different difficulty, and stale hints become SFT label noise (Sec 6). Table 9 does not cover groups with mixed parents. | i-6 now; frozen-parent groups later (iii-3). |
| 4 | The verifier shows failed asserts, so the insight names a concrete mistake. | The judge sees only "did not beat best" plus stats; 26/28 feedbacks say the experiment choice was wrong. | **Fundamental.** Sec 7.2 predicts exactly this limit for automated research. | Richer judge diagnostics; broken-start variants where mistakes are verifiable (iii-2). |
| 5 | TL;DR of about 17 words. Train Pass@1 21.4 for TL;DR vs 20.3 for the diagnostic paragraph and 13.8 for summary + diagnostic. Detailed hints help more *in context* (+41.1% vs +36.0%) (Table 3). | 27.2 words, 11/27 with numbers, about 7 restating the margin rule, 3 wrong. | **High.** The SFT target is mostly run-specific hindsight. | i-5 now; ii-5 (specific hint in context, generic rule for SFT). |
| 6 | Generator is the student with 4096 think + 4096 answer tokens (App B). Thinking makes little difference (Table 3). A stronger teacher gains 3–5 points; the information given to the generator matters most. | Same budget, but 30/30 calls end at exactly 4095 reasoning tokens. g0003-a1 hit the 8192 cap (262 s). The judge is the current policy. | **Medium.** 10.9% of wall time; self-judging makes insight NLL circular. | i-3. |
| 7 | About 0.8M backward tokens, 8 steps, lr 3e-6 full fine-tune. | 14–31k backward tokens per update, 2–4 steps, lr 2e-5 LoRA r=32. Adam state persists across updates. | **High, for attribution and stability.** | ii-7. (Clipping on almost every step matches the paper, E.3; not a deviation.) |
| 8 | SFT on every injected copy; the dedup variant is equally good (Table 2). | Same, but insight #1 of g0 got 7 of 19 SFT copies. The last insight of each group is generated (about 90 s on the critical path) but never injected or trained (`trainer.py:175`). | **Medium.** | i-4, i-6. |
| 9 | Positive-ratio filter at 75%, "not so critical" for RLTL;DR (A.3.2). | Same. With 1–2 successes, floor(n_pos/3)=0, so no negatives in g1–g3. | **Medium** (signal starvation). | Revisit with graded reward (ii-8). |
| 10 | λ=0.5; relative normalisation of the two terms not specified. | Each term is token-averaged separately (`trainer.py:308,317`), so insight tokens get 1.2–17× the per-token weight of action tokens. | **Unknown.** "SFT dominates" is not established. | Log GRPO and SFT gradient norms separately before clipping (ii-2). |
| 11 | Held-out, deconfounded Pass@1; insight-efficacy check before training (Fig 11); App B checklist; entropy (D.1). | None. | **High.** | ii-1, ii-2. |

**Already faithful (keep):**
- Separate-chat JSON judge with 5 parts.
- ≤50% insertion rule, n_fb=16, insights as user messages.
- Flipped-mask SFT on injected spans.
- Dr.GRPO token mean, no KL, clip 0.2, 2 epochs, gradient clip 1.0.
- Decoupled PPO with truncated IS.
- Blocklist mechanism.

### (b) Karpathy's autoresearch

| # | Upstream | Ours (evidence) | Matters? | Fix |
|---|---|---|---|---|
| 1 | One long session; the agent keeps its own plans, diffs and logs. | A fresh session per experiment; the closing "what I learned / next ideas" is thrown away. | **High.** Repeats: warmdown ×5, depth-9 ×4, VE-all ×2. Weight decay, softcap and Muon momentum were each considered in about 25 of 37 sessions and never tried. | A lab record written by the harness (i-8, ii-9). Notes written by the policy only under A/B (iii-7). |
| 2 | The agent reads run.log (steps, MFU, loss). | `run.sh` prints only val_bpb and VRAM (`ar_run.py:631-632`); the judge sees stats, the agent does not. | **High.** At least 18 of 37 sessions assume vocab 32768 (it is 8192, `prepare.py:45`). Token and parameter estimates are off by 2–10×. The no-op speedup (g0000-a6, steps 1926→1925) went unnoticed and led to "speedups are a weak lever". | i-8. |
| 3 | Discarded ideas stay inspectable in git. | Their "commit" is `train_sha256[:7]` (`driver.py:506`); `git show` failed in g0002-a4 and g0004-a1. g0004-a1 built on a false lineage. | **Medium.** | ii-9 (`data/runs/<id>/diff.patch` already exists). |
| 4 | The agent keeps strictly-lower results; "~0 but much simpler → keep". | Harness margin 0.0005 plus a fresh confirmation; 0 simplification attempts, yet the prompt still promises the simplicity rule. | We gain integrity: upstream's rule would have made 6 extra keeps, including 2 that did not reproduce. But the promise cannot earn reward, so RL learns never to simplify. | i-7 (drop the claim) or iii-8. |
| 5 | No stated noise level. | The prompt says σ≈0.0002 (`driver.py:73`, from `reward_sigma`); `program.md` says "a few thousandths". | **Medium.** The two are inconsistent, and 0.0002 is stale after autotune (n=4). | i-7. |
| 6 | The agent's own log. | Discard rows show the *selected* first-run value; 6 rows sit below the best at the time, which feeds "near-miss" chasing (g0004-a1 on prefetch). | **Medium.** | i-8. |
| 7 | A "think harder / radical changes / combine near-misses" paragraph. | Only in `program.md`; effectively 1 session saw it. | Low–medium. One-knob edits are not a deviation: about 12 of upstream's 15 keeps are single constants. | Open question 7. |
| 8 | GPU for quick checks. | No GPU. CPU torch 2.9.1 exists in `.venv` but was never found. The guard blocks `sed -n` reads of `prepare.py` (3 of 4 blocks; g0003-a5's blocked read would have shown the vocab size). | **Medium.** | i-7. |
| 9 | "≈12/h" nominal (5-min experiments, zero agent time). | 4.85–5.4/h. A single lane is capped at 7.4–10.9/h even with zero LLM time; the same-GPU fork runs about 5/h. | We are at parity. Gains must come from overlap and lanes. | ii-6, iii-1. |
| 10 | The agent writes its own results.tsv (it kept "seed 42→137"). | Trusted eval, sandbox, one recorded result per attempt. | We are ahead, except for the time-budget hole. | i-1. |

The 40-row history cap only affects the prompt; agents read the full `results.tsv` in their clone. Low priority.

---

## 3. Improvements, ranked by impact / effort

Every prompt, judge or reward change shifts the state distribution mid-run.
- Switch at a **group boundary**.
- Stamp a `regime` field in `rollout.json`.
- Change one coherent bundle at a time.

### (i) Do now on this machine (cheap, low risk)

**i-1. Close the time-budget hole.**
- **Why:** `TRAIN_TIME_GRACE=60` (`ar_run.py:61,565`) accepts up to 360 s of trusted training (honest runs measure 302.6–306 s).
  - Prefetching moves the second-batch anchor earlier: g0003-a7 had `t_train2`=2.7 s and 333.5 s trusted, unflagged.
  - Without `make_dataloader`, `t_train2` is None and the fallback allows up to 600 s.
  - The `training_seconds` check (`:577`) reads the agent's own printout.
  - RL will find this hole.
- **How:** in `ar_bootstrap.py`:
  - Anchor at the 2nd optimizer step via `torch.optim.optimizer.register_optimizer_step_post_hook`, and check that it fires for the custom Muon optimizer.
  - Grace ~10 s.
  - A missing trusted timer is flagged, not given a fallback.
  - Replay the new rule over all 49 ledger runs before enabling it.
  - Make the confirmation limits compile-aware: `wall_clock_excessive` and the 660 s timeout should use wall − `t_train2` (`ar_run.py:313-317,580-582`). Re-run a confirmation once if it fails only on wall time; do not void the attempt.
- **Risk:** false flags on honest runs, which the replay catches.
- **Effort:** 3–4 h.

**i-2. Stop the trainer OOM restart loop.**
- **Why:** `trainer.py:405-412` re-raises OOM before writing `FAILS`. It would restart every ~31 s forever. The longest prompt is 34k of 64k.
- **How:** increment `FAILS` before re-raising. After 2 OOMs, retry with split microbatches, otherwise skip the group and alert.
- **Risk/effort:** none / 30 min.

**i-3. Make insight generation cheaper and non-self-referential.**
- **Why:** 30/30 judge calls burn exactly 4095 thinking tokens (about 92 s with LoRA vs 63–72 s on base). Table 3 shows thinking makes little difference. A self-judge makes insight NLL circular.
- **How:**
  - `insight_thinking_budget` 4096 → 1024 (or off), `insight_max_tokens` 8192 → 2048.
  - Route the judge call in `insight.py` to version 0, which is the base model and already served.
- **Gain:** about −60 to −75 s per attempt (742 → ~670 s, +11%).
- **Risk:** this deviates from App B; record it. Spot-check 10 regenerated past insights first.
- **Effort:** config plus about 10 lines.

**i-4. Don't block on the last insight of a group.**
- **Why:** it is never injected or trained (`trainer.py:175`); this cost 281 s so far.
- **How:** in the driver, skip it at k=group_size, or run it asynchronously for the record.
- **Effort:** 30 min.
- Overlapping other insights with the next attempt is only legal when that attempt is unconditioned, which is worth about 1–2%. Not worth it.

**i-5. Constrain the judge and gate SFT targets.**
- **Why:** 11/27 hints contain values. About 7 restate the margin rule. g0002-a4 ordered an action only the harness can take. Table 3 favours TL;DR for internalisation.
- **How:** in the `insight.py:50-55` prompt:
  - "The agent cannot re-run experiments or change the keep rule."
  - "A result within about σ of the margin is noise, not a mistake."
  - TL;DR ≤20 words. Name the corrective action in the task's own vocabulary; knob names are allowed (App C.1). No numeric values, commit or attempt IDs, or "current best".
  - A regex gate (digits, IDs) keeps a failing hint *in context* but excludes it from SFT through the existing blocklist path (`trainer.py:155`).
- **Risk:** hints become vacuous ("try another knob"). Require a concrete action.
- **Effort:** 1–2 h.

**i-6. Handle stale and duplicate insights.**
- **Why:** 43/88 injected copies (about 49% of SFT copies) were written against an older best; insight #1 gets up to 7 copies.
- **How:**
  - After a keep, annotate in-context insights "(written before the best improved to X)" and fix `INSIGHT_HEADER` (`gateway.py:47`).
  - In `prepare_group` (`trainer.py:174-191`), train each insight once (dedup, Table 2), only in a context with the same parent.
- **Effort:** about 3 h.

**i-7. Make the prompt and docs honest.**
- Show a separate `noise_sigma_shown` (about 0.0005, "measured from 4 pairs, uncertain") instead of `reward_sigma` (`driver.py:73`).
- Fix `program.md` ("a few thousandths") and the "~5 sigma" / "no winner's-curse" comments (`config.py:45,47,51`, README).
- Delete the simplicity promise until it is rewardable.
- Add one line: "`.venv/bin/python` has CPU-only torch for parameter counts and shape checks."
- Let read-only `sed -n` through the guard (`guard.ts:116-117`).
- **Effort:** 1–2 h.

**i-8. Give the agent its run statistics (minimal lab record).**
- **Why:**
  - Vocab is wrong in at least 18/37 sessions.
  - g0004-a1 guessed 655M tokens; the actual was 242M.
  - No-op speedups go undetected.
  - Discard rows show the selected first-run values.
- **How:**
  - `ar_run.py:631`: print trusted steps (`trusted.evals[].train_batches`, already recorded), tokens, parameter count, vocab size and ms/step, each with the parent's value and Δ.
  - `results.tsv` / `driver.py:506`: add Δ vs the best at the time and a "within noise" flag (|Δ| < 2σ_diff). For reversed wins, show the confirmation value.
- **Risk:** prompt shift; do it at a group boundary.
- **Effort:** about 0.5 day.

**i-9. Operations.**
- vLLM watchdog keyed on the `/metrics` generation counter. Do not use "no bytes for 120–180 s": non-streaming judge calls are silent for 90–170 s.
- Bound non-streaming timeouts (`gateway.py:229` total=None; `insight.py:174-176` 3×3600 s).
- Supervisor backoff (`ctl.sh:41-50`).
- A health daemon checking attempt age, voids, GPU 2 idle time, `FAILS`, publish lag and disk.
- Disk retention:
  - LRU-cap the Inductor/Triton cache.
  - Keep the last 3 adapters plus every 10th.
  - zstd-compress attempt files.
  - Delete `scratch/archive` (6.2 GB).
  - 215 GB is free.
- **Effort:** about 1–1.5 days.

**i-10. Analysis hygiene.** Run heavy analysis with `nice`/`ionice` or off the box while runs are being timed. g0004-a5's MFU was 20.2% vs 22.5–23.0%; contention cannot be ruled out.

### (ii) Next (moderate)

**ii-1. Randomised control arms in the live loop.** (See §4.)
- **How:**
  - The driver samples the policy per attempt: v0 with p=1/3, else vN. Optionally a 25% no-insight arm.
  - Primary outcome: screen Δ vs parent, clipped to ±3e-3 (SD about 2e-3).
  - A 1e-3 shift needs about 135 attempts at a 1:2 split, about 22–28 h.
- **Risk:** base-arm rollouts are more off-policy for training; watch the IS clamp rate.
- **Effort:** about 1 day including the analysis script.

**ii-2. Probes inside the trainer process.**
- A second process will not fit: the trainer holds 51–52.5 GiB resident and peaks at 62 GiB.
- Per update:
  - KL(v_{N+1}‖v_N) and KL(v_N‖base), plus entropy, on about 20 frozen first-turn prompts.
  - Separate GRPO and SFT gradient norms before clipping.
  - Adam momentum share.
  - Log-probability of known-good (kept) vs known-bad diffs on frozen historical prompts, under base vs vN. This needs no new runs.
- Retroactive for v1–v4 from disk.
- Do not use "lp_vN − lp_base on group g+1 rollouts". They were sampled from vN, so the comparison is biased, and `lp_prox` exists only for items that passed the filter.
- **Effort:** about 1 day plus a trainer restart.
- Optional: proposal probes in vLLM idle time. These need `--max-loras ≥3` (`serve.sh:28`) and adapter refcounting (`gateway.py:333`).

**ii-3. Fix the incumbent measurement, not the candidate's.**
- **Why:** a second candidate confirmation barely helps (false keeps 0.113 → 0.112; power 0.333 → 0.348). Measuring the incumbent three times (kB=3) with one confirmation gives 5.8% false keeps and 67% power at a gain of 0.001.
- **How** (`driver.py:462-505`):
  - After a keep, run 2 fresh **unselected** replicates and set `best` to their mean.
  - The deciding confirmation is excluded: mean(confirm, extra) still carries −1.05e-4 bias.
  - Estimate σ online from the replicates; screen/confirm pairs are selection-biased.
  - Only then consider margin 0.0003.
- **Cost:** about 15 min of GPU 2 per keep, about 35% of wall time if done for every keep. Cut it by replicating only keeps with confirmation Δ < 0.002 (about 1/3 of keeps), or run replicates on GPU 3 (ii-6).
- **Effort:** about 0.5 day.

**ii-4. Two-channel insights.**
- The judge emits a run-specific `hint`, injected in context as now, and a generic ≤20-word `rule`.
- The rule is trained as a standalone (g, rule) SFT pair (SFTL;DR, Table 2). It is the only SFT target.
- This follows Table 3: detailed hints are best in context, TL;DR internalises best.
- Change `prepare_group` and the judge's JSON schema.
- **Risk:** a deviation from App A.1; record it.
- **Effort:** 1 day.

**ii-5. Size updates by data, not by repeated steps.**
- Reset the Adam moments per update, or decide deliberately to keep them, and log the momentum share.
- Accumulate ≥2 groups per update.
- Use 1 PPO epoch only together with the reset (epoch-2 clip fraction ≤0.15%).
- Set lr from a per-update KL target measured by ii-2.
- Do not replay old GRPO positives (those moves are used up) or old negatives (their meaning depended on the old best).
- **Effort:** about 0.5 day.

**ii-6. Shadow re-run on GPU 3.**
- Start a fresh-cache copy of each agent run in parallel, which gives a free unselected second measurement.
- **Gain:** together with i-3, 742 → ~557 s per attempt (+33%).
- **Prerequisites:**
  - The trainer releases GPU 3 through the lock between updates (reload takes 16 s).
  - `kill_job` and startup reaping are scoped to the runner's own process tree (`runner.py:59-66,100,237-240`).
  - One venv per GPU (`ar_gpu_pin`).
  - **Per-GPU bests**, with keep decisions only between runs on the same GPU.
  - GPU 2's noise re-calibrated under concurrent load: all 48 runs so far ran with GPU 3 essentially idle.
  - Updates may wait up to about 8 min for the lock.
- **Effort:** about 1.5–2 days.

**ii-7. Graded reward**, only after i-1 and ii-3.
- Fix the dormant ordering bug in delta mode (`driver.py:516-519`): crash or invalid gets −1, above a bad valid run at −3.
- Use a reward scale s ≈ 1e-3 (tanh, or rank over the last 32 attempts), separate from the noise σ.
- Compute it from unselected measurements only.
- This gives GRPO real negatives; then reconsider the 75% filter.
- **Risk:** noisier gradients.
- **Effort:** 0.5 day.

**ii-8. Full lab record.**
- Commit every evaluated `train.py` to `refs/experiments/<attempt_id>` in canon.git and fetch it into each clone.
- Add a parent column.
- Add a deterministic, descriptive "explored axes" index: knob → values tried → Δ, plus untouched knobs.
- **Effort:** 1 day.

**ii-9. Per-call agent thinking cap.**
- p90 is 3,340 tokens per call; the maximum is 18,927 (391 s, g0002-a5). All 7 sessions over 10k tokens failed (p≈0.15). The paper uses 768 per step.
- Mask forced end-of-think tokens from GRPO.
- **Risk:** quality. A/B it with ii-1.

### (iii) Bigger machine and research extensions

1. **Independent lanes.** Each lane gets its own branch and best, a runner bound to that lane, and sequential insights; 6–7 lanes on 8 GPUs, an estimated 25–32 attempts/h. Code changes:
   - `gateway.py`: per-lane `ST.attempt` and sockets; refcounted adapter unloading.
   - `runner.py`: one process per GPU.
   - `ar_run.py`: per-runner cache.
   - `driver.py`: `--lane` (`RUNNER_URL` L47).
   - `sandbox.sh`: per-lane forwards.
   - `trainer.py`: batch 4–8 READY groups instead of `pending[0]`, which approaches the paper's 8 tasks and ~0.8M tokens.
   - Write an integration test with a fake vLLM and runner first.
2. **Task variants as training and held-out data.**
   - Broken starts with known fixes: LR 5× too high, warmdown 0, batch 2^20, a width bug. This is the regime where the paper's verifiable insights actually apply.
   - Scale and time-budget variants need a separately pinned trusted `prepare.py`; never relax the pin.
   - This gives the deconfounded held-out Pass@1 of D.1.
3. **Frozen-parent groups within a lane**: proper GRPO groups at the same bar. The parent advances at group end to the best confirmed candidate.
4. **LoRA serving benchmark:**
   - tuned kernel configs
   - `--max-lora-rank 32` (now 64)
   - `--max-loras`
   - TP=1 vs TP=2. If TP=1 reaches ≥42 tok/s, it frees a GPU for another runner.
   - Re-check logprob parity. Keep MTP off.
5. **Stronger or separate judge with privileged diagnostics** (loss curve vs parent, tok/s, MFU); the paper reports +3–5 points from a stronger teacher. Self-play insight SFT (F.1) later.
6. **Policy-written lab notebook**, only as a lane-level A/B with a retraction path. The g0003-a3 harness-bug insight would have persisted, and the blocklist only covers SFT.
7. **Simplicity keeps**, only as a strict non-inferiority test against a replicated incumbent, with 2 confirmations and a size measure that ignores dead code. Deleting the FA3 branch or MFU code would otherwise earn free reward.
8. **Before the move: portability and preflight.**
   - The absolute project path appeared on 89 lines across about 20 files, and `RLTLDR_ROOT` was honoured only by `config.py` and the sandbox scripts. (Since fixed: the project root is `$RLTLDR_ROOT`, and machine settings live in `config.json`.)
   - uid/gid were hardcoded. (Since fixed: the sandboxes use the invoking user's ids.)
   - Assert that the GPU minor number matches the UUID; `ar_run.gpu_minor()` exists but `runner.py` does not use it.
   - `TASK_TEMPLATE` hardcodes this GPU (`driver.py:61,83`).
   - `ar_calibrate.sh` is stale (no `--no-autotune`, fresh cache or `--gpu-minor`).
   - `causal_conv1d` is missing from the trainer env.
   - Add the parity tests to `SETUP.md`.
9. **Full fine-tuning** only if LoRA capacity is shown to be the limit. It needs bf16 serving, because FP8 erases small updates, and roughly halves the runners.

---

## 4. The single most important next change

**Add a randomised control arm to the live loop (ii-1), with the in-trainer probes (ii-2).** The base policy v0 runs about 1/3 of attempts under the same prompt and insight rule, and the outcome is the clipped screen Δ vs parent.

Why this first:
- **The run cannot answer its own question.** Per-version success rates have overlapping CIs (Fisher p≈0.66). Difficulty rises over time. Persistent Adam momentum makes per-version attribution impossible. The only "learning" curve (insight NLL) is circular.
- **It makes every other change safe.** Every fix in §3 is a regime shift mid-run, and without a concurrent control a before/after comparison is uninterpretable. With per-attempt randomisation, each regime gives its own unbiased vN-vs-base estimate, because both arms face the same states.
- **It costs almost nothing in GPU time.** Base-arm attempts still do useful research (v0 kept 3/9, no worse than later versions), and their rollouts can still be trained on through truncated IS.
- **It informs the bigger machine.** About 22–28 h here (about 4–5 h on 8 GPUs) gives either a measured effect or a firm "not detectable at this update size". That tells you whether to invest in bigger updates and multi-task lanes, or in agent scaffolding (lab record, statistics).

Do i-1 (the time-budget hole) in the same restart. It is a few hours of work, and an RL-trained policy will eventually find that hole.

---

## 5. Open questions for the operator

1. **Max-autotune.** Two options:
   - *Enforce* `no_autotune`: patch `torch.compile` in the bootstrap. σ returns to about 2e-4, but you lose about +6% steps, must re-baseline, and invalidate 1.007697.
   - *Accept* it: σ is about 5e-4, each fresh confirmation costs about 1–2.5 min more, and ii-3 becomes mandatory.

   Note that compile time is outside the budget, so the loop pays for compile-heavy ideas, not the agent.
2. **Keep rule (your margin question).** Neither "lower margin" nor "2 confirmations" alone fixes it; the single-run incumbent dominates the error. Do you accept 2 unselected incumbent replicates per keep, on GPU 2 at about 10–35% of wall time or on GPU 3 after ii-6? Do you want a lower margin (~0.0003) once that is in place?
3. **Change the live run now, or freeze it?** Apply §3(i) mid-run, at group boundaries with regime tags and ideally with ii-1 in place? Or let the current design run on as a clean reference and apply the changes on the new machine?
4. **Insight design deviation.** Do you approve departing from App A.1/B: base-model judge, ≤1024 thinking tokens, constrained TL;DR, and a two-channel design where the run-specific hint goes in context and only the generic rule is SFT'd?
5. **Group semantics.** Freeze the parent per group (faithful to the paper's GRPO, slower hill-climbing), or keep greedy in-group keeps (faithful to autoresearch) and only annotate stale insights?
6. **Simplicity criterion.** Delete it from the prompt (recommended for now), or build guarded non-inferiority keeps?
7. **Exploration prompt.** Add upstream's "think harder / radical / combine near-misses" paragraph to `task.md`? It conflicts with dropping near-miss chasing; the near-miss retries kept 1/4, which is the base rate.
8. **Update regime.** Reset Adam per update and accumulate ≥2 groups (fewer, larger, attributable updates), or keep the current per-group cadence with momentum carry-over?
9. **Agent thinking cap.** Cap thinking per call (saves time; could hurt quality), tested inside the control-arm A/B?
10. **Bigger-machine design.** Should lanes be independent task variants (recommended: diversity, no stale merges, no cross-GPU comparisons) or share one branch? Should the multi-task variants count as the held-out evaluation the paper's claim depends on?

**Evidence:**
- Run data (under `$RLTLDR_ROOT/data/`): `ledger.jsonl`, `metrics_trainer.jsonl`, `attempts/*/rollout.json`, `attempts/*/calls.jsonl` and `runs/*/diff.patch`. The derived tables are in `results/` (`tools/export_results.py`).
- The RLTL;DR paper (arXiv 2609.37633): the passages on unfiltered data, Table 2, Table 3, Sec. 7.2, App. A.3.2 and App. C.1.