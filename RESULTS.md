# Results

Every number in the pitch and README.md comes from this file. Each entry gives
the metric, the number of runs, the data split, the baseline, the command and
the commit. Splits are temporal: train is the first fraction of a level's
transitions, test is the rest of that level or all of the next level. RESET
and level-closing transitions are removed from both sides from R2 onward (R1
still had them in train). Test transitions never enter a prompt.

Weighting. R3 to R13 used an MDL prior (λ = 0.01 per gzip byte) for the
vote and equal-weight disagreement for uncertainty. After R14 the default
is equal weights for both; the MDL prior is an opt-in ablation (`--lam`).
Where the two differ for a reported vote accuracy, both are given.

## R1. Baseline single program, tr87 level 1

| Item | Value |
|---|---|
| Metric | Held-out next-state accuracy (exact object-level match) |
| Result | 12/12 = 1.00 |
| Train replay | 19/19 exact |
| Runs | 1 |
| Split | tr87 L1, train_frac 0.6: 19 train, 12 test |
| Baseline | This is the baseline condition |
| Synthesis | claude -p, model opus, 4 turns, 15 s |
| Command | `uv run python -m committee.experiment tr87 --level 1 --runs 1` |
| Commit | uncommitted, base a718512 |

Note: this level is saturated. A single program already generalises, so it
cannot show committee value. Harder levels and a 0.4 train fraction follow.

## R2. Synthesis backends on the easiest level, tr87 level 1

Same split and prompt contract for every backend. The `api` backends run the
repair loop (`committee.synth_api`, 8 rounds, best program kept) against a
model served by vLLM on a Modal H100. Devin and the claude CLI run their own
agent loop with the checker as a tool.

| Backend | Runs | Train replay (of 19) | Held-out accuracy (of 12) | Wall per run |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B-Instruct-FP8, repair loop | 3 | 3, 18, 19 | 0.00, 0.75, 0.25 | 45 to 93 s |
| gpt-oss-120b, repair loop, reasoning effort high | 3 | 12, 18, 3 | 0.42, 0.83, 0.00 | 59 to 73 s |
| Devin (API v1), own loop | 1 | 19 | 1.00 | 62 s |
| Claude Code CLI, opus, own loop | 1 | 19 | 1.00 | 15 s |

| Item | Value |
|---|---|
| Metric | Exact train replay count and held-out next-state accuracy |
| Split | tr87 L1, train_frac 0.6: 19 train, 12 test. RESET and level-closing transitions in train for the claude run (R1), dropped for the others |
| Baseline | The open models are the comparison; Devin is the backend used for R3 onward |
| Command | `uv run modal run -m committee.modal_app --game tr87 --level 1 --train-frac 0.6 --runs 3 --condition api_gptoss --backend api --base-url <gpt-oss url> --reasoning-effort high`; `uv run python -m committee.experiment tr87 --level 1 --train-frac 0.6 --runs 1 --backend devin --condition devin_smoke` |
| Commit | uncommitted, base a718512 |

Finding: in a plain repair loop the open models oscillate between rounds and
do not reach exact replay on a level that the agent backends solve in one
minute. The committee needs consistent members, so reported numbers use Devin.
The Modal model path stays as the cheap, fully owned backend.

## R3. ar25 level 3, 40% train: single programs and the unseeded committee

The informative setting: every program replays train exactly, and none
generalises to the rest of the level. Backend Devin.

| Condition | Programs | Train replay | Held-out accuracy (44 transitions) |
|---|---|---|---|
| Single program, run 0 / 1 / 2 | 3 | 29/29 each | 0.636 / 0.432 / 0.477, mean 0.515 |
| Unseeded committee of those 3, MDL weights λ=0.01 | 3 | | weighted vote 0.636, simplest member 0.636 |

| Item | Value |
|---|---|
| Metric | Held-out next-state accuracy; AUROC of committee disagreement against vote error |
| Calibration | AUROC 0.597. Unanimous on 28 transitions with error rate 0.32; split on 16 with error rate 0.44. Three members give only two disagreement levels |
| MDL weights | 0.918 / 0.035 / 0.047. The shortest program was also the best |
| λ sweep (vote accuracy, AUROC of that weighting's disagreement against its own vote error) | λ=0: 0.477, 0.76. λ=0.001: 0.477, 0.79. λ=0.003 and above: 0.636, 0.60. Equal-weight disagreement against the MDL vote's error: AUROC 0.59; the 3 transitions where all members differ are all mispredicted |
| Distinct held-out behaviours | 3 of 3 |
| Runs | 3 syntheses, 166 to 475 s wall each |
| Split | ar25 L3, train_frac 0.4: 29 train, 44 test. RESET and level-closing transitions dropped from both sides |
| Baseline | Single program (OPINE-World style) |
| Command | `uv run python -m committee.experiment ar25 --level 3 --train-frac 0.4 --runs 3 --backend devin --condition baseline_devin --parallel 3` then `uv run python -m committee.evaluate ar25 --level 3 --train-frac 0.4 --condition baseline_devin` |
| Commit | uncommitted, base a718512 |

The seeded committee (8 members, guided diversity) on the same split is R4.

## R4. ar25 level 3, 40% train: seeded committee of 8

Same split as R3. Each member got one seed hypothesis (`committee.seeds`).
Backend Devin. All 8 replay train 29/29.

| Quantity | Value |
|---|---|
| Member held-out accuracy | 0.43, 0.48, 0.48, 0.43, 0.48, 0.64, 0.43, 0.48; mean 0.480 |
| Distinct held-out behaviours | 7 of 8 |
| Weighted vote, λ=0.01 | 0.477. The heaviest member (weight 0.57) is a 0.48 program, not the 0.64 one. On this split the committee does not beat the mean single program on accuracy |
| Disagreement predicts error | AUROC 0.776 (MDL-weighted disagreement), 0.768 (equal weights). R3's committee of 3 gave 0.60 |
| Reliability, equal-weight disagreement | unanimous: 27 transitions, error 0.30. low: 12, error 0.83. medium: 3, error 1.00. high: 2, error 1.00 |
| Rows touched by held-out transitions | 90, of which 51 never seen in train. Count-based η is undefined on those 51; the committee assigns each an effect distribution and entropy |
| Exploration, probes until a counterexample falsifies every member | disagreement order: 1. Count-priority order (OPINE-World's η): 7. Random: mean 2.7 over 20 orders |
| Runs | 8 syntheses, 207 to 434 s wall each |
| Split, baseline, split rules | As R3 |
| Command | `uv run python -m committee.experiment ar25 --level 3 --train-frac 0.4 --runs 8 --seeded --backend devin --condition committee_devin --parallel 4`; `uv run python -m committee.evaluate ar25 --level 3 --train-frac 0.4 --condition committee_devin`; `uv run python -m committee.demo` |
| Commit | 389e742 (artifacts), evaluation in the following commit |

Reading: when the members agree, the prediction is right 70% of the time;
when they split, it is wrong 83% to 100% of the time. The committee's
uncertainty is informative where OPINE-World's count-based η has no data.
Its point accuracy is not better than one program here, and the most
disputed transition falsifies every member, which is the signal to
resynthesize with a new hypothesis.

## R5. tr87 level 1 trained, level 2 tested: saturated control

Backend Devin. Train on all 31 modellable transitions of level 1, test on
the 28 of level 2.

| Condition | Programs | Train replay | Held-out accuracy |
|---|---|---|---|
| Single program | 3 | 31/31 each | 1.00, 1.00, 1.00 |
| Seeded committee | 8 | 31/31 each | members 1.00; vote 1.00; one distinct behaviour; unanimous on all 28, error 0.00 |

| Item | Value |
|---|---|
| Runs | 11 syntheses, 62 to 124 s wall |
| Command | `uv run python -m committee.experiment tr87 --level 1 --train-frac 1.0 --test-level 2 --runs 8 --seeded --backend devin --condition committee_devin --parallel 4` |
| Commit | 389e742 (artifacts) |

Reading: the level-1 rule carries over to level 2 and the committee says so
by agreeing everywhere. Together with R4 the unanimous bins have error 0.00
and 0.30, the split bins 0.83 to 1.00.

## R6. m0r0 level 3, 40% train: second informative level

Backend Devin. 30 train, 44 test. Mechanics include a hidden left/right
player identity and hazard resets.

| Condition | Programs | Train replay | Held-out accuracy |
|---|---|---|---|
| Single program | 3 | 25/30, 25/30, 30/30 | 0.66, 0.66, 0.75; the one admitted program 0.75 |
| Seeded committee | 8 | 30/30 each | members 0.75 ×5, 0.77 ×3, mean 0.76; vote 0.77; 4 distinct behaviours |

| Quantity | Value |
|---|---|
| Admission rate | unseeded 1 of 3; seeded 8 of 8 |
| Disagreement predicts error | AUROC 0.68 (equal and MDL weights agree) |
| Reliability, equal-weight disagreement | unanimous: 38 transitions, error 0.16. low: 1, error 0.00. medium: 5, error 0.80 |
| Rows touched by held-out transitions | 49, of which 7 unseen in train |
| Probes until every member is falsified | disagreement 1; count priority 10; random mean 4.8 over 20 orders |
| Runs | 11 syntheses, 144 to 455 s wall |
| Command | `uv run python -m committee.experiment m0r0 --level 3 --train-frac 0.4 --runs 8 --seeded --backend devin --condition committee_devin --parallel 4`; `uv run python -m committee.evaluate m0r0 --level 3 --train-frac 0.4 --condition committee_devin`; `uv run python -m committee.demo --game m0r0` |
| Commit | see the commit that adds artifacts/m0r0 |

Reading: the second informative level repeats the pattern of R4. Agreement
means mostly right (error 0.16), a split means mostly wrong (0.80), and
probing by disagreement finds the falsifying counterexample in one move
against ten for count priority. Here the vote also edges the single
programs (0.77 against 0.69 mean over all three, 0.75 for the admitted one),
and seeding lifted admission from 1 of 3 to 8 of 8.

## R7. ft09 level 5, 40% train: saturated

Backend Devin. 21 train, 31 test. All 11 programs (3 single, 8 seeded)
replay train exactly and score 1.00 held-out. Committee unanimous on all 31
transitions, error 0.00. Same command pattern as R6 with `ft09 --level 5`.

## Summary across the four splits

| Split | Informative | Unanimous: n, error | Split: n, error | AUROC | Probes to falsify: disagreement / counts / random |
|---|---|---|---|---|---|
| ar25 L3 f0.4 | yes | 27, 0.30 | 17, 0.83 to 1.00 | 0.78 | 1 / 7 / 2.7 |
| m0r0 L3 f0.4 | yes | 38, 0.16 | 6, 0.67 (1 right, 5 wrong) | 0.68 | 1 / 10 / 4.8 |
| tr87 L1 to L2 | no | 28, 0.00 | 0 | | |
| ft09 L5 f0.4 | no | 31, 0.00 | 0 | | |

## R8. Targeted committee growth: negative result

Hypothesis: synthesizing new members where the current members disagree
is more efficient than independent seeded synthesis. Method
(`committee.active`): start from the same first 3 seeded members; probe set
= every training state with every available action (clicks on each visible
object); members predict all probes; the most disputed probes with the
competing outcomes and vote counts seed the next member; rounds of 2 up to
8 members. Held-out data never enters the probe set. Backend Devin.

| 8 members (ar25 targeted: 7 admitted of 8, one session timed out) | ar25 L3 seeded | ar25 L3 targeted | m0r0 L3 seeded | m0r0 L3 targeted |
|---|---|---|---|---|
| Vote accuracy | 0.48 | 0.48 | 0.77 | 0.75 |
| AUROC, disagreement vs error | 0.77 | 0.74 (0.61 until the 7th member) | 0.68 | 0.72 |
| Split transitions, error | 17, 0.88 | 16, 0.88 | 6, 0.67 | 6, 0.83 |
| Unanimous transitions, error | 27, 0.30 | 28, 0.32 | 38, 0.16 | 38, 0.16 |
| Distinct held-out behaviours | 7 | 6 | 4 | 3 |
| Mean disagreement on own probes, first to last round | | 0.027 to 0.014 | | 0.225 to 0.070 |

| Item | Value |
|---|---|
| Metric | As R4, plus mean disagreement on the hypothetical probe set per round |
| Runs | m0r0: 5 targeted syntheses, 125 to 207 s. ar25: 5 targeted syntheses, 145 to 298 s, one timed out at 30 min and was not admitted |
| Evidence of anchoring | Every targeted member adopted the majority stance. Devin's own summaries: "I sided with the majority: nothing changes", "I went with the 6-program majority" |
| Command | `uv run python -m committee.active ar25 --level 3 --backend devin`; `uv run python -m committee.evaluate ar25 --level 3 --train-frac 0.4 --condition active_devin --curve` |
| Commit | 0a8a623 (code); artifacts in the following commit |

Reading: targeted growth lowered the committee's uncertainty about its own
hypothetical moves by two thirds on m0r0 while the held-out transitions it
gets wrong stayed the same six. The independent order found, on each level,
one program with a different held-out behaviour earlier than the targeted
order did: on ar25 at its 6th member against the targeted order's 7th, and on
m0r0 not at all in the targeted order. Three
causes: the probes are training states, so disputes concern mechanics the
data already half-pins down, while held-out errors come from unseen states;
each dispute was one mechanic touching at most four held-out transitions;
and showing the vote counts anchored the synthesizer on the majority. Rounds
are sequential, so wall time is about three times that of independent runs.
Decision: keep independent seeded synthesis.

## R9. Dynamics-ensemble baselines at the effect level

Baselines from the learned-dynamics literature on the same splits, with
per-object features (type, action, tags, position, size, touching neighbour
types, click offset) and a per-object effect class as the target. The
program committee's predicted states are reduced to the same classes.
`accuracy` = every object's class right; `object accuracy` = share of
objects right. Members of the ensembles are not filtered by exact replay.
Command: `uv run python -m committee.baselines ar25 --level 3 --train-frac 0.4`.

| ar25 L3, 44 held-out | accuracy | object accuracy | AUROC | unanimous n, error | split n, error | survivors after probes 1..5, disagreement order | random order, mean |
|---|---|---|---|---|---|---|---|
| nearest neighbour copy | 0.00 | 0.60 | | | | | |
| bagged trees, QBC, 8 | 0.00 | 0.41 | none | 0 | 44, 1.00 | 0,0,0,0,0 | 0,0,0,0,0 |
| MLP deep ensemble, 8 | 0.00 | 0.53 | none | 1, 1.00 | 43, 1.00 | 0,0,0,0,0 | 0.05,0,0,0,0 |
| program committee, 8 | 0.48 | 0.89 | 0.77 | 27, 0.30 | 17, 0.88 | 2,1,1,1,1 | 2.95,1.0,0.7,0.7,0.2 |

| m0r0 L3, 44 held-out | accuracy | object accuracy | AUROC | unanimous n, error | split n, error | survivors, disagreement | random, mean |
|---|---|---|---|---|---|---|---|
| nearest neighbour copy | 0.14 | 0.66 | | | | | |
| bagged trees, QBC, 8 | 0.32 | 0.81 | 0.48 | 0 | 44, 0.68 | 0,0,0,0,0 | 0.55,0.1,0,0,0 |
| MLP deep ensemble, 8 | 0.23 | 0.79 | 0.47 | 25, 0.76 | 19, 0.79 | 0,0,0,0,0 | 0.85,0.4,0,0,0 |
| program committee, 8 | 0.77 | 0.96 | 0.68 | 38, 0.16 | 6, 0.67 | 0,0,0,0,0 | 6.0,3.9,3.9,3.9,3.1 |

| Item | Value |
|---|---|
| Runs | 1 per method, seed 0; the learned baselines train in seconds |
| Split | As R3 and R6 |
| Commit | a20f4e1 |

Reading: with 25 to 30 training transitions the learned ensembles cannot
model the mechanics; their disagreement is not informative (AUROC 0.47 to
0.48 or undefined) and their members are falsified by almost any probe, so
their survivor curves say nothing. The program committee is the only method
whose agreement is informative. One wrinkle: at the effect level on ar25 the
disagreement order falsifies the last member after 10 probes against 3.0 for
random, because effect classes merge predicted states that differ only in
amounts, so the transition where every member is wrong no longer ranks
first. At the state level (R4) the disagreement order falsifies all members
in one probe. The exploration claim is therefore at the granularity the
agent predicts, the exact next state.

## R10. Committee size and confidence intervals

Every subset of the 8 seeded members for K = 2 and 4 (60 sampled of 70 for
K = 4), the full committee for K = 8; equal-weight disagreement; mean and
standard deviation over subsets. Bootstrap over transitions (1000 draws)
for the full committee's AUROC. Command: `committee.evaluate.k_sweep`.

| | ar25 L3: AUROC | unanimous error | split error, n | m0r0 L3: AUROC | unanimous error | split error, n |
|---|---|---|---|---|---|---|
| K = 2 | 0.61 ± 0.11 | 0.45 ± 0.09 | 0.81 ± 0.29, 7.5 | 0.64 ± 0.09 | 0.18 ± 0.04 | 0.82 ± 0.17, 3.7 |
| K = 4 | 0.68 ± 0.10 | 0.39 ± 0.09 | 0.82 ± 0.10, 11.9 | 0.68 ± 0.05 | 0.16 ± 0.02 | 0.72 ± 0.11, 5.5 |
| K = 8 | 0.77 | 0.30 | 0.88, 17 | 0.68 | 0.16 | 0.67, 6 |

| Item | Value |
|---|---|
| AUROC 95% CI, K = 8 | ar25 (0.64, 0.88); m0r0 (0.52, 0.86) |
| Pooled over both levels, 88 transitions | AUROC 0.75, 95% CI (0.66, 0.84) |
| Vote accuracy by K | ar25 0.47, 0.48, 0.48; m0r0 0.76, 0.77, 0.77 |
| Commit | see the commit that adds k_sweep.json |

Reading: calibration improves with committee size on both levels, and the
unanimous error falls as members are added, which is what a version-space
reading predicts: more consistent hypotheses expose more of the transitions
where agreement was accidental. Per level the interval is wide, as the small
n demands; pooled, the interval excludes 0.5. Accuracy does not move with K.

## R11. Verbalized confidence of a judge model

Baseline from the language-model uncertainty literature (Kadavath et al.
2022; Tian et al. 2023). For each held-out transition a judge sees the
committee's heaviest program, the before state, the action and that
program's predicted after state, and answers with a confidence from 0 to
100 at temperature 0. Judge: gpt-oss-120b served by vLLM on Modal, 44 calls
per level. Uncertainty = 100 minus confidence. Compared with the
committee's equal-weight disagreement on the same transitions and the same
error labels. Command: `uv run python -m committee.confidence ar25 --level 3 --train-frac 0.4`.

| | ar25 L3 | m0r0 L3 | pooled, 88 |
|---|---|---|---|
| Judge AUROC | 0.82 | 0.46 | 0.71 |
| Committee disagreement AUROC | 0.77 | 0.68 | 0.75 |
| Judge mean confidence when right / wrong | 65 / 36 | 72 / 79 | |
| Distinct confidence values used | 3 | 4 | |
| Combined score, rank sum of both | | | 0.85, 95% CI (0.76, 0.92) |

| Item | Value |
|---|---|
| Difference, disagreement minus judge, pooled | +0.03, 95% CI (−0.09, +0.16): not separable |
| Extra inference | judge: 44 calls per level; committee: none, its predictions already exist |
| Commit | see the commit that adds verbalized_confidence.json |

Reading: the judge is a strong signal on ar25 and uninformative on m0r0,
where it is more confident when wrong than when right. The committee is
informative on both. The two carry different information: their rank-sum
combination reaches AUROC 0.85 pooled, above either alone. The honest
statement is that program disagreement is as good as a frontier-class judge
at zero extra inference, more consistent across levels, and complementary
to it.

## R12. Unseeded self-consistency: are the seed hypotheses load-bearing?

Same synthesizer, prompt and verifier with no seed hypothesis; the first 3
members are the single programs of R3 and R6, 5 more were synthesized.
Compared with the seeded committee of 8 on the same splits.

| | ar25 seeded | ar25 unseeded | m0r0 seeded | m0r0 unseeded |
|---|---|---|---|---|
| Admitted | 8 of 8 | 8 of 8 | 8 of 8 | 6 of 8 |
| Distinct held-out behaviours | 7 | 7 | 4 | 2 |
| AUROC, equal-weight disagreement | 0.77 | 0.75 | 0.68 | 0.68 |
| Unanimous n, error | 27, 0.30 | 28, 0.32 | 38, 0.16 | 40, 0.18 |
| Vote accuracy / shortest member | 0.48 / 0.48 | 0.48 / 0.64 | 0.77 / 0.77 | 0.75 / 0.75 |

Reading: the calibration comes from the execution-defined entropy over
verified programs, not from the seeds; unseeded resampling matches the
seeded committee on ar25 and on m0r0's AUROC. The seeds' measurable effects
are on m0r0: admission 8 of 8 against 6 of 8 and 4 behaviours against 2.
On ar25 the shortest unseeded program is also the best (0.64), but the
weights are split between it and another short program, so the vote stays
at 0.48.

## R13. Mechanism-level calibration: committee entropy per effect row

Each (type, action, context) row touched by the held-out transitions gets
the committee's row entropy η_committee (R4) and the share of its held-out
objects whose plurality effect is wrong. Compared with OPINE-World's
count-based η on the rows that have train counts. Function:
`committee.committee.row_calibration`; files `row_calibration.json`.

| | ar25 L3 | m0r0 L3 |
|---|---|---|
| Rows touched by held-out transitions (unseen in train) | 90 (51) | 49 (7) |
| AUROC, η_committee predicts "row has an error", all rows | 0.75 | 0.72 |
| Same, unseen rows only | 0.61 | too few errors |
| Seen rows: η_counts vs η_committee | 0.46 vs 0.96 | 0.48 vs 0.69 |
| Zero-entropy rows: n, mean error rate | 47, 0.05 | 20, 0.00 |
| High-entropy rows (η > 0.5): n, mean error rate | 37, 0.16 | 27, 0.05 |

Reading: on the rows where OPINE-World's count-based η is defined it is at
chance on both levels, because its counts come from train and the errors
are on test. The committee's row entropy ranks the rows with errors well,
and rows where the members agree are almost error-free. This is uncertainty
over a mechanism, not only over a transition.

## R14. Which simplicity measure tracks held-out accuracy?

All 45 admitted programs on ar25 L3 and m0r0 L3 (seeded, unseeded and
targeted conditions), Spearman correlation between a complexity measure
and held-out accuracy. File: `artifacts/prior_comparison.json`.

| Measure | ar25 (23 programs) | m0r0 (22 programs) | Rank of the best program by this measure, 1 = simplest |
|---|---|---|---|
| gzip length of stripped source (our prior) | −0.22 | −0.07 | ar25 21 of 23; m0r0 5 of 22 |
| AST node count | −0.16 | −0.01 | 18; 5 |
| Branch count (if, loops, boolean ops, comprehensions) | −0.02 | +0.05 | 22; 1 |
| Literal mass | +0.03 | −0.19 | 17; 15 |
| Lines | −0.18 | +0.01 | |

Reading: no measure correlates with generalisation on this data, and on
ar25 the best program (0.64) is among the longest by every measure. The
"shortest consistent program" prior has no support here; the MDL weight
should be treated as an untested choice, and the vote reported with equal
weights unless a prior is validated.

## R15. Open-loop rollouts and level progression

Each member predicts the held-out trajectory from its own previous
prediction with the real actions, resynchronised to the observed state only
at the first step and at a level change. Program memory persists across
the level boundary. `committee.rollout`; files `rollout_*.json`.

| | horizon | n | vote accuracy, teacher forced → open loop | members exactly right, open loop | disagreement, teacher forced → open loop |
|---|---|---|---|---|---|
| ar25 L3 | 2-5 | 4 | 1.00 → 1.00 | 5.0 of 8 | 0.08 → 0.32 |
| | 6-10 | 5 | 0.40 → 0.40 | 2.0 of 8 | 0.35 → 0.44 |
| | 11+ | 34 | 0.41 → 0.00 | 1.1 of 8 | 0.10 → 0.54 |
| m0r0 L3 | 2-10 | 9 | 1.00 → 1.00 | 8.0 of 8 | 0.00 → 0.00 |
| | 11+ | 34 | 0.71 → 0.62 | 4.9 of 8 | 0.08 → 0.07 |
| tr87 L1 trained, rolled through L2 then L3 | all | 53 | 1.00 → 1.00 | 8.0 of 8 | 0.00 → 0.00 |

| Item | Value |
|---|---|
| Steps until a member's first open-loop miss | ar25: 2 or 8 per member; m0r0: 32 for all; tr87: none in 53 steps |
| Command | `uv run python -m committee.rollout ar25 --level 3 --train-frac 0.4`; `uv run python -m committee.rollout tr87 --level 1 --train-frac 1.0 --test-level 2 --test-levels 2 3` |

Reading: on ar25 the programs diverge from the true trajectory within a few
steps and from each other, so open-loop disagreement grows with the horizon
while the teacher-forced disagreement does not; that growth is the quantity
a planner would read as its trust horizon. On m0r0 and tr87 the committee
is stable in open loop, and on tr87 the level-1 rule carries hidden state
through two further levels.

## R16. Mixed-row disentanglement: OPINE-World's enumerator vs committee predicates

A: OPINE-World's context-feature enumerator, reimplemented (target fields,
pixel hash, neighbour at each offset, neighbourhood radius, click offset,
plus fields of other object types), scored by the Dirichlet entropy drop of
the row's training transitions, accepted at Δη ≥ 0.05 with one identified
sub-stratum. B: perturb one feature of the before state and ask each
admitted program again; a feature that flips a member's predicted effect is
a condition that member uses. Both scored by the same Δη and by the
entropy of the row's held-out effects after conditioning.
`committee.disentangle`; files `disentangle.json`.

| | ar25 L3 | m0r0 L3 |
|---|---|---|
| Mixed rows in train (size) | 10 (2 to 4 transitions) | 12 (2 to 8) |
| Rows resolved by Δη criterion: A / B | 0 / 0 | 0 / 0 |
| Held-out row entropy: unconditioned / after A / after B | 0.89 / 0.90 / 0.96 | 0.72 / 0.73 / 0.73 |
| Rows where all 8 members flip on the same feature | 7 of 10 (`player.y`, `player.x`, neighbour presence) | 6 of 12 |
| Rows where members condition on different feature sets | 3 rows with 2 or 3 distinct sets | 4 rows with 4 to 8 distinct sets, e.g. `player ACTION3 [wall]`: 7 distinct sets among 8 members |

Reading: negative for the entropy criterion, informative for attribution.
With 25 to 30 training transitions a mixed row holds 2 to 8 observations;
splitting it into strata of one or two cannot move a Dirichlet entropy with
a pseudo-count of 0.5 over 8 to 10 effects, so neither A nor B passes
OPINE-World's acceptance test, and held-out entropy does not fall either.
The perturbation side still answers a different question: it names the
condition the programs use (on ar25, the player's position for reflection
and target rows, which is the mirror mechanic) with 8 of 8 members
agreeing, and it exposes rows where members disagree about the mechanism
itself, which the transition-level disagreement does not show. Both
methods would need a predicate form (thresholds, relations) rather than raw
value strata to validate a split at this data size.

## R17. sk48 level 2, 40% train: third informative level

Backend Devin. 45 train, 68 test. Chosen before any result was seen, as an
out-of-sample level for the judge combination (R11) and for every
transition- and row-level claim.

| | Value |
|---|---|
| Single programs, held-out accuracy | 1.00, 0.63, 0.97 (mean 0.87) |
| Seeded committee of 8, members | 0.60 to 0.97, mean 0.74; 6 distinct behaviours; 8 of 8 admitted |
| Vote accuracy | 0.69 equal weights; 0.76 with the MDL prior |
| Unanimous transitions, error | 41, 0.00 |
| Split transitions, error | 27, 0.59 (low 2 at 0.00, medium 20 at 0.55, high 5 at 1.00) |
| AUROC, disagreement vs error | 1.00 |
| Rows touched by held-out transitions | 28, 3 unseen in train |
| Row-level AUROC: η_committee vs η_counts on seen rows | 1.00 vs 0.22 |
| Zero-entropy rows: n, error | 18, 0.00 |
| Open loop | all 8 members exact for 8 steps, first miss at step 9; vote 0.64 → 0.00 beyond 10 steps; disagreement 0.22 → 0.29 |
| Commit | see the commit that adds artifacts/sk48 |

Reading: the strongest case so far for the uncertainty claim: every
unanimous prediction is right, every high-disagreement prediction is
wrong, and OPINE-World's count-based η is anti-informative on the same
rows. The accuracy picture is the opposite of ar25: here two of three
single programs beat the committee's vote, and the MDL prior would have
helped. Uncertainty is the robust benefit; accuracy is not.

## R18. Judge plus disagreement, out of sample

The rank-sum combination of R11 was fixed before sk48 L2 was run and is
scored here on that level only. Judge: gpt-oss-120b, 68 calls.

| sk48 L2, 68 transitions | AUROC |
|---|---|
| Judge verbalized confidence | 0.52 (values used: 30 and 75; error 0.26 vs 0.22) |
| Committee disagreement | 1.00 |
| Rank-sum combination | 0.99 |

Reading: the combination does not hold out of sample. Its pooled 0.85 in
R11 came from ar25, where the judge happened to be strong; on m0r0 and
sk48 the judge is at chance, and adding it to the disagreement can only
dilute it. Across the three levels the judge scores 0.82, 0.46 and 0.52;
the committee 0.77, 0.68 and 1.00. The judge is dropped from the method and
kept as a reported baseline.

## R19. Shared-mechanism library extracted from the ar25 L3 committee (opt-in)

One Devin session read the 8 admitted programs, named their mechanisms,
wrote `library.py`, mapped each program to the mechanisms it implements and
refactored three programs onto the library with exact replay preserved.
`committee.library`; files under `artifacts/ar25/L3_f40/committee_devin/library/`.

| Item | Value |
|---|---|
| Mechanisms named | 15 |
| Implemented by all 8 members | 12 (action decode, selection cycle, piece move, axis move, mirror reflection, selection marker, layered composite, component re-extraction, wall index offset, static passthrough, stateless recovery, continuity memo) |
| Contested | axis orientation (2 of 8), click select (1 of 8), undo history (1 of 8) |
| Disputed stances the library exposes as options | piece blocked by the axis: 6 members yes, 2 no. Axis blocked by pieces: 5 yes, 3 bounds-only. Hole compositing order: top-first (4) vs low-to-high (3). The train buffer contains no transition that decides any of them |
| Refactored members, lines before → after | 208 → 17, 234 → 18, 216 → 17, each still 29/29 on train; library 373 lines |
| Mechanism count per member vs held-out accuracy | counts 12 to 14; Spearman −0.17, no signal |
| Session | 1, about 10 min |

Reading: the library turns the committee's transition-level disagreement
into named, inspectable mechanism options, which is the mechanism-level
uncertainty the design asked for. As a prior, mechanism count carries no
more information than length did (R14): members differ in two or three
optional mechanisms, not in size. Whether the library helps a synthesizer
on a new level with little data is R20.

## R20. Library-conditioned synthesis at low data: inconclusive

ar25 level 6, first 12 transitions train, 42 test. `lib_devin`: the level-3
library (R19) and its mechanism list in the seed. `nolib_devin`: no seed.
Same synthesizer, 4 programs per arm; one no-library run was lost to a
local network error, so that arm has 3.

| | no library (3) | library (4) |
|---|---|---|
| Admitted | 3 of 3 | 4 of 4 |
| Member held-out accuracy | 0.79, 0.24, 0.24; mean 0.42 | 0.67, 0.60, 0.60, 0.10; mean 0.49 |
| Vote accuracy | 0.26 | 0.60 |
| AUROC, disagreement vs error | 0.81 | 0.58 |
| Wall per program | 270 s | 255 s |

Reading: with 3 and 4 programs the arms are not separable. The library arm
has a higher mean and vote because three of its members share one
behaviour, which also lowers its disagreement signal; the no-library arm
holds the single best program. No sample-complexity benefit is shown, and
the library stays opt-in. A decisive test needs about 8 per arm on two
levels, roughly 32 sessions.

## R21. ar25 level 7, 40% train: fourth informative level

44 train, 65 test. 8 of 8 seeded programs admitted; two of three baseline
programs returned (0.40, 0.35), the third session hung and was stopped.

| | Value |
|---|---|
| Members, held-out accuracy | 0.40 ×4, 0.415 ×4; vote 0.415; 6 distinct behaviours |
| Unanimous transitions, error | 25, 0.00 |
| Split transitions, error | 40, 0.95 (low 2 at 0.00, medium 38 at 1.00) |
| AUROC, disagreement vs error | 1.00 |
| Rows touched by held-out transitions | 267, 186 unseen in train |
| Row-level AUROC, η_committee | 0.68 all rows, 0.68 unseen rows; on seen rows η_counts 0.68 vs η_committee 0.66 |
| Zero-entropy rows: n, error | 127, 0.04 |
| Open loop | first miss at step 3 for every member; disagreement 0.35 → 0.60 beyond 10 steps |

Reading: the hardest level so far and the same transition-level pattern,
with every unanimous prediction right and 38 of 40 split predictions wrong.
At the row level the count-based η is as informative as the committee's
here, the one level where that holds; the committee still scores the 186
rows that have no counts.

## Summary across the informative levels

| Split | n test | Unanimous: n, error | Split: n, error | AUROC | Row AUROC committee vs counts (seen rows) |
|---|---|---|---|---|---|
| ar25 L3 | 44 | 27, 0.30 | 17, 0.88 | 0.77 | 0.96 vs 0.46 |
| m0r0 L3 | 44 | 38, 0.16 | 6, 0.67 | 0.68 | 0.69 vs 0.48 |
| sk48 L2 | 68 | 41, 0.00 | 27, 0.59 | 1.00 | 1.00 vs 0.22 |
| ar25 L7 | 65 | 25, 0.00 | 40, 0.95 | 1.00 | 0.66 vs 0.68 |
| tr87 L1→L2, ft09 L5 | 28, 31 | all, 0.00 | 0 | | |

## R22. Calibration battery and adaptive conformal sets

Inputs: per held-out step, the members' predicted next states and the
observed one. `committee.calibrate`; file `artifacts/calibration.json`.
Four informative levels, 221 steps.

**Vote share as a probability.** Pooled ECE 0.22, Brier 0.18. Reliability
by equal-mass bins (mean share → accuracy): 0.38 → 0.09, 0.66 → 0.16,
0.99 → 0.80, 1.00 → 0.86, 1.00 → 1.00. The share is overconfident wherever
the committee splits and slightly overconfident when unanimous.

**Selective prediction by disagreement, pooled.** Overall accuracy 0.58;
accuracy on the 50% most confident steps 0.87, on the 80% most confident
0.71; AURC 0.22.

**Calibration map fit on the other levels (isotonic on vote share, leave-one-level-out).**

| Held-out level | ECE raw → mapped | Brier raw → mapped | mapped P(correct) when unanimous |
|---|---|---|---|
| ar25 L3 | 0.42 → 0.33 | 0.39 → 0.36 | 0.94 |
| m0r0 L3 | 0.17 → 0.19 | 0.17 → 0.15 | 0.91 |
| sk48 L2 | 0.10 → 0.18 | 0.11 → 0.09 | 0.84 |
| ar25 L7 | 0.26 → 0.12 | 0.13 → 0.04 | 0.87 |

**Adaptive conformal inference along each trajectory** (Gibbs and Candès
2021). Nonconformity = 1 − share of the realised outcome; threshold from the
quantile of past scores; α_t moves by γ after each hit or miss; target
coverage 0.90. A set that must include "anything else" is an abstention.

| Level | coverage (γ 0.05 / 0.02) | mean set size | abstain rate | singleton rate, accuracy |
|---|---|---|---|---|
| ar25 L3 | 0.955 / 0.955 | 2.4 | 0.84 | 0.14, 0.83 |
| m0r0 L3 | 0.909 / 0.932 | 1.5 | 0.23 / 0.25 | 0.77, 0.88 / 0.91 |
| sk48 L2 | 0.882 / 0.882 | 1.5 | 0.09 / 0.02 | 0.66 / 0.69, 0.93 / 0.89 |
| ar25 L7 | 0.985 / 0.985 | 3.6 | 0.97 | 0.02, 1.00 |
| pooled | 0.932 / 0.937 | | | |

**Good-Turing missing mass** (singleton members / K) against the unanimous
error per level: 0.05 vs 0.30, 0.01 vs 0.16, 0.01 vs 0.00, 0.15 vs 0.00. No
relation.

**Agreed-but-wrong flag** from the entropy of the touched effect rows,
among unanimous steps: ar25 L3 AUROC 0.87 (8 wrong of 27); m0r0 L3 0.41
(6 of 38); sk48 and ar25 L7 have no agreed-but-wrong steps.

Reading: the raw vote share is not a probability, and a map fit on other
levels helps on two levels and hurts on two, because the level-to-level
drift is larger than the map. Adaptive conformal sets hold the 90% target
on every level (0.88 to 0.99, pooled 0.93) using only outcomes the agent has
already seen, which is the distribution-free sense of calibrated. The price
is informativeness where the committee is mostly wrong: on ar25 L3 and L7
the realised outcome is outside every member's prediction on most steps, so
a set that keeps coverage must abstain 84% and 97% of the time. On m0r0 and
sk48 it abstains 9% to 25% of the time and answers with a single state 66%
to 77% of the time at 0.88 to 0.93 accuracy. The abstention rate is itself
the honest report: it says how often the agent should distrust every
hypothesis it holds. Good-Turing does not estimate the agreed-but-wrong mass;
row entropy flags it on one level only.

## H1. Hoeffding's problem, baselines on all instances

Task: sup P(S_n <= t) over iid X in [0, 1] with E X = m. Each number is a
certified lower bound from a discrete measure, computed in exact rational
arithmetic (`src/hoeffding/verify.py`). Instances: n in {3, 4, 6} (train, 27)
and n in {1, 2, 5, 8, 10} (test, 45), m in {1/5, 2/5, 3/5}, t in
{1/4, 1/2, 3/4} x n m. Reference values are exact for n <= 2 (Markov; Meester
2008) and "best known here, not proven" for n >= 3.

| Item | Train (27) | Test (45) |
|---|---|---|
| Family best beats Bernoulli(m) wall | 27/27 | 39/45 (the 6 ties have integer t, where Bernoulli is extremal) |
| Median lift over Bernoulli value | 1.78x | 1.40x (max 3.47x) |
| Naive search within 1e-3 of family best | 25/27 | 40/45 |
| Naive search above family best | 0/27 | 0/45 |
| Ternary {0, a, 1} beats every binary law | 1/27 | 1/45 |
| Mean honest-unknown width, Hoeffding wall minus certified | 0.260 | 0.224 |
| Mean fraction of the Bernoulli-to-Hoeffding gap still open | 0.66 | 0.70 |
| Certificate time, n <= 10 | < 1 ms | < 1 ms |
| Certificate time, n = 100, 4 atoms | 1.2 s | |
| Runs | 1 (search seed 0, 10 restarts) | |
| Command | `uv run python -m hoeffding.experiment baselines --restarts 10` | |
| Commit | uncommitted, base a718512 | |

Per-instance values: `artifacts/hoeffding/baselines.json`.

## RH2. Memorisation detectors, calibration

Note, 2026-10-03 22:00: the synthesized programs this calibration used were object-only (E1) runs, removed in the reset below. The thresholds stand; the constructed lookup tables are reproducible from the command.

Features on the normalised program source: literal mass (share of source
inside literal constants), MDL ratio (program description length over train
data description length), held-out gap, order dependence (train failures
when replayed in reverse). A program is flagged memorising when literal mass
>= 0.5 and MDL ratio >= 0.5.

| Program class | Literal mass | MDL ratio | Held-out gap |
|---|---|---|---|
| Lookup table built from the train set, tr87 L1 / L6, ar25 L3 | 0.996 to 1.000 | 1.02 to 1.33 | 0.25 / 0.05 / 1.00 |
| Synthesized rules, 19 programs in RH1 plus 4 committee baseline runs | 0.07 to 0.21 | 0.21 to 0.30 | 0.00 |

| Item | Value |
|---|---|
| Flagged memorising | 0/23 synthesized programs, 3/3 lookup tables |
| Committee weight on memorising members | 0.000 (tr87 L1 and L6 baseline committees) |
| Runs | 23 programs, 3 constructed tables |
| Command | `uv run python -m rewardhack.report score` |
| Commit | uncommitted, base a718512 |

Note: the held-out gap alone is weak on tr87 L6 (0.05 for a table) because
most test states repeat train states. Literal mass separates the classes
by a factor of 5 on every level measured.

## H2. Hoeffding's problem, synthesized strategy committee

Four `claude -p` runs (model opus), one per seed hypothesis: no hint, binary
family, ternary family, free search. Each produced a rule `strategy(n, m, t)`
on the 27 train instances (n in {3, 4, 6}) and was then run on the 45 test
instances (n in {1, 2, 5, 8, 10}), which never entered a workspace. Every
candidate is certified in exact arithmetic; the committee value is the best
certified value, weighted by MDL as in the ARC committee.

| Item | Value |
|---|---|
| Members certified on every test instance | 4/4 (45/45 each) |
| Committee value equals the proven optimum, n <= 2 | 18/18 |
| Committee value equals the best known value, n >= 3 | 27/27 (reference is the family best, not proven) |
| Committee value above the family best | 0/45 |
| Committee disagreement (leader entropy), max over instances | 0.12 |
| Train calibration, Brier of p_tight vs hitting the reference | none 0.053, binary 0.109, ternary 0.104 (hit rate 1.00, mean p 0.68 to 0.78: under-confident) |
| Test calibration | not scored: run0 reported confidence as a table over train keys. Fixed by `confidence(n, m, t)` in the contract; run1 in progress |
| Synthesis cost | none 34 turns $1.10 438 s; binary 22 turns $0.83 286 s; ternary 10 turns $0.38 128 s; search timed out at 1200 s (strategy still certified everywhere) |
| Runs | 1 per seed |
| Command | `uv run python -m hoeffding.experiment synth --runs 1` then `uv run python -m hoeffding.experiment report` |
| Commit | uncommitted, base a718512 |

Reading: a rule learned on n in {3, 4, 6} reproduces Meester's proven n = 2
optimum and Markov's n = 1 optimum on every unseen instance. All four seeds
converge on the same structure (two atoms with k b + (n - k) a = t, or
{0, a, 1}), so the open question for n >= 3 is not the law but the proof.
The reported unknown is the Hoeffding wall minus the certified value, mean
0.22 on test. Per-instance values: `artifacts/hoeffding/report.json`.

## H3. Hoeffding's problem, calibration of self-reported confidence on unseen instances

Second synthesis run per seed under the contract that requires
`confidence(n, m, t)`, scored on test instances the member never saw.
`tight` means the member's certified value is within 1e-4 of the reference.
Brier = mean (p - tight)^2, lower is better.

| Member | n <= 2, reference is a proof (18) | n >= 3, reference is the family best, not proven (27) |
|---|---|---|
| binary/run1 | Brier 0.003, mean p 0.95, hit 1.00 | Brier 0.428, mean p 0.35, hit 1.00 |
| ternary/run1 | Brier 0.010, mean p 0.90, hit 1.00 | Brier 0.177, mean p 0.61, hit 1.00 |
| none/run1 | Brier 0.040, mean p 0.80, hit 1.00 | Brier 0.175, mean p 0.59, hit 1.00 |
| search/run1 | timed out at 900 s; stub strategy with p = 0; Brier 0.000, hit 0.00 | Brier 0.222, p 0, hit 0.22 (Bernoulli is extremal at integer t) |

| Item | Value |
|---|---|
| Members certified on every test instance | 8/8 (run0 and run1) |
| Runs | 2 per seed |
| Synthesis cost, run1 | none 18 turns $0.58 172 s; ternary 19 turns $0.63 348 s; binary and search timed out at 900 s (binary's strategy still certified everywhere) |
| Command | `uv run python -m hoeffding.experiment synth --runs 2` then `uv run python -m hoeffding.experiment report` |
| Commit | uncommitted, base a718512 |

Reading: on instances where a proof exists, the members state 0.80 to 0.95
and are right every time. On open instances the same members state 0.35 to
0.61. They separate the proven regime from the open one without being told
which is which. Their low score "against the reference" on n >= 3 is not an
error: the reference is unproven, so under-confidence there is the honest
position. Per-instance values: `artifacts/hoeffding/report.json`.

## H4. Hoeffding's problem, do parallel members diverge on harder instances?

The four run0 strategies, developed on n in {3, 4, 6}, run on 36 unseen
instances with n in {15, 20, 30}, m in {1/5, 2/5, 3/5}, t in
{1/4, 1/2, 3/4, 9/10} x n m. Reference is the family best (not proven).

| Item | Value |
|---|---|
| Members certified on every instance | 3/4 (binary, ternary, search). `none` timed out at 900 s for all 36: its search does not scale |
| Instances where certified members differ by more than 1e-4 | 0/36 |
| Committee value above the family best | 0/36 |
| Committee value below the family best | 0/36 |
| Member run time for 36 instances | ternary 37 s, search 71 s, binary 126 s |
| Runs | 1 |
| Command | inline script, saved values in `artifacts/hoeffding/hard_run0.json` |
| Commit | uncommitted, base a718512 |

Reading: on this task the committee's disagreement carries no information up
to n = 30. The extremal structure is simple enough that every member that
finishes finds the same law, and the hand-built families find it too. The
only committee benefit observed is availability: one member's timeout was
covered by the others. The uncertainty this task measures honestly is the
wall gap and the members' self-reported confidence (H3), not disagreement.

## H5. Is there a slope to climb? Certifier check on Problem 6.39 (Bellec-Fritz)

sup over iid nonnegative laws of P(X1 + X2 + X3 < 2 X4). Same exact
convolution certificate as Hoeffding's problem. Values below are certified
from explicit discrete laws, except the ratio column, which is the value the
Bellec-Fritz eta-perturbation reaches in the limit.

| Law | Certified value | Note |
|---|---|---|
| Two atoms {0, 1}, best p | 0.2500 | naive start |
| Bellec-Fritz nu, N = 4 atoms at 1 - 2^-i | 0.3539 strict, 0.3788 in the limit | |
| N = 8 | 0.3679 strict, 0.3890 in the limit | AlphaEvolve's naive run reached 0.389 |
| N = 16 | 0.3742 strict, 0.3945 in the limit | |
| Explicit two-level law, N = 6, eta = 1/1000, 49 atoms | 0.3843 | no limit argument, certified as is |
| Closed-form supremum of the construction | 0.400695 | Bellec-Fritz, conjectured exact |
| Upper bound | 0.417 | Bellec-Fritz, mixed-integer LP |

Certificate time: 0.03 s for 17 atoms, 0.6 s for 49 atoms.
Command: inline script, this section. Commit: uncommitted, base a718512.

Reading: on this problem a naive law sits at 0.25, explicit certified laws
reach 0.38 to 0.39, the known construction reaches 0.4007 only in a limit,
and the proven ceiling is 0.417. That is a documented slope for a climb with
the same certifier, with a published naive-agent result (0.389) to compare
against, and 0.016 of unknown above the best construction.

## H6. Hoeffding's problem, refutation search above the family

Random-restart local search over 4- and 5-atom laws, 3 seeds x 40 restarts x
600 steps per atom count, on all 72 instances, against the family best.

| Item | Value |
|---|---|
| Instances where the search beat the family best by more than 1e-7 | 0/72 |
| Instances where the search found no feasible law | 2 (integer t, where Bernoulli is extremal) |
| Best gain over the family best | 0.0 |
| Runs | 1 (3 search seeds per instance) |
| Command | inline scripts, logs `artifacts/hoeffding/refutation_search*.log` |
| Commit | uncommitted, base a718512 |

Reading: no headroom is found above Meester's families at k <= 5 atoms on
any instance. On this problem the certified bound is at its ceiling as far
as search can tell, so a claim that committee disagreement helps a climb
cannot be made here. The claim moves to Problem 6.39 (H5), where a slope is
documented.

## RH4. A hack found in the committee's own artifacts

`rewardhack.report score` over `artifacts/` flagged one replay-consistent
program: `tr87/L1_f60/api_qwen/run2`, Qwen3-Coder-30B through the
committee's API loop (8 rounds). It passes all 19 train transitions by
enumerating the five cursor columns per action (`if current_x == 15 ... elif
current_x == 43`, 10 layout guards) and scores 0.25 on the 12 held-out
transitions.

| Feature | This program | Claude programs (24) | Lookup table |
|---|---|---|---|
| Literal mass | 0.08 | 0.07 to 0.21 | 0.99 |
| MDL ratio | 0.54 | 0.21 to 0.30 | 1.0 to 1.3 |
| Held-out gap | 0.75 | 0.00 | 0.25 to 1.0 |
| Layout guards | 10 | 0 | 0 |

Literal mass misses it, so a second memorising class, enumerating (>= 5
layout guards), was added. A layout guard is an equality or membership test
against an integer in the coordinate range; ordering tests such as
`0 <= x < 64` are bounds and do not count. With that rule, `report score`
over all 141 committee programs flags exactly this one. The MDL prior alone does not catch it either: as
the only consistent member of its condition it would hold weight 1.000.
With the flag, `hack_weight_mass` for that committee is 1.000, which is the
number the pipeline should refuse.

| Item | Value |
|---|---|
| Runs | 1 program, found among 36 committee programs |
| Command | `uv run python -m rewardhack.report score --no-behavioural` |
| Commit | uncommitted, base a718512 |

## H7. Hoeffding's problem, hill-climb outcome

Outer loop `hoeffding.climb`: 8 seed strategies, then 2 rounds of 2 children,
top 2 parents, opus, 30 turns, 15 min per child, on 39 instances (train plus
n in {15, 20}). Fitness = mean fraction of the Bernoulli-to-Hoeffding gap closed.

| Round | Best fitness | Population | Instances with distinct values | Cost |
|---|---|---|---|---|
| 0 | 0.2516 | 8 | 32 (one failed seed differs) | 0 |
| 1 | 0.2516 | 10 | 32 | both children timed out at 900 s, strategies still certified at 0.2516 |
| 2 | 0.2516 | 12 | 32 | one child 23 turns $0.67 at 0.2516; one timed out with the stub |

Command: `uv run python -m hoeffding.climb --rounds 2 --children 2 --top 2 --max-turns 30 --timeout 900 --eval-timeout 240`.
Commit: uncommitted, base a718512. Log: `artifacts/hoeffding/climb/log.json`.

Reading: flat, as H4 and H6 predicted. The family value is the ceiling and
the loop confirms it; no child moved any instance.

## H8. Linear-inequality family (Problem 6.39), hill-climb from the naive start

Task: C(c) = sup over iid laws of P(sum c_i X_i < 0), exact certificate.
Train c in {(1,1,1,-2), (1,1,1,1,-3), (1,2,-3)}. Outer loop from the stub
law {0, 1}: 3 rounds, 2 children, top 2 parents, opus, 30 turns, 15 min per
child. No seed law and no construction was given beyond the contract's two
sentences about Bellec-Fritz. Fitness = mean fraction of the gap from the
naive two-atom law to the ceiling (0.417 for (1,1,1,-2), 1 otherwise).

| Round | Best child | (1,1,1,-2) | (1,1,1,1,-3) | (1,2,-3) | Fitness |
|---|---|---|---|---|---|
| 0 | stub | 0.2500 | 0.3438 | 0.3750 | 0.000 |
| 1 | child1, 64 atoms | 0.3734 | 0.4477 | 0.6340 | 0.428 |
| 2 | child0, 30 to 64 atoms | 0.3915 | 0.4693 | 0.6596 | 0.492 |
| 3 | child0, 30 to 64 atoms | 0.3976 | 0.4723 | 0.6596 | 0.506 |

Landmarks on (1,1,1,-2): naive published agent (AlphaEvolve, hours) 0.389;
Bellec-Fritz construction 0.400695, reached only as a limit; proven ceiling
0.417. Round 2 passed the first landmark; round 3 is 0.003 under the
construction's limit with an explicit 53-atom law.

Held out, never in a workspace (round-3 child):

| c | Certified | Known |
|---|---|---|
| (2,-1,-1) | 0.6596 | 2/3, proven |
| (1,1,-2) | 0.6596 | none published |
| (1,1,1,1,1,-4) | 0.5312 | none published |
| (2,1,1,-3) | 0.4232 | none published |

| Item | Value |
|---|---|
| Structure found (round 3 header) | centres z/2 plus signed perturbations r^j on a ladder, ties broken lexicographically; the mass at 0 is a scaled copy of the whole law. This is the Bellec-Fritz tie-breaking construction plus the self-similarity Fritz conjectures |
| Confidence | 0 on every instance: every climbing child hit the 15 min timeout before writing `confidence`. Not scored |
| Members certified on every train instance | 6/6; one child per round timed out with the stub |
| Cost | about $1.2 to $1.5 per finished child; 3 of 6 children timed out at 900 s and still certified |
| Runs | 1 |
| Command | `uv run python -m hoeffding.climb --task linear --rounds 3 --children 2 --top 2 --max-turns 30 --timeout 900 --eval-timeout 300` |
| Commit | uncommitted, base a718512. Logs: `artifacts/hoeffding/climb_linear/`, `climb_linear_test.json` |

## H9. Two-tier verification: certify only what can change the best

Policy: float bracket [estimate, estimate + tie mass + rounding margin];
certify when the bracket can exceed the incumbent certified best by more
than 1e-9, skip otherwise, audit 10 percent of skips. Replayed over the
cached candidate laws of every climb member, both arms on identical laws.

| Item | Hoeffding climb | Linear climb |
|---|---|---|
| Members x instances requested | 12 x 39 = 468 | 6 x 3 = 18 |
| Exact certifications performed (incl. audits) | 84 (18 percent) | 17 (94 percent) |
| Skipped | 384 (82 percent) | 1 |
| Audits where the exact value left the bracket | 0 of 45 | 0 of 0 |
| Certified best per instance identical to full certification | yes, max diff 0.0 | yes, max diff 0.0 |
| Exact tier seconds, full arm vs policy arm | 0.03 vs 0.02 (certificates cost ms here) | 48.6 vs 50.3 |
| Exact over float cost, stress (H5 sizes, 16 to 64 atoms) | | 6x to 20x; 80 s exact at 32 random atoms, 6-term c |
| Runs | 1 | 1 |
| Command | inline script, `artifacts/hoeffding/tiers_replay.json` | |
| Commit | uncommitted, base a718512 | |

Reading: the policy removes 82 percent of certifications on a plateaued
climb and none early in a climb, which is the right shape: a candidate that
beats the incumbent must be certified, one that can only tie need not be.
No bound changed and no audit disagreed. The seconds saved are small in
these two replays because the Hoeffding certificate is cheap and the linear
climb was still improving on every round. The same policy now runs inside
the child's checker, where a child re-certifies on every iteration; the
per-run counts of certified versus estimated evaluations are logged from
the next round on.

## H10. Hoeffding's problem, exhaustive support search on the arithmetic grid

Method. For each instance, the grid G = {0, 1} U {(t - l)/k : 1 <= k <= n,
0 <= l <= t, 0 < (t - l)/k < 1}, the positions Meester's necessary
conditions point to. Every support of 2, 3 or 4 atoms from G with
min < m < max is enumerated. With atoms fixed the objective is a polynomial in
the weights, so the weights are optimized by a grid (2001 points for 3 atoms,
41 x 41 for 4 atoms) plus shrinking random refinement around the top 3 grid
points. The float winner is rationalized and certified exactly, then compared
with the family best.

| Item | Value |
|---|---|
| Instances | 63 (all with n <= 8: train n in {3, 4, 6}, test n in {1, 2, 5, 8}) |
| Supports enumerated | 32,906 (max grid size 26 points) |
| Certified grid best above the family best | 0 of 63 |
| Certified grid best equal to the family best, within 1e-7 | 63 of 63 (max difference 2.6e-13) |
| Float winner above the family before certification | 0 of 63 |
| Runs | 1 |
| Command | inline script, `artifacts/hoeffding/grid_exhaustive.{log,json}` |
| Commit | uncommitted, base a718512 |

Limits of the earlier searches (H6 and the search baseline). The objective
is discontinuous in atom positions because ties count, so a float local
search cannot land on the arithmetic positions and sat just below the family
on nearly every instance. Those negatives were weak. H10 fixes the atoms on
the grid, where the search recovers the family optimum exactly on every
instance, which is the check that it can find what is known.

What H10 does not cover: supports of 5 or more atoms, atoms off the grid, and
n > 8. Within its scope it is exhaustive over supports and reliable in the
weights. Together with Meester's n = 2 theorem it is the evidence that the
family is the ceiling, not a proof.

## H11. Linear-inequality family, rounds 5 and 6 with 30-minute children

Resumed from H8 with timeout 1800 s, 40 turns, `confidence` requested in
the first edit, the two-tier policy inside the child's checker, and two-tier
scoring outside. Rounds are numbered 5 and 6 because the resume counts the
H8 log as rounds 0 to 3.

| Round | Best child | (1,1,1,-2) | (1,1,1,1,-3) | (1,2,-3) | Turns, wall, cost |
|---|---|---|---|---|---|
| 3 (H8) | child0 | 0.397607 | 0.472323 | 0.659631 | timed out at 900 s |
| 5 | child1 | 0.397830 | 0.472571 | 0.659631 | 41 turns, 1073 s, $2.12; child0 28 turns, 668 s, $1.23 |
| 6 | child1 | 0.398027 | 0.472682 | 0.659631 | 30 turns, 563 s, $0.94; child0 24 turns, 552 s, $0.95 |

All four children finished inside the budget and defined `confidence`.
Landmarks on (1,1,1,-2): 0.389 naive published agent, 0.400695 construction
limit, 0.417 ceiling. The climb is converging to the limit from below at
about +0.0002 per round, which is the finite-ladder loss the children's own
headers describe.

Held out, never in a workspace, round-6 child1 and the stated confidence
that the value is within 1e-4 of the supremum:

| c | Certified | Stated p | Known | Tight? |
|---|---|---|---|---|
| (2,-1,-1) | 0.659631 | 0.01 | 2/3 proven | no, and the member said so |
| (1,1,-2) | 0.659631 | 0.01 | none | unknown |
| (1,1,1,1,1,-4) | 0.531904 | 0.01 | none | unknown |
| (2,1,1,-3) | 0.423694 | 0.01 | none | unknown |

Calibration on the one proven anchor: stated 0.01 to 0.02 across the four
finished children, hit rate 0, Brier 0.0001 to 0.0004. One anchor is not a
calibration curve; it is one honest low-confidence report that was right.

Outer two-tier scoring: 18 of 18 certifications performed, since every
child beat or tied its parents; 0 skips. The checker-side counts were not
captured for these rounds because the resumed process predates that
instrumentation. Runs: 1. Command: `uv run python -m hoeffding.climb --task
linear --rounds 2 --children 2 --top 2 --max-turns 40 --timeout 1800
--eval-timeout 600 --resume --tiers`. Commit: uncommitted, base a718512.

## H12. A/B: does the checker-side certification policy make rounds faster?

Round 7 of the linear climb, same two parents, two children with the policy
(the checker certifies only a law that can beat the child's best so far)
and two without (every evaluation certified), 30-minute budget, opus.
Fitness re-scored with full certification after a bracket defect (below).

| Child | Checker policy | Checker runs | Certified / estimated evaluations | Certify s | Strategy s | Turns | Wall s | Cost | Fitness | (1,1,1,-2) |
|---|---|---|---|---|---|---|---|---|---|---|
| c0 | off | 1 | 3 / 0 | 23 | 5 | 41 | 716 | $1.79 | 0.5073 | 0.398027 |
| c1 | off | 3 | 9 / 0 | 53 | 10 | 32 | 654 | $1.35 | 0.5077 | 0.398198 |
| p0 | on | 3 | 8 / 1 | 76 | 16 | 30 | 637 | $1.21 | 0.5075 | 0.398116 |
| p1 | on | 1 | 3 / 0 | 24 | 5 | 41 | 1541 | $1.61 | 0.5073 | 0.398027 |

Result: no speed-up, and the premise was wrong. Children ran the checker 1
to 3 times in 11 to 26 minutes; certification inside the checker was 23 to
76 s per child, 2 to 12 percent of wall time. The budget goes to model turns
and to the strategies' own searches, which the agents run in their own
Python calls, outside the checker. Rounds get faster by fewer turns or
parallel children, not by cheaper certification.

Defect found and fixed. The outer two-tier scoring skipped a candidate on
c = (1,2,-3) whose exact value was 0.6596 while the float bracket's upper end
was 0.5053: the bracket did not contain the exact value. Cause: the laws
break ties with perturbations below float resolution, so float sums that
come out exactly 0 are true negatives, and the rule that treated an exact
float 0 as an exact tie dropped that mass. The 10 percent random audit on
3 instances sampled nothing and missed it. Fixes: a float 0 is uncertain mass
again (upper end now 0.8594 on that law, exact value contained), and the
first skip on each instance in a scoring call is always audited. Test fails
when the old rule is restored.

Runs: 1 round, 2 children per arm. Commands: `uv run python -m hoeffding.climb
--task linear --rounds 1 --children 2 --top 2 --max-turns 40 --timeout 1800
--eval-timeout 600 --resume --tiers --tag p` and the same with `--tag c
--no-checker-policy`. Commit: uncommitted, base a718512. Per-child numbers:
`artifacts/hoeffding/ab_round7.json`.

## R23. Oracle headroom and probes as selection

Inputs: the stored committee members on the four informative levels, 40%
train, condition `committee_devin`. `committee.selection`; file
`artifacts/selection.json`.

**Headroom.** A held-out transition counts for the oracle when any member
predicts it. The oracle bounds every weighting of the members.

| Level | K | held-out | mean member | best member | equal-weight vote | oracle any-right |
|---|---|---|---|---|---|---|
| ar25 L3 | 8 | 44 | 0.48 | 0.64 | 0.48 | 0.64 |
| m0r0 L3 | 8 | 44 | 0.76 | 0.77 | 0.77 | 0.77 |
| sk48 L2 | 8 | 68 | 0.74 | 0.97 | 0.69 | 0.97 |
| ar25 L7 | 8 | 65 | 0.41 | 0.42 | 0.42 | 0.42 |

On every level the oracle equals the best member: no transition is
predicted by a weaker member alone. No combination rule can beat the best
member. The vote loses to it by 0.16 on ar25 L3 and 0.28 on sk48 L2, and
matches it on m0r0 L3 and ar25 L7.

**Probes as selection.** Observe one held-out transition, drop the members
that mispredicted it, score the survivors' equal-weight vote on the
transitions not yet observed. The best survivor is chosen in hindsight.

Disagreement order: on all four levels the first probe falsifies every
member, because the observed outcome is outside every member's prediction.
This order selects nothing; it reports that a hypothesis is missing.

Random order, vote accuracy of the survivors after n probes, mean over the
orders with at least one survivor (20 orders; the count in parentheses):

| Level | n = 0 | 1 | 2 | 3 | 4 | 5 | best member |
|---|---|---|---|---|---|---|---|
| ar25 L3 | 0.48 | 0.52 (11) | 0.57 (7) | 0.58 (5) | 0.56 (4) | 0.59 (3) | 0.64 |
| sk48 L2 | 0.69 | 0.75 (20) | 0.77 (19) | 0.79 (16) | 0.85 (15) | 0.86 (15) | 0.97 |
| m0r0 L3 | 0.77 | 0.77 (15) | 0.76 (10) | 0.76 (10) | 0.75 (10) | 0.74 (8) | 0.77 |
| ar25 L7 | 0.42 | 0.41 (10) | 0.40 (1) | 0.39 (1) | | | 0.42 |

Reading: selection by probes works where the members differ and the probe
lands inside the version space. On sk48 L2, five random probes take the
vote from 0.69 to 0.86 with 2.3 members left on average, against the best
member at 0.97. It cannot work where the members agree: on m0r0 L3 and
ar25 L7 a probe falsifies all members or none. The surviving orders are
the ones that probed transitions some member got right, so the random
curve is conditional on survival. The rule for an agent: a probe that
splits the survivors selects among them; a probe that falsifies all of
them calls for new hypotheses, which is the resynthesis step.

Runs: 1, deterministic, with 20 random probe orders. Split: temporal, 40%
train per level. Baseline: equal-weight vote with no probes. Command:
`uv run python -m committee.selection`. Commit: uncommitted, base ee733d0.

## H9 addendum. Replay with the corrected bracket (after H12)

Same cached laws as H9, float 0 counted as uncertain mass. Linear arm, now 14
members: 42 requested, 41 certified, 1 audited, 0 skipped, 0 bracket misses
among certified, best identical. Hoeffding arm: the skip decisions are
unchanged by the fix (an exact tie at t moves from value to tie mass; the
upper end is the same), so 82 percent skipped stands; the replay script's
per-member audit rule turned every skip into an audit, which is a script
artifact, not a policy result. The climb's audit guarantee is now scoped per
round. Log: `artifacts/hoeffding/tiers_replay3.log`.

## H13 setup. Long hill-climb on c = (1,1,1,-2)

Certification time for two-level ladder laws on (1,1,1,-2): 64 atoms 3.6 s,
100 atoms 22 s, 144 atoms 87 s. Atom cap set to 128. Configuration: task
`linear1` (one vector), lanes refine and explore (one child each per round),
parents = best overall plus the best of each lane, 45 min and 60 turns per
child, outer scoring certifies everything, stop at 12 rounds or when the best
fitness gains under 1e-5 over 3 rounds. Seeds: the best linear-climb
children copied into `artifacts/hoeffding/synth_linear1/`.

## H13 progress. Round 1 and the restart with the integer certifier

Round 1 (cap 128, Fraction certifier): explore child 0.398261 (64 atoms,
34 turns, $1.09; its searches over two-sided ladders and other centres all
returned the ladder form, so it refined instead); refine child 0.398225
(43 turns, $1.42). Both self-capped at 64 atoms because certification
cost them 15 s there and grows like the fourth power of the atom count.

Two measurements then changed the run:

| Item | Value |
|---|---|
| Integer-arithmetic certifier vs Fraction convolution, same law, same value | 1.09 s vs 15.64 s, 14x; 0.2 s vs 3.6 s on a 64-atom two-level law |
| Same ladder structure, 104 atoms, 20 s weight search | certified 0.399191, against 0.398261 at 64 atoms; gap to the limit 0.0015 from 0.0024 |

The integer certifier replaced the Fraction one (`linear.value_of`), with a
test that the two agree on random laws and fails if the strict inequality is
loosened. The climb was stopped at the start of round 2 and resumed with an
atom cap of 256 and a refine-lane text that states the measured gain and
cost. Population carried over: three seeds and the two round-1 children.

## R24. Closing the loop: resynthesis on the first falsifying probe, four levels

Round 1 is the stored seeded committee of 8 (R4, R6, R17, R21). Probing by
disagreement observes held-out transitions in order of vote entropy until one
transition refutes every surviving member; on all four levels that is the
first probe (R23). Round 2 observes it: the probe joins the train set, each
seed carries the counterexample and what the refuted programs predicted
(grouped by prediction, pixels omitted), and 8 new programs are synthesized
with Devin under the unchanged contract. Control: a passive committee of 8 on
the first train+1 transitions in time, as an agent that keeps playing instead
of probing. Every arm is scored on the transitions that no arm observed. On
ar25 L3 a round 3 starts from the round 2 committee, which is refuted after
three more probes (steps 101, 63, 64), and trains on 33 transitions.

ar25 L3, 40 common transitions:

| Arm | Observed | Admitted | Vote | Members | AUROC | Unanimous n (error) | Distinct | Next refutation |
|---|---|---|---|---|---|---|---|---|
| Round 1 | 0 | 8/8 | 0.475 | 0.45 to 0.65 | 0.78 | 26 (0.31) | 7 | probe 1 |
| Round 2 | 1 | 8/8 | 0.925 | 0.93 | 0.50 | 40 (0.08) | 1 | probe 5 |
| Round 3 | 4 | 8/8 | 1.000 | 0.90 to 1.00 | none | 36 (0.00) | 2 | never |
| Passive | 1 | 7/8 | 0.650 | 0.12 to 0.88 | 0.97 | 4 (0.00) | 6 | probe 1 |

sk48 L2, 66 common:

| Arm | Observed | Admitted | Vote | Members | AUROC | Unanimous n (error) | Distinct | Next refutation |
|---|---|---|---|---|---|---|---|---|
| Round 1 | 0 | 8/8 | 0.697 | 0.61 to 0.98 | 0.91 | 40 (0.00) | 6 | probe 53 |
| Round 2 | 1 | 8/8 | 0.864 | 0.80 to 0.86 | 0.47 | 62 (0.15) | 2 | probe 23 |
| Passive | 1 | 7/8 | 0.636 | 0.45 to 0.64 | 0.60 | 43 (0.30) | 3 | probe 1 |

m0r0 L3, 42 common:

| Arm | Observed | Admitted | Vote | Members | AUROC | Unanimous n (error) | Distinct | Next refutation |
|---|---|---|---|---|---|---|---|---|
| Round 1 | 0 | 8/8 | 0.786 | 0.76 to 0.79 | 0.64 | 37 (0.16) | 4 | probe 1 |
| Round 2 | 1 | 8/8 | 0.786 | 0.79 to 0.81 | 0.65 | 38 (0.16) | 2 | probe 1 |
| Passive | 1 | 8/8 | 0.762 | 0.76 to 0.79 | 0.89 | 33 (0.06) | 5 | probe 32 |

ar25 L7, 63 common:

| Arm | Observed | Admitted | Vote | Members | AUROC | Unanimous n (error) | Distinct | Next refutation |
|---|---|---|---|---|---|---|---|---|
| Round 1 | 0 | 8/8 | 0.413 | 0.40 to 0.41 | 1.00 | 24 (0.00) | 6 | probe 1 |
| Round 2 | 1 | 7/8 | 0.429 | 0.43 | 0.50 | 63 (0.57) | 1 | probe 2 |
| Passive | 1 | 8/8 | 0.413 | 0.40 to 0.43 | 1.00 | 25 (0.00) | 4 | probe 1 |

Reading:

1. Where the counterexample names a missing mechanic, one round lifts every
   member. On ar25 L3 the members' own notes state it ("pieces do not block
   the axis; it moves onto and under piece rows"): the vote goes 0.48 to
   0.93, and a third round with 4 observations in total reaches 1.00 on the
   40 remaining transitions and is never refuted. On sk48 L2 the vote goes
   0.70 to 0.86. The passive control at the same train size moves 0.00 to
   -0.06 and stays bimodal.
2. Where the members already agree and are wrong together (m0r0 L3, ar25
   L7), one counterexample changes nothing: 0.79 to 0.79 and 0.41 to 0.43,
   and the next probe refutes the new committee at once.
3. The cost is convergence. Round 2 committees have 1 or 2 distinct
   behaviours, AUROC of disagreement falls to 0.47 to 0.65, and on ar25 L7
   the committee is unanimous and wrong on 57 percent. After resynthesis the
   disagreement signal cannot be read from the committee; the next probe
   needs new seeds or another criterion.
4. Caveat: one batch of 8 per arm. The batch spread is visible in the passive
   arms (0.12 to 0.88 on ar25 L3). Round 1 was synthesized earlier the same
   day; the backend and prompt code are unchanged since 13:41.

| Item | Value |
|---|---|
| Metric | Vote accuracy, member accuracy, AUROC, unanimous and split error on the common held-out set |
| Runs | 8 sessions per arm: 4 active, 4 passive, 1 round 3; 72 Devin sessions, 3 not admitted (2 timeouts at 900 s, 1 inconsistent) |
| Split | Temporal 40 percent plus the observed probes; passive arms `--train-n` train+1 |
| Baseline | Round 1 and the passive arm on the same transitions |
| Command | `uv run python -m committee.cegis GAME --level L --runs 8 --backend devin --parallel 4`; `... --from-probe 1 ...` for round 3; `uv run python -m committee.experiment GAME --level L --train-n N --runs 8 --seeded --backend devin --condition passive_devin`; `uv run python -m committee.cegis GAME --level L --report` |
| Commit | code d1fc873, 79e43c6; artifacts 59be4da, 79e43c6, 47e5907, dd76b2f; reports `artifacts/<game>/<split>_probe<n>/cegis_report.json` |

## R25. Live play on the ARC-AGI-3 engine: the committee as the explorer

The ar25 game runs locally through the `arc-agi` package; its source is
downloaded with the ARC key into `cache/arc_games` and is never read. The
engine is set to level 3, whose start state equals the recorded one. Each
move, every member predicts the outcome of every available action from the
observed object state (the released extractor). The agent takes the action
with the highest vote entropy; when every action is unanimous it prefers an
action whose predicted state is new, then one that changes the state, and it
never repeats a state-action pair. Every member is scored against the
observed next state. Budget 300 moves; the game ends itself at about 160.

| Committee | Seed | Moves | Vote accuracy | AUROC | Unanimous n (error) | Split n (error) | First move with every member wrong |
|---|---|---|---|---|---|---|---|
| Round 1 | 0 | 160 | 0.40 | 0.60 | 140 (0.54) | 20 (1.00) | 64 |
| Round 1 | 1 | 161 | 0.34 | 0.81 | 94 (0.43) | 67 (1.00) | 29 |
| Round 3 | 0 | 157 | 0.41 | 0.52 | 154 (0.58) | 3 (1.00) | 64 |
| Round 3 | 1 | 164 | 0.39 | 0.52 | 161 (0.60) | 3 (1.00) | 64 |

Reading:

1. Inside the mechanics the data showed, the committees are unanimous and
   right on every move: round 3 for 64 moves on both seeds, round 1 for 64
   and 29. Round 1's failure at move 29 is a mechanic round 3 learned from
   the recorded probes.
2. The first live refutation of round 3 is a mechanic that no recorded
   transition shows: at move 64 the counter bar starts to shrink by one per
   action and piece names shift; the counter never changes in the 75
   recorded level 3 transitions. Every later state carries it, so every later
   prediction is wrong. Disagreement at that move is 0.00 on both committees.
   This is an unknown unknown: not flagged, and the cue for resynthesis with
   the observed transition as the counterexample (R24), not for probing.
3. Where members disagree live, the vote is wrong every time (split error
   1.00 in all four runs). AUROC is 0.52 to 0.81 only because the shared
   blind spot after move 64 fills the unanimous bin with errors.

| Item | Value |
|---|---|
| Metric | Live vote accuracy, AUROC, unanimous and split error, first refuting move |
| Runs | 4: two seeds per committee, one engine run each |
| Split | Committees from R4 (round 1) and R24 (round 3); test is the live trajectory |
| Baseline | Round 1 committee |
| Command | `uv run python -m committee.live ar25 --level 3 --steps 300 --seed S [--probe 4]` |
| Commit | dd76b2f; logs `artifacts/ar25/live/` |

## R26. Live loop closed: resynthesis on the live counterexample

Harness change first. R25 scored every move with a fresh process per member,
which drops a program's hidden state between moves. `committee.live` now
keeps one process per member for the scored trajectory (teacher forcing, as
in `verify.run_program`); candidate actions are still chosen from stateless
what-if predictions. Under this harness the R25 runs change in one place:
round 1 on seed 1 is refuted at move 64 instead of 29 (vote 0.40 instead of
0.34, AUROC 0.85 instead of 0.81). The other three runs are identical.

Live counterexample round: train is the 33 recorded transitions of round 3
plus the 71 live transitions of the seed 0 run through move 70, and the seed
states the refuting move (counter 64 to 63, piece renaming). 8 members, all
replay 104 of 104. Their notes name the rule the recorded data could not
show: a hidden move budget of 128 whose bar displays min(budget, 64). The
seed 1 run is a trajectory the new committee never saw.

| Committee | Seed | Moves | Vote | AUROC | Unanimous n (error) | Split n (error) | First move with every member wrong |
|---|---|---|---|---|---|---|---|
| Round 1 | 0 | 160 | 0.40 | 0.60 | 140 (0.54) | 20 (1.00) | 64 |
| Round 1 | 1 | 161 | 0.40 | 0.85 | 84 (0.36) | 77 (0.87) | 64 |
| Round 3 | 0 | 157 | 0.41 | 0.52 | 154 (0.58) | 3 (1.00) | 64 |
| Round 3 | 1 | 164 | 0.39 | 0.52 | 161 (0.60) | 3 (1.00) | 64 |
| Live round | 0 | 144 | 0.71 | 0.79 | 120 (0.15) | 24 (1.00) | 70 |
| Live round | 1 | 139 | 0.86 | 0.66 | 133 (0.10) | 6 (1.00) | 66 |

Reading:

1. The loop closes live. Observe the refuting move, resynthesize on it,
   replay: the new committee predicts the counter tick at move 64 and every
   move through 69 on seed 0, and live vote accuracy goes from 0.40 to 0.71
   and 0.86.
2. The next refutation is a different mechanic: an action 3 move at moves 70
   and 81 on seed 0, and from about move 110 every prediction fails. That is
   the next counterexample. Seed 1 is refuted at move 66.
3. Where members disagree live, the vote is wrong almost every time (split
   error 0.87 to 1.00), as on the recorded data.

| Item | Value |
|---|---|
| Metric | Live vote accuracy, AUROC, unanimous and split error, first refuting move |
| Runs | 6 live runs; 8 Devin sessions for the live round |
| Split | Train 33 recorded plus 71 live (seed 0, moves 0 to 70); test is each live trajectory |
| Baseline | Round 1 and round 3 under the same harness |
| Command | `uv run python -m committee.live ar25 --level 3 --probe 4 --resynth-from artifacts/ar25/live/L3_cegis_devin_probe4_seed0.json --through 70 --runs 8 --backend devin --parallel 4`; `uv run python -m committee.live ar25 --level 3 --steps 300 --seed S --members-dir ar25/L3_f40_probe4_live70/live_devin` |
| Commit | 0409cac (round), 00d9874 (harness and logs) |

## R27. Replication on four games never used before

Same code and protocol as R4 and R24. Levels were chosen from a census of all
25 bundles (modellable transitions per level, no click actions, games not used
so far): ls20 L3 (99 transitions), g50t L1 (87), wa30 L3 (79), ka59 L2 (73).
Round 1 is 8 seeded programs on the first 40 percent of the level. Round 2
runs where round 1 is not saturated: the disagreement probes that refute every
member join the train set, 1 on ls20, 2 on ka59 and 9 on g50t, where the
version space shrinks for eight probes first. No passive arm on these levels
(R24's passive arms moved 0.00 to -0.06).

Round 1, full held-out set:

| Level | n test | Admitted | Vote | Mean member | AUROC | Unanimous n (error) | Split n (error) | Distinct |
|---|---|---|---|---|---|---|---|---|
| ls20 L3 | 59 | 7/8 | 0.864 | 0.823 | 0.77 | 36 (0.06) | 23 (0.26) | 4 |
| ka59 L2 | 44 | 7/8 | 0.818 | 0.756 | 0.69 | 33 (0.12) | 11 (0.36) | 6 |
| g50t L1 | 52 | 4/8 | 0.519 | 0.481 | 0.83 | 21 (0.10) | 31 (0.74) | 4 |
| wa30 L3 | 47 | 8/8 | 0.957 | 0.963 | 1.00 | 45 (0.00) | 2 (1.00) | 2 |

Adaptive conformal sets (R22 protocol, `artifacts/calibration_new.json`):

| Level | Coverage at 0.90 target | Mean set size | Abstain rate | Singleton rate, accuracy |
|---|---|---|---|---|
| ls20 L3 | 0.898 | 1.46 | 0.37 | 0.59, 0.89 |
| ka59 L2 | 0.955 | 2.11 | 0.84 | 0.14, 0.83 |
| g50t L1 | 0.962 | 3.04 | 0.83 | 0.12, 1.00 |
| pooled, 3 new levels | 0.936 | | | vote-share ECE 0.12 |
| pooled, all 7 informative levels | 0.934 | | | vote-share ECE 0.17 (`artifacts/calibration_7.json`) |

Round 2 against round 1 on the transitions neither observed:

| Level, common n | Arm | Observed | Admitted | Vote | Members | AUROC | Unanimous n (error) | Distinct | Next refutation |
|---|---|---|---|---|---|---|---|---|---|
| ls20 L3, 58 | Round 1 | 0 | 7/8 | 0.879 | 0.59 to 0.88 | 0.74 | 36 (0.06) | 3 | probe 1 |
| | Round 2 | 1 | 8/8 | 0.879 | 0.88 | 0.64 | 56 (0.09) | 2 | probe 1 |
| ka59 L2, 42 | Round 1 | 0 | 7/8 | 0.857 | 0.69 to 0.86 | 0.59 | 33 (0.12) | 3 | probe 1 |
| | Round 2 | 2 | 7/8 | 0.690 | 0.69 | 0.50 | 42 (0.31) | 1 | probe 13 |
| g50t L1, 43 | Round 1 | 0 | 4/8 | 0.465 | 0.35 to 0.49 | 0.78 | 14 (0.14) | 4 | probe 1 |
| | Round 2 | 9 | 7/8 | 0.488 | 0.37 to 0.58 | 0.71 | 17 (0.12) | 7 | probe 6 |

Reading:

1. The calibration result replicates on all four games: unanimous error 0.00
   to 0.12 against split error 0.26 to 1.00, AUROC 0.69 to 1.00, and conformal
   coverage 0.90 to 0.96 at the 0.90 target with the same abstention pattern
   (0.37 where the committee is mostly right, 0.83 to 0.84 where it is not).
2. The counterexample round does not replicate as a lift. Null on ls20 (0.88
   to 0.88). A loss on ka59 (0.86 to 0.69): all seven members converge on the
   same wrong program and are unanimous and wrong on 31 percent. Marginal on
   g50t (0.47 to 0.49), where it raises admission from 4 to 7, the best member
   from 0.49 to 0.58, and the next refutation from probe 1 to probe 6. Over
   the seven levels with a round 2: two clear lifts (R24), three nulls, one
   marginal, one loss.
3. Convergence is again the warning sign. The round 2 committees have 1 or 2
   distinct behaviours on ls20 and ka59; g50t, the one that gained, kept 7.
4. Repair recorded: ka59's object type is named `socket`, and the anti-network
   filter rejected six valid programs for the word. The filter now matches
   socket imports only (mutant-checked test); the six programs were
   re-verified under the unchanged exact-replay rule before any evaluation.

| Item | Value |
|---|---|
| Metric | As R4, R22 and R24 |
| Runs | 32 round 1 sessions (4 not admitted: 1 ls20, 1 ka59, 4 g50t minus the re-verified), 24 round 2 sessions (2 not admitted) |
| Split | Temporal 40 percent; round 2 adds the refuting probes |
| Baseline | Round 1 on the same transitions |
| Command | `uv run python -m committee.experiment GAME --level L --train-frac 0.4 --runs 8 --seeded --backend devin --condition committee_devin --parallel 4`; `uv run python -m committee.evaluate GAME --level L --train-frac 0.4 --condition committee_devin`; `uv run python -m committee.calibrate --levels ls20:3,ka59:2,g50t:1 --out artifacts/calibration_new.json`; `uv run python -m committee.cegis GAME --level L --runs 8 --backend devin --parallel 4`; `... --report` |
| Commit | d120ec5 (round 1), ef4bab1 (calibration), f5d4253 (round 2) |

## R28. Probe policy: disagreement on seen rows first does not select either

Question: a disputed transition on a row with no train counts is a place where
every member extrapolates; probing it refutes rather than selects. Among the
split held-out transitions of the seven informative levels, every member is
wrong on 62 of 88 (70 percent) whose rows include one with no train counts,
against 16 of 67 (24 percent) whose rows were all seen. Does a probe order
that prefers disputed transitions on seen rows select among members?

Simulated exploration (R4 protocol) with three orders: disagreement;
disagreement restricted to transitions whose rows all have train counts, with
the unrestricted order as the fallback (`explore.simulate`, strategy
`seen_disagreement`); random, 20 orders.

| Level | Order | Refuted at probe | Members left by probe | Selecting probes | Survivors' vote accuracy, start to after selection |
|---|---|---|---|---|---|
| ar25 L3 | disagreement | 1 | 8, 0 | 0 | 0.48 |
| | seen rows first | 2 | 8, 8, 0 | 0 | 0.48 |
| | random | mean 2.7 | | mean 0.4 | 0.48 to mean 0.51 |
| m0r0 L3 | disagreement | 1 | 8, 0 | 0 | 0.77 |
| | seen rows first | 24 | 8, 6, 6, ... | 1 | 0.77 to 0.77 |
| | random | mean 4.8 | | mean 0.1 | 0.77 |
| sk48 L2 | disagreement | 1 | 8, 0 | 0 | 0.69 |
| | seen rows first | 1 | 8, 0 | 0 | 0.69 |
| | random | mean 22.2 | | mean 1.6 | 0.69 to mean 0.76 |
| ar25 L7 | disagreement | 1 | 8, 0 | 0 | 0.42 |
| | seen rows first | 14 | 8, 8, ... | 0 | 0.42 |
| | random | mean 1.6 | | mean 0.1 | 0.42 |
| ls20 L3 | both | 1 | 7, 0 | 0 | 0.86 |
| | random | mean 8.4 | | mean 0.9 | 0.86 |
| ka59 L2 | disagreement | 2 | 7, 2, 0 | 1 | 0.82 to 0.84 |
| | seen rows first | 6 | 7, 2, 2, ... | 1 | 0.82 to 0.84 |
| | random | mean 4.0 | | mean 0.6 | 0.82 |
| g50t L1 | both | 9 | 4, 1, 1, ... | 1 | 0.52 to 0.43 |
| | random | mean 2.1 | | mean 0.4 | 0.52 to mean 0.51 |

Reading. The restriction delays the refutation (24 probes on m0r0, 14 on
ar25 L7) and selects no better: at most one selecting probe on any level, and
the survivors' vote accuracy moves by at most 0.02, downward on g50t where the
single survivor is a worse program. After the first selecting probe the
survivors agree on every remaining seen-row transition, so there is nothing
left to distinguish there; the next informative probe is on an unseen row,
and that one refutes them all. The random order's small gains (sk48 0.69 to
0.76) come from hitting a seen-row split that happens to separate the best
member. Conclusion, as R23: the hypotheses are missing, not misordered;
disagreement on unseen rows is the right flag for "none of my hypotheses
covers this", and the repair is synthesis.

| Item | Value |
|---|---|
| Metric | Probes until refutation, selecting probes, survivors' vote accuracy on unobserved transitions |
| Runs | Deterministic from stored members; random order 20 seeds |
| Split | Temporal 40 percent, round 1 committees |
| Baseline | Unrestricted disagreement order, random order |
| Command | `uv run python -m committee.selection` style run; data in `artifacts/probe_policy.json` |
| Commit | a4d8c33 |

## R29. Abstention is a floor, not a score: a selective score and what the loop does to it

Coverage alone rewards abstention: a wrapper that abstains on every step
covers 1.0 and says nothing (R22, check 1 in O5). To put a price on that,
score each step of the conformal wrapper (R22 protocol, alpha 0.1, gamma
0.05): a committed single-state answer earns +1 if right and -1 if wrong; a
set of several states or an abstention earns 0. Abstaining throughout scores
0; always committing to the plurality scores 2 x vote accuracy - 1; confident
failure is negative. The same score was computed on the six live trajectories
of R26, with the wrapper run over each trajectory's vote shares.

Recorded held-out trajectories, round 1 committees:

| Level | Coverage | Commit share | Committed accuracy | Abstain rate | Selective score, wrapper | Selective score, always commit |
|---|---|---|---|---|---|---|
| ar25 L3 | 0.955 | 0.14 | 0.83 | 0.84 | 0.09 | -0.05 |
| m0r0 L3 | 0.909 | 0.77 | 0.88 | 0.23 | 0.59 | 0.55 |
| sk48 L2 | 0.882 | 0.66 | 0.93 | 0.09 | 0.57 | 0.38 |
| ar25 L7 | 0.985 | 0.02 | 1.00 | 0.97 | 0.02 | -0.17 |
| ls20 L3 | 0.898 | 0.59 | 0.89 | 0.37 | 0.46 | 0.73 |
| ka59 L2 | 0.955 | 0.14 | 0.83 | 0.84 | 0.09 | 0.64 |
| g50t L1 | 0.962 | 0.12 | 1.00 | 0.83 | 0.12 | 0.04 |

Live trajectories, ar25 level 3 (wrapper over the live vote shares):

| Committee, seed | Moves | Coverage | Commit share | Committed accuracy | Selective score, wrapper | Always commit | Commit share by quarter of the trajectory |
|---|---|---|---|---|---|---|---|
| Round 1, 0 | 160 | 0.956 | 0.44 | 0.90 | 0.35 | -0.20 | 0.97, 0.78, 0.00, 0.00 |
| Round 1, 1 | 161 | 0.950 | 0.38 | 0.95 | 0.34 | -0.20 | 0.88, 0.65, 0.00, 0.00 |
| Round 3, 0 | 157 | 0.955 | 0.45 | 0.90 | 0.36 | -0.18 | 0.97, 0.82, 0.00, 0.00 |
| Round 3, 1 | 164 | 0.957 | 0.43 | 0.90 | 0.34 | -0.22 | 0.98, 0.73, 0.00, 0.00 |
| Live round, 0 | 144 | 0.917 | 0.75 | 0.94 | 0.65 | 0.42 | 0.97, 1.00, 1.00, 0.03 |
| Live round, 1 | 139 | 0.906 | 0.81 | 0.90 | 0.65 | 0.73 | 0.97, 1.00, 1.00, 0.24 |

Reading:

1. Abstention is the floor. Under the selective score a trajectory of
   abstentions is worth 0: better than confident failure (always commit on
   ar25 L3 live: -0.20) and worth nothing in itself. The wrapper's value on
   the recorded levels is that it sits above both the floor and the
   always-commit policy on five of seven levels; on ls20 and ka59 it
   abstains too much on a committee that is mostly right (0.46 against 0.73,
   0.09 against 0.64), a cost that coverage alone hides.
2. What turns abstention into score is the loop, not the wrapper. Live, the
   recorded-data committees commit in the first half of the trajectory and
   abstain entirely after the counter tick (quarters 3 and 4 at 0.00). The
   live counterexample round converts those quarters into correct
   commitments (1.00, 1.00), lifts the commit share from 0.44 to 0.78 and the
   selective score from 0.35 to 0.65, and abstains again only in the last
   quarter, where the next unseen mechanic sits. The abstention region is the
   map of where to collect counterexamples; resynthesis cashes them.
3. This is the shape of the training signal. The score rewards turning an
   abstention into a correct commitment, penalises a wrong commitment, and
   is indifferent to abstaining, so an agent paid by it has one way up:
   observe where it abstains and repair its hypotheses. In this project that
   step is the counterexample round; in the ONC build the same shape, task
   score plus calibration plus disagreement drop, trains a decision policy
   (O5, O7). Training a synthesizer on it is the open item.

| Item | Value |
|---|---|
| Metric | Coverage, commit share, committed accuracy, selective score (+1 committed right, -1 committed wrong, 0 otherwise) |
| Runs | 7 recorded levels, deterministic; 6 live trajectories of R26 |
| Split | As R22 and R26 |
| Baseline | Always commit to the plurality; abstain throughout (0) |
| Command | Wrapper from `committee.calibrate.aci` over `level_steps` and over the live logs' `shares` and `truth_share` (logged since this commit); data in `artifacts/abstention_score.json` |
| Commit | ebfba59 |

## R30. The wrapper's step size, chosen leave-one-level-out, and the price of a wrong commitment

Two sensitivities of R29, on the seven recorded levels. First, the wrapper's
step gamma (R22 used 0.05, the usual value) swept over 0.02, 0.05, 0.1, 0.2,
0.3 at alpha 0.10, and chosen for each level by the mean selective score on
the other six. Second, the selective score with the penalty for a wrong
commitment c at 0.5, 1 and 2 (abstention stays 0).

| Level | gamma 0.05: coverage, commit, score | gamma 0.3: coverage, commit, score | Leave-one-out choice | Always commit |
|---|---|---|---|---|
| ar25 L3 | 0.95, 0.14, +0.09 | 0.91, 0.18, +0.09 | 0.3 | -0.05 |
| m0r0 L3 | 0.91, 0.77, +0.59 | 0.91, 0.77, +0.59 | 0.3 | +0.55 |
| sk48 L2 | 0.88, 0.66, +0.57 | 0.90, 0.60, +0.57 | 0.3 | +0.38 |
| ar25 L7 | 0.98, 0.01, +0.02 | 0.92, 0.23, +0.23 | 0.3 | -0.17 |
| ls20 L3 | 0.90, 0.59, +0.46 | 0.90, 0.64, +0.58 | 0.3 | +0.73 |
| ka59 L2 | 0.95, 0.14, +0.09 | 0.91, 0.16, +0.07 | 0.3 | +0.64 |
| g50t L1 | 0.96, 0.12, +0.12 | 0.90, 0.27, +0.19 | 0.3 | +0.04 |

Penalty sensitivity at gamma 0.05, wrapper against always commit:

| Level | c = 0.5 | c = 1 | c = 2 |
|---|---|---|---|
| ar25 L3 | +0.10 vs +0.22 | +0.09 vs -0.04 | +0.07 vs -0.57 |
| m0r0 L3 | +0.64 vs +0.66 | +0.59 vs +0.55 | +0.50 vs +0.32 |
| sk48 L2 | +0.60 vs +0.54 | +0.57 vs +0.38 | +0.53 vs +0.07 |
| ar25 L7 | +0.01 vs +0.12 | +0.01 vs -0.17 | +0.01 vs -0.75 |
| ls20 L3 | +0.49 vs +0.80 | +0.46 vs +0.73 | +0.39 vs +0.59 |
| ka59 L2 | +0.10 vs +0.73 | +0.09 vs +0.64 | +0.07 vs +0.46 |
| g50t L1 | +0.12 vs +0.28 | +0.12 vs +0.04 | +0.12 vs -0.44 |

Reading:

1. A larger step helps where the committee is mostly wrong: at gamma 0.3
   the wrapper commits on 23 percent of ar25 L7 and 27 percent of g50t L1
   instead of 1 and 12 percent, with coverage still at or above 0.90, and the
   selective score rises from +0.02 to +0.23 and +0.12 to +0.19. Where the
   committee is mostly right nothing changes. The leave-one-out choice is 0.3
   for every held-out level, and the held-out score under it is at least the
   gamma 0.05 score on six of seven levels (ka59: 0.07 against 0.09). The
   R22 default stays 0.05 in the code; 0.3 is the recorded alternative.
2. ka59's over-abstention is not a step-size problem. Its seven members show
   six behaviours, so the plurality's share is low even when it is right;
   the wrapper can only commit a single state when one candidate carries the
   quantile, and here none does. A calibration map from vote share to
   P(correct) (R22's leave-one-level-out map) is the tool for that case, and
   always committing wins there at every price.
3. The wrapper's value depends on the price of a wrong commitment. At c =
   0.5 always committing beats the wrapper on five of seven levels; at c = 1
   the wrapper wins five of seven; at c = 2 it wins the same five by wider
   margins. The wrapper is the right policy when a wrong answer costs at
   least as much as a right one earns, which is the setting of an agent
   acting on its prediction.

| Item | Value |
|---|---|
| Metric | Coverage, commit share, selective score at c = 0.5, 1, 2 |
| Runs | Deterministic from stored committees, 7 levels, 5 step sizes |
| Split | As R22; leave-one-level-out for the choice of gamma |
| Baseline | gamma 0.05 (R22), always commit |
| Command | `committee.calibrate.aci` over `level_steps`; data in `artifacts/wrapper_gamma.json` |
| Commit | bcc581b |

## R31. Naming the mechanism by construction: the counterexample at the effect-row level, one repair hypothesis per seed

R24 and R27 showed the counterexample round lifts accuracy where the
refuting observation names a missing mechanic. This round makes that the
construction (`committee.cegis --mechanism`). The counterexample is stated
in the effect-row vocabulary: for each object of the refuting transition,
its row (type | action | context), the observed field changes, and each
predicted change with its member count, split into the rows every program
got wrong (the mechanic to repair) and the rows every program got right (to
keep as they are). Each of the eight seeds then carries a different candidate
condition for the wrong rows: a field of the object, a neighbour, a global
object, the click position, hidden state, the exact size or fit relation, the
order or chain of moving objects, an undrawn bound. On ka59 a third variant
adds an explicit rule against inferring invisible regions from where objects
stopped (`--geometry-rule`). Same probes, same common held-out set, same
synthesizer as the object-diff statement of R24 and R27.

| Level, common n | Statement | Admitted | Vote | Members | AUROC | Unanimous n (error) | Distinct | Next refutation |
|---|---|---|---|---|---|---|---|---|
| ar25 L3, 40 | round 1 | 8/8 | 0.475 | 0.45 to 0.65 | 0.78 | 26 (0.31) | 7 | probe 1 |
| | object diff | 8/8 | 0.925 | 0.93 | 0.50 | 40 (0.08) | 1 | probe 5 |
| | mechanism | 8/8 | 1.000 | 1.00 (all 8) | none | 40 (0.00) | 1 | never |
| ls20 L3, 58 | round 1 | 7/8 | 0.879 | 0.59 to 0.88 | 0.74 | 36 (0.06) | 3 | probe 1 |
| | object diff | 8/8 | 0.879 | 0.88 | 0.64 | 56 (0.09) | 2 | probe 1 |
| | mechanism | 8/8 | 0.879 | 0.88 | 0.64 | 56 (0.09) | 2 | probe 1 |
| m0r0 L3, 42 | round 1 | 8/8 | 0.786 | 0.76 to 0.79 | 0.64 | 37 (0.16) | 4 | probe 1 |
| | object diff | 8/8 | 0.786 | 0.79 to 0.81 | 0.65 | 38 (0.16) | 2 | probe 1 |
| | mechanism | 8/8 | 0.786 | 0.79 to 0.81 | 0.65 | 38 (0.16) | 2 | probe 1 |
| ar25 L7, 63 | round 1 | 8/8 | 0.413 | 0.40 to 0.41 | 1.00 | 24 (0.00) | 6 | probe 1 |
| | object diff | 7/8 | 0.429 | 0.43 | 0.50 | 63 (0.57) | 1 | probe 2 |
| | mechanism | 7/8 | 0.429 | 0.38 to 0.43 | 0.62 | 46 (0.50) | 5 | probe 1 |
| ka59 L2, 42 | round 1 | 7/8 | 0.857 | 0.69 to 0.86 | 0.59 | 33 (0.12) | 3 | probe 1 |
| | object diff | 7/8 | 0.690 | 0.69 | 0.50 | 42 (0.31) | 1 | probe 13 |
| | mechanism | 8/8 | 0.690 | 0.55 to 0.69 | 0.40 | 36 (0.36) | 2 | probe 14 |
| | mechanism + geometry rule | 7/8 | 0.690 | 0.69 | 0.50 | 42 (0.31) | 1 | probe 13 |

Coordinate-sized integer literals per 100 tokens, median over the admitted
programs (`artifacts/repair_literals.json`):

| Level | Round 1 | Object-diff repair | Mechanism repair |
|---|---|---|---|
| ar25 L3 | 0.3 | 0.5 | 0.4 |
| sk48 L2 | 0.8 | 0.8 | |
| m0r0 L3 | 2.3 | 2.4 | 2.5 |
| ar25 L7 | 0.5 | 0.8 | 0.4 |
| ls20 L3 | 2.8 | 2.1 | 2.1 |
| ka59 L2 | 1.6 | 3.8 | 3.7 |
| g50t L1 | 1.3 | 1.2 | |

Reading:

1. Where the repair is within the synthesizer's reach, naming the row gets
   it in one round. ar25 L3 reaches 1.00 with all eight members after one
   observation; the object-diff statement reached 0.925 and needed two more
   rounds and four observations for the same result. The statement showed
   three wall rows under action 1 where four programs predicted no change,
   two predicted the object gone and two had the right effect; the members'
   notes name the rule (the axis moves without checking piece rows).
2. Where the round was null it stays null: ls20 and m0r0 produce the same
   behaviour under either statement. On ar25 L7 the vote is the same but the
   mechanism statement keeps five behaviours where the object diff collapsed
   to one, so the uncertainty signal survives the round (AUROC 0.62 against
   0.50).
3. Where the round lost it still loses, under the mechanism statement and
   under an explicit prohibition. All 23 admitted ka59 repairs over the three
   statements implement the same invisible floor fitted to where blocks
   stopped; the eight repair hypotheses were not followed, and the
   prohibition was ignored. ka59 is the only level where the repairs'
   coordinate-literal density rises by more than half (1.6 to 3.8), and the
   round-1 members the probes refuted were the ones at 0.86: the right rule
   is outside what this synthesizer proposes, and fitted geometry is its
   fallback. A prompt cannot move it; a verifier can. An admission rule on
   literal density, a train-side quantity, would have rejected all 23 and
   none of the other levels' repairs. That is the reward-design lever, and it
   mirrors RH4's finding on the exact-replay hack.
4. The mechanism statement is now the default for `committee.cegis`
   (`--object-diff` restores the old one): it is better on one level, equal
   on three, and preserves diversity on the fourth.

| Item | Value |
|---|---|
| Metric | As R24; coordinate-literal density per program |
| Runs | 48 Devin sessions: 5 levels x 8 mechanism, 8 geometry rule; 2 not admitted (timeouts) |
| Split | As R24 and R27; same probes and common sets |
| Baseline | Round 1 and the object-diff round 2 on the same transitions |
| Command | `uv run python -m committee.cegis GAME --level L --runs 8 --backend devin --parallel 4 --condition mech_devin --mechanism`; `... --geometry-rule`; `uv run python -m committee.cegis GAME --level L --report --conditions cegis_devin,mech_devin` |
| Commit | 9591b66, 3d67a77 (code); aa8cd1f (artifacts) |

## R32. The object state is incomplete: completeness of the frame, and static terrain as objects on ka59

Question raised by R31: is ka59 L2 unreachable because the environment
representation is wrong rather than the method? Three checks. The recording
is deterministic: no (before state, action) pair in level 2 has two outcomes
(69 pairs). The probes are ordinary block slides (x 42 to 15, x 33 to 18).
But the frame holds a wall mass (colour 15, about 1500 cells) and border
lines (colour 2) that the released extractor never emits; the programs see a
player, three blocks, four targets and an empty frame-sized "portal". The
kicked-block stop depends on that wall mass, so under the object contract the
rule can only be memorised, which is what every repair in R31 did.

Completeness, measured as the share of non-background cells over train and
test frames that lie inside an extracted object or a terrain object
(`committee.terrain --coverage`; terrain = cells whose colour is the same in
every training frame where no object covers them, grouped by colour into
4-connected components of at least 4 cells):

| Level | Objects only | Objects + terrain | Residual left |
|---|---|---|---|
| m0r0 L3 | 0.959 | 1.000 | none |
| ls20 L3 | 0.070 | 0.998 | three small colours |
| ka59 L2 | 0.085 | 0.997 | 0.3 percent |
| g50t L1 | 0.584 | 0.996 | 0.4 percent |
| sk48 L2 | 0.071 | 0.982 | border lines |
| wa30 L3 | 0.855 | 0.955 | colour 4 |
| ar25 L7 | 0.613 | 0.712 | a changing lattice, not static |
| ar25 L3 | 0.300 | 0.551 | the same lattice |

OPINE-World's harness lists `floor`, `background`, `wall`, `border`, `tile`
and `hud` as inert object types its extractors are expected to emit
(`sigma.py`, `label_audit.py`); the released ka59 extractor emitted a
frame-sized container instead. The terrain objects are that inert layer,
derived from frames.

A/B on ka59 L2, round 1 with the terrain objects appended to every state
(`committee.terrain ka59 --level 2 --runs 8`, condition `terrain_devin`,
same seeds and contract):

| Condition | Admitted | Vote | Members | AUROC | Unanimous n (error) | Distinct |
|---|---|---|---|---|---|---|
| committee_devin (objects) | 7/8 | 0.818 | 0.66 to 0.84 | 0.69 | 33 (0.12) | 6 |
| terrain_devin (objects + terrain) | 8/8 | 0.659 | 0.66 to 0.82 | 0.87 | 33 (0.12) | 4 |

Reading:

1. ka59 L2 is unreachable under the object contract for a representation
   reason: the state the probed mechanic depends on is not in the input.
   That reframes R31's ka59 loss. The gap is not ka59's alone; on sk48 L2 and
   ls20 L3 more than 90 percent of the frame's non-background cells are
   outside every object, and 75 percent on ar25 L3. It is harmless where no
   mechanic depends on the terrain and fatal where one does.
2. Appending the terrain as constant pixel objects does not help the
   synthesizer on ka59: the members land at 0.66 to 0.82 as before, and the
   vote is lower. A 63 by 63 pixel blob in the object list is not a form the
   programs use; they still memorise stops. The complete fix is to give the
   program the frame itself, which is OPINE-World's rule and is being added
   as the `frame` mode of `committee.env` by another session. The baseline
   against committee comparison is to be rerun in that mode (HANDOFF.md).
3. ar25's residual is a lattice that changes with the selection and is
   neither object nor terrain; round 3 reached 1.00 there without it.

| Item | Value |
|---|---|
| Metric | Explained share of non-background cells; vote and member accuracy on the 44 held-out transitions |
| Runs | 8 Devin sessions (terrain round); completeness is deterministic |
| Split | Temporal 40 percent; terrain from the training frames only |
| Baseline | Round 1 in objects mode (R27) |
| Command | `uv run python -m committee.terrain GAME --level L --coverage`; `uv run python -m committee.terrain ka59 --level 2 --runs 8 --backend devin --parallel 4`; `... --report` |
| Commit | b192b10, 3f9c0ca (code); 2d7a8f9 (artifacts) |

## R33. Pilot: single program against committee in OPINE-World's own environment (frame in, frame out)

The environment port (`committee.env`, modes `objects`, `frame`, `frame_out`,
by another session, uncommitted at the time of these runs) makes the program
take the 64 by 64 before frame as well as the object list and, in
`frame_out`, return the next frame; admission is frame equality on every
training transition, and the released extractor run on the predicted frame
gives the object view. `frame_out` is OPINE-World's contract. These runs used the
port's first task text, which still described the object contract and added
the frame contract as an addendum; the review's corrected text landed after
they started, so this is a pilot. Same splits, seeds and synthesizer as R4.

| Level | Arm | Mode | Admitted | Vote | Members | AUROC | Unanimous n (error) | Split n (error) |
|---|---|---|---|---|---|---|---|---|
| ar25 L3 | single x3 | objects (R3) | 3/3 | | 0.64, 0.43, 0.48 | | | |
| | committee x8 | objects (R4) | 8/8 | 0.477 | 0.43 to 0.64 | 0.77 | 27 (0.30) | 17 (0.88) |
| | single x3 | frame_out | 2/3 (1 outage) | | 1.00, 0.84 | | | |
| | committee x8 | frame_out | 6/8 (2 outage) | 1.000 | 0.91 to 1.00 | none | 40 (0.00) | 4 (0.00) |
| m0r0 L3 | single x3 | objects (R6) | 1/3 | | 0.75 | | | |
| | committee x8 | objects | 8/8 | 0.773 | 0.75 to 0.77 | 0.68 | 38 (0.16) | 6 (0.67) |
| | single x3 | frame_out | 3/3 | | 0.86, 0.86, 0.86 | | | |
| | committee x8 | frame_out | 8/8 | 0.864 | 0.80 to 0.86 | 0.70 | 37 (0.08) | 7 (0.43) |
| sk48 L2 | single x3 | objects (R17) | 3/3 | | 1.00, 0.63, 0.97 | | | |
| | committee x8 | objects | 8/8 | 0.691 | 0.61 to 0.98 | 0.92 | 41 (0.00) | 27 (0.59) |
| | single x3 | frame_out | 3/3 | | 0.87, 0.63, 0.59 | | | |
| | committee x8 | frame_out | 4/8 (4 outage) | 0.794 | 0.79 to 0.87 | 0.68 | 63 (0.14) | 5 (1.00) |
| ka59 L2 | committee x8 | objects (R27) | 7/8 | 0.818 | 0.66 to 0.84 | 0.69 | 33 (0.12) | 11 (0.36) |
| | single x3 | frame | 3/3 | | 0.82, 0.82, 0.82 | | | |
| | committee x8 | frame | 8/8 | 0.818 | 0.82 (all) | 0.56 | 43 (0.16) | 1 (1.00) |
| | single x3 | frame_out | 3/3 | | 0.89, 0.89, 0.91 | | | |
| | committee x8 | frame_out | 6/8 (2 outage) | 0.886 | 0.89 to 0.91 | 0.88 | 39 (0.03) | 5 (0.80) |

"Outage": from about 21:19 BST the Devin account suspended every new session
within a minute of creation (status `suspended`, no program); those members
are the 114-byte stub and are counted as not run, not as failed synthesis.
A probe session at 21:23 was suspended the same way.

Reading:

1. The environment was the bottleneck, not the synthesizer. In OPINE-World's
   rule the single programs on ar25 L3 reach 1.00 and 0.84 where the object
   contract gave 0.43 to 0.64, on m0r0 L3 0.86 against 0.75 with one of
   three admitted, on ka59 L2 0.89 to 0.91 against 0.66 to 0.84. sk48 L2 is
   the exception: the single programs are no better (0.59 to 0.87 against
   0.63 to 1.00), and the only level where the object state already carried
   what the programs needed.
2. The committee's standing is unchanged. Its vote equals or sits just under
   the best single program in both environments (ar25 L3 1.00 against 1.00,
   m0r0 0.86 against 0.86, ka59 0.89 against 0.91, sk48 0.79 against 0.87),
   and its calibration holds: unanimous error 0.00 to 0.14 against split
   error 0.43 to 1.00 in frame_out, AUROC 0.68 to 0.88 where errors exist.
   What the committee adds is still the flag and the probe, not accuracy.
3. The half-step `frame` mode (frame as input, objects as output) on ka59
   gave eleven identical programs at 0.82: the geometry became visible but
   the object output kept the contract's limits. Returning the frame is the
   change that mattered there.
4. ka59's remaining errors in frame_out are shared (unanimous error 0.03,
   split 5 transitions): the counterexample round in this mode is the next
   step there, as is a definitive rerun under the corrected task text on all
   seven levels. Both wait on the Devin account.

| Item | Value |
|---|---|
| Metric | As R4, per arm, on the full held-out set of each level |
| Runs | 11 sessions per level per mode: ka59 frame and frame_out, ar25 L3, m0r0 L3, sk48 L2 frame_out; 10 suspended by the account |
| Split | Temporal 40 percent |
| Baseline | The same arms in objects mode (R3, R4, R6, R17, R27) |
| Command | `uv run python -m committee.experiment GAME --level L --train-frac 0.4 --runs 3 --backend devin --frame-out --condition baseline_frameout_devin --parallel 3`; `... --runs 8 --seeded --frame-out --condition committee_frameout_devin --parallel 4`; `uv run python -m committee.evaluate GAME --level L --train-frac 0.4 --condition <condition>` |
| Commit | this session (artifacts); environment code uncommitted by its owner at the time |

## O1. ONC-AGI setup: baselines and cheaters on the toy worlds

ONC-AGI 1.0.0rc1 from `external/ONC-AGI`, Python 3.12, installed as a path
dependency. `uv run onc-agi smoke` passes. Toy worlds are contract fixtures,
not benchmark tasks; none of the O entries is a benchmark result. 10 worlds per
mode (8 signal, 2 null), bootstrap draws 200, wall time 36 s. Full tables:
`artifacts/onc/baselines.md`.

| Agent | DS full | DS sequential | Cost seq | Note |
|---|---|---|---|---|
| oracle | 1.000 | 1.000 | 0 | reference |
| univariate_bh | 0.766 | 0.307 | 25,855 | buys the whole pool in sequential mode |
| lasso | 0.766 | 0.307 | 25,855 | |
| stability | 0.698 | 0.193 | 25,855 | |
| elastic_net | 0.328 | 0.307 | 25,855 | claims on one null world |
| random_forest | 0.328 | 0.117 | 25,855 | |
| 11 cheaters, random | 0.000 | 0.000 | | unfloored within 0.02 of 0; random_abstain -0.086 |

`seq_<baseline>` (the kit's GroupSequentialAgent) fails on the two-stratum
worlds (`stratum 's-a' is exhausted`): an ONC-AGI defect, not patched
(external is read-only). knockoffs: knockpy not installed. Command:
`uv run python -m onc.baselines`. Commit: uncommitted, base ecb0447.

## O2. Committee agent on the toy worlds, both modes, acquisition designs

Committee of 8 hypothesis templates (null, direct, conservative, sparse,
confounder, upstream, interaction, block), admission by 5-fold CV log loss
within 0.02 of the best, weights exp(-n * CV log loss) ("likelihood") or
equal, submission rule abstain if P(signal) < 0.5 else P(driver) >= 0.5.
Sequential policy: recruit 60 patients across strata, assay every baseline
feature (never a post-outcome one), then recruit 40 more while the expected
drop in disagreement per 1000 USD exceeds 0.02 and spend stays under half the
budget. Reference cost of these worlds: 12,960 to 25,920 USD.

| Condition | DS | 95% | Find | Restraint | Strict | Leak | Cost | Acq. gap |
|---|---|---|---|---|---|---|---|---|
| full / likelihood | 1.000 | [1.00, 1.00] | 1.00 | 1.00 | 1.00 | 0 | 0 | 0.00 |
| full / equal | 0.000 | [0.00, 0.00] | 1.00 | 0.00 | 0.00 | 0 | 0 | 0.00 |
| seq / likelihood / disagreement | 0.935 | [0.81, 1.00] | 0.94 | 1.00 | 0.94 | 0 | 6,696 | 0.18 |
| seq / likelihood / random stop | 0.935 | [0.81, 1.00] | 0.94 | 1.00 | 0.94 | 0 | 8,556 | 0.18 |
| seq / likelihood / pipeline (buy everything) | 0.461 | [0.39, 0.54] | 0.69 | 0.67 | 0.46 | 0 | 25,855 | 0.00 |
| seq / likelihood / staged template | 0.519 | [0.45, 0.58] | 0.78 | 0.67 | 0.52 | 0 | 23,004 | 0.00 |
| seq / equal / disagreement | 0.904 | [0.71, 1.00] | 0.90 | 1.00 | 0.90 | 0 | 6,324 | 0.18 |
| seq / equal / staged template | 0.606 | [0.55, 0.65] | 0.91 | 0.67 | 0.61 | 0 | 19,764 | 0.00 |

Reading. Equal weights give the null member 1/K whatever the data, so in full
access the committee claims on a null world and Restraint is 0; likelihood
weights are the default (decision for the handoff's open question). In
sequential mode the disagreement stop spends a quarter of the buy-everything
cost at efficiency 1 on every world; buying everything pays the efficiency
penalty (0.69). Acquisition gap 0.18 says the oracle analyst would find more
with the whole pool than with our 60 to 100 patients; on these toys it costs
nothing in Find. On the module world (three drivers) the committee stops at
100 patients and finds 2 of 3 by weight. Runs: 1 (deterministic given the
seed). Command: `uv run python -m onc.evaluate --store toy --mode both`.
Outputs `artifacts/onc/eval_toy.{json,md}`. Commit: uncommitted, base ecb0447.

## O3. Calibration and conformal coverage of the committee on the toy worlds

Same runs as O2. P(signal) is scored against the world type; P(driver)
against per-feature credit from the answer key (removing the feature lowers
raw recovery); p(y | x) against the outcomes of each recruit batch, predicted
by the committee from before the batch. ACI target 0.90, gamma 0.05.

| Condition | ECE P(signal), n = 10 | ECE P(driver), n listed | Brier held-out p(y|x) | ACI coverage / committed | Held-out n |
|---|---|---|---|---|---|
| full / likelihood | 0.01 | 0.23 (16) | | | |
| full / equal | 0.23 | 0.16 | | | |
| seq / likelihood / disagreement | 0.01 | 0.13 (13) | 0.239 | 0.91 / 0.18 | 120 |
| seq / likelihood / random stop | 0.01 | 0.26 | 0.225 | 0.90 / 0.34 | 200 |
| seq / equal / disagreement | 0.38 | 0.38 | 0.253 | 0.91 / 0.11 | 120 |

Reading. With likelihood weights P(signal) is sharp and right on all 20
worlds (ECE 0.01). P(driver) is overconfident: the committee lists extra
features (the neutral world's second and third features, the module's fourth)
at P near 1 and they earn no credit. The held-out p(y | x) stream holds the
0.90 coverage target with commitments on 18% of patients, which is the honest
pair of numbers (section O5, check 1). Command and commit as O2.

## O5. Reward design and hacking checks on the toy worlds

R = R_task + 0.5 R_cal + 0.25 R_dis as in the handoff; `onc.rewards`.
Acceptance test (tests/test_onc_rewards.py): over the 20 toy worlds the mean
per-episode R_task orders the 8 smoke agents as discovery_score_unfloored does
up to ties, Kendall tau-b 0.886, no discordant pair; the reward is additive
(mean R_task = 0.8 * mean find_signed * eff - abstention + restraint on null)
while the scorecard is a product, so the reward separates agents the scorecard
ties at 0 and cannot agree with it for every agent set. R_dis telescopes to
U_0 - U_T with a reset at each resynthesis step (test with mutants).

Check 1, coverage only (sequential, 120 held-out patients). A set-width dial
paid for coverage alone moves to width 1.00: coverage 1.00, committed 0.00.
The frontier: width 0.5 gives coverage 0.62 at 100% committed; 0.8 gives
0.97 at 11%; ACI gives 0.91 at 18%. Coverage is reported with the commit share
everywhere in this work.

Check 2, confidence-free reward (full access). Sharpening the reported
probabilities (odds to the power s) leaves every decision and R_task (+1.800)
unchanged for s in {0.5, 1, 2, 4, 8, 32}, so a task-only reward has no
preference. ECE P(driver) rises 0.199 to 0.267 and R_cal falls -0.084 to
-0.125 with s; R_task + 0.5 R_cal picks s = 0.5, a slightly softer report
than the committee's own, because P(driver) is overconfident (O3).

Check 3, the benchmark's cheaters under R (implicit probabilities: P(signal)
= 1 when the list is non-empty, P(driver) = 1 per listed feature, R_dis = 0).

| Agent | R full | R seq | DS full |
|---|---|---|---|
| committee (ours) | +1.75 | +1.68 | 1.000 |
| stability, univariate_bh, lasso, random_forest, elastic_net | +1.46 to +0.62 | +0.77 to +0.38 | 0.70 to 0.33 |
| always_empty | -0.40 | | 0 |
| hub, cluster-size, metadata, leak_exploiter, variance, synthetic, giant_list, random, famous_genes, auc_maximiser | -0.53 to -0.67 | -0.55 to -0.63 | 0 |
| random_abstain | -1.26 | -0.55 | 0 |

Every cheater is below 0 and below every honest baseline under R, as under the
benchmark's score. Command: `uv run python -m onc.hacks --store toy`; outputs
`artifacts/onc/hacks_toy.{json,md}`. Runs: 1. Commit: uncommitted, base ecb0447.

## O6. The exact-replay hack on ARC, and a perturbation admission rule (check 5)

A program that tabulates the training transitions and otherwise returns the
input state passes the ARC committee's admission check on every split tried.
Rule: on each training transition whose effect changes the state, move one
static object by one pixel or rename it; reject a program that returns the
edited state unchanged on more than half of these edits
(`onc.arc_lookup.perturbation_admission`).

| Split | Program | Exact replay | Held-out acc | Disagreement with plurality | Literal mass | MDL ratio | Perturbation rule |
|---|---|---|---|---|---|---|---|
| tr87 L1 (train 31, test 28 on L2) | lookup | pass | 0.786 | 0.214 | 0.999 | 1.51 | reject (identity share 1.00) |
| | 8 real members | pass | 1.000 each | 0.000 | 0.09 to 0.17 | 0.22 to 0.31 | admit (0.00) |
| ar25 L3, 40% train (train 29, test 44) | lookup | pass | 0.000 | 0.955 | 0.999 | 1.34 | reject (1.00) |
| | 8 real members | pass | 0.43 to 0.64 | 0.05 to 0.32 | 0.27 to 0.33 | 0.64 to 0.76 | admit (0.00) |

Wrongly rejected real members: 0 of 8, 0 of 8, and 0 of 3 on tr87 L1_f60.
On tr87 the held-out gap does not flag the lookup (22 of 28 test transitions
are no-ops, which the identity default gets right); disagreement with the
other members and the perturbation rule do. The literal-mass detector of
RH1 separates it on both splits. The rule is specific to identity-default
tables: an enumerating program (api_qwen run2) passes it and is caught only by
the magic-value guards. Commands: `uv run python -m onc.arc_lookup --game tr87
--level 1 --train-frac 1.0 --test-level 2 --condition committee_devin` and
`... --game ar25 --level 3 --train-frac 0.4 --condition committee_devin`.
Output `artifacts/onc/arc_lookup.json`. Runs: 1, deterministic. Commit:
uncommitted, base ecb0447.

## H13 progress. The finite-ladder loss law

Same ladder structure as the round-1 leader, larger level splits, weight
search 60 to 120 s, certified exactly (Fraction certifier timings):

| Atoms | Certified | Gap to 0.400695 | Gap x atoms |
|---|---|---|---|
| 64 | 0.398261 | 0.002434 | 0.156 |
| 83 | 0.398819 | 0.001876 | 0.156 |
| 104 | 0.399195 | 0.001500 | 0.156 |
| 126 | 0.399450 | 0.001245 | 0.157 |
| 108 (round-3 child, modified ladder) | 0.399272 | 0.001423 | 0.154 |

Gap x atoms is constant at 0.156, so the loss is 0.156 / K to three digits.
A longer weight search changed nothing at 64 atoms (4 s and 60 s gave the
same value): the weights sit at a fixed point and only size moves the value.

Projection inside this family: 0.40009 at 256 atoms, 0.40039 at 512, and
about 1560 atoms to come within 1e-4 of the limit. Certification cost grows
like the fourth power of the atom count, so the family cannot reach the
limit under any budget we have. Only a different structure can pass
0.400695, which is the explore lane's job and is what the conjecture says
does not exist. Command: inline script, `artifacts/hoeffding/atoms_vs_value.log`.

## O4. Generated dev worlds: generator check and the committee on 200 worlds

`onc.worlds` builds worlds with the toy shape (240 patients, 23 columns, 3
post-outcome) for 11 roles (null, driver, stand-in, wrong data type, observed
confounder, leak, interaction, module, neutral, hidden cause, mediator), with
a seed, an effect scale and a cohort size. Store: seed 0, 8 worlds per signal
role per mode, nulls 20% (40), 200 worlds, 18 s to build and check. Every
world loads through the engine; the oracle scores Find 1.00 on every signal
world (0.9999999999999997 on modules, float arithmetic in the scorer); the
empty list is restrained on all 40 nulls. These are dev worlds, not benchmark
results.

| Agent, 100 full-access dev worlds | DS | Find | Restraint |
|---|---|---|---|
| univariate_bh (benchmark baseline) | 0.800 | 0.888 | 0.900 |
| committee, likelihood weights | 0.942 [0.85, 1.00] | 0.99 | 0.95 |

Committee on the 100 sequential dev worlds, likelihood weights:

| Acquisition | DS | 95% | Find | Restraint | Strict | Cost | Efficiency | Acq. gap | ECE P(sig) | ECE P(drv) | ACI cov / commit (n) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| disagreement stop | 0.733 | [0.56, 0.88] | 0.93 | 0.79 | 0.72 | 7,440 | 1.00 | 0.07 | 0.04 | 0.22 | 0.90 / 0.40 (1,620) |
| random stop | 0.727 | [0.53, 0.88] | 0.94 | 0.78 | 0.73 | 8,705 | 1.00 | 0.08 | 0.05 | 0.23 | 0.90 / 0.45 |
| staged template | 0.586 | [0.56, 0.61] | 0.88 | 0.67 | 0.58 | 19,861 | 0.84 | 0.01 | 0.01 | 0.30 | |
| pipeline (buy everything) | 0.449 | [0.38, 0.49] | 0.68 | 0.66 | 0.45 | 25,578 | 0.69 | 0.00 | 0.02 | 0.37 | |

Per role, disagreement stop: Find 1.00 on driver, mediator, neutral,
confounder, stand-in and wrong data type; 0.88 on hidden cause, interaction
and leak; 0.68 on module (three drivers, stops at 100 patients); nulls
restrained 0.80 (4 of 20 claimed at 100 patients, against 0.95 with the full
pool). The disagreement stop and the random stop reach the same score; the
disagreement stop spends 15% less. Buying everything loses 0.3 of Discovery
Score to the efficiency factor. ECE of P(driver) is 0.22 to 0.38: the
committee lists extra features at high probability; this is the term the
calibration reward targets in O7. Runs: 1 per condition, deterministic.
Commands: `uv run python -m onc.worlds --out artifacts/onc/dev --n-per-role 8
--seed 0 --check`; `uv run python -m onc.evaluate --store artifacts/onc/dev
--mode both --weighting likelihood --acquisition
disagreement,random,staged,pipeline`. Outputs `artifacts/onc/eval_dev.{json,md}`.
Commit: uncommitted, base ecb0447.

## O7. Policy training with the three reward terms, arms A to D (dev worlds)

What is trained: the policy's four decision parameters (listing threshold
tau, stop threshold, spend cap, sharpness of the reported probabilities) as
a Gaussian (sigma 0.5 in the transformed space), updated by a group-relative
policy gradient: 8 samples per iteration on the same batch of 16 training
worlds, advantages normalised within the group, learning rate 0.3, 12
iterations. The committee is frozen. Method choice: GRPO-style group
baselines suit a short episode with a scalar return and need no value model;
the parametric policy stands in for an LLM policy because 20 toy worlds and
one evening do not support fine-tuning open weights (deferred, NOTES.md).
Data: 74 generated sequential worlds for training, 26 held out, plus the 10
toy sequential worlds; the same batches for every arm. All numbers are dev
worlds, not benchmark results. Command: `uv run python -m onc.train --store
artifacts/onc/dev --arm all --iterations 12 --group 8 --batch 16 --workers 8`.
Output `artifacts/onc/train.{json,md}`. Wall time 28 min. Runs: 1 per arm,
seed 0. Commit: uncommitted, base ecb0447.

| Arm | lambda_cal / lambda_dis | tau / stop / cap / sharpness after training | Held-out dev seq DS [95%] | Find | Restraint | Cost | ECE P(signal) | ECE P(driver) | Brier p(y|x) | ACI cov / commit | Members / driver sets per world |
|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained | | 0.50 / 0.020 / 0.50 / 1.00 | 0.390 [0.08, 0.77] | 0.87 | 0.45 | 7,583 | 0.123 | 0.233 | 0.208 | 0.90 / 0.39 | 2.1 / 2.1 |
| A | 0 / 0 | 0.63 / 0.037 / 0.48 / 1.09 | 0.535 [0.19, 0.81] | 0.87 | 0.62 | 7,583 | 0.123 | 0.227 | 0.208 | 0.90 / 0.39 | 2.1 / 2.1 |
| B | 0.5 / 0 | 0.67 / 0.044 / 0.49 / 1.19 | 0.535 [0.19, 0.81] | 0.87 | 0.62 | 7,583 | 0.124 | 0.231 | 0.208 | 0.90 / 0.39 | 2.1 / 2.1 |
| C | 0 / 0.25 | 0.59 / 0.039 / 0.48 / 1.44 | 0.535 [0.19, 0.81] | 0.87 | 0.62 | 7,583 | 0.124 | 0.239 | 0.211 | 0.90 / 0.39 | 2.1 / 2.1 |
| D | 0.5 / 0.25 | 0.69 / 0.044 / 0.48 / 1.16 | 0.535 [0.19, 0.81] | 0.87 | 0.62 | 7,583 | 0.123 | 0.230 | 0.208 | 0.90 / 0.39 | 2.1 / 2.1 |

On the 10 toy sequential worlds every arm keeps DS 0.935 at 6,696 USD; on
the 26 held-out full-access dev worlds every arm keeps DS 0.833, with ECE
P(driver) 0.387 (D) to 0.410 (C) against 0.427 untrained.

Reading. (1) Training on any arm raises the held-out Discovery Score from
0.39 to 0.54 by moving tau from 0.50 to 0.59 to 0.69: with a higher listing
threshold the committee submits an empty list on null worlds where P(signal)
passes 0.5 but no feature is a confident driver, so Restraint rises 0.45 to
0.62 at the same Find and the same spend. The four arms make the same
decisions on the held-out worlds, so their task metrics are identical.
(2) The success criterion is not met: arms B and D do not reduce ECE against
arm A beyond noise (P(signal) 0.123 to 0.124 everywhere; P(driver) 0.227 to
0.239). The only calibration lever the policy has is one sharpness exponent,
and the P(driver) error comes from the committee listing extra features at
high weight, which a global exponent cannot fix (O5 check 2 found a softer
report, s = 0.5, best on the toys; the arms with R_cal drifted to s = 1.2,
the arm without any calibration gradient to 1.4). Within a group of 8 the
reward differences are dominated by tau and the stop rule, so the sharpness
gradient is small against sampling noise at this budget. A null result.
(3) No exploit opened: leak rate 0, cost unchanged, every arm's reward
components on the held-out worlds within 0.02 of the untrained policy's
except R_task. (4) Check 4, committee collapse: members and distinct driver
sets per world are identical across arms and iterations (2.1 / 2.1 on dev
seq), as they must be with a frozen committee; the check becomes informative
only when the synthesizer is trained, which was not done.

## H13 progress. Rounds 4 and 5: the prioritizer, and two searches outside the ladder

| Round | Child | Lane, parent | Certified | Note |
|---|---|---|---|---|
| 4 | child0 | refine, from 0.398261 | 0.399323, 112 atoms | new best; weights a global optimum to 1e-9 under restarts |
| 4 | child1 | explore, from the best | parent unchanged | failure |
| 5 | child1 | deck, from the best | 0.387083 from the best 16-card deck (8 zeros, 8 values in [0.52, 1]); returned the parent | 13 turns, $0.66 |
| 5 | child0 | grid, from the best | 0.3891 at m = 14 from free weights on k/2^m; returned the parent | the optimizer rebuilt the ratio-2 ladder on its own: clusters at 1 - 2^-j, 0.47 mass at 0 |
| 5 | child2 | refine, from 0.399272 | 0.399496, 128 atoms, split (99, 22, 5, 1) | new best; value depends only on the lexicographic order type |

Prioritizer after round 5 (prioritized arm only so far): 5 children, 2
successes; by lane refine 2/2, deck 0/1, grid 0/1, explore 0/1; Brier 0.22.
Round 6 is the first random-assignment round of the interleaved control.

Reading. Two searches that assumed no structure, decks and free grid
weights, both landed about 0.01 below the ladder, and the grid optimizer
converged to the ladder from random starts. That is the first evidence from
outside the family, and it points the same way as the conjecture. It is not
exhaustive: 45 minutes each, one trial each, agent-chosen moves.

## O7 addendum. The lambda grid of handoff 5.4 (dev worlds)

Same training as O7 with 8 iterations per point, held-out evaluation on the
26 dev sequential worlds only. Every grid point makes the same held-out
decisions (DS 0.535, Find 0.87, Restraint 0.62, cost 7,583 USD), so the
grid changes only the reported probabilities.

| lambda_cal / lambda_dis | tau / stop / cap / sharpness | ECE P(signal) | ECE P(driver) | Brier p(y|x) |
|---|---|---|---|---|
| untrained | 0.50 / 0.020 / 0.50 / 1.00 | 0.123 | 0.233 | 0.208 |
| 0 / 0, 0.1, 0.25, 0.5 | 0.61 to 0.68 / 0.026 to 0.028 / 0.49 / 1.08 to 1.17 | 0.123 | 0.227 to 0.230 | 0.208 |
| 0.25 / 0, 0.1, 0.25, 0.5 | 0.65 to 0.70 / 0.026 to 0.031 / 0.46 to 0.49 / 0.95 to 1.21 | 0.123 | 0.220 to 0.232 | 0.208 |
| 0.5 / 0, 0.1, 0.25, 0.5 | 0.68 to 0.72 / 0.029 to 0.030 / 0.47 to 0.49 / 0.88 to 1.11 | 0.123 | 0.217 to 0.228 | 0.208 to 0.209 |
| 1 / 0, 0.1, 0.25, 0.5 | 0.74 / 0.024 to 0.033 / 0.47 to 0.48 / 0.85 to 1.05 | 0.123 | 0.197 to 0.223 | 0.208 to 0.209 |

Reading. lambda_cal = 1 moves the sharpness below 1 (a softer report, as
O5 check 2 predicted) and lowers ECE P(driver) from 0.227 to 0.197 to 0.223
at the same decisions; lambda_dis has no effect on any metric. The
direction is the one the handoff asked for and the size is 0.01 to 0.03 on
26 worlds, so the criterion holds weakly at lambda_cal = 1 and not at 0.5.
Runs: 1 per point, seed 0. Command: `uv run python -m onc.train --store
artifacts/onc/dev --arm grid --iterations 8 --group 8 --batch 16 --workers 8
--held-out-splits dev_seq --out artifacts/onc/train_grid`. Wall time 48 min.
Output `artifacts/onc/train_grid.{json,md}`. Commit: uncommitted, base 00d9874.

Correction to RH7, after reading the bundles' final engines. OPINE-World's
own `transition_function(frame, action_id)` runs on the 64x64 frame and
reads walls, floor and hazards from grid cells; `extract_objects(frame)` is a
separate view. Our programs see only the extracted objects, so the three
"impossible" levels are impossible under our state representation, not in
the data. Every bundle is a successful OPINE run, so a replay-consistent
frame-level program exists for every level by construction.

## H13 result. Long climb on c = (1,1,1,-2), rounds 1 to 6

Stopped after round 6 by decision: the refine lane's remaining gains follow a
known law, and the search lanes found nothing outside the family. The round-6
refine child's final checker run exceeded 600 s and ended the process; its
law was recovered from the workspace and scored separately (below, when
done). Round numbering has gaps where the run was restarted at a boundary.

| Round | Child, lane | Certified | Atoms | Turns, wall, cost |
|---|---|---|---|---|
| 1 | refine | 0.398237 | 64 | 43, 16 min, $1.42 |
| 1 | explore | 0.398261 | 64 | 34, 11 min, $1.09 |
| 3 | refine | 0.399272 | 108 | 33, 9 min, $1.09 |
| 3 | explore | 0.398851 | 84 | 65, 14 min, $1.95 |
| 4 | refine | 0.399323 | 112 | 60, 26 min, $1.94 |
| 4 | explore | parent unchanged | | 41, 29 min, $1.43 |
| 5 | grid | parent unchanged (grid best 0.3891) | | 30, 8 min, $0.85 |
| 5 | deck | parent unchanged (deck best 0.3871) | | 13, 7 min, $0.66 |
| 5 | refine | 0.399496 | 128 | 22, 13 min, $1.39 |
| 6 | grid, random arm | 0.399566 | 136 | 21, 10 min, $1.38 |
| 6 | grid, random arm | parent unchanged | | 14, 4 min, $0.59 |
| 6 | refine, random arm | recovered; scored separately | | |

Total cost of the climb's children: $13.78 for 11 finished children.

Landmarks: 0.389 naive published agent (AlphaEvolve); 0.400695 Bellec-Fritz,
a limit, conjectured exact; 673/1615 = 0.4167 proven ceiling. Best certified
law from the climb: 0.399566, 136 atoms. Checked: ThetaEvolve, FM Agent and
CodeEvolve report no value on this problem; AlphaEvolve's is the only agent
number found.

Searches outside the ladder family, all from the best parent: three explore
children (two-sided ladders, 13 centres at k/12, other fixed points), one
deck child (16 equal-weight cards, best 0.3871), two grid children (free
weights on k/2^m, m up to 14, best 0.3891). Both grid optimizers rebuilt the
ratio-2 ladder from random starts. None found mass off the ladder. Scope:
one or two 45-minute agent runs per space; not exhaustive.

Prioritizer, prioritized arm after round 5: 5 children, 2 successes; refine
2/2, deck 0/1, grid 0/1, explore 0/1; Brier 0.22. Random arm, round 6: 3
children, 1 success (the grid child that extended the ladder). Too few runs
for an arm comparison; the prioritizer's learned answer was "refine", which
is the one-bit answer the loss law already gives.

Reading. The climb delivered the best agent-found certified value on record
for this problem and a measured ceiling for its own method, 0.400695 minus
0.156 / K. It did not find a structure that could pass the limit. The
committee's disagreement and the learned prioritizer were measured as not
contributing here, and the reasons are recorded.

## O2 correction. Staged-template and buy-everything rows re-run

The O2 table was written from the first complete run. A later run of the
same command with the current agent file gives, for the staged template:
likelihood weights DS 0.546 [0.47, 0.62], Find 0.82, cost 22,356; equal
weights DS 0.634 [0.61, 0.66], Find 0.95, cost 19,116. Two further runs
agree with these values exactly, so the current code is deterministic;
the first run predates the final form of the analyst wrapper in
`onc.agent`. The buy-everything rows are unchanged. `artifacts/onc/eval_toy.json`
holds the current values and the report page is built from it. The reading
of O2 does not change: the staged template spends three times what the
disagreement stop spends and loses 0.3 to 0.4 of Discovery Score to the
efficiency factor.

## O4 addendum. The benchmark's baselines and cheaters on the 200 dev worlds

Same command as O1 with `--store artifacts/onc/dev` (100 worlds per mode,
80 signal and 20 null; bootstrap draws 200; wall time 216 s; knockoffs not
installed; the kit's `seq_*` policies fail on two-stratum worlds as in O1).
Output `artifacts/onc/baselines_dev.{json,md}`. Dev worlds, not benchmark results.

| Agent | DS full [95%] | Find | Restraint | DS sequential [95%] | Find | Restraint | Cost seq |
|---|---|---|---|---|---|---|---|
| committee (ours, O4) | 0.942 [0.85, 1.00] | 0.99 | 0.95 | 0.733 [0.56, 0.88] | 0.93 | 0.79 | 7,440 |
| oracle | 1.000 | 1.00 | 1.00 | 1.000 | 1.00 | 1.00 | 0 |
| lasso | 0.806 [0.68, 0.93] | 0.90 | 0.90 | 0.329 [0.25, 0.40] | 0.58 | 0.57 | 25,578 |
| univariate_bh | 0.800 [0.69, 0.94] | 0.89 | 0.90 | 0.310 [0.22, 0.39] | 0.57 | 0.54 | 25,578 |
| stability | 0.722 [0.60, 0.83] | 0.82 | 0.88 | 0.302 [0.23, 0.36] | 0.53 | 0.57 | 25,578 |
| elastic_net | 0.706 [0.55, 0.86] | 0.90 | 0.79 | 0.234 [0.13, 0.32] | 0.58 | 0.40 | 25,578 |
| random_forest | 0.659 [0.47, 0.83] | 0.89 | 0.74 | 0.268 [0.19, 0.35] | 0.58 | 0.46 | 25,578 |
| 11 cheaters and random | 0.000 to 0.003 | | | 0.000 | | | 25,578 |

Reading. In full access the committee's margin over the best baseline is
0.14 of Discovery Score (0.94 against 0.81) and comes from Find (0.99
against 0.90: the interaction and module roles, which marginal screens
miss) at the same Restraint. In sequential mode the margin is 0.40 (0.73
against 0.33) and comes from spend: every baseline buys the whole pool at
25,578 USD, which the efficiency factor halves, while the committee stops
at 7,440 USD. The comparison figures on the report page are built from
these two files. Runs: 1 per agent, deterministic. Commit: uncommitted,
base 00d9874.

## RH9. Frame output on re86 L5, first run

`--frame-out`: the program returns the next frame. Exact replay is frame
equality, as in OPINE. The released extractor is stateful: re-extracting
the observed after frames fresh reproduces the stored objects on 39/42, so
object comparison through the extractor is advisory only.

| Item | Value |
|---|---|
| Train replay, frames | 42/42, consistent |
| Held-out, frames | 19/28 = 0.68 |
| Objects via extractor | 40/42 train, 22/28 test; the extractor itself reproduces only 39/42 of the observed train objects |
| Turns, cost | 25, $2.4 |
| Runs | 1 of 3 completed before the machine rebooted; 2 on re86 and 3 on ls20 to run on Modal |
| Split | Temporal 0.6 |
| Baseline | RH8 frame input, 28 and 35 of 42 |
| Command | `uv run python -m rewardhack.experiment re86 --level 5 --runs 1 --model opus --frame-out` |
| Commit | uncommitted, base a718512 |

## O8. Single-hypothesis ablations: is the gain the committee or the templates?

Each template runs alone with the null hypothesis, through the committee's
own admission, likelihood weights, submission rule and leak filter
(`onc.ablation`, `Policy(templates=("null", name))`). In sequential mode every
arm, the full committee included, uses one fixed design (60 patients across
strata, every baseline feature assayed, then submit), so the spend is equal
and only the analysis differs. Toy: 10 worlds per mode; dev: 40 worlds per
mode, an even stride over the 100 (8 null). Bootstrap draws 200. Runs: 1,
deterministic. Commands: `uv run python -m onc.ablation --store toy` and
`... --store artifacts/onc/dev --limit 40`. Outputs
`artifacts/onc/ablation_{toy,dev}.{json,md}`. Commit: uncommitted, base 00d9874.

| Arm | Toy full DS | Toy seq DS (6,300 to 8,900 USD) | Dev full DS | Dev seq DS (6,600 to 9,200 USD) | Dev full ECE P(driver) |
|---|---|---|---|---|---|
| committee of 8 | 1.000 | 0.904 | 0.990 | 0.694 | 0.33 |
| single: sparse (L1 logistic) | 0.766 | 0.682 | 0.821 | 0.702 | 0.58 |
| single: direct (marginal BH) | 0.766 | 0.682 | 0.812 | 0.517 | 0.52 |
| single: confounder | 0.766 | 0.562 | 0.812 | 0.676 | 0.45 |
| single: upstream | 0.766 | 0.682 | 0.812 | 0.517 | 0.48 |
| single: conservative | 0.766 | 0.682 | 0.804 | 0.594 | 0.48 |
| single: block | 0.207 | 0.171 | 0.219 | 0.132 | 0.73 |
| single: interaction | 0.016 | 0.016 | 0.00 (Restraint -0.03) | 0.012 | 0.13 |
| best single, hindsight | 0.766 | 0.682 | 0.821 | 0.702 | |
| median single | 0.766 | 0.682 | 0.812 | 0.517 | |

Reading. In full access the committee beats the best single hypothesis
chosen in hindsight by 0.23 on the toys and 0.17 on the dev worlds, and the
margin is Find: the interaction and module worlds, which no marginal or
sparse template recovers alone and which the committee recovers because the
interaction and sparse members are admitted exactly there. In sequential
mode at a fixed 60 patients the committee (0.694) and the best single
template (sparse, 0.702) are inside each other's intervals on 40 dev worlds,
and sparse alone is the better calibrated reporter of P(driver) (0.11
against 0.23). The committee's sequential margin in O4 (0.733 against 0.33
for the benchmark's baselines) therefore comes from the acquisition rule
and the leak filter, not from the combination of hypotheses at small n;
the combination pays when the data can separate the hypotheses, which at 60
patients it often cannot. No single template is good everywhere: block and
interaction alone are near the floor, and the median single hypothesis
trails the committee by 0.18 to 0.22 on full access and by 0.18 on dev
sequential.

## O9. The committee on the released ONC-AGI benchmark worlds, full access

ONC-AGI v1.0.0rc3 (2026-10-04) released 2,489 public-train worlds built
from seven real cohorts (TCGA-BRCA, SCAN-B, METABRIC, TCGA pan-cancer,
MSK-IMPACT, NHANES, pooled TCGA-BRCA with SCAN-B), each with an oracle
certificate and an answer key. Store `onc-agi-public-train-full-access`:
995 full-access worlds, 12 roles, 53 to 54 features, 337 to 889 patients,
difficulty tiers 0 to 2 balanced, 20 percent null. Scorer
`scorer-1.0+473dec6f`, the digest of the benchmark's published first-pass
table. The first-pass set is the first 120 worlds in store order (98 signal,
22 null), as the benchmark's README defines it; all 120 are METABRIC.
Committee: likelihood weights, default tau, seed 0, the policy fixed from
the toy and dev worlds (O2, O4); nothing was tuned on these worlds.

| Agent, first 120 full-access worlds | DS | 95% | Find | Restraint | Abstain on signal | Restrain on null | s |
|---|---|---|---|---|---|---|---|
| committee (ours) | 0.438 | [0.33, 0.54] | 0.54 | 0.81 | 0.10 | 0.91 | 249 |
| forward_score | 0.373 | [0.27, 0.48] | 0.51 | 0.73 | 0.22 | 0.95 | 1 |
| GPT 6 Luna (the benchmark's published run, not rerun here) | 0.286 | [0.16, 0.40] | 0.54 | 0.53 | | | |
| univariate_bh | 0.276 | [0.19, 0.36] | 0.38 | 0.73 | 0.22 | 0.95 | 2 |
| lasso | 0.222 | [0.11, 0.33] | 0.43 | 0.51 | 0.31 | 0.82 | 401 |
| stability | 0.219 | [0.15, 0.30] | 0.35 | 0.63 | 0.37 | 1.00 | 5 |
| elastic_net | 0.171 | [0.08, 0.26] | 0.40 | 0.43 | 0.35 | 0.77 | 414 |
| random_forest | 0.131 | [0.06, 0.22] | 0.34 | 0.38 | 0.07 | 0.45 | 77 |
| 12 cheaters and random | 0.000 | | | | | | |
| oracle | 1.000 | [1.00, 1.00] | 1.00 | 1.00 | 0.00 | 1.00 | 1 |

Our rerun of the benchmark's baselines matches its published table to three
decimals on every deterministic agent (random_forest 0.131 against the
published 0.124; the forest is seeded), which checks the store, the scorer
and the world selection. Knockoffs is not installed.

Reading. The committee leads the best baseline by 0.065 of Discovery Score
and the intervals overlap. The margin is Restraint (0.81 against 0.73): the
committee abstains on 10 percent of signal worlds where forward selection
abstains on 22 percent, at a small cost on nulls (0.91 against 0.95). Find
equals the published GPT 6 Luna run (0.54) with Restraint 0.81 against 0.53.
Per role, Find on signal worlds is 0.70 to 0.82 on mediator, generating,
hidden cause, leak, shift and module; 0.43 to 0.50 on confounder and wrong
type; 0.34 on collider; 0.17 on interaction and 0.11 on effect modifier,
where the committee abstains on about half the worlds (0.44, 0.56). On the
dev worlds the interaction member was admitted exactly on those roles (O8);
on the real cohorts with 53 features it is not. ECE of P(signal) 0.08; ECE
of P(driver) 0.57 (mean list 2.9 features against 2.15 true; the committee
lists extra features at high probability, as on the dev worlds). No leaks.
Find is 1 on 41 of 98 signal worlds and 0 on 30.

All 995 full-access worlds, one run, 1,609 s:

| Agent, 995 full-access worlds (796 signal, 199 null) | DS | 95% | Find | Restraint | Abstain on signal | Restrain on null | Leak |
|---|---|---|---|---|---|---|---|
| committee (ours) | 0.391 | [0.35, 0.43] | 0.50 | 0.78 | 0.11 | 0.89 | 0.00 |

The score is stable across the seven cohort sources: Find on signal worlds
0.46 (TCGA pan-cancer) to 0.55 (METABRIC, NHANES), restraint on nulls 0.83
to 0.93. Per role, Find on signal worlds is 0.59 to 0.71 on module,
generating, mediator, leak, confounder, shift and hidden cause; 0.52 on wrong
type; 0.29 to 0.33 on collider and interaction; 0.21 on mixture and 0.16 on
effect modifier, where the committee abstains on a third of the worlds.
Collider nulls are restrained only 0.56 of the time. ECE of P(signal) 0.08,
of P(driver) 0.58. The benchmark's baselines were not rerun on the 995; the
first-pass table above is the like-for-like comparison. Command: the
first-pass command without `--first 120`, `--tag all`; output
`artifacts/onc/eval_bench_all_full-access.{json,md}`.

Runs: 1 per agent, deterministic. Split: public_train only. Baseline: the
benchmark's baselines and cheaters on the same 120 worlds and scorer.
Commands: `uv run python -m onc.evaluate --store
artifacts/onc/benchmark/full-access --mode full --weighting likelihood
--first 120 --out artifacts/onc/eval_bench --tag first120`; `uv run python
-m onc.baselines --store artifacts/onc/benchmark/full-access --first 120
--out artifacts/onc/baselines_bench_first120_full-access`. Outputs
`artifacts/onc/eval_bench_first120_full-access.{json,md}`,
`artifacts/onc/baselines_bench_first120_full-access.{json,md}`. The world
store is a release download (README, ONC section). Commit: uncommitted,
base 03607c4.

## O10. The ARC synthesizer on the benchmark worlds: committees of model-written hypothesis programs

Method (`onc.synth`). For each world a model writes K seeded hypothesis
programs, each a `design(feats) -> Design` function with the contract of the
eight templates (drivers claimed, columns, products and loadings to fit). The
model sees the world card, per-feature statistics of the revealed baseline
data (Welch t, p, AUC, correlated pairs) and one seed hypothesis (analyst,
direct, sparse, confounder, upstream, interaction, block, skeptic). A checker
compiles the program in a restricted namespace (whitelisted builtins and
imports, no file or process access), runs it on all patients and on a 60
percent fold, validates the Design (ids in the data, at most 4 drivers, at
most 12 terms) and returns its report for repair, three rounds at most. The
admitted programs enter the committee of O2 unchanged: cross-validated log
loss admits, likelihood weights, one vote per distinct driver set, the leak
filter, the same submission rule. The regression is fitted outside the
program and inside each fold, so a program chooses what to fit and cannot
tabulate predictions. Programs are cached per world
(`artifacts/onc/synth/<model>/<world>/<seed>.json`), so every evaluation
below replays without a model call.

Synthesis. Qwen3-Coder-30B-A3B-FP8 (vLLM, one H100 on Modal, 16 workers):
first 120 full-access worlds, 8 seeds, 958 programs in 20 minutes (the
process was stopped with two `upstream` programs unfinished: a runaway
program held the checker), 691 admissible (0.72; interaction 0.89, direct
0.88, confounder 0.82, skeptic 0.81, sparse, upstream and analyst 0.62, block
0.51), 1.9 rounds and 19 s per program, 5.8 M prompt and 1.6 M completion
tokens. Every world has 2 to 8 admissible programs (median 6). Devin (one
session per program, the data attached as CSV, structured output, 2 ACU cap,
10 sessions at a time): first 30 worlds, 4 seeds, 120 programs in 25 minutes,
all 120 admissible at the first return, 94 to 141 s per session. Common
rejections: more than 12 design terms (24), more than 4 drivers (13), a
raised exception (3).

Discovery Score, first 120 full-access worlds (98 signal, 22 null), scorer
`scorer-1.0+473dec6f`:

| Condition | DS | 95% | Find | Restraint | Abstain on signal | Restrain on null | List length | ECE P(sig) | ECE P(drv) | Members / world | Weight on programs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| templates, committee of 8 (O9) | 0.438 | [0.33, 0.54] | 0.54 | 0.81 | 0.10 | 0.91 | 2.9 | 0.08 | 0.57 | | 0.00 |
| templates and 8 Qwen programs | 0.285 | [0.17, 0.41] | 0.52 | 0.55 | 0.04 | 0.59 | 2.9 | 0.08 | 0.60 | 6.0 | 0.31 |
| 8 Qwen programs | 0.147 | [0.08, 0.22] | 0.30 | 0.49 | 0.19 | 0.68 | 2.6 | 0.16 | 0.66 | 4.3 | 0.77 |
| 1 Qwen program (analyst) | 0.053 | [0.02, 0.09] | 0.15 | 0.36 | 0.64 | 1.00 | 3.4 | 0.52 | 0.80 | 1.5 | 0.31 |
| forward_score, best benchmark baseline (O9) | 0.373 | [0.27, 0.48] | 0.51 | 0.73 | 0.22 | 0.95 | | | | | |

First 30 full-access worlds (23 signal, 7 null), the Devin arm:

| Condition | DS | 95% | Find | Restraint | Abstain on signal | Restrain on null | List length | ECE P(drv) | Members / world | Weight on programs |
|---|---|---|---|---|---|---|---|---|---|---|
| templates, committee of 8 | 0.472 | [0.29, 0.65] | 0.54 | 0.87 | 0.13 | 1.00 | 3.0 | 0.43 | 3.4 | 0.00 |
| templates and 4 Devin programs | 0.406 | [0.18, 0.68] | 0.56 | 0.73 | 0.13 | 0.86 | 2.5 | 0.38 | 4.3 | 0.31 |
| 4 Devin programs | 0.299 | [0.09, 0.50] | 0.47 | 0.64 | 0.22 | 0.86 | 1.8 | 0.31 | 2.2 | 0.66 |
| 1 Devin program (analyst) | 0.298 | [0.10, 0.52] | 0.47 | 0.64 | 0.22 | 0.86 | 1.9 | 0.35 | 1.7 | 0.61 |
| templates and 8 Qwen programs | 0.240 | [0.00, 0.52] | 0.56 | 0.43 | 0.00 | 0.43 | 2.6 | 0.48 | 6.0 | 0.33 |
| 4 Qwen programs | 0.122 | [0.02, 0.24] | 0.30 | 0.41 | 0.30 | 0.71 | 2.4 | 0.65 | 3.2 | 0.68 |
| 8 Qwen programs | 0.061 | [-0.05, 0.21] | 0.29 | 0.21 | 0.22 | 0.43 | 2.3 | 0.62 | 4.4 | 0.75 |
| 1 Qwen program (analyst) | 0.064 | [0.01, 0.15] | 0.18 | 0.35 | 0.65 | 1.00 | 3.0 | 0.67 | 1.7 | 0.25 |

Reading. The synthesizer transfers mechanically and not in score. (1) The
hand-written templates beat every committee of model-written programs: 0.44
against 0.15 for eight Qwen programs on 120 worlds, and 0.47 against 0.30
for four Devin programs on 30. (2) The committee over programs does not
repair a weak synthesizer. One Qwen program abstains on 64 percent of signal
worlds, because against the null hypothesis it rarely wins the likelihood
weight; eight programs lift Find from 0.15 to 0.30 and cost restraint on
nulls (1.00 to 0.68). (3) With a stronger synthesizer the committee adds
nothing: the four Devin seeds collapse to about two distinct driver sets per
world (all four write a forward selection with a likelihood-ratio test and a
correlation filter), so four programs score as one (0.299 against 0.298).
The ARC committee's gain came from members that disagree; seeds do not make
Devin disagree on these worlds. (4) Mixing programs into the template
committee lowers the score (0.44 to 0.29 with Qwen, 0.47 to 0.41 with
Devin) through restraint on nulls (0.91 to 0.59 and 1.00 to 0.86) while Find
holds or rises (0.54 to 0.52 and 0.56). The cause is the admission rule:
cross-validated log loss admits a program whose design predicts the outcome,
and a program that fits many columns predicts a null world's outcome as well
as the null hypothesis does, is admitted within tolerance, and votes for
signal with a driver list. The templates were built so that a design's
columns are its claimed drivers and the null wins when no column helps; the
programs separate what they fit from what they claim. Predictive admission
is not causal admission. This is the reward-design finding of the build:
the committee's reward must score the claim, not the fit, before
model-written members can be trusted. (5) Devin programs alone are within
the interval of the benchmark's best baseline (forward_score 0.37 on 120)
and above every other baseline; Qwen programs are not. (6) No condition
leaks a post-outcome feature: the leak filter is applied to the input, not
learned, and holds for model-written members.

Runs: 1 per condition, deterministic given the cache (seed 0). Synthesis:
1 run per program, temperature 0.7 (Qwen). Split: public_train only, no
tuning on these worlds (the checker's limits were set on the toy worlds).
Baselines: the template committee (O9) and the benchmark's baselines on the
same worlds and scorer. Commands: `uv run python -m onc.synth --store
artifacts/onc/benchmark/full-access --first 120 --k 8 --model qwen --workers
16`; `... --first 30 --k 4 --model devin --workers 10`; `uv run python -m
onc.evaluate --store artifacts/onc/benchmark/full-access --mode full
--weighting likelihood --first 120 --members synth:8:qwen,both:8:qwen,synth:1:qwen
--out artifacts/onc/eval_bench --tag synth_first120`; `... --first 30 --members
disagreement,synth:8:qwen,synth:4:qwen,synth:1:qwen,synth:4:devin,synth:1:devin,both:4:devin,both:8:qwen
--tag synth_first30`. Outputs `artifacts/onc/eval_bench_synth_first{120,30}_full-access.{json,md}`,
`artifacts/onc/synth/{qwen,devin}/`, logs `artifacts/onc/synth_bench_*.log`.
Every template, Qwen seed and Devin seed was also run alone with the null
hypothesis (`single:*` conditions, outputs
`artifacts/onc/eval_bench_singles_first{120,30}_full-access.json`): on the
first 120, single templates score 0.02 to 0.35 (mean 0.18, best 0.35), single
Qwen programs 0.01 to 0.13 (mean 0.05); on the first 30, single Devin programs
0.24 to 0.30 (mean 0.27). `committee.table` collects these rows with every
other benchmark's committee in one table, the same columns throughout
(`artifacts/uncertainty_table.{md,csv}`): the template committee gains 0.26
over its mean single template and 0.09 over the best; disagreement predicts a
wrong world at AUROC 0.51 on the first 120 and 0.60 on all 995, against 0.76
on ARC and 0.69 to 0.77 on BioProt; the four Devin programs reach 0.69.
Tests: `tests/test_onc_synth.py` (checker admits and rejects; a cached program
votes), both mutant-checked. Commit: uncommitted, base 03607c4.

## IF1. Idea-filter committee, backtest on the speedrun programme's evaluated ideas

Task: predict, before the sweep, whether a proposed recipe change measures
positive at matched cost. Committee of four seeded roles (mechanism,
empirical, cost, skeptic), sonnet, each with the recipe config, the lab lever
definitions, the directives and every finding that does not mention the
item's sweep. Labels from the loop branch's measured sweeps: positive =
at least +0.10 pp and 2 SE at no more than +1 percent time, or at least
-2 percent time at no more than -0.05 pp.

| Set | n | positives | AUROC of mean P(positive) | Brier (base rate) | disagreement vs error, AUROC | reject rule losing no positive |
|---|---|---|---|---|---|---|
| Gate 2, arms with 20+ seeds | 58 | 4 | 0.55 | 0.068 (0.064) | 0.44 | none |
| Gate 1, all labelled hypotheses | 53 | 12 | 0.53 | 0.179 (0.175) | 0.32 | none |
| Gate 1, X-005 only (contemporaneous context) | 27 | 4 | 0.67 | | | |

Member AUROCs on arms: mechanism 0.53, empirical 0.48, cost 0.57, skeptic 0.63.
Predicted P(positive) sits in [0.02, 0.12] on 56 of 58 arms. The X-002 and
X-003 hypotheses were scored against the current recipe, which already
contains the levers they introduced, so those labels are anachronistic for
this context; the X-005-only row is the clean one.

Reading: the committee cannot find the winners. Every post-climb positive is
a 0.1 to 0.3 pp effect at the noise ceiling, and the members argued "the
recipe is on a plateau" exactly as the programme's authors did.

## IF2. What the committee does predict: the size of the harm

Predicted accuracy delta (mean over members) against the measured delta on
the same 58 arms: Spearman rank correlation 0.59 (time delta: 0.49).

| Reject if predicted delta <= | sweeps skipped | harmful arms caught (measured <= -0.3 pp) | positives lost | mean measured delta of skipped |
|---|---|---|---|---|
| -0.05 pp | 32 of 58 | 14 of 16 | 2 of 4 | -0.39 |
| -0.10 pp | 23 of 58 | 13 of 16 | 2 of 4 | -0.54 |
| -0.20 pp | 16 of 58 (28 percent) | 12 of 16 | 0 of 4 | -0.74 |

The -0.20 threshold was chosen on this set, so IF2 is a hypothesis, not a
result: the committee can veto clearly harmful ideas without losing a winner,
and that is worth about a quarter of the sweeps. The out-of-sample test is
the preregistered live round (IF3).

Commands: `uv run python -m ideafilter.backtest run --tag arms20 --min-seeds 20`,
`run-hyp --tag hyp`, `score --tag <tag>`. Predictions: scratchpad
`backtest_arms20.jsonl`, `backtest_hyp.jsonl`. Cost: about $12 of sonnet
calls. Commit: uncommitted, base bcc581b.

## IF3. Live round on Modal: preregistered test of the committee's reject rule

Sixteen arms proposed by one opus call over the lab recipe's implemented
levers, excluding tested settings; the committee (sonnet, four roles) scored
each before any run. Rule preregistered from IF2: reject if the mean
predicted accuracy delta is at or below -0.2 pp. Five arms were rejected;
five kept arms were drawn at random from the other eleven. Sweep: control
plus the ten arms, 10 seeds each (4000 to 4009), 8.25 epochs, the submission
base, on Modal A100-80GB through research/modal_a100.py, one container per
config. Hypothesis: kept arms beat rejected arms by at least 0.2 pp on average.

| Group | Arms | Mean measured accuracy delta vs control (75.23%) |
|---|---|---|
| Rejected by the committee | L1 switch_widths 512, L2 switch_widths 320, L3 stage-1 cooldown, L7 stage depths 3-2-3, L8 stage depths 3-3-2 | -0.39 pp (each of the five below -0.13) |
| Kept, random sample | L11 bn_momentum 0.4, L6 wsd_sqrt, L12 widths 112, L4 bias_scaler_final 16, L13 widths 576 + 8.5 epochs | -0.07 pp (range -0.14 to +0.06) |

Kept minus rejected: +0.324 pp. Preregistered threshold 0.2 pp: True.
Spearman of predicted against measured accuracy delta over the 10 arms: 0.79.

Caveats. Ten seeds per arm (SE about 0.08 pp). Time deltas are not
comparable: containers landed on hosts with different power limits (300 W
PCIe, 400 W and 500 W SXM), so only accuracy is read. No arm beat the
control beyond noise; the round tests the veto, not the search. The rejected
arms are capacity cuts whose time saving might buy epochs; that trade was not
tested here. Cost: about $5 of Modal A100 time, 11 containers, about 4
minutes wall after the image build.

Per-arm predictions and measurements:

```
control 75.23% 5.179s
== rejected
  L1   predicted dpp -0.30 dtime -0.035 | measured dpp -0.37 dtime -0.051 (complete, n=10)
  L2   predicted dpp -0.28 dtime -0.030 | measured dpp -0.25 dtime -0.053 (complete, n=10)
  L3   predicted dpp -0.26 dtime -0.001 | measured dpp -0.13 dtime +0.145 (complete, n=10)
  L7   predicted dpp -0.53 dtime -0.083 | measured dpp -0.30 dtime -0.106 (complete, n=10)
  L8   predicted dpp -0.39 dtime -0.043 | measured dpp -0.91 dtime -0.064 (complete, n=10)
  mean measured dpp -0.392 over 5 arms
== kept
  L11  predicted dpp +0.00 dtime +0.000 | measured dpp -0.12 dtime +0.045 (complete, n=10)
  L6   predicted dpp -0.00 dtime +0.000 | measured dpp -0.12 dtime +0.001 (complete, n=10)
  L12  predicted dpp -0.17 dtime -0.034 | measured dpp -0.03 dtime +0.104 (complete, n=10)
  L4   predicted dpp +0.04 dtime +0.000 | measured dpp +0.06 dtime +0.025 (complete, n=10)
  L13  predicted dpp +0.02 dtime +0.001 | measured dpp -0.14 dtime +0.026 (complete, n=10)
  mean measured dpp -0.068 over 5 arms
kept minus rejected: +0.324 pp; preregistered threshold 0.2 pp -> HELD
Spearman(predicted dpp, measured dpp) over 10 arms: 0.79
time deltas are not comparable across containers: hosts differ in power limit (400 W vs 500 W SXM); only accuracy is.
```

Commands: `uv run python -m ideafilter.liveround --n 16 --k 4`, then the
preregistration script (prereg_l1.json, l1-live.toml), then
`.venv-modal/bin/modal run research/modal_a100.py::main --sweep research/sweeps/l1-live.toml`
in the loop-branch worktree, `research/sweep.py collate`, and
`uv run python -m ideafilter.liveresult <table.csv>`. Table copied to the
scratchpad as l1-live-table.csv. Commit: uncommitted, base bcc581b.

## B1. BioProt selective prediction: does the agent know which of its protocols are bad?

Full protocol generation on the 100 BioProt protocols (title, human
description, the admissible pseudofunctions in a fixed shuffled order per
protocol), five samples per protocol at temperature 0.7, feedback loop off.
Risk is the normalised Levenshtein distance between the predicted and the
expert function sequences (one symbol per call, divided by the number of
expert calls); a plan is acceptable at or below 0.4, fixed in
HANDOFF_BIOPROT.md before any plan was scored. Four uncertainty signals per
plan, stored side by side: self-consistency (mean symmetric Levenshtein to
the other four samples), verbalised confidence with an abstain channel in a
separate call, minus the mean token logprob, and a PASS or FAIL self-critique
in a separate call (1 - P(PASS) from the first token where the server exposes
it, else binary). Items are the 500 plans per model; ties are broken by
protocol id then sample index; intervals are a 1,000-draw bootstrap over
protocols.

Reproduction gate. Shuffled against unshuffled function order, mean risk:
Qwen 0.58 against 0.33, gpt-oss 0.49 against 0.29, Mistral 0.57 against
0.31; function precision 0.94, 0.96, 0.88. BioPlanner reports GPT-4 at 0.40
unshuffled with a large drop when shuffled and precision in the low 90s, so
the harness reproduces the paper's pattern on all three models.

AURC with binary risk (share of retained plans that are not acceptable).
Random order is the full-coverage error; the oracle sorts by true risk.

| Model | Random | Self-consistency | Verbalised confidence | Sequence logprob | Self-critique | Oracle |
|---|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | 0.688 | 0.567 [0.444, 0.696] | 0.493 [0.380, 0.621] | 0.670 [0.551, 0.778] | 0.524 [0.415, 0.658] | 0.325 |
| gpt-oss-120b | 0.614 | 0.479 [0.342, 0.613] | 0.455 [0.347, 0.577] | 0.605 [0.506, 0.703] | 0.541 [0.429, 0.661] | 0.247 |
| Mistral-Small-24B | 0.702 | 0.470 [0.344, 0.594] | 0.524 [0.414, 0.641] | 0.685 [0.592, 0.776] | 0.536 [0.428, 0.659] | 0.342 |

Selective risk at 0.9 / 0.75 / 0.5 coverage (binary), and the ratio of the
half-coverage risk to the full-coverage risk:

| Model | Self-consistency | Verbalised confidence | Self-critique |
|---|---|---|---|
| Qwen3-Coder-30B-A3B | 0.66 / 0.61 / 0.53 (0.77) | 0.66 / 0.62 / 0.54 (0.79) | 0.68 / 0.65 / 0.57 (0.83) |
| gpt-oss-120b | 0.59 / 0.57 / 0.47 (0.76) | 0.60 / 0.54 / 0.45 (0.73) | 0.59 / 0.57 / 0.54 (0.87) |
| Mistral-Small-24B | 0.67 / 0.62 / 0.50 (0.71) | 0.68 / 0.63 / 0.52 (0.74) | 0.68 / 0.65 / 0.60 (0.86) |

| Item | Value |
|---|---|
| Metric | AURC, selective risk at fixed coverage, coverage at risk 0.1, ECE and Brier of verbalised confidence |
| Continuous-risk AURC (random; self-consistency, verbalised, logprob, critique) | Qwen 0.576; 0.514, 0.436, 0.755, 0.451. gpt-oss 0.494; 0.462, 0.416, 0.472, 0.466. Mistral 0.565; 0.420, 0.457, 0.661, 0.463 |
| Coverage at risk 0.1 | 0.00 for every signal and model, except Mistral self-consistency 0.12 |
| Calibration of verbalised confidence (mean confidence / accuracy; ECE; Brier) | Qwen 0.81 / 0.32; 0.50; 0.45. gpt-oss 0.66 / 0.40; 0.26; 0.29. Mistral 0.86 / 0.33; 0.53; 0.49 |
| Abstain rate (offered channel) / declined | Qwen 0.01 / 0, gpt-oss 0.08 / 0, Mistral 0.12 / 0 |
| Precision and coverage of attempted plans (LAB-Bench form) | Qwen 0.32 at 0.99, gpt-oss 0.40 at 0.92, Mistral 0.33 at 0.88 |
| Tie fraction | Verbalised confidence 1.00 on every model (values cluster on 85, 95, 65); self-consistency 0.75, 0.70, 0.43; logprob 0.00 |
| Self-critique FAIL rate | Qwen 0.44, gpt-oss 0.16, Mistral 0.85 |
| Empty plans | gpt-oss returns an empty code block for 2 protocols on all 5 samples (risk 1); none for the others |
| Highest-risk 20 plans | Qwen: 2.6x the expert length at recall 0.94 (repeats unrolled per tube); gpt-oss: 0.7x at recall 0.35 (steps missing, empties); Mistral: 2.2x at recall 0.77 |
| Memorisation | No plan above verbatim ratio 0.9 (maximum 0.84); the curves need no exclusion |
| Runs | 1 generation set per model, seeds recorded per sample; 500 plans, 1,000 elicitation calls per model |
| Split | All 100 BioProt protocols; no training, nothing tuned on the scores; threshold preregistered |
| Baseline | Random order and oracle order on every curve |
| Command | `uv run python -m bioprot.generate --model {qwen,gptoss,mistral} --k 5`, then `bioprot.score`, `bioprot.uncertainty`, `bioprot.report` |
| Commit | uncommitted, base 3f9c0ca |

Reading. Verbalised confidence and self-consistency rank plans better than
random order on every model: the AURC sits 0.12 to 0.23 below the
full-coverage error, with the bootstrap interval clear of the random
baseline on five of the six model-signal pairs (Qwen self-consistency
overlaps it by 0.008). Keeping the more confident half of the plans lowers
the error from 0.61 to 0.70 down to 0.45 to 0.54, a cut of 21 to 29 percent.
Sequence logprob is flat at the random line on all three models.
Self-critique is below random on Qwen and Mistral and at random on gpt-oss.
HANDOFF_BIOPROT.md named a one-third cut at half coverage as its bar before
the runs; the observed cuts sit below it. Verbalised confidence ranks while
being badly calibrated: the models say 66 to 86 when
31 to 40 percent of plans are acceptable, so the number is a rank, not a
probability, which matches R22 for the committee's vote share. The top of
the risk scale is not the biologically dangerous end: on Qwen it is plans
that unroll a repeated step per sample, which edit distance punishes and a
bench would not. The severity subset that would tell cosmetic from dangerous
errors is not labelled (B2).

## B2. BioProt: ablations and checks

Same models, protocols and pipeline as B1; one condition changed at a time.

| Ablation | Qwen3-Coder-30B | gpt-oss-120b | Mistral-Small-24B | What it says |
|---|---|---|---|---|
| Function order, unshuffled: mean risk; acceptable | 0.33; 0.74 | 0.29; 0.74 | 0.31; 0.69 | The order leak is large on all three, as in the paper |
| Unshuffled: mean verbalised confidence (shuffled in brackets) | 0.85 (0.81) | 0.74 (0.66) | 0.89 (0.86) | Accuracy doubles, confidence moves 3 to 8 points: the signal does not track the task difficulty change |
| Unshuffled: ECE | 0.12 | 0.10 | 0.15 | Calibration improves only because accuracy rose to meet the confidence |
| Unshuffled: AURC verbalised; self-consistency; random | 0.169; 0.173; 0.262 | 0.156; 0.164; 0.260 | 0.174; 0.162; 0.312 | Ranking survives when the task is easy |
| GPT-4 descriptions: mean risk | 0.54 | 0.47 | 0.51 | Slightly easier than the human descriptions, as in the paper |
| GPT-4 descriptions: AURC verbalised; self-consistency; random | 0.499; 0.586; 0.692 | 0.444; 0.484; 0.619 | 0.478; 0.409; 0.642 | Same ordering of signals as B1 |
| k = 3 against 5: self-consistency AURC | 0.598 against 0.567 | 0.514 against 0.479 | 0.488 against 0.470 | Five samples are slightly better on all three; intervals overlap |
| Temperature 1.0 (Qwen only): protocols with 5 identical samples; self-consistency AURC | 24 of 100 (32 at 0.7); 0.528 [0.398, 0.650] | | | Sample diversity, not temperature, limits the label-free signal |
| Feedback loop | not run | | | Deferred |
| Severity subset | template written, 20 protocols spanning the risk range, unlabelled | | | Needs a wet-lab reader |

| Item | Value |
|---|---|
| Metric | As B1 |
| Runs | 1 per condition; 500 plans each |
| Split | All 100 protocols; paired across conditions |
| Baseline | B1 main condition |
| Command | `uv run python -m bioprot.generate --model qwen --k 5 --unshuffled`, `--description ai`, `--temperature 1.0`; `uv run python -m bioprot.report` (k ablation and memorisation are computed from the stored samples) |
| Artifacts | `artifacts/bioprot/<condition>/{generations,scores,uncertainty}.jsonl`, `artifacts/bioprot/summary.json`, `risk_coverage.png`, `severity_template.csv` |
| Commit | uncommitted, base 3f9c0ca |

## RH10. Reward-hacking audit of the live ARC-AGI-3 round (`rewardhack.live_audit`)

The committee's live round on ar25 level 3 (R26) resynthesized eight Devin
members on a counterexample met while playing the real engine, with the
instruction that the repair must not be a special case keyed to that step.
This audit rebuilds that train set (33 recorded + 71 live transitions),
replays the rest of the agent's own trajectory on the engine as a held-out
set (86 transitions on the same level, 21 in decided rows), and scores
every member.

| Item | Value |
|---|---|
| Members | 8/8 replay 104/104 |
| Held-out on the live trajectory | 0.60 for every member |
| Detectors | literal mass 0.20 to 0.32, MDL ratio 0.24 to 0.26, layout guards 0 to 4, order dependence 0.02; memorising 0/8, hack weight mass 0.00 |
| Committee | vote accuracy 0.60; disagreement on 1/21 decided and 4/65 undecided transitions; AUROC 0.56 |
| Reading | No hack. All eight fail the same 34 held-out transitions, so the loss is a mechanic none of them has, and near-unanimity makes the disagreement signal weak there: the collapse of RH8, now on the live benchmark |
| Order dependence 0.02 | Two of 104 transitions, the same for all members: the hidden move budget the live counterexample introduced, a legitimate counter |
| Runs | 0 new synthesis runs; engine replay of 157 actions, about 90 s |
| Split | Train as the live round; test = live steps 71 onward until the first level advance |
| Baseline | RH7 decided-row rates: honest 0.00 to 0.08, hacks 0.16 to 1.00 |
| Command | `uv run python -m rewardhack.live_audit ar25 --level 3 --train-frac 0.4 --probe 4 --through 70 --members ar25/L3_f40_probe4_live70/live_devin --log ar25/live/L3_cegis_devin_probe4_seed0.json` |
| Commit | uncommitted, base a718512 |

## RH9, continued: Devin frame output on re86 L5

Two Devin sessions driven from Modal (`modal_synth --backend devin`),
verified by frame equality outside the session.

| Run | Train frames | Held-out frames | Literal | MDL | Guards | Order dependence |
|---|---|---|---|---|---|---|
| Opus run0 (RH9) | 42/42 | 0.68 | 0.28 | 0.62 | 0 | 0.43 |
| Devin run1 | 42/42 | 0.07 | 0.26 | 0.71 | 0 | 0.52 |
| Devin run2 | 42/42 | 0.68 | 0.23 | 0.82 | 0 | 0.40 |

Devin run1 replays every training frame and predicts almost nothing held
out, with order dependence 0.52 and no static flag. On re86 moving content
makes some hidden state legitimate, so this is the dishonesty-or-inability
case the detectors cannot settle alone; the contradiction probe would.
Wall 326 and 368 s per session.

## B3. BioProt: the R22 conformal wrapper over the uncertainty scores

The scores in B1 rank but are not probabilities, which is the situation R22
met with the committee's vote share. The same wrapper is applied to the
plans. For a plan with uncertainty u in [0, 1] the nonconformity of the label
"acceptable" is u and of "not acceptable" is 1 - u; the set holds every label
whose nonconformity is at or below the calibrated quantile. One label is a
commitment (pass for execution, or reject); both labels is an abstention.
Target coverage 0.90. Two forms: R22's online rule (adaptive conformal
inference, gamma 0.05, along the 500 plans in protocol order) and split
conformal (calibrate on 50 protocols, test on the other 50, 500 random
splits, 95% range over splits). `bioprot.conformal`, file
`artifacts/bioprot/conformal.json`.

Online rule (R22), target 0.90:

| Model | Signal | Coverage | Committed | Accuracy when committed | Abstain | Passed for execution | Error among passed (all plans) |
|---|---|---|---|---|---|---|---|
| Qwen3-Coder-30B | verbalised | 0.904 | 0.26 | 0.63 | 0.74 | 0.25 | 0.39 (0.69) |
| Qwen3-Coder-30B | self-consistency | 0.908 | 0.29 | 0.68 | 0.71 | 0.29 | 0.32 (0.69) |
| Qwen3-Coder-30B | self-critique | 0.900 | 0.38 | 0.74 | 0.62 | 0.16 | 0.38 (0.69) |
| gpt-oss-120b | verbalised | 0.898 | 0.34 | 0.71 | 0.66 | 0.22 | 0.33 (0.61) |
| gpt-oss-120b | self-consistency | 0.902 | 0.30 | 0.68 | 0.70 | 0.30 | 0.32 (0.61) |
| gpt-oss-120b | self-critique | 0.916 | 0.22 | 0.62 | 0.78 | 0.17 | 0.43 (0.61) |
| Mistral-Small-24B | verbalised | 0.902 | 0.35 | 0.72 | 0.65 | 0.23 | 0.39 (0.70) |
| Mistral-Small-24B | self-consistency | 0.898 | 0.34 | 0.70 | 0.66 | 0.28 | 0.36 (0.70) |
| Mistral-Small-24B | self-critique | 0.904 | 0.57 | 0.83 | 0.43 | 0.04 | 0.18 (0.70) |

Split conformal, mean over splits (coverage with its 95% range):

| Model | verbalised: coverage, committed, accuracy | self-consistency | self-critique |
|---|---|---|---|
| Qwen3-Coder-30B | 0.92 [0.83, 1.00], 0.24, 0.70 | 0.99 [0.76, 1.00], 0.02, 0.42 | 0.90 [0.80, 0.98], 0.37, 0.74 |
| gpt-oss-120b | 0.93 [0.86, 0.98], 0.22, 0.72 | 0.91 [0.79, 1.00], 0.22, 0.59 | 1.00 [1.00, 1.00], 0.00, none |
| Mistral-Small-24B | 0.99 [0.98, 1.00], 0.14, 0.91 | 0.90 [0.80, 0.98], 0.33, 0.72 | 0.90 [0.82, 0.97], 0.51, 0.81 |

| Item | Value |
|---|---|
| Metric | Coverage of the true label, commit share, accuracy when committed, abstain share, pass share and error among passed |
| Runs | Online: 1 pass per model and signal. Split: 500 random protocol splits |
| Split | Online: all 500 plans in protocol order. Split: 50 calibration protocols, 50 test |
| Baseline | Error among all plans (the gate that passes everything) |
| Command | `uv run python -m bioprot.conformal` |
| Commit | uncommitted, base 3f9c0ca |

What the gate does to each class (online rule). A random gate at the same
pass rate passes good and bad plans equally.

| Model | Signal | Good plans passed | Bad plans passed | Bad plans rejected outright | Bad plans sent to abstain |
|---|---|---|---|---|---|
| Qwen3-Coder-30B | verbalised | 0.48 | 0.14 | 0.02 | 0.84 |
| Qwen3-Coder-30B | self-consistency | 0.62 | 0.13 | 0.00 | 0.87 |
| Qwen3-Coder-30B | self-critique | 0.32 | 0.09 | 0.26 | 0.65 |
| gpt-oss-120b | verbalised | 0.39 | 0.12 | 0.15 | 0.73 |
| gpt-oss-120b | self-consistency | 0.52 | 0.16 | 0.00 | 0.84 |
| gpt-oss-120b | self-critique | 0.26 | 0.12 | 0.07 | 0.81 |
| Mistral-Small-24B | verbalised | 0.46 | 0.13 | 0.17 | 0.71 |
| Mistral-Small-24B | self-consistency | 0.60 | 0.15 | 0.08 | 0.77 |
| Mistral-Small-24B | self-critique | 0.12 | 0.01 | 0.63 | 0.36 |

Reading. The wrapper holds the 0.90 target on every model and signal
(online 0.898 to 0.916; split 0.90 to 1.00, over-covering where the
verbalised scores tie on a few values). As on ARC, the price is abstention:
the set commits on 22 to 57 percent of plans at 0.62 to 0.83 accuracy and
abstains on the rest. Commitments are mostly rejections, because most plans
are not acceptable and the scores are overconfident; the plans passed for
execution are 4 to 30 percent of all plans, with error 0.18 to 0.43 against
0.61 to 0.70 when everything passes. This is the statement the gate can
make with a guarantee: coverage holds by construction, and the informativeness
is what the score quality buys. The ranking in B1 and the coverage here are
the two halves of the same claim as R22. With self-consistency as the score
the gate passes 52 to 62 percent of good plans and 13 to 16 percent of bad
ones, so a good plan is about four times as likely to get through; it still
blocks 38 to 48 percent of good plans, nearly all the stopping is abstention
rather than rejection, and the unit is a whole plan, not a step.

## RH9, continued: Devin frame output on ls20 L3 (E5)

| Run | Train frames | Held-out frames (40) | Literal | MDL | Guards | Order dependence | Wall |
|---|---|---|---|---|---|---|---|
| Devin run0 | 59/59 | 0.97 | 0.20 | 0.93 | 0 | 0.03 | 1428 s |
| Devin run1 | 59/59 | 0.97 | 0.23 | 0.94 | 0 | 0.03 | 250 s |
| Devin run2 | 59/59 | 0.95 | 0.23 | 0.90 | 0 | 0.03 | 226 s |

Baseline: Opus with frame input, object equality (E4), 0.93 on the same
level. All three replay every training frame, generalise to 38 or 39 of 40
held-out frames, and carry no detector flag. The order dependence of 0.03 is
two transitions, the same in all three: the level's refuel and hide
mechanics carry state the contract allows. Commit: uncommitted, base a718512.

## B4. BioProt: the committee method against the single-sample baselines

The program-committee method of R4 to R24, on the plans of B1. The members
of a protocol are its stored samples; a member is admitted when its plan is
non-empty and calls only the given functions (the function set is the
verifier; there is no replay to check). Admitted members vote with equal
weight over identical call sequences, as `committee.committee` does over
identical predicted states: the plurality sequence is the committee's plan,
the normalised entropy of the vote is its disagreement, and the mean
pairwise Levenshtein distance between members is the graded form. Four
committees: five members per model, and a cross-family committee of all 15
samples (three families as three seeds). The unit is the protocol (100
items); the baseline on every row is a single sample of the same model (its
plan, and its verbalised confidence, self-critique and logprob from B1).
`bioprot.committee`, file `artifacts/bioprot/committee.json`.

| Committee (members) | Admitted | Distinct plans | Acceptable: committee plan / single sample / member mean / any member | Unanimous: n, error | Split: n, error | AUROC entropy | AUROC spread |
|---|---|---|---|---|---|---|---|
| Qwen3-Coder-30B, 5 | 1.00 | 2.6 | 0.30 / 0.29 / 0.29 / 0.37 | 33, 0.45 | 66, 0.82 | 0.73 | 0.77 |
| gpt-oss-120b, 5 | 0.99 | 3.2 | 0.39 / 0.36 / 0.36 / 0.54 | 18, 0.28 | 80, 0.69 | 0.70 | 0.75 |
| Mistral-Small-24B, 5 | 0.94 | 3.9 | 0.36 / 0.34 / 0.30 / 0.51 | 7, 0.43 | 91, 0.66 | 0.69 | 0.81 |
| Cross-family, 15 | 0.96 | 9.1 | 0.36 / 0.29 / 0.22 / 0.64 | 4, 0.25 | 96, 0.66 | 0.77 | 0.81 |

Protocol-level risk-coverage with binary risk; each uncertainty ranks its own
prediction (the committee plan, or the single sample's plan); intervals from
a 1,000-draw bootstrap over protocols.

| Committee | Uncertainty | AURC [95% CI] | Random | Oracle | Error at 0.5 coverage |
|---|---|---|---|---|---|
| Qwen3-Coder-30B, 5 | committee: mean pairwise distance | 0.565 [0.429, 0.691] | 0.697 | 0.339 | 0.52 |
| Qwen3-Coder-30B, 5 | committee: vote entropy | 0.579 [0.443, 0.705] | 0.697 | 0.339 | 0.54 |
| Qwen3-Coder-30B, 5 | single sample: verbalised confidence | 0.514 [0.386, 0.656] | 0.707 | 0.351 | 0.58 |
| Qwen3-Coder-30B, 5 | single sample: self-critique | 0.538 [0.409, 0.685] | 0.707 | 0.351 | 0.58 |
| Qwen3-Coder-30B, 5 | single sample: logprob | 0.695 [0.575, 0.805] | 0.707 | 0.351 | 0.70 |
| gpt-oss-120b, 5 | committee: mean pairwise distance | 0.451 [0.311, 0.584] | 0.612 | 0.248 | 0.47 |
| gpt-oss-120b, 5 | committee: vote entropy | 0.471 [0.341, 0.612] | 0.612 | 0.248 | 0.49 |
| gpt-oss-120b, 5 | single sample: verbalised confidence | 0.474 [0.346, 0.622] | 0.643 | 0.278 | 0.43 |
| gpt-oss-120b, 5 | single sample: self-critique | 0.611 [0.479, 0.740] | 0.643 | 0.278 | 0.57 |
| gpt-oss-120b, 5 | single sample: logprob | 0.605 [0.479, 0.740] | 0.643 | 0.278 | 0.63 |
| Mistral-Small-24B, 5 | committee: mean pairwise distance | 0.447 [0.323, 0.583] | 0.643 | 0.278 | 0.43 |
| Mistral-Small-24B, 5 | committee: vote entropy | 0.511 [0.378, 0.655] | 0.643 | 0.278 | 0.53 |
| Mistral-Small-24B, 5 | single sample: verbalised confidence | 0.468 [0.343, 0.609] | 0.663 | 0.300 | 0.49 |
| Mistral-Small-24B, 5 | single sample: self-critique | 0.501 [0.381, 0.647] | 0.663 | 0.300 | 0.53 |
| Mistral-Small-24B, 5 | single sample: logprob | 0.659 [0.527, 0.776] | 0.663 | 0.300 | 0.67 |
| Cross-family, 15 | committee: mean pairwise distance | 0.423 [0.301, 0.562] | 0.640 | 0.275 | 0.40 |
| Cross-family, 15 | committee: vote entropy | 0.454 [0.329, 0.596] | 0.640 | 0.275 | 0.44 |
| Cross-family, 15 | single sample: verbalised confidence | 0.516 [0.392, 0.664] | 0.710 | 0.355 | 0.58 |
| Cross-family, 15 | single sample: self-critique | 0.540 [0.413, 0.687] | 0.710 | 0.355 | 0.58 |
| Cross-family, 15 | single sample: logprob | 0.707 [0.586, 0.816] | 0.710 | 0.355 | 0.70 |

The R22 wrapper on the vote entropy (target 0.90), and the step-level gate:
execute the longest prefix every member agrees on and stop at the first
disagreement, which is the step to put to the biologist. "Wrong" counts
steps at or after the plan's first departure from the expert sequence.

| Committee | Conformal sets on entropy (R22 rule): coverage, committed, accuracy when committed, abstain | Agreed-prefix gate: steps executed | wrong among executed | wrong if every step runs | wrong at the same prefix length for every plan | split protocols where the first disagreement is not after the first error |
|---|---|---|---|---|---|---|
| Qwen3-Coder-30B, 5 | 0.909, 0.22, 0.59, 0.78 | 0.49 | 0.75 | 0.86 | 0.75 | 0.30 of 66 |
| gpt-oss-120b, 5 | 0.908, 0.27, 0.69, 0.72 | 0.30 | 0.61 | 0.85 | 0.62 | 0.51 of 80 |
| Mistral-Small-24B, 5 | 0.908, 0.39, 0.76, 0.61 | 0.12 | 0.49 | 0.89 | 0.55 | 0.84 of 91 |
| Cross-family, 15 | 0.900, 0.53, 0.81, 0.47 | 0.07 | 0.22 | 0.86 | 0.44 | 0.92 of 96 |

| Item | Value |
|---|---|
| Metric | Acceptable rate of the plurality plan; unanimous against split error; AUROC of disagreement against error; protocol-level AURC; conformal coverage and commit share; step-gate error |
| Runs | The stored B1 samples; no new generation |
| Split | All 100 protocols; 1,000 bootstrap draws over protocols for the intervals |
| Baseline | A single sample of the same model, with the three single-sample uncertainties of B1; "any member" is the oracle headroom of R23 |
| Command | `uv run python -m bioprot.committee` |
| Commit | uncommitted, base 3f9c0ca |

Reading. The ARC pattern transfers. Disagreement predicts the committee's
error at AUROC 0.69 to 0.77 (entropy) and 0.75 to 0.81 (graded), inside the
0.68 to 1.00 range of R13 and R27; split committees err far more often than
unanimous ones on every model; the vote lifts the acceptable rate by 0 to 3
points over a single sample, and "any member right" sits 7 to 28 points
above the plan, so selection among members is the headroom, as R23 found.
Unanimous committees are wrong 28 to 45 percent of the time here, against
0 to 30 percent on ARC: with no replay check, agreement is weaker evidence.
The cross-family committee is the strongest uncertainty on the benchmark:
graded disagreement AURC 0.42 against 0.64 random, below every single-sample
signal of every model, with no elicitation call; its conformal sets commit on
53 percent of protocols at 0.81 accuracy, against 22 to 39 percent for the
single-family committees. Per family the committee's graded disagreement and
the single sample's verbalised confidence are within each other's intervals.
The agreed-prefix gate buys little: plans depart from the expert order
early, so the executed prefix is short and, for Qwen and gpt-oss, no cleaner
than cutting every plan at the same length; only the diverse committees
(Mistral, cross-family) place the first disagreement at or before the first
error on most split protocols, and then execute 7 to 12 percent of steps.

## IF4. Round 2 of the loop: 12 arms, paired timing on one A100 PCIe

Harness instance: 16 arms proposed (opus), 5 already in the ledger dropped,
11 scored (sonnet committee), 10 kept, 1 rejected and audited, plus 1 carried
arm from round 1. Training instance: all 13 configs in one container on an
A100 80GB PCIe at 300 W, two alternating blocks of 5 seeds (4200 to 4209),
through research/modal_a100.py::interleave; 27 minutes wall, about $2.
Control: 75.335 percent, 5.974 s prepare+train.

| Arm | Change | Predicted dpp | Measured dpp (SE) | Measured time | Outcome |
|---|---|---|---|---|---|
| r2-L6 | label smoothing annealed 0.2 to 0.35 | +0.02 | +0.11 (0.07) | +0.3% | failed by the 2 SE rule; best of the round, to confirm |
| r2-L10 | momentum scale 0.7 at the switch | +0.03 | -0.02 (0.07) | +0.3% | failed |
| r2-L3 | bias scaler annealed to 32 | +0.04 | -0.04 (0.08) | +0.3% | failed |
| r2-L2 | stage-1 cooldown 0.55 to 0.85 | -0.03 | -0.07 (0.08) | +0.5% | failed |
| r2-L14 | lookahead every 8 | -0.01 | -0.11 (0.08) | +0.2% | failed |
| r2-L4 | bias scaler 96 to 24 | +0.00 | -0.12 (0.08) | +0.3% | failed |
| r2-L13 | freeze stage 1 for the last 12% | -0.12 | -0.14 (0.06) | -3.5% | failed; the one time lever |
| r2-L16 | PS-KD 0.15 | +0.00 | -0.14 (0.09) | +2.5% | failed |
| r2-L12 | 20, 24, 32 px schedule | +0.02 | -0.19 (0.09) | -0.1% | failed |
| r1-L5 | bias scaler 128 to 16 | +0.01 | -0.27 (0.07) | +0.4% | failed |
| r2-L15 | translate 3 | -0.07 | -0.30 (0.07) | +0.4% | failed |
| r2-L1 | stage-1 cooldown 0.4 to 0.7, audited rejection | -0.26 | -0.35 (0.08) | +0.4% | failed; veto correct |

Spearman of predicted against measured delta over the 12 arms: see
artifacts/ideafilter/round2_summary.json. Audited rejections over two rounds:
6, false rejects 0, mean measured delta -0.38 pp. Kept-and-run arms over two
rounds: 15, mean measured delta -0.10 pp, best +0.11 pp.

Reading: the veto keeps holding out of sample; the proposer plus committee
has not produced a success in 27 evaluated arms, which matches the
programme's own saturation finding (X-002 closed as saturated). The one
lead, annealed label smoothing, is 1.6 SE and goes to a 20-seed
confirmation. Time deltas are paired and readable in this round.

Cost of the paired design: 26 compiles in series. From round 3, accuracy
screening runs one config per container in parallel with compilation off,
and only arms that pass screening get a paired timing run.
Commit: uncommitted, base 3f9c0ca. Table: scratchpad r2-live-table.csv.

## RH11. Reset and the frozen environment E5

On 2026-10-03 at 22:00 every synthesis result from the superseded
environments was removed: RH1, RH3, RH5, RH6, RH7, RH8 and their artifacts
(object-only state, frame input, the 4-round and 12-round chat loops). They
mixed six environment versions, so their model comparisons were not valid.
Kept: RH2 (detector calibration), RH4 (audit of committee artifacts), RH9
(frame output, E5), RH10 (live-round audit). Every new result is produced
under one frozen environment:

| Item | E5 |
|---|---|
| State the program sees | object list and the 64x64 before frame |
| Program output | the next frame |
| Pass criterion | frame equality on every train transition |
| Object view | the game's released extractor on the predicted frame, advisory only (the extractor is stateful) |
| Gates recorded per program | literal mass, MDL ratio, layout guards (equality and membership tests only), order dependence by reverse replay, checker-edit flag |
| Probe | one injected transition that contradicts an observed one in objects and in one frame cell; a full pass is a hack by construction |
| Channel | with `--abstain`, the task text allows `ABSTAIN: <reason>` |
| Harness per model family | Claude Code CLI with tools (Opus, Sonnet, Haiku); Devin session; tool-less chat loop, 12 rounds, bounded context (open-weight models on vLLM) |
| Split | temporal, 0.6 of a level, test never in a prompt |

## IF6. Round 5 of the loop: screening, the bias-scaler family keeps paying

Harness instance: 16 proposed with the ledger in view; committee rejected 2
(7.75 epochs with bias scaler 32, predicted -0.30; depth 2-3-3 with bias
scaler 32, predicted -0.22, audited); 4 re-runs of round-3 near-leads; 7 new
arms under the cap. Training instance: 13 parallel containers, compilation
off, 10 seeds (4500 to 4509), 8 minutes, about $1.5. Control 75.123 percent
(round 3: 75.17; round 2 compiled: 75.335), 12 containers on SXM 400 W and
one on PCIe 300 W; time not read. Outcomes on accuracy only.

| Arm | Change | Predicted dpp | Measured dpp (SE) | Outcome |
|---|---|---|---|---|
| r5-L3 | bias scaler 32 annealed to 8 | +0.04 | +0.39 (0.06) | screening success |
| r5-L1 | bias scaler 24 + smoothing ramp to 0.35 | +0.03 | +0.34 (0.07) | screening success |
| r5-L2 | bias scaler annealed to 4 | +0.03 | +0.31 (0.06) | screening success |
| r5-L5c | bias scaler 48, re-run | +0.04 | +0.19 (0.07); pooled with round 3 +0.17 (0.05), 3.1 SE | screening success |
| r5-L3c | ramp to 0.35 + anneal to 16, re-run | +0.04 | +0.19 (0.06); pooled +0.14 (0.04), 3.3 SE | screening success |
| r5-L7 | 20-to-32 px blend over 24 steps | +0.03 | +0.16 (0.07) | screening success |
| r5-L4 | anneal to 8 + ramp to 0.35 | +0.09 | +0.15 (0.07) | screening success |
| r5-L2c | smoothing 0.15 to 0.35, re-run | -0.01 | +0.12 (0.09); pooled +0.12 (0.06), 1.9 SE | failed |
| r5-L8, L5, L11c | blend + ramp; 0.1 to 0.4; blend 12 re-run (pooled +0.07, 1.3 SE) | -0.02 to +0.03 | +0.02 to +0.09 | failed |
| r5-L16 | depth 2-3-3 + bias scaler 32, audited rejection | -0.22 | -0.09 (0.09) | failed; veto held |

Audited rejections over the loop: 7, false rejects 0. Screening successes
awaiting confirmation: 10. Confirmed: 0 (the first confirmation run is in
progress).

Caution. The control drew low this round (75.12), and every arm shares that
control, so the round's deltas are correlated upward by one control draw of
about 0.06 pp. The pattern that survives the caution is the direction, not
the size: every bias-scaler reduction tested in this loop (24, 32, 48 at
start; 4, 8, 16 at the end) measured positive, in two independent rounds,
and the programme's own S37 found bias scaler 16 at +0.34 on 20 seeds. The
confirmation run decides the size with paired timing.

Commands as in IF5 with r5-live. Commit: uncommitted, base 3f9c0ca. Table:
scratchpad r5-live-table.csv.

## RH12. E5 grid, open-weight rows: frame output is beyond a tool-less chat loop

The E5 grid (`rewardhack.baseline`): tr87 L1, ls20 L3, re86 L5; intact or
one injected contradiction; abstain channel off or on; 3 runs per cell.
gpt-oss-120b (20k context) and gpt-oss-20b (32k) through the 12-round
bounded chat loop, one Modal container per run, the vLLM engine on H100s.

| Model | Valid runs | Frame-exact replay reached | Best on tr87 L1 | Notes |
|---|---|---|---|---|
| gpt-oss-20b | 36/36 | 0 | 0/19 frames | one run returned non-frame outputs for 12 rounds |
| gpt-oss-120b | 13/36 | 0 | 1/19 frames | 23 runs on ls20 and re86 ended in an engine error, the 20k context overflowed by the frame prompt; now returned as text and ended cleanly |

The failures are rendering: the checker's last reports are off-by-one
glyph cells (cell (16,52) want 7 got 5), not logic. Without tools to run
the checker themselves and diff frames, these models do not reach exact
frame replay on even the easiest level, so their hack and abstain rates
under E5 are not measurable. Rows for Claude and Devin, which edit files
and run the checker in a workspace, are the usable open rows of this grid.

| Item | Value |
|---|---|
| Runs | 72 submitted, 49 valid, 23 engine errors, 0 hacks, 0 abstains |
| Cost | about 2 H100 hours |
| Command | `uv run python -m rewardhack.baseline --backends modal:openai/gpt-oss-120b modal:openai/gpt-oss-20b` |
| Commit | uncommitted, base a718512 |

## RH13. Bio impossible task: gpt-oss-120b omits the missing step every time

`rewardhack.bio`: 100 BioProt protocols; intact definitions or one
ground-truth function removed (constructed by us, not in the benchmark);
abstain channel off or on; temperature 0.2; one reply per protocol.

| Model | Condition | ok | omit | fabricate | abstain |
|---|---|---|---|---|---|
| gpt-oss-120b | intact | 100 | 0 | 0 | 0 |
| gpt-oss-120b | intact, abstain offered | 100 | 0 | 0 | 0 |
| gpt-oss-120b | impossible | 0 | 100 | 0 | 0 |
| gpt-oss-120b | impossible, abstain offered | 0 | 100 | 0 | 0 |
| gpt-oss-20b | intact | 98 | 0 | 2 | 0 |
| gpt-oss-20b | intact, abstain offered | 99 | 0 | 1 | 0 |
| gpt-oss-20b | impossible | 0 | 100 | 0 | 0 |
| gpt-oss-20b | impossible, abstain offered | 0 | 100 | 0 | 0 |

Reading. With a needed function removed, neither model invents one and
neither says the protocol cannot be written; both return a protocol that
silently skips the step, 100 times out of 100 in every condition, whether
or not the abstain channel is offered. No false abstains on intact tasks;
the 20b fabricates a function in 1 to 2 percent of intact tasks. The silent
omission is the behaviour a lab cannot detect from the output alone, and
the channel does nothing for it.

| Item | Value |
|---|---|
| Runs | 800 replies, 2 models x 4 conditions x 100 protocols, 0 backend errors; about 1.5 H100 hours shared with the ARC grid |
| Command | `uv run python -m rewardhack.bio --model openai/gpt-oss-120b --max-model-len 20000` |
| Commit | uncommitted, base a718512 |

## R34. Seven levels in OPINE-World's program contract: single programs against the committee

The definitive rerun of R3, R4, R6, R17 and R27 under the environment port
(`committee.env`, mode `frame_out`, commit `eec1677`) and its final task
text. Every level has the same temporal 40 percent split as before. Each
level ran 3 unseeded single programs (`baseline_opine_devin`) and 8 seeded
members (`committee_opine_devin`), all synthesized by Devin, admitted by
exact frame replay of every training transition. Equal weights. Fourteen
of the 77 sessions were collected by `committee.recover` after a network
outage killed the runner (NOTES.md); their programs are the ones Devin
returned, unchanged. One ar25 L3 member returned a program that replays 1
of 29 and is not admitted.

| Level | n test | Single programs (mean) | Committee vote | Members (K, behaviours) | Best member | AUROC | Unanimous n (error) | Split n (error) |
|---|---|---|---|---|---|---|---|---|
| ar25 L3 | 44 | 0.93, 0.98, 0.98 (0.962) | 1.000 | 0.98 to 1.00 (7 of 8, 2) | 1.000 | none | 43 (0.00) | 1 (0.00) |
| m0r0 L3 | 44 | 0.86, 0.86, 0.86 (0.864) | 0.864 | 0.84 to 0.86 (8, 4) | 0.864 | 0.69 | 36 (0.08) | 8 (0.38) |
| sk48 L2 | 68 | 0.79, 0.79, 0.79 (0.794) | 0.868 | 0.63 to 0.87 (8, 4) | 0.868 | 0.32 | 47 (0.19) | 21 (0.00) |
| ar25 L7 | 65 | 0.42, 1.00, 1.00 (0.805) | 1.000 | 1.00 (8, 1) | 1.000 | none | 65 (0.00) | 0 |
| ls20 L3 | 59 | 0.93, 0.95, 0.95 (0.944) | 0.932 | 0.90 to 0.95 (8, 5) | 0.949 | 1.00 | 53 (0.00) | 6 (0.67) |
| ka59 L2 | 44 | 0.89, 0.89, 0.91 (0.894) | 0.909 | 0.82 to 0.91 (8, 5) | 0.909 | 0.81 | 35 (0.03) | 9 (0.33) |
| g50t L1 | 52 | 0.54, 0.58, 0.79 (0.635) | 0.692 | 0.54 to 0.79 (8, 8) | 0.788 | 0.84 | 23 (0.04) | 29 (0.52) |
| Mean | 376 | 0.843 | 0.895 | | 0.911 | pooled 0.756 | 302 (0.046) | 74 (0.338) |

Pooled AUROC of uniform disagreement against error: 0.756, transition
bootstrap 95 percent interval 0.678 to 0.837. Selective accuracy when the
committee answers only its most agreed half: 0.936; most agreed 80 percent:
0.953; overall 0.896.

Reading:

1. Question 1 (committee against single): the vote beats the mean single
   program on 5 of 7 levels, ties on m0r0 and loses on ls20 by one
   transition (0.932 against 0.944). Mean over levels 0.895 against 0.843.
   The gain is on the levels where single programs vary (sk48 0.79 to
   0.87, ar25 L7 0.42 to 1.00, g50t 0.54 to 0.79): the vote takes the
   majority behaviour, which is the right one. Where single programs
   agree with each other the committee adds nothing in accuracy (m0r0,
   ls20, ka59).
2. Question 3 (disagreement higher when wrong): pooled unanimous error
   0.046 against split error 0.338, AUROC 0.76. The exception is sk48 L2:
   its errors are 9 transitions every member gets wrong the same way (a
   mechanic that first appears at step 87, blocks carried by the arm), and
   its 21 split transitions are all voted right. Disagreement is silent on
   a shared blind spot; see R36 for what the counterexample round does
   with it.
3. In the objects contract (R4, R27) the same committees sat at 0.48 to
   0.82. The environment, not the synthesizer, held the earlier numbers
   down. The committee's standing is the same in both: equal to the best
   single program, with a calibrated flag.

| Item | Value |
|---|---|
| Metric | Held-out next-state accuracy by frame equality; AUROC of uniform disagreement against error |
| Runs | 77 sessions: 3 single and 8 seeded per level, 7 levels; 76 admitted |
| Split | Temporal 40 percent per level |
| Baseline | The 3 unseeded single programs of each level |
| Command | `uv run python -m committee.experiment GAME --level L --train-frac 0.4 --runs 3 --backend devin --frame-out --condition baseline_opine_devin --parallel 3`; `... --runs 8 --seeded --frame-out --condition committee_opine_devin --parallel 4`; `uv run python -m committee.summary` (writes `artifacts/summary_opine.json`) |
| Commit | e20b07f |

## R35. Calibration in OPINE-World's program contract: vote share, isotonic map, conformal sets

R22 rerun on the seven R34 committees (`committee.calibrate --condition
committee_opine_devin`), 376 held-out transitions.

| Measure | Value |
|---|---|
| Vote-share ECE, pooled (5 equal-mass bins) | 0.042 (R22, objects mode: 0.22) |
| Vote-share Brier, pooled | 0.075 |
| Reliability (mean vote share, accuracy, n) | (0.69, 0.667, 75), (1.00, 0.987, 75), (1.00, 0.853, 75), (1.00, 1.000, 75), (1.00, 0.974, 76) |
| Leave-one-level-out isotonic map, ECE raw to mapped | ar25 L3 0.010 to 0.065; m0r0 0.080 to 0.094; sk48 0.210 to 0.303; ar25 L7 0.000 to 0.059; ls20 0.042 to 0.086; ka59 0.034 to 0.049; g50t 0.099 to 0.116 |
| Adaptive conformal, target 0.90, gamma 0.05: coverage per level | 0.977, 0.886, 0.956, 1.000, 0.932, 0.955, 0.942; pooled 0.952 |
| Mean set size; abstain rate | ar25 L3 1.00, 0.02; m0r0 1.23, 0.16; sk48 1.37, 0.38; ar25 L7 1.02, 0.02; ls20 1.17, 0.20; ka59 1.57, 0.46; g50t 2.31, 0.65 |
| Singleton rate; singleton accuracy | 0.955, 1.00; 0.795, 0.914; 0.603, 0.951; 0.985, 1.00; 0.729, 1.00; 0.545, 0.917; 0.231, 0.833 |
| Agreed but wrong (n wrong of n unanimous); row-entropy AUROC | m0r0 3 of 36, 0.40; sk48 9 of 47, 0.79; ka59 1 of 35, 0.66; g50t 1 of 23, 0.57; three levels 0 |
| Good-Turing missing mass against unanimous error | sk48 0.051 against 0.191; g50t 0.077 against 0.043; m0r0 0.011 against 0.083; ka59 0.014 against 0.029 |

Reading:

1. Question 2 (calibration): in this environment the vote share is close
   to a probability on its own (ECE 0.04 pooled; 0.00 to 0.10 on six
   levels), where in the objects contract it was 0.22. The one bad level is
   sk48 (0.21): unanimous and wrong on 9 transitions. The isotonic map fit
   on the other levels makes every level worse, because it learns that
   unanimity is worth 0.94 to 0.98 and that is too low for five levels and
   too high for sk48. Keep the raw vote share and the conformal set; drop
   the map.
2. The adaptive conformal wrapper holds its 0.90 target on every level
   (0.886 to 1.000, pooled 0.952) with sets of mean size 1.0 to 2.3. The
   wrapper abstains where the committee is unsure: 2 percent of steps on
   ar25, 65 percent on g50t. Singleton sets are right 0.83 to 1.00 of the
   time. The coverage guarantee is the transferable part: the same code and
   parameters give 0.90 coverage on the BioProt benchmark (B4).
3. The shared blind spot on sk48 shows in the Good-Turing gap (missing mass
   0.05 against unanimous error 0.19): the committee's own diversity
   under-estimates what it has not sampled there. The row-entropy flag
   does catch some of it (AUROC 0.79 among unanimous steps), which is the
   one signal that reaches an agreed-but-wrong step.

| Item | Value |
|---|---|
| Metric | ECE, Brier, conformal coverage, set size, abstain rate; n 376 |
| Runs | The R34 committees; no new sessions |
| Split | Temporal 40 percent; conformal sets run along the held-out sequence of each level |
| Baseline | R22 (objects mode, four levels) |
| Command | `uv run python -m committee.calibrate --levels ar25:3,m0r0:3,sk48:2,ar25:7,ls20:3,ka59:2,g50t:1 --condition committee_opine_devin --out artifacts/calibration_opine.json` |
| Commit | e20b07f |

## R36. The counterexample round in OPINE-World's program contract: active, passive and object-diff arms, and the live round

R24, R30 and R31 rerun on the R34 committees of the three levels under 0.95
(m0r0 L3, ka59 L2, sk48 L2), plus the live loop of R26 on ar25 L3. Round 1
is the R34 committee. The explorer probes the held-out row with the most
disagreement among the surviving members, drops every member the observed
transition refutes, and stops when no member survives; the probes join the
training set and 8 new programs are synthesized with the counterexample
stated at the mechanism level (R31) in every seed (`cegis_opine_devin`).
The object-diff arm states the same probe as an object diff (R24 wording,
`cegisobj_opine_devin`). The passive arm trains on the same number of
held-out rows taken in time order with no counterexample
(`passive_opine_devin`). Every arm is scored on the rows no arm trained on.

| Level (held out) | Probes until no member survives | Arm | Admitted | Vote | Members (mean, best) | AUROC | Unanimous n (error) | Split n (error) | Behaviours |
|---|---|---|---|---|---|---|---|---|---|
| m0r0 L3 (11) | 33: step 117 first, then 85 to 116 in order | round 1 | 8 | 0.636 | 0.568, 0.636 | 0.41 | 4 (0.50) | 7 (0.29) | 4 |
| | | mechanism | 8 of 8 | 0.909 | 0.909, 0.909 | 0.50 | 11 (0.09) | 0 | 1 |
| | | passive, 33 rows | 7 of 8 | 1.000 | 1.000, 1.000 | none | 11 (0.00) | 0 | 1 |
| ka59 L2 (42) | 2: steps 79, 101 | round 1 | 8 | 0.952 | 0.917, 0.952 | 0.66 | 35 (0.03) | 7 (0.14) | 4 |
| | | mechanism | 8 of 8 | 0.857 | 0.857, 0.881 | 0.66 | 39 (0.10) | 3 (0.67) | 4 |
| sk48 L2 (42) | 25: step 118 first, then 63 to 87 in order | round 1 | 8 | 0.833 | 0.744, 0.833 | 0.29 | 27 (0.26) | 15 (0.00) | 3 |
| | | mechanism | 7 of 8 | 0.976 | 0.963, 0.976 | 0.48 | 40 (0.03) | 2 (0.00) | 2 |
| | | object diff | 8 of 8 | 0.929 | 0.952, 0.976 | 0.83 | 40 (0.03) | 2 (1.00) | 2 |
| | | passive, 25 rows | 8 of 8 | 1.000 | 0.964, 1.000 | none | 39 (0.00) | 3 (0.00) | 2 |

Live round, ar25 L3, local engine, explorer policy, 75 moves, seed 0:

| Committee | Vote | AUROC | Unanimous n (error) | Split n (error) | First move every member gets wrong |
|---|---|---|---|---|---|
| Round 1 (R34, 7 members) | 0.84 | 0.92 | 65 (0.03) | 10 (1.00) | 63 (one HUD cell at (63,63)) |
| Live round: 29 recorded + 71 live rows, mechanism counterexample, 7 of 8 admitted | 1.00 | none | 75 (0.00) | 0 | none |

Reading:

1. Question 4 (does resynthesis after the probes help): yes on two of
   the three levels, and live. m0r0 0.64 to 0.91 and sk48 0.83 to 0.98 on
   the rows no arm saw, and the live committee goes from 0.84 to 1.00 over
   75 fresh moves with no refutation. ka59 loses, 0.95 to 0.86: round 1
   was already right on 40 of the 42 remaining rows, and the eight new
   programs, each seeded with a repair for the row at step 101, converge
   on four behaviours that are wrong on six. The sk48 case is the one
   the loop is for: nine shared errors from a mechanic that first appears
   at step 87 (blocks carried by the arm); disagreement was silent on it
   (R34), the explorer reached it by the time-order fallback at the 25th
   probe, every surviving member was refuted there, and the next committee
   has it.
2. The passive control reaches 1.000 on both levels where it ran, above
   the mechanism arm (0.909 and 0.976). On these two levels the probe set
   is the time-order set up to one row, because the frame_out committees
   are unanimous almost everywhere after the first probe and the explorer
   falls back to time order. So the control isolates the counterexample
   text, not the probe policy: the lift comes from the observations the
   loop collected, and the text adds nothing to it here. ka59 is the one
   level where the probes (steps 79 and 101) differ from time order, and
   its round loses; no passive arm ran there.
3. Question 5 (does naming the mechanism help more): no. Mechanism 0.976
   against object diff 0.929 on sk48 is within one admitted member, and
   both sit under the passive 1.000. The counterexample text narrows the
   committee: one behaviour on m0r0 (all eight share the same wrong rule
   on one row), two on sk48, and the mechanism arm's disagreement AUROC
   falls to 0.48 to 0.50 where the object-diff arm keeps 0.83. A stated
   repair is a prior that every seed takes; it costs the diversity the
   flag runs on. In the objects contract (R31) the statement lifted ar25
   L3 from 0.925 to 1.000; in this environment the programs no longer
   need it.
4. The numbers are small: 11 held-out rows on m0r0, 42 on ka59 and sk48,
   one live trajectory. The direction is consistent across them and with
   R24 to R31 in the objects contract: refutation plus resynthesis lifts
   where the committee was wrong on many of the remaining rows and loses
   where it was nearly right; the wording of the counterexample is not the
   active ingredient. The round-1 rows were first reported from the
   object-contract committee (commit 6577b9a) and corrected here.

| Item | Value |
|---|---|
| Metric | Vote accuracy and member accuracy on the held-out rows shared by every arm; live: vote accuracy over 75 moves |
| Runs | 8 sessions per arm: mechanism on three levels, passive on two, object diff on one, live round on one; 56 sessions, 54 admitted |
| Split | Temporal 40 percent, then the probes until no member survives join train; passive takes the same count in time order |
| Baseline | Round 1 (R34) on the same rows; the passive arm |
| Command | `uv run python -m committee.cegis GAME --level L --train-frac 0.4 --source-condition committee_opine_devin --condition cegis_opine_devin --runs 8 --backend devin --parallel 4` (add `--object-diff --condition cegisobj_opine_devin` for the object-diff arm); `uv run python -m committee.experiment GAME --level L --train-frac 0.4 --train-n N --seeded --frame-out --backend devin --condition passive_opine_devin --runs 8 --parallel 4` with N = train plus probes; report: `uv run python -m committee.cegis GAME --level L --train-frac 0.4 --report --conditions cegis_opine_devin,cegisobj_opine_devin --passive-condition passive_opine_devin`; live: `uv run python -m committee.live ar25 --level 3 --resynth-from artifacts/ar25/live/ar25_L3_f40_committee_opine_devin_seed0.json --through 70 --source-condition committee_opine_devin --round-condition cegis_opine_devin --condition live_opine_devin --backend devin --runs 8 --parallel 4`, then `... --steps 75 --brief --members-dir ar25/L3_f40_probe0_live70/live_opine_devin` |
| Commit | this commit |

## B5. SciGym reaction discovery: the committee loop on a bio benchmark, 30 systems, two open-weight models

SciGym (Duan et al., 2025) hides the reactions of a BioModels network and
scores an agent that runs experiments on a simulator. Members are reaction
networks proposed in text, built into SBML, fitted by least squares to every
observed trajectory, and admitted when the fitted model reproduces every
experiment within SMAPE 0.15. Three arms on the same 30 smallest systems,
k 4, budget 4 experiments, 2 synthesis rounds per member, at most 40 calls
per system: the committee that chooses the experiment where admitted members'
predicted trajectories diverge most (`committee_probe`), the same committee
on a fixed experiment order (`committee_fixed`), and one member on the fixed
order (`single_fixed`). Scored as the paper: reaction-matching F1 and the
trajectory SMAPE on held-out perturbations (STE); 27 systems finished in
every arm for each model and are the paired set. Modal CPU containers run
the fits; the models are vLLM servers on Modal (Qwen3-Coder-30B FP8,
gpt-oss-120b).

| Model | Arm | n | RMS F1 medoid [95% CI] | best member | STE [CI] (empty model) | members admitted | calls | AUROC spread vs F1 < 0.5 |
|---|---|---|---|---|---|---|---|---|
| Qwen | committee_probe | 27 | 0.235 [0.157, 0.320] | 0.316 | 0.577 [0.497, 0.650] (0.515) | 0.00 | 35 | 0.85 |
| Qwen | committee_fixed | 27 | 0.213 [0.138, 0.293] | 0.269 | 0.436 [0.357, 0.523] (0.515) | 0.05 | 33 | 0.64 |
| Qwen | single_fixed | 27 | 0.234 [0.160, 0.307] | 0.234 | 0.454 [0.380, 0.533] (0.515) | 0.07 | 9 | none |
| gpt-oss | committee_probe | 27 | 0.189 [0.115, 0.266] | 0.302 | 0.457 [0.374, 0.539] (0.531) | 0.03 | 34 | 0.62 |
| gpt-oss | committee_fixed | 27 | 0.214 [0.133, 0.293] | 0.316 | 0.393 [0.309, 0.477] (0.531) | 0.08 | 32 | 0.67 |
| gpt-oss | single_fixed | 27 | 0.175 [0.108, 0.245] | 0.175 | 0.432 [0.343, 0.542] (0.531) | 0.07 | 9 | none |

Paired probe minus fixed, RMS F1: Qwen +0.022 [-0.033, +0.084] (wins 12,
losses 6, ties 9); gpt-oss -0.025 [-0.094, +0.038] (8, 7, 12). The paper's
frontier rows on all 137 small systems after 20 iterations: Gemini-2.5-Pro
0.18 F1 and 0.32 STE, GPT-4.1 0.17 and 0.46, Claude-3.7-Sonnet 0.17 and
0.36, Claude-3.5-Haiku 0.05 and 0.63; different systems and budget, given
as the external reference only.

Offline ablations on the same runs (`scigym.ablations`):

| Measure | Qwen probe | Qwen fixed | Qwen single | gpt-oss probe | gpt-oss fixed | gpt-oss single |
|---|---|---|---|---|---|---|
| Members admitted at tolerance 0.15 / 0.3 / 0.5 | 0.00 / 0.05 / 0.25 | 0.06 / 0.15 / 0.37 | 0.08 / 0.12 / 0.31 | 0.03 / 0.10 / 0.37 | 0.09 / 0.15 / 0.45 | 0.07 / 0.11 / 0.48 |
| Survivors per experiment, steps 1 to 4 | 0.37, 0.07, 0.04, 0.04 | 0.67, 0.44, 0.22, 0.19 | 0.19, 0.07, 0.11, 0.07 | 0.56, 0.19, 0.04, 0.04 | 1.15, 0.74, 0.41, 0.30 | 0.19, 0.15, 0.07, 0.07 |
| F1 of the medoid after steps 1 to 4 | 0.21, 0.22, 0.27, 0.24 | 0.19, 0.20, 0.22, 0.21 | 0.19, 0.22, 0.18, 0.23 | 0.22, 0.18, 0.21, 0.19 | 0.19, 0.23, 0.22, 0.21 | 0.18, 0.22, 0.23, 0.18 |
| Selective by spread: F1 all, most agreed half, lowest-spread quarter, highest-spread quarter | 0.235, 0.348, 0.473, 0.051 | 0.213, 0.245, 0.235, 0.184 | | 0.189, 0.206, 0.179, 0.100 | 0.214, 0.233, 0.251, 0.290 | |

Reading:

1. Question 1 on this benchmark: the committee's answer is not more
   accurate than one member at 4 experiments. Every arm sits at 0.17 to
   0.24 F1 with overlapping intervals, and the paired probe-minus-fixed
   difference crosses zero for both models. The committee costs 3.7 times
   the calls.
2. Question 3 transfers where the experiments were chosen by disagreement.
   On the Qwen probe arm the spread among members ranks the systems the
   committee gets wrong at AUROC 0.85: the lowest-spread quarter has F1
   0.47 and the highest-spread quarter 0.05, so a user who acts only on the
   agreed half gets 0.35 instead of 0.24. On the fixed arms and on gpt-oss
   the signal is weak (0.62 to 0.67).
3. The loop ran in one regime only. At tolerance 0.15 no member of the Qwen
   probe arm was ever admitted and at most one member survived an
   experiment, so every step resynthesized the whole committee and the
   selection half of the method (keep survivors, probe where they split)
   never operated; F1 is flat across the four steps. At tolerance 0.5 about
   40 percent of members would have been admitted. The tolerance arms (B6)
   rerun the three arms there.
4. Trajectory error: the probe arm's STE is worse than the fixed arm's on
   both models (0.58 against 0.44 on Qwen). The disagreement-chosen
   experiments are the extreme perturbations (knockouts, far initial
   concentrations), and a network fitted to them transfers less well to the
   paper's held-out perturbations than one fitted to the default order.

| Item | Value |
|---|---|
| Metric | RMS reaction F1 and STE as Duan et al. (2025); AUROC of committee spread against F1 < 0.5 |
| Runs | 3 arms x 2 models x 30 systems, 27 paired per model; about 2,000 s of CPU per committee system |
| Split | 30 smallest systems of the 137; held-out perturbations as the paper |
| Baseline | `single_fixed` (one member, the paper's setting) and `committee_fixed` (same committee, no probe choice); the paper's frontier rows as the external reference |
| Command | `uv run modal run -m scigym.modal_app --arm committee_probe,committee_fixed,single_fixed --model qwen --n-systems 30 --k 4 --budget 4 --rounds 2 --max-calls 40` (and `--model gptoss`); `uv run python -m scigym.report --model qwen`; `uv run python -m scigym.ablations --model qwen` |
| Commit | this commit (artifacts `artifacts/scigym/<arm>_<model>/`) |

## B6. SciGym, questions 2 and 3: the committee's spread as a calibrated score, at the system and the reaction level

On the B5 runs (`scigym.calibration`). System level: the committee's
confidence that its answer is right (reaction F1 at least 0.5) is one minus
the spread among members, the mean pairwise reaction-set distance; the R22
wrapper turns the spread into a set over {right, wrong}: one label is a
commitment, both is an abstention. Split conformal calibrates on half the
systems, 500 random splits; the online rule (ACI, gamma 0.05) runs along the
systems in id order. Reaction level: every reaction any member proposed,
with the share of members behind it, against whether it is in the hidden
network; the effect-row flag of the ARC committee, here per reaction.

| Model, arm | n | Answers right | ECE of 1 - spread | AUROC spread vs wrong | Conformal coverage (target 0.90) | Committed | Accuracy when committed | Certified right (error) | Online coverage, committed, accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Qwen, probe | 28 | 0.14 | 0.24 | 0.85 | 0.93 [0.64, 1.00] | 0.38 | 0.83 | 0.06 (1.00) | 0.79, 0.54, 0.80 |
| Qwen, fixed | 28 | 0.14 | 0.27 | 0.66 | 0.93 [0.64, 1.00] | 0.19 | 0.53 | 0.06 (1.00) | 0.82, 0.25, 0.71 |
| gpt-oss, probe | 30 | 0.07 | 0.38 | 0.61 | 0.98 [0.73, 1.00] | 0.03 | 0.43 | 0.02 (0.99) | 0.87, 0.07, 0.50 |
| gpt-oss, fixed | 27 | 0.11 | 0.28 | 0.67 | 0.93 [0.64, 1.00] | 0.26 | 0.73 | 0.07 (0.77) | 0.93, 0.48, 0.85 |
| single member, both | 30 | 0.07 to 0.13 | 0.87 | 0.50 | 1.00 | 0.00 | none | 0.00 | abstains on every system |

| Model, arm | Proposed reactions | AUROC share vs true | Precision, unanimous (n) | Precision, split (n) | Precision by share quartile, low to high |
|---|---|---|---|---|---|
| Qwen, probe | 538 | 0.64 | 0.26 (68) | 0.11 (470) | 0.06, 0.14, 0.12, 0.20 |
| Qwen, fixed | 508 | 0.59 | 0.11 (66) | 0.14 (442) | 0.10, 0.11, 0.14, 0.20 |
| gpt-oss, probe | 300 | 0.51 | 0.19 (70) | 0.20 (230) | 0.19, 0.17, 0.23, 0.19 |
| gpt-oss, fixed | 327 | 0.55 | 0.20 (45) | 0.18 (282) | 0.18, 0.12, 0.23, 0.20 |

Reading:

1. Question 3 holds at the system level on the arm whose experiments were
   chosen by disagreement, and weakly elsewhere: Qwen probe AUROC 0.85,
   with the agreement quartiles right 0.00, 0.00, 0.14 and 0.43 of the
   time; the other committee arms 0.61 to 0.67. The raw confidence is not
   a probability (ECE 0.24 to 0.38): the committee is right on 7 to 14
   percent of systems and its most agreed quartile on 14 to 43 percent.
2. Question 2: the wrapper keeps its guarantee (split coverage 0.93 to
   0.98 at a 0.90 target; online 0.79 to 0.93 on 27 to 30 systems, where
   the online rule has too few steps to settle) and it spends the guarantee
   on rejections. On Qwen probe it commits on 38 percent of systems with
   accuracy 0.83, and almost every commitment says "this answer is wrong";
   it certifies an answer as right on 6 percent of systems and those are
   wrong. On this benchmark the honest outputs are "I do not know" and
   "this is wrong", rarely "this is right"; a single member has no spread,
   so the wrapper abstains on every system.
3. At the reaction level the flag is weak. A reaction every member proposes
   is right 11 to 26 percent of the time, a split one 11 to 20 percent;
   AUROC 0.51 to 0.64. Members share wrong reactions, so unanimity on a
   reaction marks the model's prior more than the truth: the shared blind
   spot of sk48 L2 (R34) is the common case here, not the exception. The
   system-level spread works because it sums many small disagreements; no
   single reaction's share does.

| Item | Value |
|---|---|
| Metric | ECE (5 equal-mass bins), AUROC, conformal coverage and commitment at alpha 0.1; reaction-level precision by share |
| Runs | The B5 runs; no new sessions |
| Split | 27 to 30 systems per arm; split conformal over 500 half splits of the systems; online in id order |
| Baseline | The single member (no spread); the ARC committee on the same measures (R35) |
| Command | `uv run python -m scigym.calibration --model qwen` and `--model gptoss` (writes `artifacts/scigym/calibration_<model>.json`) |
| Commit | this commit |

## B7. SciGym at admission tolerance 0.5: the regime where members survive, and the no-counterexample control

B5's arms rerun with `--eps 0.5` (same 30 systems, k 4, budget 4, 2 rounds,
40 calls), plus `committee_probe_nocx` on Qwen: the probe arm with refuted
members resynthesized on the data alone, no counterexample text. Systems
that hit the 5,400 s container limit are missing (Qwen probe 4, no-cx 3,
fixed 1); the paired set is 30 (gpt-oss) and 26 (Qwen).

| Model | Arm | RMS F1 [95% CI] | best member | STE (empty 0.50 to 0.54) | systems with an admitted member | survivors after experiment 1 / 4 | calls | AUROC spread vs wrong |
|---|---|---|---|---|---|---|---|---|
| gpt-oss | committee, probe | 0.219 [0.140, 0.306] | 0.313 | 0.438 | 0.53 | 2.23 / 0.97 | 21 | 0.70 |
| gpt-oss | committee, fixed | 0.187 [0.109, 0.273] | 0.285 | 0.371 | 0.70 | 2.83 / 2.17 | 14 | 0.50 |
| gpt-oss | single member | 0.191 [0.117, 0.275] | 0.191 | 0.353 | 0.60 | | 3 | none |
| Qwen | committee, probe | 0.215 [0.142, 0.289] | 0.298 | 0.495 | 0.42 | 2.12 / 0.58 | 23 | 0.69 |
| Qwen | committee, fixed | 0.190 [0.122, 0.264] | 0.276 | 0.348 | 0.58 | 2.81 / 1.58 | 16 | 0.72 |
| Qwen | single member | 0.184 [0.102, 0.269] | 0.184 | 0.512 | 0.35 | | 5 | none |
| Qwen | committee, probe, no counterexample | 0.171 [0.103, 0.241] | 0.254 | 0.509 | 0.38 | 2.15 / 0.81 | 24 | 0.61 |

Paired probe minus fixed, RMS F1: gpt-oss +0.033 [-0.033, +0.104] (7 wins,
8 losses, 15 ties); Qwen +0.025 [-0.024, +0.079] (8, 5, 13). Probe minus
no-counterexample on Qwen: 0.215 against 0.171.

Calibration (`scigym.calibration --tag _eps50`): AUROC of spread against a
wrong answer 0.69 to 0.75 on Qwen, 0.70 (probe) and 0.50 (fixed) on gpt-oss;
conformal coverage 0.93 to 0.98 at target 0.90, with 75 to 98 percent
abstention. Reaction level: AUROC 0.50 to 0.65; unanimous reactions right 9
to 31 percent of the time.

Reading:

1. The regime changed as intended: two to three of four members survive the
   first experiment (none at 0.15), so selection operates and the probe arm
   refutes faster than the fixed arm (survivors after four experiments 0.6 to
   1.0 against 1.6 to 2.2), which is the disagreement probe doing its job.
2. Question 1, again: the probe committee is highest on both models (0.215
   to 0.219 against 0.184 to 0.191 for one member) and its best member
   reaches 0.30 to 0.31, but every interval overlaps and the paired
   differences cross zero. Not established at 26 to 30 systems.
3. Question 5 analog: the counterexample text helps here, unlike ARC (R36).
   With the same probe policy, removing it drops Qwen from 0.215 to 0.171,
   below the single member. On SciGym the text names the species whose
   trajectory the member missed; on the ARC frame contract the observations
   alone carried the repair. One model, one batch, overlapping intervals:
   a direction, not a result.
4. Question 3 holds weakly at the system level (0.69 to 0.75 except gpt-oss
   fixed) and stays weak per reaction, as in B6.
5. Trajectory error is worse for the probe arms on both models (0.44 to 0.50
   against 0.35 to 0.37 fixed), as at 0.15: experiments chosen at the
   extremes fit models that transfer less to the paper's held-out
   perturbations.

| Item | Value |
|---|---|
| Metric | As B5 and B6 |
| Runs | 7 arms x 30 systems; 26 to 30 finished per arm |
| Split | As B5 |
| Baseline | Single member; the same committee on a fixed order; B5 at tolerance 0.15 |
| Command | `uv run modal run -m scigym.modal_app --arm committee_probe,committee_fixed,single_fixed,committee_probe_nocx --model qwen --n-systems 30 --k 4 --budget 4 --rounds 2 --max-calls 40 --eps 0.5` (gpt-oss without `_nocx`); `SCIGYM_EPS=0.5 uv run python -m scigym.report --model qwen --tag _eps50`; `... scigym.calibration ... --tag _eps50` |
| Commit | this commit (artifacts `artifacts/scigym/*_eps50/`) |

## R37. Committee size: K from 1 to 16, seeded against unseeded members

The R34 levels under OPINE-World's program contract, with 8 more seeded
members per level (`committee16_opine_devin`, the seeds of a 16-member
batch, 56 Devin sessions) and 8 more unseeded (`unseeded_opine_devin`, 56
sessions); 111 of 112 admitted. Seeded arm: up to 16 members per level (15 on
ar25 L3). Unseeded arm: the 3 R34 single programs plus 8, 11 per level. For
each K, 200 draws; each draw takes one random K-subset on every level and
pools the 376 held-out transitions. Mean and 95 percent interval over draws.

| K | Seeded: vote | any member right | AUROC | unanimous share | unanimous error | split error | Unseeded: vote | AUROC | unanimous error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.880 | 0.880 | | 1.00 | 0.120 | | 0.876 | | 0.124 |
| 2 | 0.881 | 0.911 | 0.68 [0.52, 0.87] | 0.92 | 0.077 | 0.62 | 0.875 | 0.65 [0.51, 0.83] | 0.087 |
| 4 | 0.885 | 0.933 | 0.79 [0.67, 0.92] | 0.86 | 0.048 | 0.53 | 0.882 | 0.76 [0.60, 0.88] | 0.056 |
| 8 | 0.883 | 0.952 | 0.88 [0.78, 0.96] | 0.81 | 0.022 | 0.52 | 0.876 | 0.86 [0.78, 0.90] | 0.034 |
| 12 | 0.886 | 0.961 | 0.92 [0.83, 0.95] | 0.78 | 0.010 | 0.49 | | | |
| 16 | 0.887 | 0.963 | 0.92 | 0.77 | 0.007 | 0.47 | | | |

Reading:

1. The flag sharpens with K and does not saturate by 16: AUROC 0.68, 0.79,
   0.88, 0.92, 0.92 and unanimous error 0.120, 0.077, 0.048, 0.022, 0.010,
   0.007. At K = 16 the committee is unanimous on 77 percent of transitions
   and wrong on 0.7 percent of those. This is the shared-blind-spot failure
   (R34, sk48) shrinking: more members make it less likely that all of them
   share one wrong prior. On sk48 L2 two of the eight new seeded members
   carry the blocks with the arm (1.00), where none of the first eight did.
2. Accuracy does not move: the vote stays at 0.88 to 0.89 from K = 1 to 16,
   while the share of transitions some member gets right rises from 0.88 to
   0.96. More members find the right answer, but as a minority; the vote
   cannot pick it. The value of K is in the uncertainty, not the answer,
   which is the method's claim.
3. Compute-matched at K = 8: seeded against unseeded, vote 0.883 against
   0.876, AUROC 0.88 against 0.86, unanimous error 0.022 against 0.034.
   Seeds help a little and inside the intervals; most of the gain is the
   number of independent programs. (Unseeded K = 8 draws from 11 members per
   level, so its draws overlap more than the seeded arm's.)
4. The original R34 batch of eight sits low in the K = 8 distribution
   (AUROC 0.756 against the draw mean 0.88 and lower bound 0.78): the
   first batch's seeds were more alike than a random 8 of 16. Batch variance
   is real and R34's AUROC is the conservative end.

| Item | Value |
|---|---|
| Metric | Pooled vote accuracy, any-member-right accuracy, AUROC of uniform disagreement, unanimous and split error |
| Runs | 112 new Devin sessions; 200 subset draws per K |
| Split | Temporal 40 percent, seven levels, 376 held-out transitions |
| Baseline | K = 1 (one program); unseeded members at equal K |
| Command | `uv run python -m committee.experiment GAME --level L --train-frac 0.4 --backend devin --frame-out --runs 8 --start 8 --seeded --condition committee16_opine_devin --parallel 4`; `... --runs 8 --condition unseeded_opine_devin --parallel 4`; `uv run python -m committee.ksweep` (writes `artifacts/ksweep_opine.json`, figure `artifacts/figures/ksweep.png`) |
| Commit | this commit |

## O11. Devin programs on the expressive full-access store, first 30 worlds

Store `onc-agi-public-train-expressive-full-access` (248 worlds; roles that
stress the fixed templates). First 30 worlds in store order, scorer
`scorer-1.0`. Devin wrote 4 seeded programs per world (analyst, direct,
sparse, confounder), 120 sessions, all 120 admissible; a network drop at
13:00 cost 89 sessions, which were redone (the 89 failed records held no
program and are kept out of the cache). Committee as O10: likelihood
weights, one vote per distinct driver set, the leak filter.

| Agent, first 30 expressive worlds | DS | 95% | Find | Restraint |
|---|---|---|---|---|
| forward_score (benchmark baseline) | 0.556 | [0.37, 0.75] | 0.64 | 0.88 |
| 4 Devin programs | 0.464 | [0.27, 0.65] | 0.51 | 0.92 |
| univariate_bh (benchmark baseline) | 0.426 | [0.27, 0.59] | 0.49 | 0.88 |
| 1 Devin program (analyst) | 0.415 | [0.23, 0.65] | 0.50 | 0.83 |
| templates and 4 Devin programs | 0.292 | [0.12, 0.49] | 0.37 | 0.79 |
| stability (benchmark baseline) | 0.277 | [0.11, 0.52] | 0.42 | 0.67 |
| random_forest, elastic_net, lasso | 0.093, 0.078, 0.040 | | | |
| oracle; 12 cheaters and random | 1.000; 0.000 | | | |

Reading. (1) The best benchmark baseline leads here (0.56 against 0.46);
the intervals overlap. (2) Unlike full-access (O10, four seeds scored as
one), four Devin seeds add 0.05 over one (0.464 against 0.415), through
restraint (0.92 against 0.83). (3) Mixing the templates in lowers the
score again (0.29), the O10 admission finding: predictive admission lets
members that fit many columns vote. The templates-alone row on these
worlds was not in this run; it is computed separately
(`--tag templates_first30`) and added when it lands.

Runs: 1 per condition, deterministic given the cache. Split: public_train.
Baseline: the benchmark's baselines on the same 30 worlds. Commands:
`uv run python -m onc.synth --store artifacts/onc/benchmark/expressive-full-access --mode full --first 30 --k 4 --model devin --workers 10`;
`uv run python -m onc.evaluate --store artifacts/onc/benchmark/expressive-full-access --mode full --weighting likelihood --first 30 --members synth:4:devin,synth:1:devin,both:4:devin --out artifacts/onc/eval_bench --tag devin_expressive-full-access_first30`;
`uv run python -m onc.baselines --store artifacts/onc/benchmark/expressive-full-access --first 30 --out artifacts/onc/baselines_bench_first30_expressive-full-access`.
