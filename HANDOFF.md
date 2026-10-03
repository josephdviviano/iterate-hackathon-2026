# Handoff: program committee build

Written 2026-10-03 17:15 BST. Deadline about 2026-10-04 11:00 BST. No new
features after about 08:00; those hours go to repairs, the video, README
and the submission. Read CLAUDE.md first; then DESIGN_DOC.md, RESULTS.md,
NOTES.md and research/baseline_review.md. The write-up page for the human
reader is https://claude.ai/artifact/RZK96oVH1iZXjRnY4ndH8C (private;
version 6).

## State

- Branch `jdv`, 30 commits, head `0749a24`. Nothing pushed. `main` holds
  only the two pre-build commits. Merging `jdv` into `main` is the user's
  call and is required for "the demo runs from main".
- Uncommitted edits under `src/hoeffding`, `src/rewardhack`, `tests/` and
  `artifacts/hoeffding` belong to two other sessions (see "Parallel builds"
  in DESIGN_DOC.md). Do not commit or edit them. Shared files
  (README.md, RESULTS.md, NOTES.md, DESIGN_DOC.md, pyproject.toml) are
  append-only; use Edit, never Write, on them.
- `external/conceptualizer` is untracked on purpose (pre-event code, no
  `.git`). `external/opine-world` is a submodule and read-only.
- Tests: `uv run pytest` (17 pass). Every committee test was checked
  against a deliberate mutant; keep that rule.

## What the method is now

Sample K = 8 programs with Devin, each from a data-derived seed; keep the
ones that replay every train transition exactly; equal-weight plurality for
the prediction; normalised vote entropy for ranking uncertainty; adaptive
conformal sets along the trajectory for the calibrated statement; probe
where the committee disagrees. Everything else is an opt-in ablation:
MDL prior (`--lam`), targeted growth (`committee.active`), disentanglement
(`committee.disentangle`), mechanism library (`committee.library`), judge
(`committee.confidence`), open-loop mode (`committee.rollout`).

The one-sentence description the reviews endorse: query by committee
applied to programmatic world models, with the first measurement that it
is calibrated on ARC-AGI-3 object data.

## Credentials and services

- Devin: `DEVIN_API_KEY` in `.env.committee` (gitignored) and the Modal
  secret `devin-auth`. About 110 sessions used today; balance unknown, the
  consumption API is not enabled for this account; check the dashboard.
- Modal: `vllm-auth` secret (`VLLM_API_KEY`, `OPENAI_BASE_URL`); two
  deployed servers, `committee-llm` (Qwen3-Coder-30B FP8) and
  `committee-llm-gptoss` (gpt-oss-120b); both scale to zero after 15 min
  and answer 503 while cold. URLs in `.env.committee` as
  `OPENAI_BASE_URL` and `OPENAI_BASE_URL_GPTOSS`. Neither model admits
  programs on hard levels (R2); they serve the judge baseline.
- No ARC-AGI-3 API key: no live play is possible until one exists.
- The Claude account is not to be used for synthesis (user decision).

## Artifact layout and commands

`artifacts/<game>/L<level>_f<frac>[_T<test>][_n<train_n>]/<condition>/run<k>/`
holds `program.py`, `meta.json`; `test_preds.json` is gitignored and is
rebuilt on demand by `committee.evaluate.load_runs`. Condition names in
use: `baseline_devin`, `committee_devin`, `unseeded_devin`,
`active_devin`, `lib_devin`, `nolib_devin`, `api_*`, `devin_smoke`.

```
uv run python -m committee.experiment GAME --level L --train-frac 0.4 --runs 8 --seeded --backend devin --condition committee_devin --parallel 4
uv run python -m committee.evaluate GAME --level L --train-frac 0.4 --condition committee_devin [--curve]
uv run python -m committee.rollout GAME --level L --train-frac 0.4
uv run python -m committee.baselines GAME --level L --train-frac 0.4
uv run python -m committee.calibrate            # all four informative levels
uv run python -m committee.demo --game ar25 --level 3 --train-frac 0.4
```

Known trap: a Devin session can stall in `working`; the loop waits for
`--timeout` (1800 s) and the thread then returns a stub. One ar25 L7
baseline process hung past that and had to be killed (`pkill -f
"committee.experiment ar25 --level 7"`). `committee.synth_devin` nudges
blocked sessions and sessions that reported ALL PASS without output.

## Results index (RESULTS.md)

| Entry | One line |
|---|---|
| R1, R2 | tr87 L1 saturated; open models fail exact replay in a repair loop, Devin and claude solve it |
| R3, R4 | ar25 L3: single programs 0.52 mean; seeded committee AUROC 0.78, unanimous error 0.30, split 0.83 to 1.00; 51 of 90 rows unseen; probe by disagreement falsifies in 1 vs 7 |
| R5, R7 | tr87 L1 to L2 and ft09 L5 saturated, committee unanimous and right |
| R6 | m0r0 L3: AUROC 0.68, unanimous 0.16, split 0.80; seeds lift admission 1 of 3 to 8 of 8 |
| R8 | Targeted growth negative: members adopt the majority stance |
| R9 | Learned ensembles (trees, MLP) cannot model the mechanics at 25 to 30 transitions |
| R10 | AUROC rises with K; pooled 0.75, CI 0.66 to 0.84 |
| R11, R18 | Judge confidence: strong on ar25, chance elsewhere; combination fails out of sample |
| R12 | Unseeded resampling matches seeded calibration; seeds help admission and diversity on m0r0 |
| R13 | Row-level entropy predicts rows with errors (0.75, 0.72, 1.00) where count-based η is at chance |
| R14 | No complexity measure tracks held-out accuracy over 45 programs; equal weights default |
| R15 | Open-loop rollouts; disagreement grows with horizon where programs diverge; memory carries across levels on tr87 |
| R16 | Entropy-based row splitting has no power at 2 to 8 observations; perturbation attributes rows to conditions |
| R17, R21 | sk48 L2 and ar25 L7: AUROC 1.00, unanimous error 0.00 |
| R19 | Library: 15 mechanisms, 12 unanimous, 3 contested with named disputed stances |
| R20 | Library-conditioned synthesis at 12 train transitions: inconclusive at 3 and 4 per arm |
| R22 | Vote share is not a probability (ECE 0.22); adaptive conformal sets hold 0.90 coverage on all four levels with abstention where the committee is wrong |

Also measured, not yet in RESULTS: on every informative level the oracle
"any member right" equals the best single member (0.64, 0.77, 0.97, 0.42),
so combining members has no headroom; the problems are selection and
absent hypotheses. Record this as R23 if it is used.

## Unexplored directions

1. **Close the CEGIS loop.** After the first falsifying probe, resynthesize
   a committee conditioned on the counterexample and measure the next
   committee's held-out error. Everything measured so far stops at
   "falsified after one probe". This is the most important missing piece
   and the one that scales with the synthesizer rather than with our
   bookkeeping. One level, 8 to 16 sessions.
2. **Live play.** Needs an ARC-AGI-3 API key and a harness around the
   `arc-agi` package: planner over the committee, conformal set as the
   trust signal, abstention as the cue to explore. Without it the pitch
   cannot claim actions saved in a game.
3. **Factored seeds, no anchoring.** Assign stances on the library's
   contested mechanisms across members by a covering design (fractional
   factorial), so per-mechanism marginals are read off the committee and
   one distinguishing probe settles one mechanism. Opt-in condition; not
   enumeration (the user ruled that out: it does not scale).
4. **Version-space-spanning probes with rollouts.** Opt-in only, by the
   user's decision; the member sees the majority prediction and is asked
   for a consistent program that differs if the data permits. Must not mix
   with the base committee's results.
5. **Decisive library A/B.** 8 per arm on two levels at 12 train
   transitions, about 32 sessions. R20 was 3 and 4 per arm.
6. **Wider benchmark.** 111 levels have 25+ modellable transitions. Tiers:
   screen 17 stratified levels with 2 single programs each (34 sessions),
   committees on the informative ones (64 to 80), ablations on 3 to 4
   (40 to 60), cross-level on 4 games (32). Fixed-count split
   (`--train-n 40 --test-n 80`) is implemented for this; defaults
   unchanged. Blocked on the Devin budget.
7. **Conformal sets inside exploration.** Abstention rate as the trigger to
   observe; set size as the planning horizon. Simulation is possible
   offline with the rollout machinery.
8. **Agreed-but-wrong.** Row entropy flags it on ar25 L3 (0.87) only.
   Good-Turing does not. Open-loop divergence as a feature is untested.
9. **Soft committee (no verifier).** Never run; too few inconsistent
   programs. Would need weaker synthesis on purpose.
10. **Selection among members.** The vote loses to the best member on
    ar25 L3 and sk48. Nothing in train identifies it; probes do. A
    "probes as selection" curve (accuracy of the survivors after n probes)
    is cheap from stored data.
11. **Devin backend on the Modal fan-out.** Code path exists, untested.
12. **ECE and Brier with the conformal set** as the probability object
    (coverage-based), and cluster bootstrap by level once more levels exist.

## Prioritized next steps

| Priority | Step | Cost | Why first |
|---|---|---|---|
| 1 | Merge `jdv` into `main` (user), run `uv run python -m committee.demo` from main | minutes | Work rule 2 |
| 2 | Record R23 (oracle = best member) and the probes-as-selection curve from stored data | 1 h, no sessions | Closes the "combine the programs" question with numbers |
| 3 | Close the CEGIS loop on ar25 L7 or ar25 L3 | 2 h, 8 to 16 sessions | The missing mechanism of the method |
| 4 | Demo video (90 s, cached demo) and README final pass; project description | 2 h | Submission items |
| 5 | Decisive library A/B or the benchmark screening tier | 1 to 3 h, 32 to 34 sessions | Only if budget is confirmed |
| 6 | Factored seeds as an opt-in condition | 2 h, 8 sessions | Scalable successor to seeds and targeted growth |

Stop features at 08:00 BST. Keep the last three hours for repairs, the
video, README.md and the submission.

## Decisions the user has made (do not relitigate)

- Independent seeded synthesis is the construction method; targeted growth
  is dropped. Every added component stays opt-in until an A/B shows a gain.
- Equal weights are the default; priors are ablations.
- Online conformal prediction is the calibration route, chosen for
  generality beyond ARC.
- Enumeration of mechanism combinations is out: it does not scale.
- Do not over-index on the current levels; later levels are harder and
  there is no live play yet.
- PDFs are tracked; predicted-state dumps are not.
- No Claude account for synthesis; Devin and Modal only.
