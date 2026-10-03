# Handoff: the night plan, written 2026-10-03 22:00 BST

Deadline about 2026-10-04 11:00 BST. No new features after 08:00; those
hours go to repairs, the video, README.md and the submission. Read
CLAUDE.md first; then RESULTS.md (R3 to R33), NOTES.md and this file. The
write-up page is https://claude.ai/artifact/RZK96oVH1iZXjRnY4ndH8C (private,
version 19; every number on it points to a RESULTS entry).

## State

- Branch `jdv`, head `e953262`, pushed to origin. `main` holds only the
  pre-event commits; the user's decision is "pull from main, push jdv only".
  Merging into `main` is the user's call and is needed before the demo can
  be said to run from main (work rule 2).
- Uncommitted work in the tree belongs to other sessions and must not be
  committed, edited or reverted here:
  - `src/committee/` (22 files) and `src/committee/env.py`: the environment
    port by session `iterate-10` (modes `objects`, `frame`, `frame_out`).
    Its review fixes landed at 21:27, 62 tests pass, demo and stored
    evaluations replay unchanged. It commits only when the user tells it to.
    Step P0 below is that commit.
  - `src/onc`, `src/ideafilter`, `src/hoeffding`, `src/rewardhack` and their
    tests and artifacts: the ONC-AGI, idea-filter, Hoeffding and
    reward-hacking builds.
  - RESULTS.md and NOTES.md carry uncommitted entries from those sessions.
    To commit your own appended entry without taking theirs, stage a blob
    built from `git show HEAD:RESULTS.md` plus your own tail (`git
    hash-object -w --stdin`, then `git update-index --cacheinfo
    100644,<hash>,RESULTS.md`), then commit.
- Tests for the committee build: `uv run python -B -m pytest -q -p
  no:cacheprovider tests/test_committee.py tests/test_explore.py
  tests/test_loader.py` (16 pass). Every test was checked against a
  deliberate mutant; keep that rule. Clear `__pycache__` before a mutant
  check, or a same-length edit runs from the stale `.pyc`.
- Demo: `uv run python -m committee.demo --game ar25 --level 3 --train-frac
  0.4`, offline, about 2.3 s, seven sections including the loop and both
  round-2 statements. Live demo: `uv run python -m committee.live ar25
  --level 3 --steps 75 --brief --members-dir ar25/L3_f40_probe4_live70/live_devin`
  plays 75 moves on the local engine in about 15 s.

## What is established (objects mode unless stated)

| Claim | Numbers | Entries |
|---|---|---|
| The committee does not predict better than one program | the oracle "any member right" equals the best member on every level | R3, R23 |
| It knows when it is wrong | unanimous error 0.00 to 0.30, split error 0.26 to 1.00, AUROC 0.68 to 1.00 on 8 levels of 6 games | R4, R6, R17, R21, R27 |
| Calibrated statement | vote share is not a probability (ECE 0.22); adaptive conformal sets hold 0.90 on all 8 levels (0.88 to 0.99, pooled 0.93) | R22, R27 |
| Probing refutes, it does not select | the first disagreement probe refutes every member on every level; seen-rows-first does not select either | R23, R28 |
| Resynthesis on the refuting observation | ar25 L3 0.48 to 0.93 to 1.00 in three rounds; sk48 0.70 to 0.86; null on m0r0 and ls20; marginal on g50t; loss on ka59; convergence kills the disagreement signal | R24, R27 |
| Mechanism-level counterexample (now the default) | ar25 L3 to 1.00 in one round; equal elsewhere; ka59 still converges on fitted geometry | R31 |
| Live play on the engine | rounds 1 and 3 unanimous and right for 64 moves, refuted by an unrecorded counter mechanic with disagreement 0.00; the live counterexample round lifts 0.40 to 0.71 and 0.86 | R25, R26 |
| Abstention is a floor; the loop converts it | selective score; live commit share 0.44 to 0.78, score 0.35 to 0.65; gamma 0.3 leave-one-out; the wrapper wins when a wrong answer costs at least a right one | R29, R30 |
| The object state is incomplete | objects explain 0.07 to 0.96 of the frame; terrain objects reach 0.96 to 1.00 on six levels; ka59's wall mass is not in the state; terrain objects do not help the synthesizer | R32 |
| OPINE's own environment (frame in, frame out), pilot | single programs ar25 L3 1.00 and 0.84, m0r0 0.86, ka59 0.89 to 0.91; the committee vote equals or sits under the best single; calibration holds (unanimous 0.00 to 0.14, split 0.43 to 1.00) | R33 |

R33 is a pilot: the port's first task text, and ten members lost to a Devin
suspension (21:19 to at least 21:56, every new session suspended within a
minute). The user reports Devin is usable again. Be judicious.

## The plan

Budget: 101 Devin sessions in the base plan, 33 more on one contingency.
A session is 2 to 10 minutes; run arms at `--parallel 4` and keep about 20
sessions concurrent at most (that worked all day). Set a time limit per
step; at the limit record the result, positive or negative.

### P0. Commit the environment and prove Devin runs (0 to 1 session, 20 min)

1. The user tells `iterate-10` to commit the env port. Pull it into this
   tree (same branch; `git status` clean in `src/committee`).
2. Run the test suite and the demo. Push.
3. One probe session: `uv run python -m committee.experiment tr87 --level 1
   --train-frac 0.6 --runs 1 --backend devin --condition devin_probe2`. If
   its meta says `status: suspended` and the program is the 114-byte stub,
   stop and tell the user; nothing below can run.

### P1. The comparison in OPINE's environment under the final text (44 sessions, about 1 h)

Conditions `baseline_opine_devin` (3 unseeded) and `committee_opine_devin`
(8 seeded), mode `--frame-out`. Never add to the pilot conditions
(`*_frameout_devin`): `check_mode` refuses mixed modes, nothing refuses
mixed task texts, so the names must differ.

```
uv run python -m committee.experiment GAME --level L --train-frac 0.4 --runs 3 --backend devin --frame-out --condition baseline_opine_devin --parallel 3
uv run python -m committee.experiment GAME --level L --train-frac 0.4 --runs 8 --seeded --backend devin --frame-out --condition committee_opine_devin --parallel 4
uv run python -m committee.evaluate GAME --level L --train-frac 0.4 --condition committee_opine_devin
```

1. ka59 L2 first (11 sessions). It measures the text effect against the
   pilot: if the single mean and the vote are within 0.03 of R33's, the
   pilot stands for ar25 L3, m0r0 L3 and sk48 L2; if not, rerun those three
   (33 sessions, the contingency).
2. ar25 L7, ls20 L3, g50t L1 (33 sessions), the levels with no frame data.
3. Record R34: per level, single programs and committee in objects mode and
   in the OPINE environment side by side (admission, vote, members, AUROC,
   unanimous and split error), plus the conformal battery:
   `uv run python -m committee.calibrate --levels ar25:3,m0r0:3,sk48:2,ar25:7,ls20:3,ka59:2,g50t:1 --condition committee_opine_devin --out artifacts/calibration_opine.json`
   (`--condition` is in the port). No kill criterion: this is the
   measurement the user asked for.

### P2. The counterexample round in OPINE's environment (about 40 sessions, 1 h)

Only on levels where the round-1 vote under P1 is below 0.95 (ar25 L3 is at
1.00 and is skipped). Expected: ka59, m0r0, sk48, g50t, ar25 L7. The
mechanism statement is the default, the mode is inherited from the source
committee, and there is no passive arm (R24 settled that).

```
uv run python -m committee.cegis GAME --level L --train-frac 0.4 --source-condition committee_opine_devin --condition cegis_opine_devin --dry-run
uv run python -m committee.cegis GAME --level L --train-frac 0.4 --source-condition committee_opine_devin --condition cegis_opine_devin --runs 8 --backend devin --parallel 4
uv run python -m committee.cegis GAME --level L --train-frac 0.4 --source-condition committee_opine_devin --condition cegis_opine_devin --report
```

A level whose committee is never refuted (dry run says so) is skipped.
Record R35: does the loop still lift accuracy once the environment is
right, and does the new committee converge as before? Stop rule: one round
per level; a second round only where the first lifted the vote by 0.10 or
more.

### P3. Live play in OPINE's environment (0 to 8 sessions, 45 min)

`committee.live` inherits the mode from the committee it loads (per
`iterate-10`); test with `--steps 20` before anything else.

```
uv run python -m committee.live ar25 --level 3 --steps 300 --seed 0 --members-dir ar25/L3_f40/committee_opine_devin
uv run python -m committee.live ar25 --level 3 --steps 300 --seed 1 --members-dir ar25/L3_f40/committee_opine_devin
```

If the committee is refuted within the game's 160 moves, one live
counterexample round (8 sessions) as in R26, then replay. Record R36. This
is the live demo in the proper environment; keep the 75-move `--brief`
command for the pitch.

### P4. Batch variance (8 sessions, 30 min)

A second committee batch on m0r0 L3 in the OPINE environment,
`committee_opine2_devin`, same seeds. Record R37: vote, AUROC and unanimous
error for batch 1 and batch 2. This bounds the caveat under every claim in
this project.

### P5. A second synthesizer on Modal (0 Devin sessions)

The feasibility job writes to `artifacts/tr87/L1_f60/modal_feasibility.log`
(gpt-oss-120b woke at 21:55; Qwen was still cold). If a model admits
programs in frame-out mode on tr87 L1 and ka59 L2, run one committee of 8
on ka59 and on ar25 L3 through the fan-out:
`uv run modal run -m committee.modal_app --game ka59 --level 2 --train-frac 0.4 --runs 8 --seeded --frame-out --condition committee_opine_gptoss`
and record R38, the calibration result with a second synthesizer. If
neither model admits, record the negative in one line under R33.

### After the experiments (no sessions)

1. RESULTS entries R34 to R38, each with metric, runs, split, baseline,
   command and commit. Every number in README.md and on the page comes from
   there.
2. README.md: the results section gets the OPINE-environment table and the
   loop result in that environment. The credits already list arc-agi.
3. CLAUDE.md track table: the "no live play" gap is closed (R25, R26);
   update that row and the known-gaps line.
4. Report page: one section for R34 to R38. Read it before publishing;
   another session edits its ONC part.
5. HANDOFF.md: replace this plan with what was run.
6. The user merges `jdv` into `main`; run the demo from `main`; push.
7. Video (90 s: demo sections 2, 3, 6 and 7, then the 75-move live run),
   the project description, the submission.

## Decisions to keep (do not relitigate)

- Independent seeded synthesis builds the committee; equal weights; every
  added component stays opt-in until an A/B shows a gain. The
  mechanism-level counterexample is the default statement since R31.
- OPINE's own environment is `frame_out` (frame in, frame out, admitted by
  frame equality). `frame` is the half-step and gave eleven identical
  programs at 0.82 on ka59; do not use it for comparisons.
- Online conformal prediction is the calibration route; the selective score
  (+1 committed right, -1 committed wrong, 0 abstain or set) is the one
  number that prices abstention. gamma 0.05 stays the code default; 0.3 is
  the recorded leave-one-out alternative.
- No Claude account for synthesis: Devin, or the open models on Modal.
- Terrain objects (R32) are superseded by the frame modes; keep the
  completeness measurement, do not develop them further.
- Do not tune on test data: thresholds come from train-side quantities; the
  held-out set is never shown to anything that selects.
- Static terrain, probe policies and prompt-level prohibitions were
  measured and dropped (R28, R31, R32). A verifier-side admission term on
  literal density is the recorded next lever against fitted geometry, not
  applied to any reported number.

## Credentials and services

- `.env.committee` (gitignored) holds `DEVIN_API_KEY`, `VLLM_API_KEY`,
  `OPENAI_BASE_URL` (Qwen server), `OPENAI_BASE_URL_GPTOSS` and
  `ARC_API_KEY`; `synth_api.load_env_file` loads it.
- Devin: about 250 sessions used today. A suspended session shows
  `status: suspended` in meta and returns the stub. Sessions can stall in
  `working`; `--timeout` (default 900 s) bounds them.
- Modal: apps `committee-llm` (Qwen3-Coder-30B FP8) and
  `committee-llm-gptoss` (gpt-oss-120b) scale to zero after 15 min and
  answer 503 while cold. R2: neither reached exact replay in objects mode.
- ARC-AGI-3 games are downloaded to `cache/arc_games` (gitignored) with the
  key; `committee.live` runs the local engine; game source is never read.

## Artifact layout

`artifacts/<game>/L<level>_f<frac>[_probe<n>][_live<through>]/<condition>/run<k>/`
with `program.py` and `meta.json`; `test_preds.json` and `test_objs.json`
are gitignored and rebuilt on demand. Conditions in use: `baseline_devin`,
`committee_devin`, `unseeded_devin`, `active_devin`, `cegis_devin`,
`mech_devin`, `mechgeo_devin`, `passive_devin`, `terrain_devin`,
`baseline_frame_devin`, `committee_frame_devin`, `baseline_frameout_devin`,
`committee_frameout_devin`, `live_devin`. Reports: `cegis_report.json` in
each probe directory, `artifacts/calibration*.json`,
`artifacts/abstention_score.json`, `artifacts/wrapper_gamma.json`,
`artifacts/probe_policy.json`, `artifacts/repair_literals.json`,
`artifacts/ar25/live/*.json`.

## Known traps

- The anti-network filter once matched the word `socket` (a ka59 object
  type); it matches imports only now. The port's workspace checker applies
  the same filter.
- `cegis --report` scans `*_probe<n>` directories, skips names such as
  `probe4_live70`, and compares conditions by name (`--conditions a,b`).
- The live harness keeps one process per member for the scored trajectory;
  hidden state matters (round 1, seed 1 moved from refutation at move 29 to
  64 when that was fixed).
- Other sessions edit RESULTS.md, NOTES.md, README.md, CLAUDE.md and
  pyproject.toml: use Edit, never Write, on them, and stage only your own
  lines.
