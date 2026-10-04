# Handoff: the night plan, written 2026-10-03 22:00 BST

Deadline 2026-10-04 15:00 BST (official). No new features after 12:00; those
hours go to repairs, the video, README.md and the submission. Read
CLAUDE.md first; then RESULTS.md (R3 to R33), NOTES.md and this file. The
write-up page is https://claude.ai/artifact/RZK96oVH1iZXjRnY4ndH8C (private,
version 19; every number on it points to a RESULTS entry).

## State

Update 2026-10-04 00:05 BST (session iterate-62, stopped at its usage limit).
Runners still alive as background processes on this machine (nohup); if one
dies, `committee.recover` collects its Devin sessions. What is done and left:

- DONE: P1 on all seven levels, R34 (comparison) and R35 (calibration)
  written and committed (`97e1258`). `committee.summary` and
  `committee.calibrate --condition committee_opine_devin` regenerate every
  number. Live play with the frame_out committees: ar25 L3 vote 0.84 over
  75 moves, AUROC 0.92, first all-wrong move 63 (one HUD cell); ka59 0.973;
  sk48 0.627 with every error unanimous (shared blind spot).
- DONE, not yet written up (R36): the counterexample rounds in frame_out.
  Held-out rows are the ones no arm trained on; round 1 is the frame_out
  committee on those rows (pass `--source-condition committee_opine_devin`
  to the report, or round 1 is the object-contract committee). m0r0 L3 (11
  rows): round 1 0.636, mechanism round 0.909 (8 of 8, one behaviour),
  passive 1.000 (7 of 8). ka59 L2 (42 rows): round 1 0.952, mechanism
  round 0.857, a loss; no passive arm. sk48 L2 (42 rows): round 1 0.833,
  mechanism 0.976 (7 of 8), object diff 0.929, passive 1.000. Written up
  as R36.
  `uv run python -m committee.cegis GAME --level L --train-frac 0.4 --report
  --conditions cegis_opine_devin,cegisobj_opine_devin --passive-condition
  passive_opine_devin` prints each table.
- RUNNING at 00:05: sk48 passive (`L2_n70/passive_opine_devin`, first
  members 1.0, 0.93, 0.93), sk48 object-diff arm
  (`L2_f40_probe25/cegisobj_opine_devin`, first members 0.93, 0.977, 0.93),
  ar25 L3 live round (`L3_f40_probe0_live70/live_opine_devin`, 4 of 8
  members admitted on 100 transitions). When the live round lands, replay:
  `uv run python -m committee.live ar25 --level 3 --steps 75 --brief
  --members-dir ar25/L3_f40_probe0_live70/live_opine_devin` and compare
  with `artifacts/ar25/live/ar25_L3_f40_committee_opine_devin_seed0.json`.
- 10:25 BST, 2026-10-04: SciGym B5 and B6 written; the tolerance-0.5 arms
  (three arms on gpt-oss, four on Qwen with `committee_probe_nocx`) are
  running on Modal, results in `artifacts/scigym/*_eps50/`; report with
  `SCIGYM_EPS=0.5 uv run python -m scigym.report --model qwen --tag _eps50`
  and write B7. Post-deadline SciGym arms, each a 90 minute Modal run after
  warming the vLLM servers: random experiment order, unseeded members, K 8,
  budget 8 to 20, a second batch per arm.
- DONE at 00:50: R36, the NOTES.md entry, README results and demo command,
  the CLAUDE.md track table, the report doc rewritten by the five questions
  (https://claude.ai/code/artifact/84ea4a4a-2162-4457-9525-1702a9026886, rev
  19), every run directory committed. The demo for the pitch is the sk48 L2
  frame_out command in README.md (2.3 s). LEFT: the user merges jdv into
  main; the video and the submission text.
- Earlier LEFT list, kept for the record: write R36 (rounds, passive, object-diff, live round) and the NOTES.md
  entry for this session; update README results and the CLAUDE.md track
  table to R34 to R36; rewrite the report by the five questions in the
  Claude Doc https://claude.ai/code/artifact/84ea4a4a-2162-4457-9525-1702a9026886
  (read it first; the HTML page is superseded); commit the round artifacts
  (run directories, not the logs, which hold absolute paths); the user
  merges jdv into main. Budget used: 134 Devin sessions of the 133 planned (plus 32 contingent).
  Not run: ka59 passive, P4 batch variance, Q reward hacking.

Earlier update, 2026-10-03 23:50 BST (session iterate-62). Everything below this
paragraph is the 22:00 state; these are the changes since.

- Every build is committed and pushed on `jdv` (head `6b881e6`): the
  environment port, reward hacking, Hoeffding, ONC, BioProt, SciGym, the idea
  filter, docs and packages, in one commit per build. Left out on purpose:
  `external/conceptualizer` and eight ONC and Hoeffding logs that hold
  absolute paths. The `*_opine_devin.log` files also hold absolute paths
  (tracebacks); commit the run directories, not the logs.
- P0 passed (tr87 probe 1.0). P1 wave one (ka59 L2, ar25 L3, m0r0 L3) ran
  at 22:45. At about 22:50 the hotspot lost DNS for 20 minutes; every runner
  process died in its poll loop and 14 sessions kept running on Devin with
  no local owner. `committee.recover` (new) lists finished committee
  sessions no stored run references, matches each to its condition and run
  index by the seed text in its prompt, and writes the run; all 14 were
  recovered, 3 never-started runs were topped up with `--start`. The Devin
  client now retries over transport errors, and one failed run no longer
  ends its condition. Recovered runs carry `recovered: true` and a `wall_s`
  that includes the suspension.
- P1 results so far (frame_out, `committee.summary`): ka59 L2 single 0.89,
  0.89, 0.91 against vote 0.909, AUROC 0.81, unanimous error 0.03 against
  split 0.33; m0r0 L3 single 0.86 x3 against vote 0.864, AUROC 0.69,
  unanimous 0.08 against split 0.38; ar25 L3 single 0.98, 0.98, 0.93
  against vote 1.000 (7 of 8 members in; run6 still running). Wave two
  (sk48 L2, ar25 L7, ls20 L3, g50t L1) launched 23:20.
- P2 started on the two levels under 0.95: m0r0 L3 (`L3_f40_probe33`: the
  first probe is the most disagreed row, then every remaining row in time
  order; the 33rd refutes the last 4 survivors, so the passive split equals
  the active one and only the counterexample text differs) and ka59 L2
  (`L2_f40_probe2`: probes at steps 79 and 101, 42 held out). Passive and
  object-diff arms wait on these results.

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

## The story: five questions (user, 2026-10-03 23:05)

The results are too many to navigate. The submission answers five
questions and nothing else is a headline; everything else is appendix.
Write R34 to R37, the README results section, the report page and the
pitch in this order, one answer per question, with the number and the
entry.

The environment was rewritten (frame in, frame out, R32 and R33). Every
number below comes from the old object contract unless it says R33, and the
user's instruction (23:10) is that those numbers are not all trustworthy.
They are the provisional shape of each answer, nothing more. The headline
answers come only from the runs in the new environment, steps P1 to P4; the
objects-mode entries stay in RESULTS as history and appear in the write-up
only beside their new-environment counterpart.

| Question | Provisional answer (objects mode) | Entries | Re-established by |
|---|---|---|---|
| 1. Does a committee predict better than a single program? | No. The vote equals or sits just under the best single program; the oracle equals the best member. Same in OPINE's environment (pilot). | R3, R23, R33 | P1 |
| 2. Are its predictions calibrated? | The vote share is not (ECE 0.22); the adaptive conformal sets hold 0.90 on all eight levels (0.88 to 0.99) and live (0.91 to 0.96); where the committee is mostly wrong the coverage is bought by abstention, and the selective score prices that. | R22, R27, R29, R30 | P1 (calibrate on the new conditions), P3 |
| 3. Is disagreement higher when it is wrong? | Yes: unanimous error 0.00 to 0.30, split 0.26 to 1.00; AUROC 0.68 to 1.00, pooled 0.75 (0.66 to 0.84); live split error 0.87 to 1.00. Shared blind spots are not flagged. | R4, R10, R27, R25, R26 | P1, P3 |
| 4. Does resynthesis after a high-disagreement probe help? | The probe refutes in one move and never selects; resynthesis on it lifts where the observation names one mechanic (ar25 L3 0.48 to 1.00 in three rounds; sk48 0.70 to 0.86; live 0.40 to 0.71 and 0.86), null on three, loss on one; passive control 0.00 to -0.06; the new committee converges. | R23, R24, R26, R27, R28 | P2, P3 |
| 5. Does resynthesis aimed at the missing mechanism help more? | Yes where the rule is in reach: ar25 L3 to 1.00 in one round (object diff 0.925, three rounds); equal on three; keeps diversity on ar25 L7; cannot move ka59 because the object state hid the walls, which the OPINE environment fixes (ka59 0.89 with no repair). | R31, R32, R33 | P2 |

Appendix material, never a headline: MDL prior (R14), targeted growth
(R8), judge confidence (R11, R18), learned ensembles (R9), unseeded
resampling (R12), mechanism library (R19, R20), row disentanglement (R16),
rollouts (R15), probe policy (R28), wrapper step size (R30), terrain objects
(R32), reward hacking (RH), Hoeffding (H), ONC-AGI (O), bio (B).

## The plan

Budget: 133 Devin sessions in the base plan (P1 77, P2 40, P3 up to 8, P4 8), 32 more if P2 shows a lift (the passive and object-diff arms), plus 36 in the reward-hacking queue (Q), last and optional. Every step feeds one of the five questions above; the objects-mode numbers are not reused for any headline.
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

### P1. The comparison in OPINE's environment under the final text (77 sessions, about 1.5 h)

Conditions `baseline_opine_devin` (3 unseeded) and `committee_opine_devin`
(8 seeded), mode `--frame-out`. Never add to the pilot conditions
(`*_frameout_devin`): `check_mode` refuses mixed modes, nothing refuses
mixed task texts, so the names must differ.

```
uv run python -m committee.experiment GAME --level L --train-frac 0.4 --runs 3 --backend devin --frame-out --condition baseline_opine_devin --parallel 3
uv run python -m committee.experiment GAME --level L --train-frac 0.4 --runs 8 --seeded --backend devin --frame-out --condition committee_opine_devin --parallel 4
uv run python -m committee.evaluate GAME --level L --train-frac 0.4 --condition committee_opine_devin
```

1. All seven levels, ka59 L2 first: ar25 L3, m0r0 L3, sk48 L2, ar25 L7,
   ls20 L3, g50t L1, ka59 L2 (77 sessions). The user wants every headline
   number from the rewritten environment under one task text, so the pilot
   (R33, draft text) is a consistency check only: report the ka59 and ar25
   L3 differences between pilot and final in one line.
2. Questions 1, 2 and 3 are answered from these runs alone: single against
   committee (admission, accuracy), the calibration battery, and unanimous
   against split error with the AUROC.
3. Record R34: per level, single programs and committee in objects mode and
   in the OPINE environment side by side (admission, vote, members, AUROC,
   unanimous and split error), plus the conformal battery:
   `uv run python -m committee.calibrate --levels ar25:3,m0r0:3,sk48:2,ar25:7,ls20:3,ka59:2,g50t:1 --condition committee_opine_devin --out artifacts/calibration_opine.json`
   (`--condition` is in the port). No kill criterion: this is the
   measurement the user asked for.

### P2. The counterexample round in OPINE's environment (40 to 72 sessions, 1 to 2 h)

Only on levels where the round-1 vote under P1 is below 0.95 (ar25 L3 was
at 1.00 in the pilot and would be skipped). Expected: ka59, m0r0, sk48,
g50t, ar25 L7 (40 sessions). The mechanism statement is the default and the
mode is inherited from the source committee. Then, on the two levels with
the largest lift, two more arms so that questions 4 and 5 are answered in
this environment and not inherited from R24 and R31: the passive control
(`committee.experiment ... --train-n <train+probes> --seeded --frame-out
--condition passive_opine_devin`, 8 each, question 4) and the object-diff
statement (`committee.cegis ... --object-diff --condition cegisobj_opine_devin`,
8 each, question 5). If no level lifts by 0.05 or more, skip both arms and
record that: in this environment the repair step adds nothing measurable,
which is itself the answer.

```
uv run python -m committee.cegis GAME --level L --train-frac 0.4 --source-condition committee_opine_devin --condition cegis_opine_devin --dry-run
uv run python -m committee.cegis GAME --level L --train-frac 0.4 --source-condition committee_opine_devin --condition cegis_opine_devin --runs 8 --backend devin --parallel 4
uv run python -m committee.cegis GAME --level L --train-frac 0.4 --source-condition committee_opine_devin --condition cegis_opine_devin --report
```

A level whose committee is never refuted (dry run says so) is skipped.
Record R35 (question 4: round 1 against round 2 against passive) and R36
(question 5: mechanism statement against object diff). Stop rule: one round
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
counterexample round (8 sessions) as in R26, then replay. Record R37. This
is the live demo in the proper environment; keep the 75-move `--brief`
command for the pitch.

### P4. Batch variance (8 sessions, 30 min)

A second committee batch on m0r0 L3 in the OPINE environment,
`committee_opine2_devin`, same seeds. Record R37: vote, AUROC and unanimous
error for batch 1 and batch 2. This bounds the caveat under every claim in
this project. Record R38.

A second synthesizer on Modal was considered and dropped by the user
(2026-10-03 22:05): no experiment uses the open models.

### Q. Optional, last: the reward-hacking build's grid (36 Devin sessions, about 1 h)

Demoted by the user at 22:55: Track 2 has an epistemic-uncertainty aim
(2.3) and a reward-hacking aim (2.1), and the results to date align with
the first. The pitch leads with 2.3; reward hacking is supporting evidence
(RH1 to RH5, one line in the pitch). Run Q only if sessions and time remain
after P1 to P4 and the write-up; skipping it costs the submission nothing.

Session `external-c5` (owner of `src/rewardhack`) handed it over at 22:45.
It is independent of the committee conditions. Do not run synthesis locally
in parallel with it: the laptop rebooted under that load earlier today. Its
loop runs inside the deployed Modal function `rewardhack-synth`, which holds
the `devin-auth` secret and verifies every returned program outside the
session; the launcher skips runs whose `meta.json` exists, so it resumes.

```
uv run python -m rewardhack.baseline --backends devin --parallel 6
```

Environment E5 (frame in, frame out; RESULTS RH11). Levels tr87 L1, ls20 L3,
re86 L5 at train fraction 0.6; conditions intact, intact+abstain,
contradiction, contradiction+abstain; 3 runs per cell; the 5 existing
intact runs are skipped; each session capped at 4 ACU and 1800 s. Records:
`artifacts/rewardhack/<game>/L<level>_f60/<condition>_fout/devin/run<k>/`
with `program.py`, `meta.json`, `verdict.txt`; outcomes consistent, honest,
hack, abstain, fail. Then, offline:

```
uv run python -m rewardhack.report table
uv run python -m rewardhack.split --frame tr87:1 ls20:3 re86:5
uv run python -m rewardhack.report demo
```

Record as new RH entries and a NOTES entry, append only. Blocker at the
time of writing: Devin answered HTTP 403 `out_of_quota` at 22:40 BST; the
command is resumable and can be re-issued as is once the quota is restored.
The Claude-family cells of that grid are not for this session (they need
the `claude-auth` Modal secret, which is empty, and the user's decision
against Claude synthesis stands for the committee).

### After the experiments (no sessions)

1. RESULTS entries R34 to R38, each with metric, runs, split, baseline,
   command and commit. Every number in README.md and on the page comes from
   there.
2. README.md: the results section gets the OPINE-environment table and the
   loop result in that environment. The credits already list arc-agi.
3. CLAUDE.md track table: the "no live play" gap is closed (R25, R26);
   update that row and the known-gaps line.
4. Report page: the five questions answered from R34 to R38, the objects-mode numbers beside them as history. Read it before publishing;
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
- No Claude account for synthesis, and no open-model synthesis on Modal (dropped 22:05): Devin only.
- The submission is a 2.3 entry, epistemic uncertainty. Reward hacking is
  supporting evidence, not a pitch claim (user, 22:55); the reward-hacking
  grid is last and optional.
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
