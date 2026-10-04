# Hackathon project

Do only work that increases the score or decreases the risk of disqualification.


## Project Tracking
NOTES.md should contain log entries with the following structure:

- [USERNAME] - SHORT TITLE - HASH
  A one line description of the conversation content:
  DONE:
    - Work item A
    - Work item B
  DEFERRED:
    - Work not done, and WHY.
  ABANDONED:
    - Work decisions we've decided to do previously but were abandoned, with reasoning (including evidence).


## Status

- Track: 2.3 Epistemological agents, selected 2026-10-03. See DESIGN_DOC.md.
- Deadline: 2026-10-04 15:00 BST (official, per the user at 09:40 BST).
- Repository: private. The papers in research/pdfs are open access. Do not raise licence or privacy caveats about them.

## External Resources

- `external/opine-world` - programmatic world modelling for ARC
- `conceptualizer` - half-finished research project on a distributional version of the tiny reasoning network architecture, which is based on GRAM.

## Rules

A rule violation or unfair play can disqualify the team. If a task does not obey a rule, stop and tell the user.

1. Write all code during the event. Do not copy code from earlier projects. The repository must have no commits from before the event.
2. Credit each open-source library, API and pre-trained model in README.md when you add it.
3. Do not train or tune on test data. Do not use defects in an evaluation to increase the score.

## Criteria

| Criterion (20 points each) | What we show |
|---|---|
| Technicality | A non-trivial method that operates end to end, measured against a baseline. |
| Creativity | One new idea that the pitch gives in one sentence. |
| Usefulness | A named user who makes a decision from the output. |
| Demo | An actual result, live, in 90 seconds, with no failure. |
| Track / sponsor alignment | Each requirement of the selected track or challenge, by name. |

## Work rules

1. First, make the simplest system that operates end to end. Then make one part better at a time.
2. The demo must operate from main at all times. One command starts it. Run this command before each merge into main.
3. Record each result in RESULTS.md with its metric, number of runs, data split, baseline, command and commit. Each number in the pitch and README.md must come from this file.
4. Set a time limit before each experiment. At the limit, stop and record the result, positive or negative.
5. Run GPU jobs and long jobs on Modal. Pre-compute each step that is too slow for the live demo.
6. Do not commit keys, tokens or absolute paths.
7. Other persons and sessions edit this repository in parallel. Do not run `git reset --hard`, force-push or other commands that can delete their work.
8. Do not start a new feature in the last 3 hours. Use those hours for repairs, the video, README.md and the submission.
9. All resources in `external/` should be used as read-only references. For any feature that draws on that work, we should reimplement it in our own codebase, do not copy the code verbatim.


## Submission and rounds

Submit these items before the deadline. Late teams are not eligible.

- The GitHub repository link. README.md gives the problem, approach, demo command, results and credits.
- A 2-minute demo video.
- A short project description.

| Round | Time | Format |
|---|---|---|
| 1, all teams | 5 min | Pitch 1:30: problem, approach, high-level scheme. Live demo 1:30. Q&A 2:00. |
| 2, 8 to 10 finalists, on stage | 3 min | Pitch 1:30. Demo 1:30. No questions. |

The organizers call finalists one at a time. Keep the demo prepared to start immediately.

## Track

**Track 2, Originator.** Agents that do science and know when they are wrong or
reward hacking. The track also asks for environments, rewards and benchmarks
for bio agents. Sub-tracks, in the organizers' words:

1. Benchmark science agents: evals that catch research agents reward hacking.
2. Lab hardware, automation and safety: safe, standard control of lab equipment, with testable results.
3. Epistemological agents: agents that flag what they do not know, with calibrated uncertainty, falsification and reward design.

We enter 2.3. The goal in one sentence: an agent that proposes hypotheses,
says where it does not know, tests there, accepts refutation, and cannot game
its own reward. Each work item names the requirement it serves. Each claim
points to an entry in RESULTS.md.

| Requirement | What we show | Evidence |
|---|---|---|
| Does science | A committee of programs: hypothesize (K seeded programs), predict, probe where they disagree, accept refutation, resynthesize on the probes. Rerun in OPINE-World's program contract on seven levels, with the loop closed on three levels and live. Hoeffding's problem is the second science task, certified in exact arithmetic. | R34, R36, H2 |
| Flags what it does not know | Disagreement on effect rows never seen in train, where count-based ontology error is undefined. Conformal sets that abstain. Unanimous error 0.05 against split error 0.34 over seven levels. | R34, R35 (R4, R13, R22 in the earlier object contract) |
| Calibrated uncertainty | AUROC of vote entropy against error, 0.76 pooled at K 8, rising to 0.92 at K 16 with unanimous error 0.007 (R37). In OPINE-World's program contract the vote share is close to a probability (ECE 0.04; 0.22 in the object contract). Adaptive conformal sets hold 0.90 coverage on seven levels (pooled 0.95). Self-reported confidence on Hoeffding scored by Brier. | R34, R35, R37, H3 |
| Falsification | Exact replay admits members. Probes drop refuted members until none survives. Round 2 resynthesizes on the probes: it lifts two of three levels and the live run, loses the one that was nearly right; the passive control lifts as much, so the gain is from the observations, not the counterexample wording. | R34, R36 |
| Reward design | The reward is exact replay with anti-tabulation checks (`committee.verify`). Disagreement is the exploration signal. An abstain channel in the task text. Exact certification, not float scores, on Hoeffding. | RH1, H2, H12 |
| Knows when it reward hacks (track header), supporting evidence only | Injected-contradiction 2x2 across Claude and open-weight synthesizers. Held-out gap, literal-mass and MDL-ratio detectors. One hack found in our own artifacts. Demoted by the user on 2026-10-03: one line in the pitch, no further sessions unless everything else is done. | RH1 to RH5 |
| Sub-track 1 | Not entered. The 2x2 is evidence for the header, not a 2.1 submission. | RH1, RH3, RH5 |
| Sub-track 2 | Not addressed. Out of scope. | |
| Bio agents | BioProt protocol generation as a selective-prediction benchmark: three open-weight models write 100 lab protocols, four uncertainty signals are scored by risk-coverage curves against random and oracle baselines, with an abstain channel and ECE and Brier. Verbalised confidence and self-consistency beat random on every model; sequence logprob does not; keeping the more confident half of the plans cuts the error by 21 to 29 percent; the R22 conformal wrapper over the same scores holds 0.90 coverage and commits on 22 to 57 percent of plans. The committee method transfers: disagreement predicts error at AUROC 0.69 to 0.77, and a cross-family committee of 15 is the strongest uncertainty on the benchmark with no elicitation call. | B1 to B4, R22 |

Known gaps, to state in the pitch and not hide: reward design is spread over
three builds and must be told as one design; live play is one game and one
level (R25, R26, R36); the bio evidence is one benchmark (B1 to B4); the
counterexample wording is not the active ingredient of the repair, the
observations are (R36); shared blind spots are silent to disagreement (sk48
L2, R34). The pitch
leads with 2.3, epistemic uncertainty; the results align with that aim, and
reward hacking is supporting evidence (user decision, 2026-10-03 22:55).

- **Modal challenge (optional).** Best use of Modal, in any part of the project. Deferred, see NOTES.md.
