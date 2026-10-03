# Hackathon project

Do only work that increases the score or decreases the risk of disqualification.

## Status

- Track: not selected. After selection, delete the other tracks below.
- Deadline: approximately 2026-10-04 11:00 BST. Replace with the official time.
- Repository: private. The papers in research/pdfs are open access. Do not raise licence or privacy caveats about them.

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

## Tracks (select one)

- **1 AI Automated Discovery of Algorithms.**
  - 1.1 Autoresearch framework: build your own. Show it on a problem from Tao et al.'s repository or another suitable benchmark. Judged on novelty and inventiveness, performance, interpretability and ease of use. Progress on an open maths problem is welcome but optional.
  - 1.2 Neural Network Speedrun: the fastest training time on the specified dataset and hardware wins. The system must meet the quality threshold on hidden test data.
- **2 Originator.** Agents that do science and know when they are incorrect or reward hacking. Also environments, rewards and benchmarks for bio agents.
  - 2.1 Benchmark science agents: evals that catch research agents that reward hack.
  - 2.2 Lab hardware, automation and safety: safe, standard control of lab equipment, with testable results.
  - 2.3 Epistemological agents: agents that flag what they do not know, with calibrated uncertainty, falsification and reward design.
- **3 Drug and protein design, by Serova.** Predict peptide-HLA complex stability with existing protein foundation models. Do better than NetMHCstabpan, a sequence-based, domain-specific, supervised model trained on peptide-HLA sequence pairs.
- **4 Materials manufacturing, by Polaron.** Compare electron microscopy images of a supplier's battery electrode batches against an approved baseline batch. Detect a meaningful change and quantify which material KPIs cause it. Give a verdict (accept, investigate or reject) with its uncertainty, and explain it to a materials expert. The supplied batches mix acceptable and defective variation. An unseen batch arrives approximately 8 hours into day one. Judged on KPI quality, accuracy on the new batch, interpretability, honest handling of uncertainty and real-world usability for a QC decision.
- **Modal challenge.** Best use of Modal, in any part of the project.
- **Devin challenge.** Devin reproduces a key result of a published scientific paper. Then go further than that result.
