# Human directives

Verbatim requests from the team lead (Bradley Segal) that govern this programme.

## 2026-10-03, request 1

> Review https://github.com/AIDDA-Institute/CIFAR-100-speedrun, appraise it using the
> writing-tools framework and present a proposal. Systematically check parallel approaches
> and mine what people have leveraged for this in the past

Outcome: `research/PROPOSAL.md` (appraisal, prior-art mining, hypotheses H1–H6, probes P0–P4).

## 2026-10-03, request 2

> Create a forked repo and tickets backlog using writing-tools research development and
> exploration and prepare for scaled exploration

Outcome: this fork (`BradSegal/CIFAR-100-speedrun`, branch `research`), the programme
workspace under `tickets/PROGRAMME`, and the exploration tooling under `research/`.

## 2026-10-03, request 3

> Use a single gpu and systematically hill climb and explore the speedrun using research-tools.
> Work until convergence

Interpretation recorded in the programme: exploration runs on one local GPU (GPU 1, one slot).
With no A100 available (blockers B-001/B-002), time is estimated by a local proxy (eager wall
time on the otherwise idle GPU) and every proxy-based selection carries a reopening condition
for A100 calibration. Convergence means no accessible lever lowers proxy time at matched
single-view accuracy beyond seed noise.
