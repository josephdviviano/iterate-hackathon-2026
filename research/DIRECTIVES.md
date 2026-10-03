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

## 2026-10-03, request 4

> You can only use a single GPU, the other GPU will be used by a competing GPT agent that you
> must compete against to hill climb the task

Effect: this team uses GPU 1 only. `research/allowed-devices` lists the permitted CUDA
indices and `research/sweep.py` refuses sweep files naming any other device. GPU 0 and the
competing agent's processes are never touched. Because the competitor shares the host's CPU,
power and PCIe, each sweep launch logs GPU utilisation and host load, and every comparison
uses an in-sweep control.

## 2026-10-03, request 5

> Ensure we've tested and hill climbed all strategies from the parallel task fully. This should
> include novel algorithm alternatives. modal is authed for timing tests

Effect: task T-018 and exploration X-003 audit every strategy from the CIFAR-10 record lineage
(airbench, airbench94_muon, airbench96_faster, hiverge, Fable/Fulcrum, hlb-CIFAR10, Page), the
competitor's programme and novel alternatives against the converged recipe;
`research/strategy-coverage.md` holds the matrix. Modal A100 runs need a payment method on the
Modal account (both A100 variants refuse to launch without one).
