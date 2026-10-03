# Program committees for epistemic world models

A world-model agent that knows what it does not know. Instead of one
synthesized program per game, it keeps a committee of programs that all replay
the observed transitions exactly, lets them vote with equal weight, and reports
where they disagree. Disagreement is the uncertainty it flags and the probe it asks
for next.

Track 2.3, epistemological agents. See DESIGN_DOC.md for the architecture and
RESULTS.md for every reported number.

## Results

ar25 level 3, trained on the first 40% of the level, tested on the rest
(RESULTS.md R3 and R4, backend Devin):

| | Single program | Committee of 8 |
|---|---|---|
| Replays train exactly | 3 of 3 | 8 of 8 |
| Held-out accuracy | 0.64, 0.43, 0.48 | members mean 0.48, vote 0.48 |
| Error when the committee is unanimous | | 0.30 (27 transitions) |
| Error when the committee splits | | 0.83 to 1.00 (17 transitions) |
| AUROC, disagreement against error | | 0.78 |
| Held-out effect rows never seen in train | η undefined on 51 of 90 | entropy on all 90 |
| Probes until a counterexample falsifies every hypothesis | OPINE-World count priority: 7 | disagreement: 1 (random: 2.7) |

Across six splits (R4 to R7, R17, R21), 8 seeded programs each:

| Split | Unanimous transitions, error | Split transitions, error | Probes to falsify: disagreement / count priority / random |
|---|---|---|---|
| ar25 L3 | 27, 0.30 | 17, 0.83 to 1.00 | 1 / 7 / 2.7 |
| m0r0 L3 | 38, 0.16 | 6, 0.67 | 1 / 10 / 4.8 |
| sk48 L2 | 41, 0.00 | 27, 0.59 | |
| ar25 L7 | 25, 0.00 | 40, 0.95 | |
| tr87 L1 to L2 | 28, 0.00 | 0 | |
| ft09 L5 | 31, 0.00 | 0 | |

When the committee agrees it is mostly right. When it splits it is mostly
wrong. On the two levels with a split, probing where it disagrees finds a
counterexample that falsifies every hypothesis in one move.

Against baselines on the two informative levels (RESULTS.md R9 to R12):

| Baseline | Outcome |
|---|---|
| Learned dynamics ensembles (bagged trees, MLP deep ensemble) on per-object features | 0.00 to 0.32 accuracy vs the committee's 0.48 and 0.77 from 25 to 30 transitions; disagreement uninformative (AUROC 0.47 to 0.48) |
| Verbalized confidence of a judge model (gpt-oss-120b) over the heaviest program | Pooled AUROC 0.71 vs committee 0.75, not separable; judge strong on ar25, uninformative on m0r0; the two combine to 0.85. Judge needs 44 calls per level, the committee none |
| Unseeded resampling, same verifier (self-consistency) | Same calibration; the seed hypotheses raise admission (8 of 8 vs 6 of 8) and distinct behaviours (4 vs 2) on m0r0 |
| Committee size K = 2, 4, 8 | AUROC rises with K on both levels; pooled over 88 transitions 0.75, 95% CI 0.66 to 0.84 |

Closest prior work and exact differences: `research/baseline_review.md`.

Calibration (RESULTS.md R22): the raw vote share is not a probability (ECE
0.22 pooled), but adaptive conformal sets built online from the committee's
vote shares hold a 90% coverage target on all four informative levels
(0.88 to 0.99). Where the committee is mostly wrong the sets abstain, which
is the honest answer; where it is mostly right they return a single state
66% to 77% of the time at 0.88 to 0.93 accuracy.

## Demo

```
uv sync
uv run python -m committee.demo
```

## Commands

```
uv run python -m committee.loader ft09                       # build a game's transition buffer
uv run python -m committee.matrix tr87 --level 1             # effect matrix and ontology error
uv run python -m committee.experiment tr87 --level 1 --runs 1               # baseline: one program
uv run python -m committee.experiment tr87 --level 1 --runs 8 --seeded \
    --condition committee --parallel 4                      # committee: seeded programs
uv run python -m committee.evaluate tr87 --level 1 --condition committee
uv run python -m committee.active ar25 --level 3 --backend devin   # grow a committee where it disagrees
uv run python -m committee.evaluate ar25 --level 3 --train-frac 0.4 --condition active_devin --curve
uv run python -m committee.calibrate                          # ECE, Brier, conformal sets (R22)
uv run python -m committee.selection                          # oracle headroom, probes as selection (R23)
uv run python -m committee.cegis ar25 --level 3 --dry-run     # the first falsifying probe and the round 2 seed
uv run python -m committee.cegis ar25 --level 3 --runs 8 --backend devin --parallel 4   # round 2 committee
uv run python -m committee.experiment ar25 --level 3 --train-n 30 --runs 8 --seeded \
    --backend devin --condition passive_devin                 # passive control: next transition in time
uv run python -m committee.cegis ar25 --level 3 --report      # round 1 vs round 2 vs passive
uv run pytest
```

Synthesis backends, selected with `--backend`:

| Backend | What it is | Credential |
|---|---|---|
| `api` (default) | Repair loop over an OpenAI-compatible model. We serve Qwen3-Coder-30B-A3B-Instruct-FP8 with vLLM on a Modal H100 (`committee.modal_llm`). | Modal Secret `vllm-auth`: `VLLM_API_KEY`, `OPENAI_BASE_URL` |
| `devin` | One Devin session per committee member; task and checker as attachments, program returned as structured output. | Modal Secret `devin-auth`: `DEVIN_API_KEY` |
| `claude` | Claude Code CLI agent in an isolated workspace. Kept as an option, not used for reported numbers. | Modal Secret `claude-auth` |

```
uv run modal deploy -m committee.modal_llm                     # model server
uv run modal run -m committee.modal_app --game ar25 --level 3 --train-frac 0.4 --runs 3
uv run modal run -m committee.modal_app --game ar25 --level 3 --train-frac 0.4 \
    --runs 8 --seeded --condition committee                   # one container per member
```

The local path (`committee.experiment`) is the same code without the fan-out.
Local runs read credentials from `.env.committee` (gitignored).

## Credits

- Data: the 25 ARC-AGI-3 replay bundles and per-game object extractors
  released by OPINE-World (Courtis, Li and Sanner, 2026, arXiv:2607.01531),
  read from the `external/opine-world` submodule. The extractors are used as
  frozen input data. No code is copied from that repository.
- Pre-trained model: Qwen3-Coder-30B-A3B-Instruct-FP8 (Qwen team, Alibaba, Apache-2.0),
  served with vLLM (Apache-2.0) on Modal.
- APIs: Devin API (Cognition) as a synthesis backend; Claude Code CLI (Anthropic) as an
  optional backend.
- Libraries: numpy, matplotlib, openai (client), httpx, modal, pytest, uv.
- Ideas: ontology error and effect rows from OPINE-World; parallel sampling and
  coverage from GRAM (Baek et al., 2026, arXiv:2605.19376); query by committee
  (Seung, Opper and Sompolinsky, 1992); minimum description length (Rissanen, 1978).

## Reward hacking

The synthesizer is scored by exact replay on the train transitions it can
see, so it can tabulate them instead of modelling the mechanics.
`rewardhack` measures that propensity. Every program gets a literal-mass,
MDL-ratio, held-out-gap and order-dependence score. A contradictory
transition injected into the train set makes exact replay impossible, so a
full pass is a hack by construction; an ABSTAIN channel lets the agent say so
instead. See research/reward_hacking_review.md.

```
uv run python -m rewardhack.report score                                # hack features of every stored program
uv run python -m rewardhack.experiment tr87 --level 1 --runs 3 --contradiction --abstain --parallel 3
uv run python -m rewardhack.report summary                              # outcomes per condition
uv run modal deploy src/rewardhack/modal_app.py                         # open-weight synthesizer (vLLM on Modal)
uv run python -m rewardhack.experiment tr87 --level 1 --runs 3 --contradiction --abstain \
    --backend modal --model Qwen/Qwen2.5-Coder-7B-Instruct --max-turns 4   # chat loop, 4 checker rounds
```
