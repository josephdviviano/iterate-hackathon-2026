# Design

## Problem

OPINE-World learns an ARC-AGI-3 game as one Python program, admitted when it replays every observed transition. Its uncertainty measure, the ontology error η, is a Dirichlet posterior on effect counts per (type τ, action a, local context μ) row. On rows with no observations η is uninformative, so the agent cannot say what it does not know about states it has not seen.

## Idea

Replace the single synthesized world model with a committee of exact-replay-consistent programs, weighted by simplicity. Committee disagreement on unobserved rows is both the uncertainty report and the exploration target.

Track 2.3. The user is the exploration agent, and the human operator behind it, who decides which action to spend next.

## Data flow

```
replay bundle (frames, actions, level steps, released extractor)
      |
      v
   loader ----------> object buffer D = [(s, a, s')]  -- temporal split --> D_train | D_test
      |                                                                      |
      v                                                                      |
 effect matrix (tau, a, mu) + eta_counts                                     |
      |                                                                      |
      v                                                                      |
 seed hypotheses (K) ---> synthesizer (claude -p) ---> K programs             |
                                                           |                 |
                                                           v                 |
                                      verifier: sandboxed exact replay on D_train
                                                           |
                                                           v
                                      consistent set ---> MDL weights ---> committee
                                                                              |
                     +--------------------------------------------------------+
                     v
   predictive distribution on (s, a) in D_test
     - accuracy: simplest, weighted vote, single-program baseline
     - calibration: vote entropy vs misprediction
     - eta_committee on rows eta_counts cannot score
     - active-learning simulator: probe max-disagreement first
                     |
                     v
   RESULTS.md, cached artifacts, 90-second demo
```

## Components

| Module | Responsibility | Reimplemented from |
|---|---|---|
| `loader` | Rebuild 64x64 frames from the bundle deltas. Run the game's released extractor in a subprocess to produce object-level states. Emit `D` and the level split. | OPINE bundle schema in `docs/build_site_data.py` (read only) |
| `matrix` | Key each object transition by (τ, a, μ). Effect signature from changed fields. Dirichlet posterior and η from counts. | OPINE §03 and §04 of the project page |
| `seeds` | Produce K distinct seed hypotheses from `D_train`: simplest rule per type, one per candidate split of the worst mixed row, hidden-state accumulator, interaction-rule variant. | Our design. Guided diversity, after the GRAM ablation that unguided noise collapses |
| `synth` | One `claude -p` call per seed. Prompt holds `D_train` in object form, the seed, and the engine contract. Output is one program with `transition_function(state, action)`. | OPINE engine contract, rewritten |
| `verify` | Run each program on `D_train` in a subprocess with a timeout and no network. Keep programs with exact replay. Reject any program that reads the buffer file. | OPINE exact-replay rule |
| `committee` | Weight w_i ∝ exp(−λ · L_i), L_i = gzip length of the AST-normalized source. Predictive distribution over next states as a weighted vote. Disagreement = entropy of the vote. | Our design. MDL prior |
| `evaluate` | Held-out accuracy, calibration (AUROC, reliability curve), coverage of distinct behaviours, active-learning curves against random and η-priority orders. | Coverage after GRAM §4.2 |
| `demo` | One command. Replays cached results for one game in 90 seconds. No live LLM call. | |
| `modal_app` | Fan-out of synthesis runs on Modal: one container per member, verifier in the same container, artifacts written locally in the shared layout. | |
| `modal_llm` | vLLM server for an open-weights coder model on a Modal H100, OpenAI-compatible, behind an API key. | Modal vLLM example |
| `synth_api` | Repair loop over any OpenAI-compatible model: task and transitions in, program out, checker report back in, up to N rounds. The default synthesizer. | CEGIS (Solar-Lezama et al., 2006) |
| `synth_devin` | One Devin session per member. Attachments carry the transitions and checker; the program comes back as structured output. | Devin API v1 |
| `experiment`, `explore` | Local orchestration with the same artifact layout; simulated exploration by disagreement against random and count-priority orders. | Query by committee (Seung, Opper and Sompolinsky, 1992) |

## Evaluation protocol

| Item | Choice |
|---|---|
| Games | Development on ft09, lp85, tr87. Then m0r0, ar25. |
| Split | Temporal. Train = first fraction of a level's transitions. Test = the rest of that level, or all of the next level (cross-level split). RESET and level-closing transitions are removed from both sides because their outcome is a new layout, not a function of the state. Test never enters a prompt. |
| Baseline | One program from one `claude -p` call with no seed. Same prompt otherwise. |
| Committee | K = 8 seeds. Programs that fail exact replay are dropped and counted. |
| Metrics | Next-state accuracy. AUROC of disagreement against misprediction. Coverage. Transitions to committee collapse. |
| Runs | At least 3 synthesis runs per game per condition. Report mean and spread. |
| Record | Every number goes in RESULTS.md with metric, runs, split, baseline, command and commit. |

## Fixed and out of scope

- The released per-game extractor is frozen input data. We study the transition layer only. Credited in README as data, not code.
- No live games. No ARC API. No OPINE-World harness.
- No GRAM code. No library learning. See NOTES.md for the reasons.
- Phase 4, context splits from program predicates, only after Phases 1 to 3 are recorded.

## Planned layout

```
src/committee/   loader.py matrix.py seeds.py synth.py verify.py committee.py evaluate.py demo.py
prompts/         synthesis prompt and seed templates
artifacts/       cached programs, verdicts, metrics per game (git-tracked, small)
RESULTS.md       every reported number
NOTES.md         log of done, deferred and abandoned work
```

## Phases

| Phase | Goal | Deliverable |
|---|---|---|
| 1 | Offline data. Load a replay bundle, run the released extractor, build the object buffer and the (τ, a, μ) effect matrix with η counts. Temporal train/test split. | `loader`, `matrix`, tests |
| 2 | Committee. K seeded `claude -p` calls, sandboxed exact-replay verifier, MDL weights, weighted vote and disagreement. | `seeds`, `synth`, `verify`, `committee` |
| 3 | Evaluation and demo. Held-out accuracy vs single-program baseline, calibration AUROC, coverage, active-learning curves on ft09, lp85, tr87. Numbers in RESULTS.md. 90-second cached demo. | `evaluate`, `demo`, RESULTS.md |
| 4 | Deferred. Context splits from program predicates. | |

"After Phase 3 is recorded" means the first numbers exist in RESULTS.md and the demo runs from main.

## Track 2.3 requirements

| Requirement | Evidence |
|---|---|
| Flag what they do not know | Committee disagreement on unobserved (τ, a, μ) rows, where η is uninformative |
| Calibrated uncertainty | AUROC and reliability curve of vote entropy against misprediction |
| Falsification | Exact replay drops inconsistent programs. The temporal held-out split falsifies survivors |
| Reward design | Disagreement is the exploration reward. MDL weight is the prior |
| Know when reward hacking (Track 2 header) | `src/rewardhack`: held-out gap, literal-mass and MDL-ratio detectors on every program, abstain output under an injected contradiction. See research/reward_hacking_review.md §6 |

## Parallel builds

Three builds run in parallel in this repository and must not edit each other's files:

| Build | Owns |
|---|---|
| Program committee (core) | `src/committee/`, `tests/test_loader.py`, `tests/test_committee.py`, `tests/test_explore.py`, `artifacts/<game>/`, `cache/` |
| Reward hacking | `src/rewardhack/`, `tests/test_rewardhack.py`, `artifacts/rewardhack/`, `research/reward_hacking_review.md` |
| Uncertainty bound estimation | (owner to fill in) |
| OPINE-World variant with the ARC harness | (owner to fill in) |

Shared files (`pyproject.toml`, `README.md`, `NOTES.md`, `RESULTS.md`, `DESIGN_DOC.md`) take append-only edits.
