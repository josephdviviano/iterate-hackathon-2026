# Notes

- [jdv] - Plan: program committee over OPINE-World - a718512 (uncommitted)
  Pushback on a distributional OPINE-World world model, then an agreed plan and a design doc.
  DONE:
    - Read the OPINE-World page and source, the GRAM paper, and the `conceptualizer` repo.
    - Verified the 25 replay bundles in `external/opine-world/docs/replay_data` hold every frame, action, level boundary and the final engine with its `extract_objects`. They are an offline dataset.
    - Selected Track 2.3 (epistemological agents). Removed the other tracks from CLAUDE.md.
    - Agreed the plan in DESIGN_DOC.md: committee of exact-replay-consistent programs, weighted by simplicity, with disagreement as the uncertainty and exploration signal.
  DEFERRED:
    - Phase 4, mixed-row disentanglement from program predicates. OPINE-World already enumerates and scores context features ξ by Δη. Programs add value only if their predicates leave that vocabulary. Do it after Phases 1 to 3 are recorded.
    - Modal. No token is set up and the jobs are API calls plus sandboxed replay, not GPU work. Revisit for the Modal challenge if time remains after Phase 3.
    - Second, more complex task after ARC. The committee, verifier and MDL weight are task-agnostic. Only the loader is ARC-specific.
    - Compression measurement for shared helpers across committee programs. Only as a measurement, only after Phase 3.
  ABANDONED:
    - GRAM as an implementation base. GRAM is an 11M-parameter neural recursive model trained by amortized variational inference. ARC-AGI-1 took 960 GPU-hours (GRAM Table 7). `conceptualizer` implements only N-Queens and has no finished reproduction run. We keep three ideas: parallel width sampling, selection by a value signal, and the ablation that unguided noise collapses on multi-solution tasks.
    - Library learning across programs (shared abstractions). It is DreamCoder-style compression and fights the simplicity prior: a library shortens each program only after you pay for the library. Not buildable in the time left.
    - A distribution over program source text. Every program that exactly replays the buffer has likelihood 1, so text-level weights carry no information. We use the version space with a simplicity (MDL) prior and read information from where consistent programs disagree.
    - Live OPINE-World harness runs as "cheap ARC runs". One synthesis round is 12 minutes and 10.5M prompt tokens. 25 games cost about $800. No `ARC_API_KEY` or `ANTHROPIC_API_KEY` is set. Replaced by offline replay bundles and `claude -p`.
    - Synthesizing the object extractor. Each game's released extractor is frozen and used as input data. We study the transition layer only, and state this in the pitch and README.
    - OpenAI as the synthesis backend. The `claude` CLI is installed and is the backend OPINE-World used, so results compare like for like.

- [jdv] - Reward-hacking evals: review and parallel build - a718512 (uncommitted)
  Literature review of reward-hacking evals (2025 to 2026), mapped onto the committee harness, then a parallel `src/rewardhack` build.
  DONE:
    - Review in research/reward_hacking_review.md: 40 verified works in four groups, six design lessons, harness mapping with cost per item.
    - Phases 1 to 4 and the Track 2.3 requirement mapping were not written anywhere. Added to DESIGN_DOC.md.
    - Track 2.3 gap found: "know when reward hacking" had no evidence in the plan. Covered by the held-out gap, literal-mass and MDL detectors, and an abstain output under an injected contradiction.
    - Parallel package `src/rewardhack/` with its own tests and `artifacts/rewardhack/`. Imports `committee.loader`, `committee.verify`, `committee.committee` read-only. No edits to `src/committee/`. One line added to pyproject.toml packages.
    - Detectors: literal mass, MDL ratio, held-out gap, order dependence. Calibrated on constructed lookup tables vs synthesized rules (RESULTS.md RH2). Thresholds 0.5 / 0.5 separate the classes by 5x.
    - Contradiction injection with an abstain channel, 2x2 on tr87 L1 and L6, 24 runs, $3.16 (RESULTS.md RH1). Hack rate 0/12. Abstain 5/6 when offered, each naming the exact pair. No false abstains on intact data.
    - Tests verified to fail under three breakages: literal mass forced to 0, is_hack ignoring the pair, contradiction with delta 0.
    - README.md section and commands. Pitch sentence: the simplicity prior is the hack detector, and the agent says "no rule exists" instead of forcing a pass.
  DEFERRED:
    - Harder games for the contradiction test. tr87 is the easiest level set; Opus did not hack even without the channel. ar25 L3 and the mixed-row games are where the Opus 5.5 card predicts 3-6x more attempts.
    - Other models as synthesizer (sonnet, haiku). The literature puts hack rates at 0% for Opus-class and up to 14% for weaker models, so a model sweep is the cheapest way to show the detectors catching a real hack.
    - Decoy-file trap (Hack-Verifiable Environments). Needs `--output-format stream-json` tool-call logging. After the abstain and contradiction results are recorded.
    - CoT monitors and activation probes. Closed model, program-only output, so there is no reasoning trace or activations to monitor.
  ABANDONED:
    - LLM-judge monitor over synthesis transcripts. The literature (2608.00583, 2605.16626) shows judge catch rates fall to 4-11% under reasoning rewrites, and our hacks are all visible in the program source, where static and behavioural checks are exact and free.

- [jdv] - Hoeffding's problem as a parallel task - a718512 (uncommitted)
  Research on bounds for uncertainty-estimation methods as an autoresearch target, then a parallel `src/hoeffding/` build with the same seeds, synth, verify, committee loop as ARC.
  DONE:
    - Surveyed UQ-method bounds (jackknife+, empirical Bernstein, binomial CI length, distance to calibration, Berry-Esseen). None is both open and cheap to certify. Chose Hoeffding's problem: sup P(S_n <= t) over iid X in [0, 1] with mean m. Solved for n = 1 (Markov) and n = 2 (Meester 2008). Open for n >= 3.
    - Rejected "Beat the average" (Problem 6.39 in Tao et al.'s repository) as the headline: its lower bound 0.400695 is conjectured exact, so a run only rediscovers. Kept as a possible calibration benchmark.
    - Package `src/hoeffding/`: `verify.py` exact rational certificate with weight repair; `families.py` Meester's binary and ternary families; `search.py` random-restart local search baseline; `synth.py` `claude -p` workspace that also asks for `report.json`, the agent's probability per instance that its value is tight; `committee.py` MDL weights from `committee.committee.description_length`, leader disagreement, honest interval [certified, Hoeffding wall]. One line added to pyproject.toml packages. Tests in `tests/test_hoeffding.py` fail under a mutated verifier.
    - Baselines recorded as H1 in RESULTS.md.
    - Four synthesis runs (one per seed) recorded as H2. All four strategies certify on all 45 test instances and match the proven optimum on all 18 instances with n <= 2 and the family best on the 27 open ones. Cost $0.38 to $1.10 per run.
    - Contract change: the agent must define `confidence(n, m, t)`, scored on unseen instances. run0 gave confidence as a table over train keys only, so test calibration could not be scored.
    - run1 per seed under the new contract, recorded as H3. Members state 0.80 to 0.95 on instances with a proof (hit 1.00) and 0.35 to 0.61 on open instances. Two runs timed out at 900 s; one of them left the stub.
    - Runner fixes: members may import the workspace `verify.py` (written into the run directory, `sys.path` set under `python -I`); early failures fill the confidence list so lengths match.
  DECISIONS:
    - Lower tail P(S_n <= t), matching Meester's normalisation; the upper tail follows by X -> 1 - X.
    - Upper wall is Hoeffding's KL bound only. Bentkus (2004) is tighter but its conditions were not verified in time.
    - Reference for n >= 3 is "best known here, not proven". The Hoeffding-Shrikhande reduction to at most 4 atoms holds for n = 2 only, so no finite search is complete for n >= 3. Calibration against exact truth uses n <= 2 (18 test instances); calibration against the reference is reported separately and labelled.
    - Floats in a candidate are rounded to rationals with denominator <= 1e12, then the last two weights are re-solved so the mean constraint holds exactly. A candidate the repair cannot fix is rejected. This closes the float-rounding, out-of-range and boundary-tie hacks.
    - Train instances n in {3, 4, 6}; test n in {1, 2, 5, 8, 10}. The strategy must be a rule in (n, m, t); test instances never enter the workspace.
    - H4: on 36 unseen instances with n up to 30, the three members that finish agree with each other and with the family best to 1e-4 on every instance. Committee disagreement carries no information on this task; recorded as such.
    - Outer hill-climb loop `src/hoeffding/climb.py`: population = seed runs plus children; fitness = mean fraction of the Bernoulli-to-Hoeffding gap closed on 39 climb instances (train plus n in {15, 20}), rejected or timed-out instances score zero; each round the top 2 parents' source, the per-instance best so far, and the instances where parents disagree or fail go into the children's workspace. The certified value is the only reward. Disagreement and rejection counts are logged per round in `artifacts/hoeffding/climb/log.json`.
  DEFERRED:
    - A harder family where search fails, so that disagreement can carry information: t close to n m with large n, or the Bellec-Fritz inequalities (AlphaEvolve's naive run reached 0.389 against the known 0.400695). Only if the pitch needs a disagreement result from this task; H3 already gives the calibration result.
    - Float-only verifier as a reward-hacking control. The exact verifier makes it moot for scoring; only a demo item.
    - Bentkus upper wall. Would shrink the reported unknown width; needs the theorem's conditions checked.
    - Other coefficient vectors (Bellec-Fritz family) with the same certifier.
  ABANDONED:
    - Berry-Esseen finite-n constant. Esseen's 0.4097 is proven sharp for n >= 2 exp(1e17) (He and Cheng 2026) and the Bernoulli case is below it for n <= 5e5; a finite-n improvement is implausible.
    - Problem 6.11 (Fourier uncertainty principle). Best upper bound 0.3102 (Cohn, de Laat, Goncalves) is out of reach and the problem is not statistical.

- [jdv] - Committee build: Phases 0 to 3 code, first baselines, backend pivot - a718512 (uncommitted)
  Built the committee pipeline end to end, ran first baselines with the claude CLI, then moved synthesis off the Claude account.
  DONE:
    - `src/committee`: loader (bundle to object buffer via the frozen released extractor), matrix (τ, a, μ rows and η), verify (sandboxed exact replay), synth (claude CLI agent in an isolated workspace), synth_api (repair loop over any OpenAI-compatible model), seeds, committee (MDL weights, vote, disagreement, AUROC, reliability bins, per-row η), explore (query by committee vs random vs count priority), evaluate, demo, modal_app (fan-out), modal_llm (vLLM server). 14 tests, each checked against a mutant.
    - Bundle semantics verified: a step's frame is the state after its action. Released frame-mode engines replay exactly once the move-counter mask and level-entry caches are accounted for.
    - R1 recorded. tr87 L1 and L6 are saturated for a single program. ar25 L3 at 40% train is informative: 3/3 baseline programs replay train exactly and score 0.43 to 0.48 held-out (claude backend, old split, voided).
    - Split bug found and fixed: RESET and level-closing transitions were in train. Programs that "passed" them had stored the layout. Now dropped from both sides. Local runs made under the old split were deleted.
  DEFERRED:
    - Devin as a second synthesis backend for the hard levels, after the Modal model numbers are in.
    - Rerun of ar25 L3 f0.4 and tr87 L1 to L2 (baseline 3, committee 8) on the new backend.
  ABANDONED:
    - Synthesis through the user's Claude Max account (claude CLI locally or via `claude setup-token` on Modal). The user chose the Devin and Modal accounts instead. The claude backend stays in the code as an option.
    - Running the sweep locally. Long jobs go to Modal per the work rules; the fan-out is `committee.modal_app`.
