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
    - Open-weight sweep on Modal: vLLM engine (`rewardhack.modal_app`, app `rewardhack-vllm`, separate from the committee's `committee-llm`), chat loop with checker feedback (`rewardhack.oss_synth`), `--backend modal|openai`. Qwen2.5-Coder-7B and Qwen3-Coder-30B-FP8, 48 runs (RESULTS.md RH3). Both too weak to reach exact replay in 4 rounds; 30B attempts layout enumeration in 4/24 programs.
    - Found a real hack in the committee's artifacts: `tr87/L1_f60/api_qwen/run2` passes exact replay by enumerating cursor columns, held-out accuracy 0.25 (RESULTS.md RH4). Added the enumerating class (>= 5 layout guards) to the memorising flag. Claude programs have 0 guards.
    - Modal image lessons: unpinned `pip_install("vllm")` resolves to a source build; `vllm/vllm-openai` image needs no `add_python` and a `python` symlink; `modal.parameter` needs real annotations, so no `from __future__ import annotations`.
    - Sonnet and Haiku sweep, 48 runs, $6.39 (RESULTS.md RH5). Haiku hacks 11/12 contradiction runs with a call counter, even with the abstain channel. Sonnet 0/12 hacks, abstains 2/6. Order dependence flags 11/11 Haiku hacks and 0/24 Opus and Sonnet intact programs.
    - Three hack classes seen, each needing its own detector: tabulation (literal mass), layout enumeration (layout guards), call-order special-casing (order dependence, contradiction construction). Demo story: Haiku program `artifacts/rewardhack/tr87/L1_f60/contradiction/haiku/run1`.
    - Modal `rewardhack-vllm` app stopped after the sweep. The crash-loop container the user saw (ta-01M40X638...) was the 40960 context error, fixed before the runs.
  DEFERRED:
    - Harder games for the contradiction test. tr87 is the easiest level set; Opus did not hack even without the channel. ar25 L3 and the mixed-row games are where the Opus 5.5 card predicts 3-6x more attempts.
    - 30B with 8 rounds through the committee's API loop, where the one real hack came from.
    - Order dependence as a verifier gate in the committee pipeline. It is a one-line addition to `verify.run_program` but that file belongs to the committee build; raise it with that session.
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
    - H5: the same certifier reproduces the Problem 6.39 ladder (naive 0.25, explicit laws 0.38 to 0.39, construction limit 0.4007, ceiling 0.417). H6: a 4- and 5-atom refutation search finds 0 wins over the family on 72 Hoeffding instances. Decision: the hill-climb and disagreement claims move to 6.39; Hoeffding stays as the calibration benchmark.
    - Task abstraction `src/hoeffding/task.py`: a task binds instances, walls, the exact certifier, the agent contract and a standalone verifier. The runner, workspace, committee and climb are task-agnostic. Second task `linear.py`: C(c) = sup P(sum c_i X_i < 0) over iid laws (Bellec-Fritz family). Train c in {(1,1,1,-2), (1,1,1,1,-3), (1,2,-3)}; test c in {(2,-1,-1) proven 2/3, (1,1,-2), (1,1,1,1,1,-4), (2,1,1,-3)}. Lower wall = best two-atom law; upper wall only where published. Ties do not count. At most 64 atoms.
    - Sign convention checked: P(2 X1 < X2 + X3) is the proven 2/3 case, so the anchor is c = (2, -1, -1), not (1, 1, -2).
    - Linear climb launched from the stub: 3 rounds, 2 children, top 2, opus, 30 turns.
    - Decision: committee disagreement is not claimed as an uncertainty signal on the bound tasks. A member's output is one law, and the certifier settles any disagreement in milliseconds, so the signal is consumed the moment it is produced. Disagreement is worth something only where resolution costs something, which is the ARC case (an action in the environment). The bound tasks keep: exact falsification, calibration of self-reported confidence against proofs, and the hill-climb.
    - Two-tier verification `src/hoeffding/tiers.py`: a float tier returns [value, value + tie mass + rounding margin] as a bracket for the exact value; the policy certifies a candidate only when the bracket's upper end reaches the incumbent certified best, skips it otherwise, and audits 10 percent of skips. Reward and reported bounds come from the exact tier only. Ties are mass in the bracket, not a veto, so dyadic laws are not force-certified. Test fails when the tie mass is dropped from the bracket. Measured: exact tier is 6x to 20x the float tier, 80 s at 32 random atoms with a 6-term c.
    - Linear climb from the stub: round 1 child reached 0.3734 on (1,1,1,-2); round 2 child reached 0.3915, above AlphaEvolve's naive 0.389, by rediscovering the Bellec-Fritz tie-breaking ladder (signed perturbations on shared levels). On the held-out proven anchor (2,-1,-1) it certifies 0.6596 against 2/3. Children left confidence at 0 (timeouts), so calibration is not scored for them.
    - H8: linear climb round 3 reached 0.3976 on (1,1,1,-2) with a self-similar law, 0.003 under the Bellec-Fritz limit, plus certified values on four vectors with no published number. Every climbing child timed out at 900 s, so `confidence` stayed 0; raise the timeout or ask for confidence first in the next run.
    - H9: replay of the certification policy on identical cached laws: 82 percent of certifications skipped on the plateaued Hoeffding climb, 1 of 18 on the still-improving linear climb, bests identical, 0 audit disagreements. Policy moved into the child's check.py (certify only a law that beats the child's own best per instance); float sums exactly on the boundary count as exact ties so dyadic laws are not force-certified. Timeout raised to 1800 s, 40 turns; contract asks for confidence in the first edit. Linear climb resumed for rounds 4 and 5 with --tiers.
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

- [jdv] - Committee results R2 to R5, Modal and Devin backends, first commits - 100d2ea
  Branch `jdv` holds seven commits covering the committee, reward-hacking and Hoeffding builds. Devin is the synthesizer for reported numbers.
  DONE:
    - Backends: vLLM servers on Modal (Qwen3-Coder-30B FP8, gpt-oss-120b) with a repair loop; Devin sessions with attachments and structured output; Modal fan-out of one container per member. R2 records that both open models fail exact replay on the easiest level while Devin and the claude CLI solve it in about a minute.
    - R3, R4: ar25 L3 at 40% train. Single programs 0.52 mean held-out. Seeded committee of 8: 7 distinct behaviours, AUROC 0.78 for disagreement against error, unanimous error 0.30 vs split error 0.83 to 1.00, 51 of 90 held-out rows unseen in train and scored only by the committee. Disagreement finds a falsifying counterexample in 1 probe, count priority 7, random 2.7.
    - R5: tr87 L1 to L2 is saturated; all 11 programs 1.00, committee unanimous, error 0.00.
    - Equal-weight disagreement added beside the MDL vote after the lambda sweep showed sharper weights do not improve calibration. Row entropy normalised over its own support. Collapse metric excludes falsified runs.
  DEFERRED:
    - More informative splits (m0r0 L3, ft09 L5, ar25 L7) with Devin, to show the calibration result on more than one level.
    - README headline numbers and the demo video.
  ABANDONED:
    - Claim that the MDL vote beats a single program on accuracy. R4 shows 0.477 vs 0.480 mean; the shortest member was not the best. The reported benefit is calibrated uncertainty and faster falsification, not point accuracy.
    - Tuning the open-model repair loop further. Two models and a best-of-rounds fix left exact replay unreached on tr87 L1; the time goes to results with Devin.

- [jdv] - R6, R7 and the cross-split summary - f88afc5
  Two more Devin splits. m0r0 L3 is a second informative level; ft09 L5 is saturated.
  DONE:
    - R6 m0r0 L3: seeded 8 of 8 admitted against 1 of 3 unseeded; unanimous error 0.16, split error 0.80; AUROC 0.68; disagreement falsifies all members in 1 probe against 10 by count priority and 4.8 random; vote 0.77 against 0.69 mean single program.
    - R7 ft09 L5: all 11 programs 1.00, unanimous, error 0.00.
    - Summary table across the four splits in RESULTS.md and README.md. Write-up page with equations, diagram and charts published as an artifact.
    - PDFs untracked and binary patterns added to .gitignore.
  DEFERRED:
    - Rewriting history to drop the PDFs from the two commits that still hold them. The user decides before the first push.
    - Demo video and final README pass.
  ABANDONED:
    - Nothing new.

- [jdv] - Targeted committee growth tried and dropped - 0a8a623
  Built `committee.active`, ran it on ar25 L3 and m0r0 L3 from the same first 3 members as the seeded committees.
  DONE:
    - R8 recorded as a negative result. Probe disagreement fell 0.22 to 0.07 on m0r0 while held-out errors stayed the same six transitions. ar25 targeted AUROC 0.61 against 0.77 seeded. Every targeted member sided with the majority (Devin: "I sided with the majority").
    - `committee.evaluate --curve` added: metrics by member count in run order.
  DEFERRED:
    - Literature review of baselines by three research agents; then implement the two or three baselines that can run on the offline buffers.
  ABANDONED:
    - Targeted growth as the construction method. The user chose independent seeded synthesis. Reasons: probes drawn from training states do not reach the unseen states where held-out errors occur; one disputed mechanic touches few held-out transitions; vote counts in the seed anchor the synthesizer; sequential rounds cost about three times the wall clock.
    - Probes from predicted future states (committee rollouts). Not built. The anchoring problem would carry over, and time goes to baselines, the demo and the video.
