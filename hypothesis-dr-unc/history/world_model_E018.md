# World model

_Last updated after: E018_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E012 (FEASIBLE, tip; NOTE translate is already 2, inherited from E011): airbench96-style (whiten 2x2, widths 128/256/576 x3 convs residual, GELU, BN, Nesterov SGD lr 9 / wd 0.012 / mom 0.85, LS 0.1, Lookahead, alt flip + translate 2, no cutout, bs 1024, bf16 autocast, compile max-autotune), 11 epochs: accuracy 0.7567, time 7.70 s. Margin +0.37 pp.

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->
- Per-seed accuracy sd ~0.15-0.3 pp -> 3-seed mean sd ~0.1-0.17 pp; differences < 0.3 pp are noise. Per-trial time spread ~0.05-0.1 s; 3-seed mean time differences > ~0.1 s are real.
- time = mean(prepare+train) over 3 seeds (0,1,2); CUDA syncs at boundaries are charged. First trial can carry lazy-init cost (E000: std 0.17 s on a 0.1 s mean), so anything lazy (cudnn autotune, compile, allocator growth) must be warmed in untimed build.
- Harness overhead per run ~13 s wall for the trivial recipe (process spawn, data load, env checks).
- Reference point (README): ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe. Our GPU is A100 SXM (somewhat faster clocks/power than PCIe).

## Where the cost goes
<!-- what dominates the objective, and why -->
- (E006) prepare = 0.06 s steady state, 0.15 s on the first trial (lazy first-use cost not warmed in build). Whitening-subset size is irrelevant to time.
- (E001) train loop ~98% of time: ~19 ms/step at bs 1024 for the 128/384/576 net (~0.9 s/epoch, ~2.9 TFLOP/step -> ~150 TFLOPS, about half of bf16 peak, so mostly compute-bound). prepare ~0.07 s (first trial 0.15 s). Time ~ linear in epochs x FLOPs/epoch.

## Established facts
<!-- each with the experiments that support it, e.g. "(E003, E007)" -->
- airbench96-style 128/384/576 at 8.5 epochs, no TTA, reaches only ~73.0% on CIFAR-100 (seed sd ~0.3 pp). The unverified "75.6% at 9 epochs" lore likely used TTA. (E001)
- At 10 epochs the net under-fits: removing cutout 12 and LS 0.2 -> 0.1 gave +1.26 pp at equal epochs (worth ~1.4 epochs). Regularization strength is a first-order lever in this regime. (E004)
- Width trade-off: 128/256/384 at 10 ep = 0.7402 in 6.23 s vs 128/384/576 0.7550 in 8.94 s -> 0.55 pp per second saved, cheaper than cutting epochs (~0.9 pp/s). Time scales ~0.70x for 0.60x FLOPs (small convs less efficient). (E007) BUT at 12 ep the narrow net reached only 0.7456 (+0.27 pp/epoch): it is near its ceiling (~75%), so narrow+longer does NOT reach the edge cheaply. (E009)
- Progressive resizing (first 50% of steps at 24 px, bilinear) at 10 ep: -0.55 pp, -1.42 s (-16%) -> 0.39 pp/s, the cheapest trade measured. The 24 px phase runs at ~0.58x the cost of 32 px. (E008)
- Resize (50% at 24 px) at 11 ep: 0.7526 / 8.24 s (E010) vs 10 ep 0.7495 / 7.52 s (E008): +0.31 pp/epoch. Per-epoch returns near 75% are ~0.3 pp/ep for both narrow (E009) and wide-with-resize nets: epochs are now an expensive lever (~0.4 pp/s).
- Curriculum 24/28/32 in thirds: 0.7512 / 6.40 s, same trade as 50%@24 (0.42 pp/s). Resize schedules all sit on one ~0.4 pp/s line: accuracy cost ~ proportional to FLOPs removed. (E018)
- Resize+AA on E012 net at 12 ep: 0.7521 / 6.89 s, only +0.11 pp over 11 ep (E013). Near 75% the epoch slope is ~0.1-0.3 pp/ep: the recipe is approaching a ceiling around 75.5-76% for this architecture/regularization; accuracy must come from elsewhere (capacity where it is cheap, optimizer/schedule). (E017)
- Resize on the E012 net (50% at 24 px, antialias): -0.57 pp, -1.39 s (-18%) -> 6.32 s at 0.7510. Antialias does not reduce the resize cost vs non-AA on the wide net (E008: -0.55). Resize is a consistent ~0.4 pp/s lever; combine with +1 epoch. (E013)
- Narrowing only block 2 (384 -> 256) keeps accuracy: 128/256/576 at 11 ep = 0.7567 / 7.70 s vs 128/384/576 at 10 ep 0.7585 / 8.96 s. Block-3 width is what matters; block 2 is the FLOP hog with little capacity value. Time/epoch 0.70 s vs 0.89 s. (E012)
- bs 1024 -> 768 at equal epochs (per-example lr fixed): accuracy unchanged, +5% time. Sample-limited, not step-limited; larger batches might trade throughput for little accuracy. (E015)
- LS 0.1 -> 0.0 costs -0.31 pp (E014): LS ~0.1 is near optimal; the LS axis is exhausted.
- Translate 4 -> 2 at 10 ep: +0.35 pp (~2 sd), time unchanged: weak regularization is better in this under-fitting regime. (E011)
- Alternating (derandomized) flip beats random flip by ~0.4 pp at 10 ep (lit:C0002 transfers to CIFAR-100). (E005)
- max-autotune compile (cudagraphs) gives only -2.5% time: launch/Python overhead is small; the loop is compute-bound. (E003)
- Accuracy vs epochs around 8.5-10 ep: ~0.87 pp/epoch; time ~0.95 s/epoch, near-zero intercept. Edge by epochs alone ~11-12 ep (~11-11.5 s). (E001, E002)

## Lessons from failures
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->
- (E007) I overestimated the capacity cost of narrowing at 10 ep... (E009) ...and then over-generalized: per-epoch slope depends on how close a net is to its ceiling. Measure the slope per architecture; don't transfer it. 128/256/384 slope at 10-12 ep is 0.27 pp/ep vs 0.87 for the wide net at 8.5-10.

## Dead ends
- Proxy-loss hard-example filtering (airbench96_faster-style, 32/64/64 proxy, keep 50% of 2048): -2.9 pp and +3.4 s at matched main steps. Dead. (E016)
- Random flip (E005), LS 0 (E014), bs 768 (E015), whitening subset (E006): no gain.

## Open questions and surprises
- Will the slope hold to 11-12 ep or flatten?
- Do the remaining regularizers (translate 4, LS 0.1, flip) still over-regularize at ~9-10 epochs? Which half of E004 (cutout vs LS) carried the gain?
- Intermediate widths (e.g. 128/320/480 or 128/384/512) may sit at a better point between capacity and cost. Alternatively, narrow only the cheap-to-narrow part.
- Combine the cheap levers: resize + narrow + a few more epochs. Resize fraction/size (e.g. 60-70% at 24 px, or 20 px) unexplored. Further regularization reduction (LS 0-0.05, translate 2) untested. Note D002 misreads E004 as 8.5 ep; it was 10 ep (+1.26 pp at equal epochs vs E003).

## Changelog
<!-- one line per experiment: "E007 (I012, H2.1.3): <result> -> <what changed in this model>" -->
E000 (baseline): template acc 0.012, time 0.10 s -> harness works; floor of fixed overhead is ~0.1 s; need a real recipe.
E001 (I001, H1.1.1): airbench96 128/384/576 8.5 ep: acc 0.7304, 7.77 s -> refutes "near the edge at 8.5 epochs"; edge needs ~+2.3 pp; time is compute-bound train loop.
E002 (I002, H1.1.2): 10 ep: acc 0.7435, 9.24 s -> slope 0.87 pp/ep, 0.98 s/ep; epochs-only edge ~11.5 ep.
E003 (I004, H2.2.1): max-autotune: acc 0.7424, 9.01 s -> -2.5% time, refutes >=8%; overheads are not where the time is; FLOPs are.
E004 (I005, H4.1.1): no cutout + LS 0.1 at 10 ep: acc 0.7550, 8.94 s -> FEASIBLE; regularization was too strong for the under-fitting regime.
E005 (I006, H4.3.1): random flip: acc 0.7509 (-0.41 pp), 9.01 s -> alternating flip helps; discarded.
E006 (I007, H2.3.1): whitening on 5k images: acc 0.7540, 8.95 s -> no time change; prepare is not a lever (except ~0.03 s first-trial warmup); discarded.
E007 (I008, H1.2.1): 128/256/384 at 10 ep: acc 0.7402, 6.23 s -> 0.55 pp/s trade-off beats epochs; narrow+longer is the frontier; discarded (infeasible).
E008 (I011, H3.2.1): 50% steps at 24 px: acc 0.7495, 7.52 s -> 0.39 pp/s, best trade so far; infeasible alone, discarded; combine with +epochs.
E009 (I010, H1.2.1): 128/256/384 at 12 ep: acc 0.7456, 7.44 s -> narrow net saturates (0.27 pp/ep); narrow+longer is not the route; discarded.
E010 (I012, H3.2.1): resize + 11 ep: acc 0.7526, 8.24 s -> 0.04 pp short; epoch slope ~0.3 pp/ep near the edge; discarded; need margin from elsewhere.
E011 (I013, H4.1): translate 2: acc 0.7585 (+0.35 pp), 8.96 s -> kept (forced) for margin; resize+translate 2 at 10-11 ep should now clear 0.753.
E012 (I014, H1.2.1): 128/256/576 11 ep: acc 0.7567, 7.70 s -> NEW BEST; block 2 width is cheap to cut, block 3 width matters.
E013 (I015, H3.2.1): resize+AA on E012: acc 0.7510, 6.32 s -> AA no help; resize -18% time for -0.57 pp; discarded (infeasible); resize + 12 ep is the obvious next.
E014 (I016, H4.1): LS 0: acc 0.7536 (-0.31 pp), 7.71 s -> LS 0.1 stays; discarded.
E015 (I017, H4): bs 768: acc 0.7563, 8.09 s -> sample-limited, no accuracy gain; discarded; try larger batches.
\nE016 (I018, H3.1.1): proxy filtering: acc 0.7274, 11.09 s -> dead end; small proxy nets are latency-bound and hard-example selection hurts on 100 classes.\n
E017 (I021, H3.2.2): resize+AA, 12 ep: acc 0.7521, 6.89 s -> 0.09 pp short; epochs nearly exhausted (+0.1 pp/ep); discarded.
E018 (I022, H3.2.3): 24/28/32 curriculum: acc 0.7512, 6.40 s -> same 0.4 pp/s line as 50%@24; discarded.
