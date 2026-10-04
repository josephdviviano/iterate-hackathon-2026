# World model

_Last updated after: E007_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E004 (FEASIBLE): airbench96-style (whiten 2x2, 128/384/576 x3 convs residual, GELU, BN, Nesterov SGD lr 9 / wd 0.012 / mom 0.85, LS 0.1, Lookahead, alt flip + translate 4, NO cutout, bs 1024, bf16 autocast, torch.compile max-autotune), 10 epochs: accuracy 0.7550, time 8.94 s. Margin over 0.753: +0.2 pp (~0.25 epoch).

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
- Width trade-off: 128/256/384 at 10 ep = 0.7402 in 6.23 s vs 128/384/576 0.7550 in 8.94 s -> 0.55 pp per second saved, cheaper than cutting epochs (~0.9 pp/s). Capacity is not the binding constraint at <= 10 ep; narrow + more epochs is the better frontier. Time scales ~0.70x for 0.60x FLOPs (small convs less efficient). (E007)
- Alternating (derandomized) flip beats random flip by ~0.4 pp at 10 ep (lit:C0002 transfers to CIFAR-100). (E005)
- max-autotune compile (cudagraphs) gives only -2.5% time: launch/Python overhead is small; the loop is compute-bound. (E003)
- Accuracy vs epochs around 8.5-10 ep: ~0.87 pp/epoch; time ~0.95 s/epoch, near-zero intercept. Edge by epochs alone ~11-12 ep (~11-11.5 s). (E001, E002)

## Lessons from failures
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->
- (E007) I overestimated the capacity cost of narrowing. At <= 10 epochs the net is step-limited, not capacity-limited: try narrowing before cutting epochs.

## Dead ends

## Open questions and surprises
- Will the slope hold to 11-12 ep or flatten?
- Do the remaining regularizers (translate 4, LS 0.1, flip) still over-regularize at ~9-10 epochs? Which half of E004 (cutout vs LS) carried the gain?
- How far does narrowing go? 128/256/384 at ~12 ep is the next point (I010); even narrower (96/192/288?) may be better still.
- Progressive resizing (I011) untested. Further regularization reduction (LS 0-0.05, translate 2) untested. Note D002 misreads E004 as 8.5 ep; it was 10 ep (+1.26 pp at equal epochs vs E003).

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
