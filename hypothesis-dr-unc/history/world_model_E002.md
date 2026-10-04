# World model

_Last updated after: E002_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E002 airbench96-style (whiten 2x2, 128/384/576 x3 convs residual, GELU, BN, Nesterov SGD lr 9 / wd 0.012 / mom 0.85, LS 0.2, Lookahead, alt flip + translate 4 + cutout 12, bs 1024, bf16 autocast, torch.compile default), 10 epochs: accuracy 0.7435, time 9.24 s. NOT feasible (needs +1.0 pp).

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->
- time = mean(prepare+train) over 3 seeds (0,1,2); CUDA syncs at boundaries are charged. First trial can carry lazy-init cost (E000: std 0.17 s on a 0.1 s mean), so anything lazy (cudnn autotune, compile, allocator growth) must be warmed in untimed build.
- Harness overhead per run ~13 s wall for the trivial recipe (process spawn, data load, env checks).
- Reference point (README): ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe. Our GPU is A100 SXM (somewhat faster clocks/power than PCIe).

## Where the cost goes
<!-- what dominates the objective, and why -->
- (E001) train loop ~98% of time: ~19 ms/step at bs 1024 for the 128/384/576 net (~0.9 s/epoch, ~2.9 TFLOP/step -> ~150 TFLOPS, about half of bf16 peak, so mostly compute-bound). prepare ~0.07 s (first trial 0.15 s). Time ~ linear in epochs x FLOPs/epoch.

## Established facts
<!-- each with the experiments that support it, e.g. "(E003, E007)" -->
- airbench96-style 128/384/576 at 8.5 epochs, no TTA, reaches only ~73.0% on CIFAR-100 (seed sd ~0.3 pp). The unverified "75.6% at 9 epochs" lore likely used TTA. (E001)
- Accuracy vs epochs around 8.5-10 ep: ~0.87 pp/epoch; time ~0.95 s/epoch, near-zero intercept. Edge by epochs alone ~11-12 ep (~11-11.5 s). (E001, E002)

## Lessons from failures
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->

## Dead ends

## Open questions and surprises
- Will the slope hold to 11-12 ep or flatten?
- Would capacity (width) or regularization tuning buy the 2.3 pp more cheaply than epochs?

## Changelog
<!-- one line per experiment: "E007 (I012, H2.1.3): <result> -> <what changed in this model>" -->
E000 (baseline): template acc 0.012, time 0.10 s -> harness works; floor of fixed overhead is ~0.1 s; need a real recipe.
E001 (I001, H1.1.1): airbench96 128/384/576 8.5 ep: acc 0.7304, 7.77 s -> refutes "near the edge at 8.5 epochs"; edge needs ~+2.3 pp; time is compute-bound train loop.
E002 (I002, H1.1.2): 10 ep: acc 0.7435, 9.24 s -> slope 0.87 pp/ep, 0.98 s/ep; epochs-only edge ~11.5 ep.
