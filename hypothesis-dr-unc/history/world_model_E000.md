# World model

_Last updated after: E000_

## Current best
<!-- the current recipe in a line or two, and its metrics -->
E000 template (16-ch conv, 64 images, 3 SGD steps): accuracy 0.012, time 0.10 s. NOT feasible (needs accuracy >= 0.753).

## Measurement
<!-- what the metrics mean in practice: run-to-run noise, what a real difference looks like -->
- time = mean(prepare+train) over 3 seeds (0,1,2); CUDA syncs at boundaries are charged. First trial can carry lazy-init cost (E000: std 0.17 s on a 0.1 s mean), so anything lazy (cudnn autotune, compile, allocator growth) must be warmed in untimed build.
- Harness overhead per run ~13 s wall for the trivial recipe (process spawn, data load, env checks).
- Reference point (README): ResNet9-style, 40 epochs, width 64 -> 75.36% in 59.3 s on A100 PCIe. Our GPU is A100 SXM (somewhat faster clocks/power than PCIe).

## Where the cost goes
<!-- what dominates the objective, and why -->

## Established facts
<!-- each with the experiments that support it, e.g. "(E003, E007)" -->

## Lessons from failures
<!-- distilled from post-mortems: what went wrong, and the rule that follows -->

## Dead ends

## Open questions and surprises

## Changelog
<!-- one line per experiment: "E007 (I012, H2.1.3): <result> -> <what changed in this model>" -->
E000 (baseline): template acc 0.012, time 0.10 s -> harness works; floor of fixed overhead is ~0.1 s; need a real recipe.
