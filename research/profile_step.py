"""Kernel-level profile of one timed trial (prepare + train) of a submission.

    python research/profile_step.py --submission-path research/lab_recipe --data-root data \
        --params '{"compile_mode": "max-autotune-no-cudagraphs"}'

Builds the submission (warm-up included), runs one untimed trial to settle clocks and
allocators, then profiles a second trial with ``torch.profiler`` and prints the kernels with the
most self CUDA time plus totals by kernel family. CUDA graphs hide kernels from the profiler, so
profile ``max-autotune-no-cudagraphs`` to see the same generated kernels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from benchmark.api import BuildContext
from benchmark.data import load_split
from benchmark.worker import load_submission, seed_everything

FAMILIES = (
    ("conv-dgrad", r"dgrad|convolve_common_engine_float_NHWC.*dgrad|wgrad_alg0|bwd_data"),
    ("conv-wgrad", r"wgrad|bwd_filter"),
    ("conv/gemm", r"sm80_xmma|cutlass|gemm|implicit_convolve|conv|winograd|ampere_"),
    ("sgd", r"fused_sgd|multi_tensor|foreach"),
    ("triton", r"^triton_"),
    ("pool", r"pool|max_"),
    ("copy", r"copy|Memcpy|Memset|elementwise|vectorized|index|gather|scatter|cat"),
)


def family(name: str) -> str:
    for label, pattern in FAMILIES:
        if re.search(pattern, name, re.I):
            return label
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--params", default="{}")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--bandwidth",
        action="store_true",
        help="instrument Inductor kernels and print achieved GB/s per kernel (slow; no profile)",
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    if args.bandwidth:
        import torch._inductor.config as inductor_config  # noqa: PLC0415

        inductor_config.profile_bandwidth = True
    module = load_submission(args.submission_path)
    state = module.build(BuildContext(device, json.loads(args.params)))
    data = load_split(args.data_root, train=True)
    if args.bandwidth:
        seed_everything(0)
        module.prepare(state, data, 0)
        module.train(state)
        torch.cuda.synchronize()
        return
    for seed, profiled in ((0, False), (1, True)):
        seed_everything(seed)
        if not profiled:
            module.prepare(state, data, seed)
            module.train(state)
            continue
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            module.prepare(state, data, seed)
            module.train(state)
            torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total = sum(e.self_device_time_total for e in events)
    print(f"total self CUDA time {total / 1e3:.1f} ms over {len(events)} kernels")
    by_family: dict[str, float] = defaultdict(float)
    for e in events:
        by_family[family(e.key)] += e.self_device_time_total
    for label, us in sorted(by_family.items(), key=lambda kv: -kv[1]):
        print(f"  {label:12s} {us / 1e3:9.1f} ms  {100 * us / total:5.1f}%")
    print(f"\ntop {args.top} kernels by self CUDA time")
    for e in sorted(events, key=lambda e: -e.self_device_time_total)[: args.top]:
        share = 100 * e.self_device_time_total / total
        print(
            f"{e.self_device_time_total / 1e3:8.1f} ms {share:5.1f}% calls {e.count:6d}  "
            f"{e.key[:110]}"
        )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
