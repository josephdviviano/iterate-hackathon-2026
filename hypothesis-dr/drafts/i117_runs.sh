#!/bin/bash
# I117: fresh-process determinism and time of autotuned vs default compile (seeds 0-2), each with a private cold cache.
cd /home/jovyan/work/arena3/hypothesis-dr
run() {
  name=$1; params=$2
  nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv,noheader | tr '\n' ' ' > /tmp/i117_$name.load
  TORCHINDUCTOR_CACHE_DIR=/tmp/i117_ind_$name TRITON_CACHE_DIR=/tmp/i117_tri_$name CUDA_VISIBLE_DEVICES=3 taskset -c 20-23 \
    uv run python -m benchmark.run --submission arena --n 3 --seed 0 --no-accuracy-target --results-root drafts/i117/$name --params "$params" > /tmp/i117_$name.log 2>&1
  rm -rf /tmp/i117_ind_$name /tmp/i117_tri_$name
}
run A1 '{}'
run B1 '{"cudnn_benchmark": false, "compile_mode": "default"}'
run A2 '{}'
run B2 '{"cudnn_benchmark": false, "compile_mode": "default", "extra_warmup": true}'
echo done
