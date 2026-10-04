#!/usr/bin/env bash
# vLLM policy server: Qwen3.8-27B-FP8, TP=2 on GPUs 0,1, versioned LoRA adapters + dev endpoints (localhost only).
# --logprobs-mode processed_logprobs: returned logprobs are those of the actual sampling distribution
# (after temperature/top-k/top-p), i.e. exact behaviour-policy log-probs for the trainer's IS weights.
# Policy adapters (policy-vN) are (re)registered at runtime by the gateway; never use load_inplace (vLLM 0.30 bug).
# Checkpoint and served name come from config (model_dir, base_model_name); the serving env from $RLTLDR_SERVE_PY.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)     # code (rltldr/ package); config root is $RLTLDR_ROOT or this
SERVE_PY=${RLTLDR_SERVE_PY:-$HOME/envs/serve/bin/python}
MODEL_DIR=$(PYTHONPATH="$HERE" "$SERVE_PY" -m rltldr.config get model_dir)
SERVED_NAME=$(PYTHONPATH="$HERE" "$SERVE_PY" -m rltldr.config get base_model_name)
source "$(dirname "$SERVE_PY")/activate"
export CUDA_VISIBLE_DEVICES=${SERVE_GPUS:-0,1}
export VLLM_USE_DEEP_GEMM=0
# FlashInfer JIT must use a consistent CUDA 13.0 toolkit: container CUDA_PATH/nvcc is 12.8 (too old for sm_120f).
# $RLTLDR_CUDA_HOME (default ~/envs/cuda13) is a symlink overlay of the venv's nvidia/cu13 wheels (nvcc/nvvm/crt/cccl
# pinned to 13.0) with lib64 + libcuda stub (docs/SETUP.md).
CUDA13=${RLTLDR_CUDA_HOME:-$HOME/envs/cuda13}
export CUDA_HOME=$CUDA13 CUDA_PATH=$CUDA13 PATH=$CUDA13/bin:$PATH
export CUDA_CACHE_MAXSIZE=4294967296
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port ${VLLM_PORT:-8000} \
  --tensor-parallel-size 2 \
  --max-model-len ${MAX_MODEL_LEN:-262144} \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --language-model-only \
  --enable-prefix-caching \
  --logprobs-mode processed_logprobs \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-lora --max-lora-rank ${MAX_LORA_RANK:-64} --max-loras 2 \
  "$@"
