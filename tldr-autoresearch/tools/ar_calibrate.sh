#!/bin/bash
# Noise calibration for autoresearch on the agent GPU (config agent_gpu_uuid): N unmodified-baseline runs into a
# SEPARATE ledger (data/calib_ledger.jsonl, AR_ATTEMPT_ID=calib-<i>), then (with --baseline) one run into the MAIN
# ledger (AR_ATTEMPT_ID=baseline) so its train_sha256 becomes the root parent of the driver. Calls tools/ar_run.py
# directly with the runner's settings (--gpu-minor, --no-autotune if configured, a fresh compile cache per run);
# the agent-facing <repo>/run.sh submits to the runner daemon instead. Run it before starting the driver, while the
# runner is stopped (both use the agent GPU). Summarise with
#   python3 tools/ar_stats.py data/calib_ledger.jsonl
# Usage: ar_calibrate.sh [N=5] [--baseline]
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)  # code (rltldr/ package, tools/ar_run.py)
SERVE_PY=${RLTLDR_SERVE_PY:-$HOME/envs/serve/bin/python}
TRUSTED_PY=${RLTLDR_TRUSTED_PY:-/usr/bin/python3}
cfg() { PYTHONPATH="$HERE" "$SERVE_PY" -m rltldr.config get "$1"; }   # honours $RLTLDR_ROOT / $RLTLDR_CONFIG
N=${1:-5}
REPO=$(cfg repo) || exit 1
DATA=$(cfg data) || exit 1
CALIB_LEDGER=$DATA/calib_ledger.jsonl
MAIN_LEDGER=$(cfg ledger) || exit 1
PREPARE_SHA=$(cfg prepare_sha256) || exit 1
GPU=$(cfg agent_gpu_uuid) || exit 1
GPU_MINOR=$(cfg agent_gpu_minor) || exit 1
[[ $GPU =~ ^GPU-[0-9a-fA-F-]+$ ]] || { echo "agent_gpu_uuid='$GPU' is not a GPU UUID: set it in config.json" >&2; exit 1; }
EXTRA=()
[ "$(cfg no_autotune)" = true ] && EXTRA+=(--no-autotune)
mkdir -p "$DATA"
cd "$REPO" || exit 1
# ar <ledger> <attempt id> <desc>: one run through the trusted wrapper, fresh compile cache. env -i proves it
# needs nothing from the caller's environment (PATH, venv, CUDA_*).
ar() {
  local cache; cache=$(mktemp -d -p "$DATA" cache-XXXXXX)
  env -i HOME="$HOME" PATH=/usr/bin:/bin AR_ATTEMPT_ID="$2" "$TRUSTED_PY" -I "$HERE/tools/ar_run.py" \
    --repo "$REPO" --ledger "$1" --gpu "$GPU" --gpu-minor "$GPU_MINOR" --prepare-sha "$PREPARE_SHA" \
    --cache-dir "$cache" "${EXTRA[@]}" --desc "$3"
  local rc=$?
  rm -rf "$cache"
  return $rc
}
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "repo not clean; calibration must run the committed baseline" >&2; exit 1
fi
echo "calibrating at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD), $(date -u +%FT%TZ)"
for i in $(seq 1 "$N"); do
  echo "=== calib-$i start $(date -u +%T)"
  ar "$CALIB_LEDGER" "calib-$i" "baseline (noise calibration $i/$N)"
  echo "=== calib-$i exit=$? end $(date -u +%T)"
done
if [ "${2:-}" = "--baseline" ]; then
  echo "=== baseline (main ledger) start $(date -u +%T)"
  ar "$MAIN_LEDGER" baseline "baseline"
  echo "=== baseline exit=$? end $(date -u +%T)"
fi
