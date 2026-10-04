#!/usr/bin/env bash
# Control script for the RLTL;DR autoresearch loop.
#   ./ctl.sh start [component...]    start (supervised, auto-restart) — default: all, in dependency order
#   ./ctl.sh stop  [component...]    stop (driver first, so the current attempt finishes cleanly)
#   ./ctl.sh status                  processes, GPU usage, gateway state, latest metrics
#   ./ctl.sh logs <component>        tail -f the component log
#   ./ctl.sh report                  research progress: best val_bpb trajectory, group metrics, insights, updates
# Components: vllm (GPUs 0,1 = $SERVE_GPUS) | gateway | runner (trains on config agent_gpu_uuid) |
#             trainer (config trainer_gpu_uuid) | driver (pi attempts, sandboxed)
# Machine-specific settings: $RLTLDR_ROOT/config.json (see config.example.json). Python envs: $RLTLDR_SERVE_PY
# (default ~/envs/serve/bin/python: vLLM, gateway, runner, driver) and $RLTLDR_TRAIN_PY (default ~/envs/train/bin/python).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)     # code (rltldr/ package)
ROOT="${RLTLDR_ROOT:-$HERE}"
RUN=$ROOT/run; LOGS=$ROOT/logs
mkdir -p "$RUN" "$LOGS"
export RLTLDR_ROOT=$ROOT
SERVE_PY=${RLTLDR_SERVE_PY:-$HOME/envs/serve/bin/python}
TRAIN_PY=${RLTLDR_TRAIN_PY:-$HOME/envs/train/bin/python}
ALL="vllm gateway runner trainer driver"

cfg() { PYTHONPATH="$HERE" "$SERVE_PY" -m rltldr.config get "$1"; }   # resolved config value

cmd_for() {
  case "$1" in
    vllm)     echo "$ROOT/serve.sh" ;;
    gateway)  echo "$SERVE_PY -m rltldr.gateway" ;;
    trainer)  local gpu; gpu=$(cfg trainer_gpu_uuid) || return 1
              [[ $gpu =~ ^GPU-[0-9a-fA-F-]+$ ]] || {
                echo "trainer_gpu_uuid='$gpu' is not a GPU UUID: set it in $ROOT/config.json (nvidia-smi -L)" >&2; return 1; }
              echo "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TRAIN_PY -m rltldr.trainer" ;;
    runner)   echo "$SERVE_PY -m rltldr.runner" ;;
    driver)   echo "$SERVE_PY -m rltldr.driver" ;;
    *) echo "unknown component $1" >&2; return 1 ;;
  esac
}

wait_http() {  # url timeout_s
  local t=0; until curl -sf "$1" >/dev/null 2>&1; do sleep 5; t=$((t+5)); [ $t -ge "$2" ] && return 1; done
  return 0
}

start_one() {
  local c=$1 cmd; cmd=$(cmd_for "$c") || return 1
  if [ -f "$RUN/$c.pid" ] && kill -0 "$(cat "$RUN/$c.pid")" 2>/dev/null; then echo "$c already running"; return 0; fi
  rm -f "$RUN/$c.stop"
  # supervisor loop: restart on exit (backoff) until a stop file appears
  setsid nohup bash -c "
    cd $ROOT
    while [ ! -f $RUN/$c.stop ]; do
      echo \"=== [\$(date -Is)] starting $c\" >> $LOGS/$c.log
      $cmd >> $LOGS/$c.log 2>&1 &
      echo \$! > $RUN/$c.child.pid; wait \$!; rc=\$?
      echo \"=== [\$(date -Is)] $c exited rc=\$rc\" >> $LOGS/$c.log
      [ -f $RUN/$c.stop ] && break
      sleep 15
    done" > /dev/null 2>&1 &
  echo $! > "$RUN/$c.pid"
  echo "started $c (supervisor pid $(cat "$RUN/$c.pid"))"
  case "$c" in
    vllm)    wait_http http://127.0.0.1:8000/v1/models 1800 && echo "  vllm ready" || echo "  vllm NOT ready (see logs/vllm.log)";;
    gateway) wait_http http://127.0.0.1:8100/health 120 && echo "  gateway ready" || echo "  gateway NOT ready";;
    runner)  wait_http http://127.0.0.1:8200/control/state 900 && echo "  runner ready" || echo "  runner NOT ready";;
  esac
}

stop_one() {
  local c=$1
  touch "$RUN/$c.stop"
  if [ -f "$RUN/$c.child.pid" ]; then
    local p; p=$(cat "$RUN/$c.child.pid")
    if kill -0 "$p" 2>/dev/null; then
      kill -TERM "$p" 2>/dev/null
      # driver: SIGTERM finishes the current attempt (can take ~10 min); others get 60 s
      local lim=60; [ "$c" = driver ] && lim=1800
      local t=0; while kill -0 "$p" 2>/dev/null && [ $t -lt $lim ]; do sleep 2; t=$((t+2)); done
      kill -0 "$p" 2>/dev/null && { echo "  force-killing $c"; pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; }
    fi
  fi
  [ -f "$RUN/$c.pid" ] && kill "$(cat "$RUN/$c.pid")" 2>/dev/null
  rm -f "$RUN/$c.pid" "$RUN/$c.child.pid"
  echo "stopped $c"
}

status() {
  for c in $ALL; do
    if [ -f "$RUN/$c.child.pid" ] && kill -0 "$(cat "$RUN/$c.child.pid")" 2>/dev/null; then
      printf "%-9s RUNNING (pid %s)\n" "$c" "$(cat "$RUN/$c.child.pid")"
    else printf "%-9s stopped\n" "$c"; fi
  done
  echo; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  echo; echo "gateway:"; curl -s http://127.0.0.1:8100/control/state 2>/dev/null | head -c 600; echo
  echo; echo "driver state:"; [ -f $ROOT/data/driver_state.json ] && $SERVE_PY -c "
import json; s=json.load(open('$ROOT/data/driver_state.json'))
print(f\"group {s['group']} attempt {s['k']}/8  attempts={s['n_attempts']} kept={s['n_kept']}  best val_bpb={s['parent']['val_bpb']:.6f} ({s['parent']['commit'][:7]})\")
print('insights this group:', [i['hint'][:90] for i in s['insights']])"
  echo; echo "last driver metrics:"; tail -n 3 $ROOT/data/metrics_driver.jsonl 2>/dev/null | cut -c1-400
  echo; echo "last trainer metrics:"; tail -n 2 $ROOT/data/metrics_trainer.jsonl 2>/dev/null | cut -c1-400
}

action=${1:-status}; shift || true
comps=${*:-$ALL}
case "$action" in
  start) for c in $comps; do start_one "$c"; done ;;
  stop)  rev=""; for c in $comps; do rev="$c $rev"; done; for c in $rev; do stop_one "$c"; done ;;
  restart) for c in $comps; do stop_one "$c"; start_one "$c"; done ;;
  status) status ;;
  logs) tail -n 100 -f "$LOGS/${1:-driver}.log" ;;
  report) /usr/bin/python3 "$ROOT/tools/report.py" --data "$(cfg data)" "$@" ;;
  *) echo "usage: $0 start|stop|restart|status|logs|report [component...]"; exit 1 ;;
esac
