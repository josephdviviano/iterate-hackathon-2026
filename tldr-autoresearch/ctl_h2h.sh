#!/usr/bin/env bash
# Control script for the head-to-head (h2h) run: base model vs RLTL;DR v5, two continuous autoresearch
# sessions that share the vLLM server of ./ctl.sh; each arm has its own nanochat GPU, runner, gateway socket,
# sandbox, pi session and repo (see rltldr/h2h_config.py; machine settings in h2h_config.json, template
# h2h_config.example.json). One-time setup: tools/h2h/make_adapters.py, then tools/h2h/setup.py --create-start.
#   ./ctl_h2h.sh start   [component...]  start, supervised (auto-restart with back-off); default: all, in order
#   ./ctl_h2h.sh stop    [component...]  stop; default: all, both agents first (at the same time), then the rest
#   ./ctl_h2h.sh restart [component...]  stop, then start in dependency order
#   ./ctl_h2h.sh status                  processes, vLLM, GPUs, gateway + runner state, per-arm progress
#   ./ctl_h2h.sh logs    [component...]  tail -f the component logs (default: all)
#   ./ctl_h2h.sh dashboard               build dashboard_h2h/index.html once (tools/h2h_dashboard/watch.sh: live)
# Components (start order): gateway | runner-base | runner-v5 | agent-base | agent-v5. Both agents are started
# back to back, only after the gateway and their runners answer. vLLM is never started or stopped here.
# Stopping a runner kills its in-flight training run (the ledger records it as interrupted).
# Env: RLTLDR_SERVE_PY (python of the serving env, default ~/envs/serve/bin/python).
# Test hooks: RLTLDR_ROOT (data root), H2H_CONFIG (ports, arms, vllm_url), H2H_CMD_<component, '-' -> '_'>
# (command to run instead of the real component), H2H_STOP_TIMEOUT (seconds before force-kill), FORCE=1 (skip
# the agent preconditions).
set -uo pipefail
CODE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
ROOT=${RLTLDR_ROOT:-$CODE}
export RLTLDR_ROOT=$ROOT
RUN=$ROOT/run/h2h; LOGS=$ROOT/logs/h2h
SERVE_PY=${RLTLDR_SERVE_PY:-$HOME/envs/serve/bin/python}
SYS_PY=/usr/bin/python3

# shared config (ports, arms, paths) from rltldr/h2h_config.py, honouring RLTLDR_ROOT / H2H_CONFIG
CFG_SH=$(PYTHONPATH=$CODE $SYS_PY - <<'EOF'
import re, shlex
from rltldr.h2h_config import load_h2h_config
c = load_h2h_config()
print("ARMS=" + shlex.quote(" ".join(a.name for a in c.arms)))
print(f"GW_PORT={int(c.gateway_port)}")
print("VLLM_URL=" + shlex.quote(c.vllm_url.rstrip("/")))
for a in c.arms:
    assert re.fullmatch(r"[A-Za-z0-9]+", a.name), f"arm name {a.name!r} must be alphanumeric"
    for k, v in (("RPORT", int(a.runner_port)), ("GPUMINOR", int(a.gpu_minor)), ("GPUUUID", a.gpu_uuid),
                 ("REPO", a.repo), ("SESS", a.session_dir), ("ADAPTER", a.adapter_dir),
                 ("SERVED", a.served_model)):
        print(f"{k}_{a.name}=" + shlex.quote(str(v)))
EOF
) || { echo "cannot read rltldr/h2h_config.py" >&2; exit 1; }
eval "$CFG_SH"
RUNNERS=""; AGENTS=""
for a in $ARMS; do RUNNERS="$RUNNERS runner-$a"; AGENTS="$AGENTS agent-$a"; done
ALL="gateway$RUNNERS$AGENTS"
arm_of() { echo "${1#*-}"; }
var() { local v="${1}_$2"; echo "${!v}"; }

cmd_for() {
  local hook="H2H_CMD_${1//-/_}"
  if [ -n "${!hook:-}" ]; then echo "${!hook}"; return 0; fi
  case "$1" in
    gateway)  echo "$SERVE_PY -m rltldr.h2h_gateway" ;;
    runner-*) echo "$SERVE_PY -m rltldr.h2h_runner --arm $(arm_of "$1")" ;;
    agent-*)  echo "$SERVE_PY -m rltldr.h2h_supervisor --arm $(arm_of "$1")" ;;
  esac
}

valid_comp() { case " $ALL " in *" $1 "*) return 0;; esac; echo "unknown component '$1' (have: $ALL)" >&2; return 1; }
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
pid_of() { [ -f "$RUN/$1.$2" ] && cat "$RUN/$1.$2" 2>/dev/null; }     # pid_of <component> pid|child.pid
http_ok() { curl -sf -m 5 "$1" >/dev/null 2>&1; }
wait_http() {  # url timeout_s component
  local t=0
  until http_ok "$1"; do
    alive "$(pid_of "$3" pid)" || return 1
    sleep 3; t=$((t + 3)); [ $t -ge "$2" ] && return 1
  done
  return 0
}
ready_url() {
  case "$1" in
    gateway)  echo "http://127.0.0.1:$GW_PORT/health" ;;
    runner-*) echo "http://127.0.0.1:$(var RPORT "$(arm_of "$1")")/control/state" ;;
  esac
}

# ---- preconditions ---------------------------------------------------------------------------------------
check_old_system() {   # the RLTL;DR runner/trainer (./ctl.sh) use the same GPUs: never run both systems at once
  local c busy=""
  for c in runner trainer driver gateway; do alive "$(cat "$ROOT/run/$c.child.pid" 2>/dev/null)" && busy="$busy $c"; done
  [ -z "$busy" ] && return 0
  echo "refusing: old RLTL;DR components are running:$busy (stop them with ./ctl.sh stop$busy)" >&2; return 1
}
check_vllm() {
  http_ok "$VLLM_URL/v1/models" && return 0
  echo "refusing: vLLM does not answer at $VLLM_URL (it is managed by ./ctl.sh: ./ctl.sh start vllm)" >&2; return 1
}
check_adapters() {
  local a d ok=0
  for a in $ARMS; do
    d=$(var ADAPTER "$a")
    if [ ! -f "$d/adapter_config.json" ] || ! ls "$d"/*.safetensors >/dev/null 2>&1; then
      echo "refusing: adapter for arm $a missing at $d (run: $SERVE_PY $CODE/tools/h2h/make_adapters.py)" >&2; ok=1
    fi
  done
  return $ok
}
check_gpu() {   # runner-<arm>: its GPU must be configured in h2h_config.json
  local a; a=$(arm_of "$1")
  [ -n "$(var GPUUUID "$a")" ] && [ "$(var GPUMINOR "$a")" -ge 0 ] && return 0
  echo "refusing $1: gpu_uuid/gpu_minor of arm $a not set (h2h_config.json; see h2h_config.example.json)" >&2; return 1
}
check_agent() {   # agent-<arm>: its repo + pi config exist, the gateway and its runner answer
  local a; a=$(arm_of "$1")
  [ -d "$(var REPO "$a")/.git" ] && [ -d "$(var SESS "$a")/agent" ] || {
    echo "refusing $1: $(var REPO "$a") or $(var SESS "$a")/agent missing (run: python3 $CODE/tools/h2h/setup.py --create-start)" >&2; return 1; }
  [ "${FORCE:-0}" = 1 ] && return 0
  http_ok "$(ready_url gateway)" || { echo "refusing $1: gateway not answering on :$GW_PORT (start it first)" >&2; return 1; }
  http_ok "$(ready_url "runner-$a")" || { echo "refusing $1: runner-$a not answering (start it first)" >&2; return 1; }
}

# ---- start / stop ----------------------------------------------------------------------------------------
start_one() {
  local c=$1 cmd
  cmd=$(cmd_for "$c")
  if alive "$(pid_of "$c" pid)"; then echo "$c already running"; return 0; fi
  mkdir -p "$RUN" "$LOGS"
  rm -f "$RUN/$c.stop" "$RUN/$c.child.pid"; echo 0 > "$RUN/$c.restarts"
  # supervisor loop: restart on exit until a stop file appears; back-off 15 s, doubling (max 5 min) while the
  # component keeps dying within 5 min of its start
  setsid nohup bash -c '
    c=$1; cmd=$2; RUN=$3; LOGS=$4; cd "$5" || exit 1
    delay=0; n=0
    while [ ! -f "$RUN/$c.stop" ]; do
      echo "=== [$(date -Is)] starting $c" >> "$LOGS/$c.log"
      t0=$(date +%s)
      ( eval "exec $cmd" ) >> "$LOGS/$c.log" 2>&1 < /dev/null &
      echo $! > "$RUN/$c.child.pid"; wait $!; rc=$?
      echo "=== [$(date -Is)] $c exited rc=$rc" >> "$LOGS/$c.log"
      [ -f "$RUN/$c.stop" ] && break
      n=$((n + 1)); echo $n > "$RUN/$c.restarts"
      if [ $(( $(date +%s) - t0 )) -ge 300 ] || [ $delay -eq 0 ]; then delay=15; else delay=$((delay * 2)); fi
      [ $delay -gt 300 ] && delay=300
      echo "=== [$(date -Is)] restarting $c in ${delay}s (restart $n)" >> "$LOGS/$c.log"
      for ((i = 0; i < delay; i++)); do [ -f "$RUN/$c.stop" ] && break; sleep 1; done
    done
    rm -f "$RUN/$c.child.pid"' "h2h-$c" "$c" "$cmd" "$RUN" "$LOGS" "$CODE" > /dev/null 2>&1 < /dev/null &
  echo $! > "$RUN/$c.pid"
  echo "started $c (supervisor pid $!, log $LOGS/$c.log)"
  local url; url=$(ready_url "$c")
  if [ -n "$url" ]; then
    local lim=120; [[ $c == runner-* ]] && lim=1200          # first start: clone + uv sync of the workspace
    if wait_http "$url" $lim "$c"; then echo "  $c ready"; else echo "  $c NOT ready (see $LOGS/$c.log)"; return 1; fi
  fi
}

descendants() {  # all descendants of pid $1 (computed before anything is killed, so none escape by reparenting)
  ps -e -o pid=,ppid= | awk -v root="$1" '{ kids[$2] = kids[$2] " " $1 }
    END { q = root; while (q != "") { split(q, a, " "); q = ""; for (i in a) { n = split(kids[a[i]], k, " ");
          for (j = 1; j <= n; j++) { print k[j]; q = q " " k[j] } } } }'
}

signal_one() {   # stop file + SIGTERM to the component (its own orderly shutdown)
  local c=$1 p; touch "$RUN/$c.stop"
  p=$(pid_of "$c" child.pid); alive "$p" && kill -TERM "$p" 2>/dev/null
  return 0
}

finish_one() {   # wait for the component to exit, force-kill its whole tree after the timeout
  local c=$1 p lim t=0
  case "$c" in agent-*) lim=120;; runner-*) lim=90;; *) lim=30;; esac
  lim=${H2H_STOP_TIMEOUT:-$lim}
  p=$(pid_of "$c" child.pid)
  while alive "$p" && [ $t -lt "$lim" ]; do sleep 1; t=$((t + 1)); done
  if alive "$p"; then
    local tree; tree="$p $(descendants "$p" | tr '\n' ' ')"
    echo "  $c did not exit within ${lim}s: force-killing its process tree ($(echo $tree | wc -w) processes)"
    # sandboxed parts of the tree run as root: sudo; only pids from this component's own tree are named
    sudo -n kill -KILL $tree 2>/dev/null || kill -KILL $tree 2>/dev/null
  fi
  local sp; sp=$(pid_of "$c" pid)
  alive "$sp" && kill -TERM "$sp" 2>/dev/null
  t=0; while alive "$sp" && [ $t -lt 10 ]; do sleep 1; t=$((t + 1)); done
  rm -f "$RUN/$c.pid" "$RUN/$c.child.pid"
  echo "stopped $c"
}

stop_set() {   # stop the given components: agents first (together), then runners (together), then the gateway
  local phase c todo
  for phase in agent runner gateway; do
    todo=""
    for c in "$@"; do [[ $c == $phase* ]] && todo="$todo $c"; done
    [ -z "$todo" ] && continue
    for c in $todo; do signal_one "$c"; done
    for c in $todo; do finish_one "$c"; done
  done
}

start_set() {  # start the given components in dependency order; agents last, back to back
  local c ordered="" agents="" rc=0
  for c in $ALL; do case " $* " in *" $c "*) ordered="$ordered $c";; esac; done
  check_old_system || return 1
  for c in $ordered; do
    case "$c" in
      gateway) { check_vllm && check_adapters && start_one gateway; } || return 1 ;;
      runner-*) { check_gpu "$c" && start_one "$c"; } || rc=1 ;;
      agent-*) agents="$agents $c" ;;
    esac
  done
  [ -z "$agents" ] && return $rc
  check_vllm || return 1
  for c in $agents; do check_agent "$c" || return 1; done    # all or none: the arms start together
  for c in $agents; do start_one "$c"; done
  return $rc
}

# ---- status ----------------------------------------------------------------------------------------------
uptime_of() {
  local s; s=$(ps -o etimes= -p "$1" 2>/dev/null | tr -d ' ')
  [ -z "$s" ] && return
  printf "%dh%02dm" $((s / 3600)) $((s % 3600 / 60))
}

show_json() {  # compact view of a JSON state document: scalars on one line, one line per nested object
  $SYS_PY -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    print("  (no answer)"); sys.exit()
def fmt(v):
    if isinstance(v, float): return f"{v:.4g}"
    if isinstance(v, (dict, list)): return json.dumps(v, separators=(",", ":"))[:80]
    return str(v)
def show(d, ind="  "):
    if not isinstance(d, dict):
        print(ind + fmt(d)); return
    flat = [f"{k}={fmt(v)}" for k, v in d.items() if not isinstance(v, dict)]
    if flat: print(ind + " ".join(flat)[:400])
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{ind}{k}:"); show(v, ind + "  ")
show(d)'
}

status() {
  local c p sp
  echo "components:"
  for c in $ALL; do
    p=$(pid_of "$c" child.pid); sp=$(pid_of "$c" pid)
    if alive "$p"; then printf "  %-12s RUNNING  pid %-8s up %-7s restarts %s\n" "$c" "$p" "$(uptime_of "$p")" "$(cat "$RUN/$c.restarts" 2>/dev/null || echo 0)"
    elif alive "$sp"; then printf "  %-12s RESTARTING (supervisor pid %s, restarts %s)\n" "$c" "$sp" "$(cat "$RUN/$c.restarts" 2>/dev/null || echo 0)"
    else printf "  %-12s stopped\n" "$c"; fi
  done
  echo; printf "vLLM (%s): " "$VLLM_URL"
  curl -sf -m 5 "$VLLM_URL/v1/models" 2>/dev/null | $SYS_PY -c '
import json, sys
try:
    print("up · models: " + ", ".join(m.get("id", "?") for m in json.load(sys.stdin).get("data", [])))
except ValueError:
    print("DOWN")' || echo "DOWN"
  echo; echo "GPUs (index, used MiB, util):"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  local a apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null)
  for a in $ARMS; do
    printf "  arm %-5s GPU %s: " "$a" "$(var GPUMINOR "$a")"
    [ -n "$(var GPUUUID "$a")" ] || { echo "not configured"; continue; }
    echo "$apps" | grep -F "$(var GPUUUID "$a")" | awk -F', ' '{printf "pid %s (%s) ", $2, $3} END {if (!NR) printf "idle"; print ""}'
  done
  echo; echo "gateway state (:$GW_PORT):"
  curl -sf -m 5 "http://127.0.0.1:$GW_PORT/control/state" 2>/dev/null | show_json
  for a in $ARMS; do
    echo "runner-$a state (:$(var RPORT "$a")):"
    curl -sf -m 5 "$(ready_url "runner-$a")" 2>/dev/null | show_json
  done
  echo; echo "progress:"
  $SYS_PY "$CODE/tools/h2h_dashboard/build.py" --status 2>&1 | sed 's/^/  /'
}

# ---- main ------------------------------------------------------------------------------------------------
action=${1:-status}; shift || true
comps=${*:-$ALL}
case "$action" in
  start|stop|restart|logs) for c in $comps; do valid_comp "$c" || exit 1; done ;;
esac
case "$action" in
  start)   start_set $comps ;;
  stop)    stop_set $comps ;;
  restart) stop_set $comps; start_set $comps ;;
  status)  status ;;
  logs)    files=""; for c in $comps; do files="$files $LOGS/$c.log"; done; exec tail -n 50 -F $files ;;
  dashboard) exec $SYS_PY "$CODE/tools/h2h_dashboard/build.py" "$@" ;;
  *) echo "usage: $0 start|stop|restart|status|logs|dashboard [component...]   components: $ALL"; exit 1 ;;
esac
