#!/usr/bin/env bash
# Frozen-policy reruns in the training-time harness: v5 (no updates), one experiment per fresh pi session,
# history table + keep rule with confirmation re-runs, groups of 8 — arm x1 WITH insights, arm x2 WITHOUT
# insights, each on its own GPU. Each arm is an independent instance of rltldr/{gateway,runner,driver} configured
# by data/h2h/<arm>/rl_config.json (ports, sockets, data, canon repo, GPU); agents run in tools/h2h_sandbox.sh
# (arms from data/frz/arms.json), training runs with ar_run --isolate. vLLM stays under ./ctl.sh.
#   FRZ_GPU_x1=<uuid>:<minor> FRZ_GPU_x2=<uuid>:<minor> ./ctl_frz.sh init [--seed <commit>]
#                                    write data/frz/arms.json and data/h2h/<arm>/rl_config.json from the templates
#                                    in examples/frz/ (@RLTLDR_ROOT@ -> the project root; existing files are kept,
#                                    FORCE=1 rewrites them). --seed: also create each arm's seed repo (the
#                                    rl_config branch at <commit> of canon.git), from which the driver builds the
#                                    arm's canon.git on its first start
#   ./ctl_frz.sh start|stop|restart|status|logs [component...]
#   ./ctl_frz.sh baseline <arm>      one baseline run of the start commit (needed once, before the driver)
# Components: gateway-x1 runner-x1 driver-x1 gateway-x2 runner-x2 driver-x2
# Env: RLTLDR_ROOT (project root; default: this script's directory), RLTLDR_SERVE_PY (python of the serving env,
# default ~/envs/serve/bin/python), STOP_TIMEOUT (seconds a driver may take to stop).
set -uo pipefail
CODE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
ROOT=${RLTLDR_ROOT:-$CODE}
RUN=$ROOT/run/frz; LOGS=$ROOT/logs/frz
mkdir -p "$RUN" "$LOGS"
export RLTLDR_ROOT=$ROOT H2H_CONFIG=$ROOT/data/frz/arms.json
PY=${RLTLDR_SERVE_PY:-$HOME/envs/serve/bin/python}
SYS_PY=/usr/bin/python3
ARMS="x1 x2"
ALL="gateway-x1 runner-x1 gateway-x2 runner-x2 driver-x1 driver-x2"

arm_of() { echo "${1#*-}"; }
cfg_of() { echo "$ROOT/data/h2h/$(arm_of "$1")/rl_config.json"; }
port_of() { $SYS_PY -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$(cfg_of "$1")" "$2"; }

cmd_for() {
  case "${1%%-*}" in
    gateway) echo "env RLTLDR_CONFIG=$(cfg_of "$1") $PY -m rltldr.gateway" ;;
    runner)  echo "env RLTLDR_CONFIG=$(cfg_of "$1") $PY -m rltldr.runner" ;;
    driver)  echo "env RLTLDR_CONFIG=$(cfg_of "$1") $PY -m rltldr.driver" ;;
    *) echo "unknown component $1" >&2; return 1 ;;
  esac
}

wait_http() { local t=0; until curl -sf "$1" >/dev/null 2>&1; do sleep 5; t=$((t+5)); [ $t -ge "$2" ] && return 1; done; return 0; }

start_one() {
  local c=$1 cmd; cmd=$(cmd_for "$c") || return 1
  if [ -f "$RUN/$c.pid" ] && kill -0 "$(cat "$RUN/$c.pid")" 2>/dev/null; then echo "$c already running"; return 0; fi
  [ -f "$(cfg_of "$c")" ] || { echo "$c: $(cfg_of "$c") missing (run ./ctl_frz.sh init)" >&2; return 1; }
  rm -f "$RUN/$c.stop"
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
  echo "started $c"
  case "${c%%-*}" in
    gateway) wait_http "http://127.0.0.1:$(port_of "$c" gateway_port)/health" 120 && echo "  $c ready" || echo "  $c NOT ready";;
    runner)  wait_http "http://127.0.0.1:$(port_of "$c" runner_port)/control/state" 900 && echo "  $c ready" || echo "  $c NOT ready";;
  esac
}

stop_one() {
  local c=$1
  touch "$RUN/$c.stop"
  if [ -f "$RUN/$c.child.pid" ]; then
    local p; p=$(cat "$RUN/$c.child.pid")
    if kill -0 "$p" 2>/dev/null; then
      kill -TERM "$p" 2>/dev/null
      local lim=60; [ "${c%%-*}" = driver ] && lim=${STOP_TIMEOUT:-1800}
      local t=0; while kill -0 "$p" 2>/dev/null && [ $t -lt $lim ]; do sleep 2; t=$((t+2)); done
      kill -0 "$p" 2>/dev/null && { echo "  force-killing $c"; pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; }
    fi
  fi
  [ -f "$RUN/$c.pid" ] && kill "$(cat "$RUN/$c.pid")" 2>/dev/null
  rm -f "$RUN/$c.pid" "$RUN/$c.child.pid"
  echo "stopped $c"
}

baseline() {   # one unmodified run of the start commit, recorded as attempt "baseline" in the arm's ledger
  local a=$1 c="runner-$1"; local port; port=$(port_of "$c" runner_port)
  local commit; commit=$(git -C "$ROOT/data/h2h/$a/canon.git" rev-parse "$(port_of "$c" branch)")
  curl -sf -X POST "http://127.0.0.1:$port/control/attempt" -H 'Content-Type: application/json' \
       -d "{\"id\":\"baseline\",\"parent_commit\":\"$commit\"}" >/dev/null || { echo "attempt failed"; return 1; }
  $PY - "$port" "$ROOT/data/h2h/$a/canon.git" "$commit" <<'EOF'
import json, subprocess, sys, urllib.request
port, bare, commit = sys.argv[1:4]
src = subprocess.run(["git", "-C", bare, "show", f"{commit}:train.py"], capture_output=True, text=True, check=True).stdout
req = urllib.request.Request(f"http://127.0.0.1:{port}/run", data=json.dumps({"desc": "baseline", "train_py": src}).encode(),
                             headers={"Content-Type": "application/json"})
print(json.loads(urllib.request.urlopen(req, timeout=None).read())["output"])
EOF
  curl -sf -X POST "http://127.0.0.1:$port/control/attempt_end" -H 'Content-Type: application/json' -d '{"id":"baseline"}' >/dev/null
}

render_configs() {   # examples/frz/*.json -> data/frz/arms.json, data/h2h/<arm>/rl_config.json
  $SYS_PY - "$ROOT" "$CODE/examples/frz" "$ARMS" "${FORCE:-0}" <<'EOF'
import json, os, sys
root, ex, arms, force = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4] == "1"
gpu = {}
for a in arms:
    uuid, minor = os.environ[f"FRZ_GPU_{a}"].rsplit(":", 1)
    gpu[a] = (uuid, int(minor))


def render(src, dst, fill):
    if os.path.exists(dst) and not force:
        print(f"kept {dst} (exists; FORCE=1 rewrites it)")
        return
    with open(src) as f:
        d = {k: v for k, v in json.loads(f.read().replace("@RLTLDR_ROOT@", root)).items() if not k.startswith("_")}
    fill(d)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".tmp", "w") as f:
        f.write(json.dumps(d, indent=1) + "\n")
    os.replace(dst + ".tmp", dst)
    print(f"wrote {dst}")


def fill_arms(d):
    for arm in d["arms"]:
        arm["gpu_uuid"], arm["gpu_minor"] = gpu[arm["name"]]


def fill_rl(a):
    def fill(d):
        d["agent_gpu_uuid"], d["agent_gpu_minor"] = gpu[a]
    return fill


render(os.path.join(ex, "arms.json"), os.path.join(root, "data", "frz", "arms.json"), fill_arms)
for a in arms:
    render(os.path.join(ex, f"{a}.rl_config.json"), os.path.join(root, "data", "h2h", a, "rl_config.json"), fill_rl(a))
EOF
}

init() {
  local seed="" a v d branch
  if [ "${1:-}" = --seed ]; then seed=${2:-}; [ -n "$seed" ] || { echo "init: --seed needs a commit" >&2; return 1; }; fi
  for a in $ARMS; do
    v="FRZ_GPU_$a"
    [[ ${!v:-} =~ ^[^:]+:[0-9]+$ ]] || {
      echo "init: set $v=<gpu uuid>:<gpu minor> (uuid from nvidia-smi -L; minor = N of /dev/nvidiaN)" >&2; return 1; }
    export "${v?}"
  done
  render_configs || return 1
  for a in $ARMS; do
    d=$ROOT/data/h2h/$a
    mkdir -p "$d/rl"
    # tools/frz_dashboard reads the h2h arm layout (<arm>/ledger.jsonl); the RL harness writes rl/ledger.jsonl
    [ -e "$d/ledger.jsonl" ] || [ -L "$d/ledger.jsonl" ] || ln -s rl/ledger.jsonl "$d/ledger.jsonl"
    [ -n "$seed" ] || continue
    if [ -e "$d/repo" ] || [ -e "$d/canon.git" ]; then echo "kept $d/repo (the arm is already seeded)"; continue; fi
    branch=$(port_of "x-$a" branch)
    rm -rf "$d/repo.new"
    if git init -q "$d/repo.new" && git -C "$d/repo.new" fetch -q --no-tags "$ROOT/canon.git" "$seed" &&
       git -C "$d/repo.new" checkout -q -b "$branch" FETCH_HEAD && mv "$d/repo.new" "$d/repo"; then
      echo "seeded $d/repo: $branch at $(git -C "$d/repo" rev-parse --short HEAD)"
    else
      rm -rf "$d/repo.new"; echo "init: seeding $d/repo at $seed failed" >&2; return 1
    fi
  done
  echo "next: ./ctl_frz.sh start runner-x1 runner-x2, ./ctl_frz.sh baseline <arm> for each arm, then ./ctl_frz.sh start"
}

status() {
  for c in $ALL; do
    if [ -f "$RUN/$c.child.pid" ] && kill -0 "$(cat "$RUN/$c.child.pid")" 2>/dev/null; then
      printf "%-11s RUNNING (pid %s)\n" "$c" "$(cat "$RUN/$c.child.pid")"
    else printf "%-11s stopped\n" "$c"; fi
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  for a in $ARMS; do
    [ -f "$ROOT/data/h2h/$a/rl_config.json" ] || { echo "$a not initialized (./ctl_frz.sh init)"; continue; }
    $SYS_PY -c "
import json,os
d='$ROOT/data/h2h/$a/rl'; c=json.load(open('$ROOT/data/h2h/$a/rl_config.json'))
s=json.load(open(d+'/driver_state.json')) if os.path.exists(d+'/driver_state.json') else None
print('$a', 'insights' if c['insights_enabled'] else 'no-insights', 'GPU', c['agent_gpu_minor'], end=' ')
print(f\"group {s['group']} k {s['k']} attempts={s['n_attempts']} kept={s['n_kept']} best={s['parent']['val_bpb']:.6f} insights_now={len(s['insights'])}\" if s else 'no driver state yet')"
  done
}

action=${1:-status}; shift || true
case "$action" in
  init)    init "$@" ;;
  start)   for c in ${*:-$ALL}; do start_one "$c"; done ;;
  stop)    comps=${*:-$ALL}; rev=""; for c in $comps; do rev="$c $rev"; done; for c in $rev; do stop_one "$c"; done ;;
  restart) for c in ${*:-$ALL}; do stop_one "$c"; start_one "$c"; done ;;
  status)  status ;;
  logs)    tail -n 100 -f $(for c in ${*:-$ALL}; do echo "$LOGS/$c.log"; done) ;;
  baseline) baseline "$1" ;;
  *) echo "usage: $0 init [--seed <commit>]|start|stop|restart|status|logs|baseline [component|arm...]"; exit 1 ;;
esac
