#!/usr/bin/env bash
# Run the research agent (pi) as the calling user inside private mount + PID + network namespaces.
#   sandbox.sh <writable_session_dir> <cmd> [args...]
# Inside the sandbox:
#   * no GPU: every /dev/nvidia<N> is masked. Training runs are executed by the runner daemon outside
#     (./run.sh -> 127.0.0.1:8200 -> runner.sock), itself in a separate single-GPU sandbox (train_sandbox.sh).
#   * no network except loopback: 127.0.0.1:8100 -> gateway (chat only), 127.0.0.1:8200 -> runner (/run only).
#   * the whole filesystem is read-only (incl. /usr, /opt, $HOME, harness code, ledger, .git internals of
#     nothing but the agent's own throwaway clone) except the agent's repo clone and its pi session dir;
#     /tmp, /var/tmp and /dev/shm are private tmpfs.
#   * separate PID namespace (killed as a whole with the sandbox); the caller's uid/gid (taken with id -u / id -g
#     BEFORE sudo; refused for root) with no_new_privs and no capabilities (no sudo).
# Root: $RLTLDR_ROOT, else the directory above this script. The agent repo: $AGENT_REPO, else $ROOT/autoresearch.
set -euo pipefail
ROOT=${RLTLDR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_UID=$(id -u) RUN_GID=$(id -g)
[ "$RUN_UID" != 0 ] || { echo "sandbox.sh: run it as the unprivileged user, not as root" >&2; exit 2; }
SESSION_DIR=$1; shift
mkdir -p "$SESSION_DIR"
exec sudo -n --preserve-env unshare --mount --pid --net --fork --kill-child --mount-proc --propagation private -- bash -c '
  set -e
  ROOT="$1"; SESSION_DIR="$2"; REPO="$3"; RUN_UID="$4"; RUN_GID="$5"; shift 5
  [[ $RUN_UID =~ ^[1-9][0-9]*$ && $RUN_GID =~ ^[0-9]+$ ]] || { echo "sandbox.sh: bad uid/gid" >&2; exit 2; }
  source "$ROOT/tools/sandbox_lib.sh"
  lo_up
  mask_gpus
  lockdown_fs "$REPO" "$SESSION_DIR"
  [ -d "$REPO/.venv" ] && mount --bind "$REPO/.venv" "$REPO/.venv" && mount -o remount,bind,ro "$REPO/.venv"
  setpriv --reuid="$RUN_UID" --regid="$RUN_GID" --init-groups --no-new-privs --inh-caps=-all -- \
    /usr/bin/python3 -I "$ROOT/tools/uds_forward.py" "8100:$ROOT/run/gateway.sock" "8200:$ROOT/run/runner.sock" &
  for i in $(seq 50); do (exec 3<>/dev/tcp/127.0.0.1/8100) 2>/dev/null && break; sleep 0.1; done
  exec setpriv --reuid="$RUN_UID" --regid="$RUN_GID" --init-groups --no-new-privs --inh-caps=-all -- "$@"
' sandbox "$ROOT" "$SESSION_DIR" "${AGENT_REPO:-$ROOT/autoresearch}" "$RUN_UID" "$RUN_GID" "$@"
