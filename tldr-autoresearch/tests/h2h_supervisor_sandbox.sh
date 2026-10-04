#!/usr/bin/env bash
# Test stand-in for tools/h2h_sandbox.sh, selected through H2H_SANDBOX by tests/test_h2h_supervisor.py.
#   h2h_supervisor_sandbox.sh <arm> <cmd> [args...]
# Same calling convention and process structure as the real sandbox (sudo unshare: private mount + pid + net
# namespaces, loopback only, the invoking user's uid/gid with no_new_privs, 127.0.0.1:8100 / :8200 forwarded to
# Unix sockets), but without the filesystem lockdown, and the forwarded sockets come from the environment instead
# of the arm:
#   H2H_TEST_GATEWAY_SOCK, H2H_TEST_RUNNER_SOCK   fake gateway / runner sockets
#   H2H_TEST_ARGV_LOG                             optional: every invocation's argv is appended as one JSON line
set -euo pipefail
TOOLS=$(cd "$(dirname "$0")/../tools" && pwd)
RUID=$(id -u) RGID=$(id -g)       # taken before sudo: the command runs as the invoking user again
if [ -n "${H2H_TEST_ARGV_LOG:-}" ]; then
  /usr/bin/python3 -I -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@" >> "$H2H_TEST_ARGV_LOG"
fi
shift   # the arm: the sockets come from the environment here
exec sudo -n unshare --mount --pid --net --fork --kill-child --mount-proc --propagation private -- bash -c '
  set -e
  TOOLS="$1"; GW="$2"; RS="$3"; RUID="$4"; RGID="$5"; shift 5
  source "$TOOLS/sandbox_lib.sh"
  lo_up
  setpriv --reuid="$RUID" --regid="$RGID" --init-groups --no-new-privs --inh-caps=-all -- \
    /usr/bin/python3 -I "$TOOLS/uds_forward.py" "8100:$GW" "8200:$RS" &
  for i in $(seq 50); do (exec 3<>/dev/tcp/127.0.0.1/8100) 2>/dev/null && break; sleep 0.1; done
  exec setpriv --reuid="$RUID" --regid="$RGID" --init-groups --no-new-privs --inh-caps=-all -- "$@"
' h2h-test-sandbox "$TOOLS" "$H2H_TEST_GATEWAY_SOCK" "$H2H_TEST_RUNNER_SOCK" "$RUID" "$RGID" "$@"
