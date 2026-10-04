#!/usr/bin/env bash
# Run a command as the calling user inside the sandbox of one head-to-head (h2h) arm.
#   h2h_sandbox.sh <arm> <cmd> [args...]
#
# The jail of tools/sandbox.sh (private mount + PID + network namespaces, loopback only, every /dev/nvidia*
# masked, the whole filesystem read-only, private tmpfs /tmp /var/tmp /dev/shm, the caller's uid/gid (id -u / id -g,
# taken before sudo; refused for root) with no_new_privs and no capabilities; 127.0.0.1:8100 -> the arm's gateway
# socket, :8200 -> the arm's runner socket) plus information hiding: the caller's home ($HOME, called ~ below) is
# covered by a read-only tmpfs and only these paths are put back:
#   ~/.local/{bin,lib,opt,share/uv}  ro  pi, node, uv and the venvs' interpreter (not ~/.local/share/{claude,
#                                        jupyter,pki}, ~/.local/state: credentials, server tokens)
#   ~/.cache/autoresearch, ~/.cache/uv  ro  data shards + tokenizer; uv cache (if present)
#   $ROOT/tools/run_client.py        ro  run.sh calls it (at this host path)
#   $ROOT/tools/uds_forward.py       ro  the sandbox's loopback forwarder (started after the covering)
#                                        Only these two FILES of tools/: the rest (dashboards, READMEs, reports,
#                                        the harness, this script, the build spec) names the experiment and its
#                                        arms. Files are bound by inode: a running sandbox keeps the old version
#                                        of a file that is replaced on the host (edits in place are seen).
#   the arm's socket dir(s)          ro
#   the arm's repo                   rw  (its .venv read-only)
#   the arm's session dir            rw  pi agent dir + session files
# Nothing else under ~ is visible: no run data, canon.git, the other arm, logs, ~/.claude*, ~/.pi,
# models, envs, scratch. Arm paths come from rltldr/h2h_config.py (honours RLTLDR_ROOT and H2H_CONFIG); the
# socket paths can be overridden with H2H_GATEWAY_SOCK / H2H_RUNNER_SOCK (tests). Only ~ is hidden: keep the project
# root ($RLTLDR_ROOT) under it, as on the source machine, or run data, the other arm and logs stay visible (ro).
#
# Layout inside (H2H_SANDBOX_LAYOUT):
#   neutral (default)  the arm-specific paths appear at arm-independent places, identical in both arms:
#                        repo -> ~/autoresearch, session dir -> ~/.session,
#                        sockets -> ~/.sock/{gateway,runner}/<name>.sock
#                      pi writes its cwd into the system prompt, so the host path (.../data/h2h/<arm>/repo) would
#                      tell the model which arm it is in and make the two arms' prompts differ. Host paths of the
#                      repo and session dir in <cmd> args (including K=V arguments of env) and the caller's cwd are
#                      rewritten to the inside paths, so callers keep passing host paths.
#   host               everything appears at its host path, nothing is rewritten.
#
# The command starts in the (rewritten) caller's cwd with a clean environment: HOME (= ~), PATH (~/.local/bin and the
# system dirs), LANG, USER, LOGNAME (the caller's user name);
# pass anything else through the command (`/usr/bin/env -i K=V ... pi ...`). PID 1 inside is a bash that runs the
# command in the foreground (stdin/stdout pass through untouched), reaps orphans (a session runs for days) and
# exits with the command's status. Killing the sudo or the unshare process (SIGKILL via `sudo kill`) takes the
# whole sandbox down: sudo -> unshare -> PID 1 are linked by parent-death signals (setpriv --pdeathsig; dropping
# to the caller's uid clears the one unshare --kill-child sets, so PID 1 re-arms it with `--pdeathsig keep`).
# Give it pipes (or /dev/null) as stdin/stdout/stderr: an inherited fd of a host file can be reopened inside through
# /proc/self/fd/N. Not hidden: /proc/self/mountinfo shows each bind mount's host source path, i.e. the arm's dir
# name (.../data/h2h/<arm>/repo): its `root` field is the path inside the source filesystem, which no bind or fd
# trick changes; only arm dirs with arm-neutral names on the host would close it.
#
# Hiding technique: every path to re-expose is opened as an fd (directory or file) by root inside the new mount
# namespace BEFORE anything is covered; after the lockdown a tmpfs is mounted over ~, each path is
# bind-mounted from /proc/self/fd/N onto its inside path (an empty dir or file created in the tmpfs; mount -c:
# canonicalising the fd link would bind the empty tmpfs entry) and the fds are closed. This is the staging-rbind
# technique without a staging tree (nothing to lazily unmount) and it also works for paths under /tmp (tests),
# which the private /tmp would otherwise shadow. The working
# directory is re-resolved at the end: a cwd inherited from before the covering would still point into the hidden
# tree (`cd ..` from it would list rltldr/data).
set -euo pipefail
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
HIDE=${HOME:-}; HIDE=${HIDE%/}         # the caller's home: covered by the tmpfs
RUN_UID=$(id -u) RUN_GID=$(id -g) RUN_USER=$(id -un)
LAYOUT=${H2H_SANDBOX_LAYOUT:-neutral}
N_REPO=$HIDE/autoresearch N_SESSION=$HIDE/.session N_SOCK=$HIDE/.sock    # neutral inside paths

die() { echo "h2h_sandbox: $*" >&2; exit 2; }
[ "$RUN_UID" != 0 ] || die "run it as the unprivileged user (it drops to the caller's uid), not as root"
[[ $HIDE == /?* && $HIDE != *[$'\t\n']* && -d $HIDE ]] || die "HOME must be a clean absolute dir other than /: '$HIDE'"
[[ $RUN_USER =~ ^[A-Za-z0-9._-]+$ ]] || die "bad user name: $RUN_USER"
[ $# -ge 2 ] || die "usage: h2h_sandbox.sh <arm> <cmd> [args...]"
ARM=$1; shift
[[ $ARM =~ ^[A-Za-z0-9_-]+$ ]] || die "bad arm name: $ARM"
case $LAYOUT in neutral|host) ;; *) die "H2H_SANDBOX_LAYOUT must be 'neutral' or 'host', not '$LAYOUT'" ;; esac

CFG=$(/usr/bin/python3 -I - "$TOOLS/../rltldr" "$ARM" <<'PY'
import shlex, sys
sys.path.insert(0, sys.argv[1])
from h2h_config import load_h2h_config
try:
    a = load_h2h_config().arm(sys.argv[2])
except KeyError as e:
    sys.exit(f"h2h_sandbox: {e.args[0]}")
for k, v in (("REPO", a.repo), ("SESSION_DIR", a.session_dir), ("GATEWAY_SOCK", a.gateway_sock),
             ("RUNNER_SOCK", a.runner_sock)):
    print(f"{k}={shlex.quote(v)}")
PY
) || exit 2
eval "$CFG"
GATEWAY_SOCK=${H2H_GATEWAY_SOCK:-$GATEWAY_SOCK}
RUNNER_SOCK=${H2H_RUNNER_SOCK:-$RUNNER_SOCK}
for p in "$REPO" "$SESSION_DIR" "$GATEWAY_SOCK" "$RUNNER_SOCK"; do
  [[ $p == /* && $p != *[$'\t\n']* ]] || die "not a clean absolute path: $p"
done
[ -d "$REPO" ] || die "arm repo $REPO does not exist (tools/h2h/setup.py creates it)"
GW_DIR=$(dirname "$GATEWAY_SOCK") RN_DIR=$(dirname "$RUNNER_SOCK")
mkdir -p "$SESSION_DIR" "$GW_DIR" "$RN_DIR"

# what to put back: "mode<TAB>host path<TAB>inside path", sorted by inside path (parents before children)
EXPOSE=()
expose() { [ -e "$2" ] && EXPOSE+=("$1"$'\t'"$2"$'\t'"${3:-$2}"); return 0; }
for d in .local/bin .local/lib .local/opt .local/share/uv .cache/autoresearch .cache/uv; do expose ro "$HIDE/$d"; done
for f in run_client.py uds_forward.py; do   # single files: nothing else of tools/ (see the header)
  [ -f "$TOOLS/$f" ] || die "missing $TOOLS/$f"
  expose ro "$TOOLS/$f"
done
if [ "$LAYOUT" = neutral ]; then
  expose ro "$GW_DIR" "$N_SOCK/gateway"
  expose ro "$RN_DIR" "$N_SOCK/runner"
  expose rw "$REPO" "$N_REPO"
  expose rw "$SESSION_DIR" "$N_SESSION"
  GW_IN=$N_SOCK/gateway/$(basename "$GATEWAY_SOCK") RN_IN=$N_SOCK/runner/$(basename "$RUNNER_SOCK")
  REPO_IN=$N_REPO
  # host repo/session paths in the command and the cwd -> inside paths (whole path components only)
  mapfile -d '' -t ARGS < <(/usr/bin/python3 -I - "$REPO" "$N_REPO" "$SESSION_DIR" "$N_SESSION" "$PWD" "$@" <<'PY'
import os, re, sys
pairs = []
for src, dst in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    for s in {src.rstrip("/"), os.path.realpath(src)}:
        pairs.append((re.compile(r"(?<![\w./-])" + re.escape(s) + r"(?![\w.-])"), dst))
out = []
for arg in sys.argv[5:]:
    for rx, dst in pairs:
        arg = rx.sub(lambda m: dst, arg)
    out.append(arg)
sys.stdout.write("\0".join(out) + "\0")
PY
  )
  [ ${#ARGS[@]} -eq $(($# + 1)) ] || die "argument rewriting failed"
  CWD_IN=${ARGS[0]}
  set -- "${ARGS[@]:1}"
else
  expose ro "$GW_DIR"
  [ "$RN_DIR" = "$GW_DIR" ] || expose ro "$RN_DIR"
  expose rw "$REPO"
  expose rw "$SESSION_DIR"
  GW_IN=$GATEWAY_SOCK RN_IN=$RUNNER_SOCK REPO_IN=$REPO CWD_IN=$PWD
fi
for s in "$GW_IN" "$RN_IN"; do [ ${#s} -le 107 ] || die "socket path too long for AF_UNIX (108 bytes): $s"; done
EXPOSE_SPEC=$(printf '%s\n' "${EXPOSE[@]}" | LC_ALL=C sort -t$'\t' -k3,3)

exec sudo -n setpriv --pdeathsig KILL unshare --mount --pid --net --fork --kill-child --mount-proc --propagation private -- /bin/bash -c '
  set -Eeuo pipefail
  trap '\''echo "h2h_sandbox: setup failed (line $LINENO: $BASH_COMMAND)" >&2'\'' ERR
  TOOLS=$1 HIDE=$2 EXPOSE_SPEC=$3 REPO_IN=$4 CWD_IN=$5 GW_IN=$6 RN_IN=$7 RUN_UID=$8 RUN_GID=$9 RUN_USER=${10}; shift 10
  [[ $RUN_UID =~ ^[1-9][0-9]*$ && $RUN_GID =~ ^[0-9]+$ ]] || { echo "h2h_sandbox: bad uid/gid" >&2; exit 2; }
  source "$TOOLS/sandbox_lib.sh"
  mkpath() {   # create a directory and its missing parents, owned by the caller like the real tree
    local d="" c
    IFS=/ read -ra parts <<< "${1#/}"
    for c in "${parts[@]}"; do d="$d/$c"; [ -d "$d" ] || install -d -o "$RUN_UID" -g "$RUN_GID" -m 0755 "$d"; done
  }
  mkmountpoint() {   # <fd> <path>: an empty directory or file (like the target of the fd) to bind it onto
    if [ -d "/proc/self/fd/$1" ]; then
      mkpath "$2"
    else
      mkpath "${2%/*}"
      [ -e "$2" ] || install -o "$RUN_UID" -g "$RUN_GID" -m 0644 /dev/null "$2"
    fi
  }
  # 1. open what will be put back while the whole tree is still visible
  FDS=() MODES=() DSTS=()
  while IFS=$'\''\t'\'' read -r mode src dst; do
    exec {fd}<"$src"
    FDS+=("$fd") MODES+=("$mode") DSTS+=("$dst")
  done <<< "$EXPOSE_SPEC"
  # 2. the tools/sandbox.sh jail (no rw dirs yet), with every NVIDIA device node masked
  lo_up
  for d in /dev/nvidia* /run/nvidia-persistenced/socket; do
    if [ -c "$d" ] || [ -S "$d" ]; then mount --bind /dev/null "$d"; fi
  done
  lockdown_fs
  # 3. cover the home directory and put the listed paths back from their fds
  mount -t tmpfs -o "size=1m,mode=0755,uid=$RUN_UID,gid=$RUN_GID,nosuid,nodev" tmpfs "$HIDE"
  for i in "${!FDS[@]}"; do
    fd=${FDS[$i]}
    mkmountpoint "$fd" "${DSTS[$i]}"
    mount -c --bind "/proc/self/fd/$fd" "${DSTS[$i]}"
    mount -o "remount,bind,${MODES[$i]},nosuid,nodev" "${DSTS[$i]}"
    exec {fd}<&-
  done
  if [ -d "$REPO_IN/.venv" ] && [ ! -L "$REPO_IN/.venv" ]; then
    mount --bind "$REPO_IN/.venv" "$REPO_IN/.venv"
    mount -o remount,bind,ro,nosuid,nodev "$REPO_IN/.venv"
  fi
  mount -o remount,ro "$HIDE"
  # 4. re-resolve the cwd in the new tree (the inherited one still points into the covered directory)
  cd -- "$CWD_IN" 2>/dev/null || cd -- "$REPO_IN"
  # 5. loopback forwarders to the arm sockets, then drop to the caller (clean env, bash as reaping PID 1)
  DROP=(setpriv --reuid="$RUN_UID" --regid="$RUN_GID" --init-groups --no-new-privs --inh-caps=-all --pdeathsig keep --)
  ENV=(/usr/bin/env -i HOME="$HIDE" PATH="$HIDE/.local/bin:/usr/local/bin:/usr/bin:/bin" LANG=C.UTF-8
       USER="$RUN_USER" LOGNAME="$RUN_USER")
  "${DROP[@]}" "${ENV[@]}" /usr/bin/python3 -I "$TOOLS/uds_forward.py" "8100:$GW_IN" "8200:$RN_IN" \
    </dev/null >&2 &
  for i in $(seq 100); do   # listening sockets of this netns: 0x1FA4 = 8100, 0x2008 = 8200
    t=$(cat /proc/net/tcp)
    [[ $t == *":1FA4 00000000:0000 0A"* && $t == *":2008 00000000:0000 0A"* ]] && break
    sleep 0.05
  done
  exec "${DROP[@]}" "${ENV[@]}" /bin/bash -c '\''"$@"; exit $?'\'' h2h-init "$@"
' h2h_sandbox "$TOOLS" "$HIDE" "$EXPOSE_SPEC" "$REPO_IN" "$CWD_IN" "$GW_IN" "$RN_IN" "$RUN_UID" "$RUN_GID" \
  "$RUN_USER" "$@"
