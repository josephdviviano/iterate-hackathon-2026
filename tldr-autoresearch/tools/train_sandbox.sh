#!/usr/bin/env bash
# Run an (agent-written, untrusted) training command in private mount + PID + network namespaces.
#   train_sandbox.sh <gpu_minor> <rw_dir>[:<rw_dir>...] [options] -- <cmd> [args...]
# Inside: only /dev/nvidia<gpu_minor> usable; no network at all; the whole filesystem read-only except the
# given dirs (compile caches, the run's trusted-record dir); private /tmp, /var/tmp, /dev/shm; separate PID
# namespace (killed as a whole); the caller's uid/gid (id -u / id -g, taken before sudo; refused for root) with
# no_new_privs and no capabilities. train.py can only train on its GPU and print.
#
# Options (none given = exactly the behaviour above; tools/ar_run.py --isolate uses all of them):
#   --hide-home DIR            after the lockdown cover DIR (the caller's home) with a fresh private tmpfs (rw, 1 GB,
#                              discarded with the sandbox); only the --expose / --overlay paths are put back.
#                              Nothing else under DIR (run data, canon.git, the other arm, logs, ~/.claude, ...)
#                              is visible. The rw dirs keep working but are hidden if they lie under DIR.
#   --expose ro|rw:SRC[:DST]   put SRC (dir or file) back at DST (default: SRC), read-only or read-write
#   --overlay SRC:DST          an ephemeral copy-on-write view of the directory SRC at DST: reads see SRC, writes go
#                              to a tmpfs layer that is discarded with the sandbox (SRC itself is never modified,
#                              so nothing written in one run reaches the next)
#   --fd N:read|append|consume:PATH
#                              open PATH as fd N (3..9) for the command before anything is hidden; consume = read
#                              and unlink (a one-time secret that must not stay reachable by path)
#   --mask PATH                bind /dev/null over the file PATH (e.g. data only the trusted judge may read: it gets
#                              it through --fd instead)
#   --chdir DIR                working directory of the command (default: the caller's)
# Exposure technique (as tools/h2h_sandbox.sh): every SRC is opened as an fd by root inside the new mount namespace
# while the whole tree is visible; after the tmpfs covers DIR, each one is bind-mounted from /proc/self/fd/N onto its
# DST (mount -c: canonicalising the fd link would bind the empty tmpfs dir) and the fd is closed.
# Root: $RLTLDR_ROOT, else the directory above this script (sandbox_lib.sh is sourced from $ROOT/tools).
set -euo pipefail
ROOT=${RLTLDR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
die() { echo "train_sandbox: $*" >&2; exit 2; }
RUN_UID=$(id -u) RUN_GID=$(id -g)
[ "$RUN_UID" != 0 ] || die "run it as the unprivileged user (it drops to the caller's uid), not as root"
[ $# -ge 2 ] || die "usage: train_sandbox.sh <gpu_minor> <rw_dir>[:<rw_dir>...] [options] -- <cmd> [args...]"
GPU_MINOR=$1; RW=$2; shift 2
HIDE="" CHDIR="" SPEC=""
clean() { [[ $1 == /* && $1 != *[$'\t\n']* ]] || die "not a clean absolute path: $1"; }
add() { SPEC+="$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\n'; }    # kind, a, b, c
while [ $# -gt 0 ]; do
  case $1 in
    --hide-home) clean "$2"; HIDE=$2; shift 2 ;;
    --expose)
      IFS=: read -r mode src dst <<< "$2"
      [[ $mode == ro || $mode == rw ]] || die "--expose needs ro|rw:SRC[:DST], not $2"
      dst=${dst:-$src}; clean "$src"; clean "$dst"; [ -e "$src" ] || die "--expose: $src does not exist"
      add expose "$mode" "$src" "$dst"; shift 2 ;;
    --overlay)
      IFS=: read -r src dst <<< "$2"
      clean "$src"; clean "$dst"; [ -d "$src" ] || die "--overlay: $src is not a directory"
      add overlay "$src" "$dst" ""; shift 2 ;;
    --fd)
      IFS=: read -r n mode path <<< "$2"
      [[ $n =~ ^[3-9]$ ]] || die "--fd: N must be 3..9, not $n"
      [[ $mode == read || $mode == append || $mode == consume ]] || die "--fd: bad mode $mode"
      clean "$path"; add fd "$n" "$mode" "$path"; shift 2 ;;
    --mask) clean "$2"; add mask "$2" "" ""; shift 2 ;;
    --chdir) clean "$2"; CHDIR=$2; shift 2 ;;
    --) shift; break ;;
    *) break ;;                       # no "--": the command starts here (old call style)
  esac
done
if [ -z "$HIDE" ] && grep -q '^expose' <<< "$SPEC"; then die "--expose needs --hide-home"; fi
exec sudo -n --preserve-env unshare --mount --pid --net --fork --kill-child --mount-proc --propagation private -- bash -c '
  set -e
  ROOT="$1"; GPU_MINOR="$2"; RW="$3"; HIDE="$4"; SPEC="$5"; CHDIR="$6"; RUN_UID="$7"; RUN_GID="$8"; shift 8
  [[ $RUN_UID =~ ^[1-9][0-9]*$ && $RUN_GID =~ ^[0-9]+$ ]] || { echo "train_sandbox: bad uid/gid" >&2; exit 2; }
  source "$ROOT/tools/sandbox_lib.sh"
  mkpath() {   # create a mountpoint (dir, or empty file if $2=file) and its missing parents, owned by the caller
    local d="" c
    IFS=/ read -ra parts <<< "${1#/}"
    for c in "${parts[@]:0:${#parts[@]}-1}"; do d="$d/$c"; [ -d "$d" ] || install -d -o "$RUN_UID" -g "$RUN_GID" -m 0755 "$d"; done
    if [ "$2" = file ]; then [ -e "$1" ] || install -o "$RUN_UID" -g "$RUN_GID" -m 0644 /dev/null "$1"
    else [ -d "$1" ] || install -d -o "$RUN_UID" -g "$RUN_GID" -m 0755 "$1"; fi
  }
  # 1. open what will be put back while the whole tree is still visible
  EFD=() EMODE=() EDST=() OVL=()
  while IFS=$'"'"'\t'"'"' read -r kind a b c; do
    case $kind in
      expose) exec {fd}<"$b"; EFD+=("$fd") EMODE+=("$a") EDST+=("$c") ;;
      overlay) OVL+=("$a"$'"'"'\t'"'"'"$b") ;;
    esac
  done <<< "$SPEC"
  mask_gpus "$GPU_MINOR"
  IFS=: read -ra dirs <<< "$RW"
  lockdown_fs "${dirs[@]}"
  # 2. fds for the command (after the lockdown: append targets must lie in a rw dir; see sandbox_lib.sh)
  while IFS=$'"'"'\t'"'"' read -r kind a b c; do
    [ "$kind" = fd ] || continue
    case $b in
      read) eval "exec $a<\"\$c\"" ;;
      append) eval "exec $a>>\"\$c\"" ;;
      consume) eval "exec $a<\"\$c\""; rm -f -- "$c" ;;
    esac
  done <<< "$SPEC"
  # 3. ephemeral copy-on-write layers (on the private /tmp tmpfs), mounted while their lower dirs are visible
  i=0
  for o in "${OVL[@]}"; do
    src=${o%%$'"'"'\t'"'"'*} dst=${o#*$'"'"'\t'"'"'}
    s=/tmp/.overlay$i; i=$((i + 1))
    install -d -o "$RUN_UID" -g "$RUN_GID" -m 0755 "$s/upper"; install -d -m 0700 "$s/work"; install -d -m 0755 "$s/merged"
    mount -t overlay overlay -o "lowerdir=$src,upperdir=$s/upper,workdir=$s/work" "$s/merged"
    exec {fd}<"$s/merged"; EFD+=("$fd") EMODE+=(rw) EDST+=("$dst")
  done
  # 4. cover the home directory and put the listed paths back (parents before children)
  if [ -n "$HIDE" ]; then
    mount -t tmpfs -o "size=1g,mode=0755,uid=$RUN_UID,gid=$RUN_GID,nosuid,nodev" tmpfs "$HIDE"
    for i in $(for j in "${!EDST[@]}"; do printf "%s\t%s\n" "${EDST[$j]}" "$j"; done | LC_ALL=C sort | cut -f2); do
      fd=${EFD[$i]}
      if [ -d "/proc/self/fd/$fd" ]; then mkpath "${EDST[$i]}" dir; else mkpath "${EDST[$i]}" file; fi
      mount -c --bind "/proc/self/fd/$fd" "${EDST[$i]}"
      mount -o "remount,bind,${EMODE[$i]},nosuid,nodev" "${EDST[$i]}"
      exec {fd}<&-
    done
  fi
  while IFS=$'"'"'\t'"'"' read -r kind a b c; do
    if [ "$kind" = mask ] && [ -e "$a" ]; then mount --bind /dev/null "$a"; fi
  done <<< "$SPEC"
  if [ -n "$CHDIR" ]; then cd -- "$CHDIR"; elif [ -n "$HIDE" ]; then cd -- "$PWD" 2>/dev/null || cd /; fi
  exec setpriv --reuid="$RUN_UID" --regid="$RUN_GID" --init-groups --no-new-privs --inh-caps=-all -- "$@"
' train_sandbox "$ROOT" "$GPU_MINOR" "$RW" "$HIDE" "$SPEC" "$CHDIR" "$RUN_UID" "$RUN_GID" "$@"
