# Shared by sandbox.sh (agent) and train_sandbox.sh (training). Sourced INSIDE the new mount namespace as
# root, before privileges are dropped.
#   lockdown_fs <rw_dir>...   every mount read-only (incl. /usr, /opt, /etc, $HOME), fresh private
#                             tmpfs on /tmp, /var/tmp, /dev/shm (host /dev/shm holds vLLM's IPC buffers),
#                             then the given directories re-exposed read-write
#   mask_gpus [keep_minor]    bind /dev/null over every /dev/nvidia<N> except /dev/nvidia<keep_minor>
#   lo_up                     bring up the loopback interface of the new network namespace (with /usr/bin/python3,
#                             or $RLTLDR_TRUSTED_PY if that is set and /usr/bin/python3 fails)
lockdown_fs() {
  awk '{print $5}' /proc/self/mountinfo | grep -vE '^/(proc|sys|dev)(/|$)' | sort -u | while read -r m; do
    mount -o remount,bind,ro "$m" 2>/dev/null || true
  done
  for t in /tmp /var/tmp /dev/shm; do mount -t tmpfs -o size=32g,mode=1777,nosuid,nodev tmpfs "$t"; done
  local w
  for w in "$@"; do
    [ -n "$w" ] || continue
    mkdir -p "$w" 2>/dev/null || true
    mount --bind "$w" "$w" && mount -o remount,bind,rw "$w"
  done
}
mask_gpus() {
  local d
  for d in /dev/nvidia[0-9]*; do
    [ -n "${1:-}" ] && [ "$d" = "/dev/nvidia$1" ] && continue
    mount --bind /dev/null "$d"
  done
}
lo_up() {
  local py code="
import socket, fcntl, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
fcntl.ioctl(s, 0x8914, struct.pack('16sH14s', b'lo', 0x1 | 0x8 | 0x40, bytes(14)))"
  for py in /usr/bin/python3 ${RLTLDR_TRUSTED_PY:+"$RLTLDR_TRUSTED_PY"}; do
    "$py" -I -c "$code" 2>/dev/null && return 0
  done
  echo "lo_up: could not bring up the loopback interface (no working /usr/bin/python3; set RLTLDR_TRUSTED_PY)" >&2
  return 1
}
