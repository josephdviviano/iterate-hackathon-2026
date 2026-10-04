#!/usr/bin/env python3
"""
gpu_watchdog.py - SIGKILL agent-spawned processes that hold a CUDA context on a GPU they may not use.

Optional extra layer of the GPU-pinning defence (docs/AR_RUN.md, "GPU pinning"); not started by ctl.sh: the agent
sandbox (tools/sandbox.sh) already masks every /dev/nvidia*. It only ever touches processes that are provably
the agent's:
  - their environment contains the marker (default AR_AGENT=1, set when launching pi), or
  - they descend from --root-pid (the pi process).
Anything else (vLLM, the policy trainer, other sessions) is never signalled.
nvidia-smi inside this container reports container-visible PIDs, so /proc lookups work.

  AR_AGENT=1 pi ... &                                   # launch the agent with the marker
  python3 tools/gpu_watchdog.py --allowed <agent GPU UUID> --root-pid <pi pid> --incidents data/incidents.jsonl
Use --once for a single sweep, --dry-run to only report.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def gpu_procs():
    """[(pid, gpu_uuid, used_mib)]; empty if nvidia-smi fails (logged by the caller)."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"nvidia-smi exit {r.returncode}")
    out = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            out.append((int(parts[0]), parts[1], parts[2] if len(parts) > 2 else ""))
    return out


def has_marker(pid, marker):
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            return marker in f.read().split(b"\0")
    except OSError:
        return False


def is_descendant(pid, root):
    for _ in range(512):
        if pid == root:
            return True
        if pid <= 1:
            return False
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])  # field 4 = ppid
        except (OSError, ValueError, IndexError):
            return False
    return False


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()[:300]
    except OSError:
        return ""


def log(path, rec):
    line = json.dumps(rec)
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allowed", action="append", required=True, help="allowed GPU UUID (repeatable)")
    ap.add_argument("--root-pid", type=int, default=0, help="agent root PID (e.g. pi); 0 = marker only")
    ap.add_argument("--marker", default="AR_AGENT=1", help="env entry identifying agent processes")
    ap.add_argument("--incidents", default="", help="append-only JSONL incident log")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report, do not kill")
    a = ap.parse_args()
    marker = a.marker.encode()
    allowed = set(a.allowed)
    while True:
        try:
            procs = gpu_procs()
        except Exception as e:  # keep watching; nvidia-smi can transiently fail
            log(a.incidents, {"ts": time.time(), "event": "nvidia_smi_error", "error": str(e)[:300]})
            procs = []
        for pid, gpu_uuid, mib in procs:
            if gpu_uuid in allowed or pid == os.getpid():
                continue
            if not (has_marker(pid, marker) or (a.root_pid and is_descendant(pid, a.root_pid))):
                continue
            rec = {"ts": time.time(), "event": "wrong_gpu", "pid": pid, "gpu_uuid": gpu_uuid,
                   "used_mib": mib, "cmdline": cmdline(pid)}
            if a.dry_run:
                rec["action"] = "dry_run"
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                    rec["action"] = "killed"
                except OSError as e:
                    rec["action"] = f"kill_failed: {e}"
            log(a.incidents, rec)
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
