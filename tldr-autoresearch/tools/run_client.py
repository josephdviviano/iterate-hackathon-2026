#!/usr/bin/env python3
"""./run.sh "<description>" — submit the current train.py to the experiment runner and print the result.

Training happens on the dedicated GPU outside the agent's sandbox (~6 minutes). Output: the runner's summary
(`val_bpb: X`, `peak_vram_mb: Y`, ...) on success, or `status: <error>` with the end of the training log on
failure. Exit code: 0 ok, 1 failed run, 3 refused.

Besides the description and train.py, the request carries `head` = `git rev-parse HEAD` of the current repo
(empty if that fails), so the runner's ledger can be matched to the agent's own commits. git runs with hooks
and fsmonitor disabled. Standard library only. AR_RUNNER_URL overrides the runner address (tests).
"""
import json
import os
import subprocess
import sys
import urllib.request

RUNNER_URL = os.environ.get("AR_RUNNER_URL") or "http://127.0.0.1:8200"


def git_head():
    try:
        r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
                            "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


desc = " ".join(sys.argv[1:]).strip()
if not desc:
    print('usage: ./run.sh "<short description of the experiment>"', file=sys.stderr)
    sys.exit(2)
with open("train.py") as f:
    src = f.read()
body = json.dumps({"desc": desc, "train_py": src, "head": git_head()}).encode()
req = urllib.request.Request(RUNNER_URL.rstrip("/") + "/run", data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=None) as r:
        d = json.loads(r.read())
except Exception as e:  # runner unreachable
    print(f"status: harness_error\ncould not reach the experiment runner: {e}")
    sys.exit(3)
print(d.get("output", ""))
sys.exit(int(d.get("rc", 1)))
