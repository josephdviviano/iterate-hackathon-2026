"""Experiment runner daemon (the only process that trains on the agent GPU, config agent_gpu_uuid).

The agent's `./run.sh "<desc>"` (inside its sandbox: no GPU, no general network) POSTs its current train.py
here. For each run the daemon
  1. checks out the attempt's parent commit (fetched from the driver-owned canon.git) in a pristine runner
     workspace (own clone, own venv, pristine prepare.py — nothing the agent can modify),
  2. writes the submitted train.py,
  3. executes tools/ar_run.py: training in a second sandbox (agent GPU only, no network, read-only fs) under the
     trusted evaluation bootstrap; GPU pinning, timeout, snapshot + diff, exactly one ledger line per run,
     per-attempt run cap, one recorded ok run per attempt,
  4. returns ar_run's compact summary (val_bpb / peak_vram_mb, or status + reason).
The attempt id and parent come from the driver (/control/attempt on the trusted TCP port), never from the
client. State survives restarts (data/runner_state.json); leftover training processes are reaped at start.
"""
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rltldr.config import load_config, require_gpu_uuid  # noqa: E402
from rltldr.io_utils import read_json, write_json_atomic  # noqa: E402

CFG = load_config()
WS = CFG.runner_ws                                   # pristine clone used for every run
CANON = CFG.canon_dir
AR_RUN = os.path.join(CFG.root, "tools", "ar_run.py")
PY = os.environ.get("RLTLDR_TRUSTED_PY", "/usr/bin/python3")   # trusted stdlib-only python that runs ar_run.py
UV = os.environ.get("UV_BIN", "uv")
HOME = os.path.expanduser("~")
SOCK = os.path.join(CFG.run_dir, "runner.sock")
STATE_PATH = CFG.path("runner_state.json")
PORT = CFG.runner_port
GIT_ENV = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
log = logging.getLogger("runner")

STATE = {"attempt": None, "job": None}   # job: {"proc", "attempt_id", "t0", "desc"}
LOCK = None


def git(*args):
    r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=WS, capture_output=True, text=True,
                       env=GIT_ENV)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def save_state():
    write_json_atomic(STATE_PATH, {"attempt": STATE["attempt"]})


def gpu2_pids():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    return [int(l.split(",")[0]) for l in out.strip().splitlines()
            if l.strip() and l.split(",")[1].strip() == CFG.agent_gpu_uuid]


def sudo_kill(pids):
    if pids:
        subprocess.run(["sudo", "-n", "kill", "-KILL", *map(str, pids)], capture_output=True)


def descendants(root):
    children = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    children.setdefault(int(f.read().rsplit(")", 1)[1].split()[1]), []).append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    out, stack = [], [root]
    while stack:
        for c in children.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


def kill_job(job):
    """Stop a run: TERM ar_run (it writes its ledger line), then make sure nothing survives."""
    p = job["proc"]
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(10)
    sudo_kill(descendants(p.pid) + [p.pid])
    sudo_kill(gpu2_pids())          # this daemon is the only user of the agent GPU


def harness_error(msg):
    return web.json_response({"rc": 3, "output": f"status: harness_error\n{msg}"})


async def run(request):
    try:
        d = await request.json()
    except json.JSONDecodeError:
        return harness_error("bad request")
    att = STATE["attempt"]
    if not att:
        return web.json_response({"rc": 3, "output": "status: refused\nno experiment session is active"})
    trusted = request.app["trusted"]
    max_runs = int(d["max_runs"]) if trusted and "max_runs" in d else CFG.max_runs_per_attempt
    # confirmation re-runs compile into a fresh cache: a cached re-run would reuse the same kernels and not be
    # an independent measurement
    fresh_cache = bool(d.get("fresh_cache")) and trusted
    desc = " ".join(str(d.get("desc") or "").split())[:300] or "(no description)"
    src = d.get("train_py")
    if not isinstance(src, str) or not src.strip():
        return web.json_response({"rc": 3, "output": "status: refused\ntrain.py content missing"})
    async with LOCK:                          # one training run at a time on the agent GPU
        if STATE["attempt"] is not att:       # the session ended while we waited
            return web.json_response({"rc": 3, "output": "status: refused\nthe experiment session is over"})
        try:
            git("checkout", "-q", "-f", att["parent_commit"])
            git("clean", "-q", "-ffd")
        except RuntimeError as e:
            log.error("workspace checkout failed: %s", e)
            return harness_error("the runner could not prepare its workspace")
        with open(os.path.join(WS, "train.py"), "w") as f:
            f.write(src)
        env = {"PATH": f"{HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin", "HOME": HOME,
               "AR_ATTEMPT_ID": att["id"], "AR_MAX_RUNS": str(max_runs), "LANG": "C.UTF-8",
               "RLTLDR_ROOT": CFG.root}
        cmd = [PY, "-I", AR_RUN, "--repo", WS, "--ledger", CFG.ledger, "--gpu", CFG.agent_gpu_uuid,
               "--gpu-minor", str(CFG.agent_gpu_minor), "--prepare-sha", CFG.prepare_sha256, "--desc", desc]
        if CFG.no_autotune:
            cmd.append("--no-autotune")
        if CFG.runner_isolate:
            cmd.append("--isolate")
        cache = None
        if fresh_cache:
            cache = tempfile.mkdtemp(prefix="cache-", dir=CFG.data)
            cmd += ["--cache-dir", cache]
        else:
            os.makedirs(CFG.cache_dir, exist_ok=True)
            cmd += ["--cache-dir", CFG.cache_dir]
        log.info("run for %s: %s", att["id"], desc)
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=WS, env=env, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        STATE["job"] = {"proc": proc, "attempt_id": att["id"], "t0": time.time(), "desc": desc}
        try:
            out, _ = await proc.communicate()
        finally:
            STATE["job"] = None
            if cache:
                shutil.rmtree(cache, ignore_errors=True)
        txt = out.decode("utf-8", "replace").strip()
        log.info("run for %s finished rc=%s: %s", att["id"], proc.returncode, txt[:300].replace("\n", " | "))
        return web.json_response({"rc": proc.returncode, "output": txt})


async def ctl_attempt(request):
    d = await request.json()
    # the parent comes from the driver-owned canonical repo; verify it exists before the agent starts
    try:
        git("fetch", "-q", CANON, f"+{CFG.branch}:refs/remotes/canon/{CFG.branch}")
        git("cat-file", "-e", f"{d['parent_commit']}^{{commit}}")
    except RuntimeError as e:
        return web.json_response({"error": f"parent commit not available: {e}"}, status=500)
    STATE["attempt"] = {"id": d["id"], "parent_commit": d["parent_commit"]}
    save_state()
    return web.json_response({"attempt": STATE["attempt"]})


async def ctl_attempt_end(request):
    d = await request.json()
    att = STATE["attempt"]
    if att and att["id"] == d.get("id"):
        STATE["attempt"] = None
        save_state()
    job = STATE["job"]
    if job and job["attempt_id"] == d.get("id"):
        log.warning("attempt %s ended with a run in flight; killing it", d.get("id"))
        await asyncio.get_running_loop().run_in_executor(None, kill_job, job)
    return web.json_response({"ok": True})


async def ctl_state(_):
    job = STATE["job"]
    return web.json_response({"attempt": STATE["attempt"],
                              "job": {k: v for k, v in job.items() if k != "proc"} if job else None})


def ensure_workspace():
    if not os.path.isdir(os.path.join(WS, ".git")):
        subprocess.run(["git", "clone", "-q", "--template=", "--branch", CFG.branch, CANON, WS], check=True,
                       env=GIT_ENV)
    if not os.path.isdir(os.path.join(WS, ".venv")):
        subprocess.run([UV, "sync", "--frozen"], cwd=WS, check=True)


def make_app(trusted: bool) -> web.Application:
    app = web.Application(client_max_size=64 << 20)
    app["trusted"] = trusted
    routes = [web.post("/run", run)]
    if trusted:
        routes += [web.post("/control/attempt", ctl_attempt), web.post("/control/attempt_end", ctl_attempt_end),
                   web.get("/control/state", ctl_state)]
    app.add_routes(routes)
    return app


async def serve():
    # trusted: TCP on localhost (driver). untrusted: Unix socket forwarded into the agent sandbox as
    # 127.0.0.1:8200 — /run only (attempt id, parent and run cap are always the driver's).
    global LOCK
    LOCK = asyncio.Lock()
    os.makedirs(os.path.dirname(SOCK), exist_ok=True)
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    runners = []
    for trusted, site in ((True, lambda r: web.TCPSite(r, "127.0.0.1", PORT)), (False, lambda r: web.UnixSite(r, SOCK))):
        r = web.AppRunner(make_app(trusted), access_log=None)
        await r.setup()
        await site(r).start()
        runners.append(r)
    log.info("runner listening on 127.0.0.1:%d (trusted) and %s (sandbox); workspace %s", PORT, SOCK, WS)
    try:
        await asyncio.Event().wait()
    finally:
        for r in runners:
            await r.cleanup()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    require_gpu_uuid(CFG, "agent_gpu_uuid")
    ensure_workspace()
    left = gpu2_pids()
    if left:   # training orphaned by a previous runner instance (its ledger line may be missing)
        log.warning("reaping %d leftover agent-GPU process(es): %s", len(left), left)
        sudo_kill(left)
    STATE["attempt"] = (read_json(STATE_PATH, {}) or {}).get("attempt")
    if STATE["attempt"]:
        log.info("restored active attempt %s", STATE["attempt"]["id"])
    asyncio.run(serve())


if __name__ == "__main__":
    main()
