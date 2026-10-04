"""Head-to-head (h2h) experiment runner: one daemon per arm, the only process that trains on that arm's GPU.

    $RLTLDR_SERVE_PY rltldr/h2h_runner.py --arm base|v5        (normally started by ./ctl_h2h.sh)

The arm's agent runs `./run.sh "<desc>"` inside its sandbox (no GPU, loopback only). tools/run_client.py POSTs the
current train.py, the description and the agent's own `git rev-parse HEAD` to 127.0.0.1:8200, which the sandbox
forwards to this daemon's Unix socket (arm.runner_sock: `POST /run` only). For each run the daemon
  1. resets its pristine workspace (arm.runner_ws: its own clone of canon `h2h/start`, detached at the start
     commit, with its own .venv; nothing the agent can modify): `git checkout -f <start>`, `git clean -ffd`,
  2. writes the submitted train.py,
  3. executes tools/ar_run.py --isolate: training in the train sandbox on this arm's GPU only (venv pinned to
     arm.gpu_uuid), under the trusted evaluation bootstrap (trusted val_bpb and training time); exactly one line
     per run in arm.ledger (desc = the client's description, agent_head = the client-reported HEAD), snapshots in
     arm.runs_dir. --isolate hides everything under $HOME from train.py except the workspace, the data and
     the interpreter, at arm-neutral paths (the workspace appears at ~/autoresearch, like the agent's own repo),
     and gives it arm.cache_dir only as a copy-on-write layer that is discarded after the run (a warm compile
     seed; nothing a run writes reaches the next one),
  4. returns ar_run's output: upstream's full summary block, or `status: <...>` + reason, with any arm-specific
     host path or GPU id replaced by a neutral one (both arms' agents must see the same kind of text). A run the
     harness itself interrupted (/control/kill, a runner restart) is reported as `status: harness_error` with the
     advice to resubmit, not as a crash of train.py.
Unlike rltldr/runner.py there are no attempts and no run caps (continuous session): a run is accepted whenever
none is in flight for this arm; concurrent submissions queue on a lock. A run continues if its client hangs up
(the result is still recorded in the ledger).

Trusted control on TCP 127.0.0.1:arm.runner_port: GET /health, GET /control/state (incl. free disk space of the
data dir), POST /control/kill (kill the in-flight run; ar_run still writes its ledger line), POST /run. Everything
this daemon kills is its own process tree or a process on this arm's GPU, never anything of the other arm. Leftovers of a previous instance (its
in-flight ar_run, processes on this arm's GPU) are reaped at start; SIGTERM stops an in-flight run cleanly and
exits. State: arm.dir/runner_state.json.
Executables: ar_run.py runs under $RLTLDR_TRUSTED_PY (default /usr/bin/python3, standard library only), uv is
$UV_BIN (default `uv` on PATH).
"""
import argparse
import asyncio
import fcntl
import logging
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
import time

from aiohttp import web

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code root (tools/ live here)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "tools"))
from ar_gpu_pin import ensure_pin  # noqa: E402
from rltldr.h2h_config import Arm, H2HConfig, load_h2h_config  # noqa: E402
from rltldr.io_utils import read_json, write_json_atomic  # noqa: E402

AR_RUN = os.path.join(ROOT_DIR, "tools", "ar_run.py")
PY = os.environ.get("RLTLDR_TRUSTED_PY") or "/usr/bin/python3"   # trusted interpreter for ar_run.py (stdlib only)
UV = os.environ.get("UV_BIN") or "uv"
HOME = pwd.getpwuid(os.getuid()).pw_dir    # the home ar_run.py --isolate hides (it uses the same lookup)
SAFE_PATH = f"{HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
BASE_ENV = {"PATH": SAFE_PATH, "HOME": HOME, "LANG": "C.UTF-8"}
GIT_ENV = {**BASE_ENV, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0"}
HEAD_RE = re.compile(r"[0-9a-f]{1,40}")
CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
MAX_TRAIN_PY = 4 << 20       # bytes
MAX_DESC = 300               # chars
KILL_GRACE = 15.0            # s for ar_run to write its ledger line after TERM before its tree is SIGKILLed
DISK_WARN_GB = 30.0          # log a warning before each run when the data dir's filesystem has less free space
N_REPO = f"{HOME}/autoresearch"             # the workspace inside the train sandbox (= the agent's repo path)
N_CACHE = f"{HOME}/.cache/ar-compile"       # the compile cache inside the train sandbox
N_DIR = f"{HOME}/.h2h"                      # any other path under an arm's data dir
N_GPU = "GPU-00000000-0000-0000-0000-000000000000"   # neutral stand-in for the arm's GPU UUID
H2H_PATH_RE = re.compile(r"data/h2h/[^/\s'\"]+")
log = logging.getLogger("h2h_runner")


# ------------------------------------------------------------------------------------------- processes
def proc_table():
    """{pid: (ppid, starttime)} for every process visible in /proc."""
    out = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    fields = f.read().rsplit(")", 1)[1].split()
                out[int(d)] = (int(fields[1]), int(fields[19]))   # fields 4 (ppid) and 22 (starttime)
            except (OSError, ValueError, IndexError):
                pass
    return out


def tree_snapshot(root, table=None):
    """[(pid, starttime)] of root's descendants (not root itself)."""
    table = table if table is not None else proc_table()
    children = {}
    for pid, (ppid, _) in table.items():
        children.setdefault(ppid, []).append(pid)
    out, stack = [], [root]
    while stack:
        for c in children.get(stack.pop(), []):
            out.append((c, table[c][1]))
            stack.append(c)
    return out


def still_alive(snapshot):
    """The snapshot's pids that still are the same processes (guards against pid reuse)."""
    table = proc_table()
    return [pid for pid, st in snapshot if table.get(pid, (None, None))[1] == st]


def sudo_kill(pids):
    pids = sorted(set(pids) - {os.getpid()})
    if pids:
        subprocess.run(["sudo", "-n", "kill", "-KILL", *map(str, pids)], capture_output=True)


def gpu_pids(gpu_uuid):
    """Compute processes on one physical GPU (nvidia-smi reports this container's pids)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1] == gpu_uuid:
            pids.append(int(parts[0]))
    return pids


def is_zombie_or_gone(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except (OSError, IndexError):
        return True


def git(cwd, *args):
    r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", *args], cwd=cwd,
                       capture_output=True, text=True, env=GIT_ENV, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:500]}")
    return r.stdout.strip()


def sha256_file(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def clean_desc(desc):
    return " ".join(CTRL_RE.sub(" ", str(desc or "")).split())[:MAX_DESC] or "(no description)"


def clean_head(head):
    """Client-reported git HEAD (untrusted): <= 40 hex chars, else empty."""
    h = head.strip().lower() if isinstance(head, str) else ""
    return h if HEAD_RE.fullmatch(h) else ""


def failure(status, msg, rc=3):
    return {"rc": rc, "output": f"status: {status}\n{msg}"}


def disk_free_gb(path):
    try:
        return round(shutil.disk_usage(path).free / 1e9, 1)
    except OSError:
        return None


def neutralizer(cfg):
    """f(text) -> text with every arm's host paths and GPU ids replaced by arm-independent ones."""
    subs = []
    for a in cfg.arms:
        for host, neutral in ((a.runner_ws, N_REPO), (a.cache_dir, N_CACHE), (a.dir, N_DIR)):
            for h in {host, os.path.realpath(host)}:
                subs.append((h, neutral))
        if a.gpu_uuid:
            subs.append((a.gpu_uuid, N_GPU))
    subs.sort(key=lambda kv: -len(kv[0]))       # longest (most specific) first

    def scrub(text):
        for h, n in subs:
            text = text.replace(h, n)
        return H2H_PATH_RE.sub("data/h2h/arm", text)
    return scrub


# ---------------------------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, cfg: H2HConfig, arm: Arm):
        self.cfg, self.arm = cfg, arm
        self.ws = arm.runner_ws
        self.state_path = os.path.join(arm.dir, "runner_state.json")
        self.start_commit = None
        self.lock = None              # asyncio.Lock (created inside the event loop)
        self.job = None               # in-flight run: {"proc", "pid", "desc", "head", "t0", "killing"}
        self.closing = False
        self.waiting = 0              # submissions queued behind the in-flight run
        self.n_runs = 0
        self.last = None
        self.started_at = time.time()
        self.scrub = neutralizer(cfg)

    # ---- state ------------------------------------------------------------------------------
    def job_public(self):
        j = self.job
        if not j:
            return None
        return {"pid": j["pid"], "desc": j["desc"], "head": j["head"], "t0": j["t0"],
                "elapsed_s": round(time.time() - j["t0"], 1)}

    def save_state(self):
        write_json_atomic(self.state_path, {
            "arm": self.arm.name, "pid": os.getpid(), "started_at": self.started_at,
            "start_commit": self.start_commit, "gpu_uuid": self.arm.gpu_uuid, "n_runs": self.n_runs,
            "job": self.job_public(), "last": self.last, "disk_free_gb": disk_free_gb(self.arm.dir),
            "updated": time.time()})

    # ---- workspace --------------------------------------------------------------------------
    def ensure_workspace(self):
        """Own clone of canon `h2h/start` (only objects reachable from it), detached at the start commit, with
        its own venv (`uv sync --frozen`) pinned to this arm's GPU."""
        cfg, ws = self.cfg, self.ws
        start = git(cfg.canon_dir, "rev-parse", "--verify", f"refs/heads/{cfg.start_ref}^{{commit}}")
        if not os.path.isdir(os.path.join(ws, ".git")):
            if os.path.exists(ws):
                raise RuntimeError(f"{ws} exists but is not a git clone; move it away")
            tmp = ws + ".partial"
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(os.path.dirname(ws), exist_ok=True)
            log.info("cloning %s (%s) into %s", cfg.canon_dir, cfg.start_ref, ws)
            git(None, "clone", "-q", "--template=", "--no-local", "--single-branch", "--branch", cfg.start_ref,
                "--no-tags", cfg.canon_dir, tmp)
            os.rename(tmp, ws)
        try:
            git(ws, "cat-file", "-e", f"{start}^{{commit}}")
        except RuntimeError:
            git(ws, "fetch", "-q", "--no-tags", cfg.canon_dir,
                f"+refs/heads/{cfg.start_ref}:refs/remotes/origin/{cfg.start_ref}")
        self.start_commit = start
        self.reset_workspace()
        sha = sha256_file(os.path.join(ws, "prepare.py"))
        if sha != cfg.prepare_sha256:
            raise RuntimeError(f"prepare.py at {cfg.start_ref} has sha256 {sha}, expected {cfg.prepare_sha256}")
        venv = os.path.join(ws, ".venv")
        # every start (a no-op when the venv is complete; repairs one whose creation was interrupted)
        log.info("syncing the workspace venv (uv sync --frozen)")
        subprocess.run([UV, "sync", "--frozen", "--quiet"], cwd=ws, env=BASE_ENV, check=True, stdin=subprocess.DEVNULL)
        for p in ensure_pin(venv, self.arm.gpu_uuid):
            log.info("installed GPU pin %s", p)

    def reset_workspace(self):
        git(self.ws, "checkout", "-q", "-f", "--detach", self.start_commit)
        git(self.ws, "clean", "-q", "-ffd")

    # ---- reaping ----------------------------------------------------------------------------
    def is_our_ar_run(self, pid):
        """pid is an ar_run.py writing to this arm's ledger (so never the other arm's run)."""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().decode("utf-8", "replace").split("\0")
        except OSError:
            return False
        return AR_RUN in argv and "--ledger" in argv and self.arm.ledger in argv

    def stop_ar_run(self, pid, snapshot=None):
        """TERM an ar_run (it kills its sandbox and writes the ledger line), give it KILL_GRACE seconds, then
        SIGKILL whatever is left of its tree and of this arm's GPU processes. Blocking."""
        snapshot = list(snapshot or []) + tree_snapshot(pid)
        try:
            os.killpg(pid, signal.SIGTERM)       # ar_run leads its own session/process group
        except OSError:
            pass
        t_end = time.time() + KILL_GRACE
        while time.time() < t_end and not is_zombie_or_gone(pid):
            time.sleep(0.2)
        if not is_zombie_or_gone(pid):
            log.warning("ar_run %d did not exit within %.0f s after TERM; killing its tree", pid, KILL_GRACE)
            snapshot += tree_snapshot(pid) + [(pid, proc_table().get(pid, (None, None))[1])]
        sudo_kill(still_alive(snapshot))
        left = gpu_pids(self.arm.gpu_uuid)
        if left:
            log.warning("killing %d process(es) left on %s: %s", len(left), self.arm.gpu_uuid, left)
            sudo_kill(left)

    def reap_leftovers(self):
        try:
            st = read_json(self.state_path, {}) or {}
        except (OSError, ValueError) as e:
            log.warning("ignoring unreadable %s: %s", self.state_path, e)
            st = {}
        self.n_runs = int(st.get("n_runs") or 0)
        self.last = st.get("last")
        old = st.get("job") or {}
        if old.get("pid") and self.is_our_ar_run(old["pid"]):
            log.warning("a previous runner left run %r in flight (ar_run pid %d); stopping it",
                        old.get("desc"), old["pid"])
            self.stop_ar_run(old["pid"])
        left = gpu_pids(self.arm.gpu_uuid)
        if left:
            log.warning("reaping %d leftover process(es) on %s: %s", len(left), self.arm.gpu_uuid, left)
            sudo_kill(left)

    # ---- runs -------------------------------------------------------------------------------
    def ar_run_cmd(self, desc):
        a, cfg = self.arm, self.cfg
        cmd = [PY, "-I", AR_RUN, "--repo", self.ws, "--ledger", a.ledger, "--runs-dir", a.runs_dir,
               "--cache-dir", a.cache_dir, "--gpu", a.gpu_uuid, "--gpu-minor", str(a.gpu_minor),
               "--prepare-sha", cfg.prepare_sha256, f"--desc={desc}", "--full-summary", "--isolate"]
        if cfg.no_autotune:
            cmd.append("--no-autotune")
        return cmd

    async def submit(self, desc, src, head):
        """Queue behind the in-flight run, then run. Shielded from client hang-ups by the caller."""
        self.waiting += 1
        try:
            await self.lock.acquire()
        finally:
            self.waiting -= 1
        try:
            return await self.run_one(desc, src, head)
        finally:
            self.lock.release()

    async def run_one(self, desc, src, head):
        if self.closing:
            return failure("harness_error", "the experiment runner is restarting; submit the run again")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.reset_workspace)
            with open(os.path.join(self.ws, "train.py"), "w", encoding="utf-8", errors="replace") as f:
                f.write(src)
        except (RuntimeError, OSError) as e:
            log.error("workspace reset failed: %s", e)
            return failure("harness_error", "the runner could not prepare its workspace")
        free = disk_free_gb(self.arm.dir)
        if free is not None and free < DISK_WARN_GB:
            log.warning("only %.1f GB free on the data filesystem", free)
        env = {**BASE_ENV, "RLTLDR_ROOT": ROOT_DIR, "AR_MAX_RUNS": "0", "AR_AGENT_HEAD": head}
        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(
            *self.ar_run_cmd(desc), cwd=self.ws, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        job = self.job = {"proc": proc, "pid": proc.pid, "desc": desc, "head": head, "t0": t0, "killing": None}
        self.save_state()
        log.info("run started (ar_run pid %d, head %s): %s", proc.pid, head[:12] or "-", desc)
        try:
            out, _ = await proc.communicate()
            if job["killing"] is not None:      # finish the kill (incl. the GPU sweep) before the next run
                await asyncio.shield(job["killing"])
        finally:
            self.job = None
        txt = out.decode("utf-8", "replace").strip()
        rc = proc.returncode
        self.n_runs += 1
        m = re.search(r"^val_bpb:\s+([0-9.]+)\s*$", txt, re.M)
        st = re.match(r"status: (\S+)", txt)
        self.last = {"desc": desc, "head": head, "rc": rc, "t0": t0, "t1": time.time(),
                     "wall_s": round(time.time() - t0, 1), "status": st.group(1) if st else ("ok" if m else None),
                     "val_bpb": float(m.group(1)) if m and not st else None, "killed": job["killing"] is not None}
        self.save_state()
        log.info("run finished rc=%s in %.0f s: %s", rc, time.time() - t0, txt[:400].replace("\n", " | "))
        if job["killing"] is not None:
            return failure("harness_error", "the experiment harness interrupted this run (a runner restart or an "
                           "operator stop; not a problem with train.py). Submit the same experiment again.")
        if rc not in (0, 1, 3) or not txt.startswith(("---", "val_bpb:", "status: ")):
            log.error("ar_run failed unexpectedly (rc=%s):\n%s", rc, txt[-4000:])
            return failure("harness_error", "the experiment runner failed; the run was not recorded properly")
        return {"rc": rc, "output": self.scrub(txt)}

    async def kill_job(self, reason):
        job = self.job
        if not job:
            return None
        info = self.job_public()
        if job["killing"] is None:
            log.warning("killing the in-flight run (%s): %s", reason, job["desc"])
            snap = tree_snapshot(job["pid"])           # taken while the tree is intact
            job["killing"] = asyncio.get_running_loop().run_in_executor(None, self.stop_ar_run, job["pid"], snap)
        await asyncio.shield(job["killing"])
        return info

    # ---- HTTP -------------------------------------------------------------------------------
    async def h_run(self, request):
        try:
            d = await request.json()
            if not isinstance(d, dict):
                raise ValueError("not an object")
        except Exception:  # noqa: BLE001  (malformed JSON / encoding)
            return web.json_response(failure("harness_error", "bad request"))
        src = d.get("train_py")
        if not isinstance(src, str) or not src.strip():
            return web.json_response(failure("refused", "train.py content missing"))
        if len(src.encode("utf-8", "replace")) > MAX_TRAIN_PY:
            return web.json_response(failure("refused", f"train.py is larger than {MAX_TRAIN_PY >> 20} MB"))
        task = asyncio.ensure_future(self.submit(clean_desc(d.get("desc")), src, clean_head(d.get("head"))))
        return web.json_response(await asyncio.shield(task))

    async def h_state(self, _request):
        return web.json_response({
            "arm": self.arm.name, "pid": os.getpid(), "started_at": self.started_at, "gpu_uuid": self.arm.gpu_uuid,
            "start_commit": self.start_commit, "busy": self.job is not None, "job": self.job_public(),
            "queued": self.waiting, "n_runs": self.n_runs, "last": self.last,
            "disk_free_gb": disk_free_gb(self.arm.dir)})

    async def h_kill(self, _request):
        info = await self.kill_job("control/kill")
        return web.json_response({"killed": info is not None, "job": info})

    async def h_health(self, _request):
        return web.json_response({"ok": True, "arm": self.arm.name, "busy": self.job is not None})

    def make_app(self, trusted):
        app = web.Application(client_max_size=MAX_TRAIN_PY * 2 + (1 << 20))
        routes = [web.post("/run", self.h_run)]
        if trusted:
            routes += [web.get("/health", self.h_health), web.get("/control/state", self.h_state),
                       web.post("/control/kill", self.h_kill)]
        app.add_routes(routes)
        return app

    async def serve(self):
        self.lock = asyncio.Lock()
        sock = self.arm.runner_sock
        os.makedirs(self.arm.sock_dir, mode=0o755, exist_ok=True)
        if os.path.exists(sock):
            os.unlink(sock)
        runners = []
        sites = ((True, lambda r: web.TCPSite(r, "127.0.0.1", self.arm.runner_port)),
                 (False, lambda r: web.UnixSite(r, sock)))
        for trusted, site in sites:
            r = web.AppRunner(self.make_app(trusted), access_log=None)
            await r.setup()
            await site(r).start()
            runners.append(r)
        self.save_state()
        log.info("runner[%s] listening on 127.0.0.1:%d (trusted) and %s (sandbox); workspace %s @ %s; GPU %s",
                 self.arm.name, self.arm.runner_port, sock, self.ws, self.start_commit[:12], self.arm.gpu_uuid)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for s in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(s, stop.set)
        try:
            await stop.wait()
            log.info("shutting down")
            self.closing = True
            await self.kill_job("runner shutdown")
        finally:
            for r in runners:
                await r.cleanup()
            if os.path.exists(sock):
                os.unlink(sock)
            self.save_state()


def main():
    ap = argparse.ArgumentParser(description="h2h experiment runner for one arm (see module docstring)")
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s runner[{a.arm}]: %(message)s")
    cfg = load_h2h_config()
    cfg.arm(a.arm).check_gpu()
    r = Runner(cfg, cfg.arm(a.arm))
    os.makedirs(r.arm.dir, exist_ok=True)
    # one runner per arm: a second instance must not reap the first one's run or reset its workspace
    lock_f = open(os.path.join(r.arm.dir, "runner.lock"), "a+")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error("another runner for arm %s is running (%s is locked)", a.arm, lock_f.name)
        sys.exit(1)
    r.reap_leftovers()          # before touching the workspace a leftover run may still be using
    r.ensure_workspace()
    asyncio.run(r.serve())


if __name__ == "__main__":
    main()
