#!/usr/bin/env python3
"""
test_h2h_runner.py - GPU-free integration test of rltldr/h2h_runner.py (+ tools/run_client.py).

Both arms' runners are started for real (serve env) against a throw-away root (RLTLDR_ROOT + H2H_CONFIG override:
test ports 18210/18220, sockets and data under the temp root, the real canon.git read-only) and driven through
their Unix sockets exactly like the sandboxed run_client. The submitted train.py files are fakes that crash or
sleep without ever creating a CUDA context; they still run through the real ar_run.py + train sandbox on the
arm's GPU minor. Checks: workspace (own clone of h2h/start only, detached, venv pinned to the arm's GPU),
ledger fields (desc, agent_head, no attempt, no cap; descriptions starting with '-'), refusals, state (incl. free
disk), queueing, /control/kill and runner restarts (reported as harness_error: resubmit), cross-arm isolation,
arm-neutral output (no host path / arm name in what the agent sees), single instance per arm, reaping after a
runner crash, SIGTERM shutdown, run_client.py end-to-end. Runs execute in ar_run's --isolate sandbox, where every
path is arm-neutral, so a run's processes are found through its sandbox's command line (the run's trusted dir).

Must not run while the production h2h runners are active (it refuses if the arms' GPUs are busy or their runner
ports listen). Needs the production h2h config (h2h_config.json with the arms' GPUs), canon.git with h2h/start,
the serving env ($RLTLDR_SERVE_PY, default ~/envs/serve/bin/python) and passwordless sudo.

  python3 tests/test_h2h_runner.py        # ~3-5 min
"""
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("RLTLDR_ROOT") or CODE
SERVE_PY = os.environ.get("RLTLDR_SERVE_PY") or os.path.expanduser("~/envs/serve/bin/python")
HOME = os.path.expanduser("~")
RUNNER = os.path.join(CODE, "rltldr", "h2h_runner.py")
CLIENT = os.path.join(CODE, "tools", "run_client.py")
UDS_FWD = os.path.join(CODE, "tools", "uds_forward.py")
sys.path.insert(0, CODE)
from rltldr.h2h_config import load_h2h_config  # noqa: E402

PROD = load_h2h_config()
ARMS = {a.name: a for a in PROD.arms}
PORTS = {"base": 18210, "v5": 18220}
CRASH = "import sys\nprint('starting', flush=True)\nraise ValueError('fake failure for the runner test')\n"
SLEEP = "import time\nprint('training...', flush=True)\ntime.sleep(90)\n"
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""), flush=True)


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=900):
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def post_sock(sock, body, raw=None):
    c = UnixHTTPConnection(sock)
    c.request("POST", "/run", body=raw if raw is not None else json.dumps(body),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"null")


def http_tcp(port, method, path, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request(method, path)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"null")


def ledger(root, arm):
    p = os.path.join(root, "data", "h2h", arm, "ledger.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def bootstrap_procs(runs_dir):
    """ar_bootstrap.py processes (judge + child) of the in-flight run(s) whose files live in runs_dir: descendants of
    the processes whose command line names a path under runs_dir (sudo/unshare of the run's sandbox)."""
    runs = runs_dir.rstrip("/").encode() + b"/"
    kids, argvs = {}, {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argvs[int(pid)] = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            ppid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(pid))
    stack = [pid for pid, argv in argvs.items() if any(a.startswith(runs) for a in argv)]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid not in seen:
            seen.add(pid)
            stack.extend(kids.get(pid, []))
    return sorted(p for p in seen if any(a.endswith(b"ar_bootstrap.py") for a in argvs.get(p, [])))


def ar_run_procs(ledger_path):
    out = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if any(a.endswith(b"ar_run.py") for a in argv) and ledger_path.encode() in argv:
            out.append(int(pid))
    return out


def wait_for(fn, timeout, step=0.3):
    t = time.time()
    while time.time() - t < timeout:
        v = fn()
        if v:
            return v
        time.sleep(step)
    return fn()


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def preconditions():
    for a in PROD.arms:
        a.check_gpu()
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    busy = [l for l in out.splitlines() if any(a.gpu_uuid in l for a in PROD.arms)]
    if busy:
        return f"arm GPUs are busy: {busy}"
    for a in PROD.arms:
        if port_open(a.runner_port):
            return f"production runner port {a.runner_port} is listening"
    return None


class Env:
    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="h2h_runner_test_", dir=os.path.join(ROOT, "scratch"))
        self.cfg_path = os.path.join(self.root, "h2h_config.json")
        arms = []
        for a in PROD.arms:
            arms.append({"name": a.name, "served_model": a.served_model, "adapter_dir": a.adapter_dir,
                         "gpu_uuid": a.gpu_uuid, "gpu_minor": a.gpu_minor, "runner_port": PORTS[a.name]})
        with open(self.cfg_path, "w") as f:
            json.dump({"canon_dir": PROD.canon_dir, "arms": arms}, f)
        self.procs = {}
        self.logs = {}

    def env(self):
        return {"PATH": f"{HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin", "HOME": HOME,
                "LANG": "C.UTF-8", "RLTLDR_ROOT": self.root, "H2H_CONFIG": self.cfg_path}

    def sock(self, arm):
        return os.path.join(self.root, "run", "h2h", arm, "runner.sock")

    def ws(self, arm):
        return os.path.join(self.root, "data", "h2h", arm, "runner_ws")

    def runs(self, arm):
        return os.path.join(self.root, "data", "h2h", arm, "runs")

    def ledger_path(self, arm):
        return os.path.join(self.root, "data", "h2h", arm, "ledger.jsonl")

    def start(self, arm, wait=True):
        log = open(os.path.join(self.root, f"runner_{arm}.log"), "ab")
        p = subprocess.Popen([SERVE_PY, RUNNER, "--arm", arm], env=self.env(), stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        self.procs[arm] = p
        if wait:
            ok = wait_for(lambda: p.poll() is not None or (os.path.exists(self.sock(arm)) and port_open(PORTS[arm])),
                          600, 1.0)
            if p.poll() is not None:
                raise RuntimeError(f"runner {arm} exited: {self.tail(arm)}")
            assert ok
        return p

    def tail(self, arm, n=3000):
        try:
            return open(os.path.join(self.root, f"runner_{arm}.log"), errors="replace").read()[-n:]
        except OSError:
            return ""

    def stop_all(self):
        for p in self.procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in self.procs.values():
            try:
                p.wait(timeout=40)
            except subprocess.TimeoutExpired:
                p.kill()


def out_of(res, key):
    """The runner output a background submission got ("" if it failed or is missing)."""
    v = res.get(key)
    return v[1].get("output", "") if isinstance(v, tuple) and isinstance(v[1], dict) else ""


def submit_bg(sock, body, results, key):
    def go():
        try:
            results[key] = post_sock(sock, body)
        except Exception as e:  # noqa: BLE001
            results[key] = ("error", repr(e))
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def main():
    why = preconditions()
    if why:
        print(f"refusing to run: {why}")
        return 2
    E = Env()
    head = "0123456789abcdef0123456789abcdef01234567"
    try:
        t0 = time.time()
        E.start("base")
        E.start("v5")
        print(f"(both runners up after {time.time() - t0:.0f} s)", flush=True)

        # ---- workspace -------------------------------------------------------------------------
        start = subprocess.run(["git", "--git-dir", PROD.canon_dir, "rev-parse", f"{PROD.start_ref}^{{commit}}"],
                               capture_output=True, text=True).stdout.strip()
        oct3 = subprocess.run(["git", "--git-dir", PROD.canon_dir, "rev-parse", "autoresearch/oct3"],
                              capture_output=True, text=True).stdout.strip()
        for arm in ("base", "v5"):
            ws = E.ws(arm)
            g = lambda *a: subprocess.run(["git", *a], cwd=ws, capture_output=True, text=True)
            head_ws = g("rev-parse", "HEAD").stdout.strip()
            detached = g("symbolic-ref", "-q", "HEAD").returncode != 0
            has_oct3 = g("cat-file", "-e", f"{oct3}^{{commit}}").returncode == 0
            refs = g("for-each-ref", "--format=%(refname)").stdout.split()
            pin = open(os.path.join(ws, ".venv", "lib", "python3.10", "site-packages", "sitecustomize.py")).read()
            check(f"{arm}: workspace detached at h2h/start, no other branches' objects, venv pinned to its GPU",
                  head_ws == start and detached and not has_oct3 and all("oct3" not in r for r in refs)
                  and ARMS[arm].gpu_uuid in pin and not any(a.gpu_uuid in pin for a in PROD.arms if a.name != arm),
                  (head_ws, start, detached, has_oct3, refs))

        # ---- single instance per arm ------------------------------------------------------------
        first = E.procs["base"]
        p2 = E.start("base", wait=False)
        E.procs["base"] = first
        rc2 = p2.wait(timeout=60)
        st = http_tcp(PORTS["base"], "GET", "/health")
        check("second runner for the same arm exits (lock) and leaves the first one alone",
              rc2 == 1 and st == (200, {"ok": True, "arm": "base", "busy": False}), (rc2, st))

        # ---- a run through the untrusted socket ---------------------------------------------------
        code, d = post_sock(E.sock("base"), {"desc": "fake\ncrash \x00 run", "train_py": CRASH, "head": head.upper()})
        L = ledger(E.root, "base")
        e = L[-1] if L else {}
        check("run: client gets ar_run's output (status crash + tail), rc 1",
              code == 200 and d["rc"] == 1 and d["output"].startswith("status: crash")
              and "fake failure for the runner test" in d["output"], d)
        check("run: one ledger line: desc (sanitized), agent_head, no attempt, no cap, arm GPU, arm runs dir",
              len(L) == 1 and e["desc"] == "fake crash run" and e["agent_head"] == head and e["attempt_id"] is None
              and e["max_runs"] == 0 and e["gpu"] == ARMS["base"].gpu_uuid
              and e["log_path"].startswith(os.path.join(E.root, "data", "h2h", "base", "runs") + "/")
              and e["parent_rev"] == start, {k: e.get(k) for k in ("desc", "agent_head", "attempt_id", "max_runs",
                                                                  "gpu", "log_path", "parent_rev")})
        ws_train = open(os.path.join(E.ws("base"), "train.py")).read()
        check("run: workspace holds the submitted train.py at the start commit",
              ws_train == CRASH and subprocess.run(["git", "rev-parse", "HEAD"], cwd=E.ws("base"), capture_output=True,
                                                   text=True).stdout.strip() == start)
        check("run: nothing written to the other arm", not ledger(E.root, "v5"))
        frames = [l for l in d["output"].splitlines() if l.lstrip().startswith('File "')]
        check("run: the agent's output names only neutral paths (no host path, no arm dir, no 'data/h2h')",
              any(f'"{HOME}/autoresearch/train.py"' in l for l in frames) and E.root not in d["output"]
              and "data/h2h" not in d["output"] and "runner_ws" not in d["output"], d["output"])

        # descriptions that look like options (`./run.sh -baseline`) are recorded as given
        for desc in ("-baseline", "--lr=0.04"):
            code, d = post_sock(E.sock("base"), {"desc": desc, "train_py": CRASH, "head": head})
            check(f"run: description {desc!r} is passed to ar_run as one value and recorded",
                  d["rc"] == 1 and d["output"].startswith("status: crash") and ledger(E.root, "base")[-1]["desc"] == desc,
                  (d, ledger(E.root, "base")[-1].get("desc")))

        code, d = post_sock(E.sock("base"), {"desc": "bad head", "train_py": CRASH, "head": "../../etc/passwd"})
        check("run: malformed head -> agent_head null", ledger(E.root, "base")[-1]["agent_head"] is None)
        n = len(ledger(E.root, "base"))
        r1 = post_sock(E.sock("base"), {"desc": "x"})
        r2 = post_sock(E.sock("base"), None, raw=b"{not json")
        check("refusals: missing train.py -> refused rc 3, bad JSON -> harness_error; no ledger lines",
              r1[1]["rc"] == 3 and r1[1]["output"].startswith("status: refused")
              and r2[1]["output"].startswith("status: harness_error") and len(ledger(E.root, "base")) == n, (r1, r2))

        # ---- state, queueing, kill, cross-arm isolation ------------------------------------------
        res = {}
        th_a = submit_bg(E.sock("base"), {"desc": "sleep base", "train_py": SLEEP, "head": head}, res, "a")
        th_v = submit_bg(E.sock("v5"), {"desc": "sleep v5", "train_py": SLEEP, "head": head}, res, "v")
        busy = wait_for(lambda: http_tcp(PORTS["base"], "GET", "/control/state")[1]["busy"]
                        and bootstrap_procs(E.runs("base")) and bootstrap_procs(E.runs("v5")), 60)
        th_b = submit_bg(E.sock("base"), {"desc": "queued crash", "train_py": CRASH, "head": head}, res, "b")
        time.sleep(1.5)
        st = http_tcp(PORTS["base"], "GET", "/control/state")[1]
        check("state: busy with the in-flight job, 1 queued submission, free disk reported",
              busy and st["busy"] and st["job"]["desc"] == "sleep base" and st["job"]["head"] == head
              and st["queued"] == 1 and st["start_commit"] == start and st["disk_free_gb"] > 0, st)
        v5_before = bootstrap_procs(E.runs("v5"))
        t = time.time()
        code, k = http_tcp(PORTS["base"], "POST", "/control/kill")
        dt_kill = time.time() - t
        th_a.join(30)
        check("kill: /control/kill returns within a few seconds; client told the harness interrupted it (resubmit)",
              k["killed"] and k["job"]["desc"] == "sleep base" and dt_kill < 10
              and out_of(res, "a").startswith("status: harness_error")
              and "interrupted this run" in out_of(res, "a") and res["a"][1]["rc"] == 3, (dt_kill, k, res.get("a")))
        check("kill: nothing of base's run survives", not bootstrap_procs(E.runs("base")), bootstrap_procs(E.runs("base")))
        check("isolation: v5's run is untouched by base's kill",
              v5_before and bootstrap_procs(E.runs("v5")) == v5_before, (v5_before, bootstrap_procs(E.runs("v5"))))
        th_b.join(120)
        L = ledger(E.root, "base")
        check("queue: the queued run ran after the kill; ledger: interrupted line, then the queued run",
              out_of(res, "b").startswith("status: crash")
              and L[-2]["desc"] == "sleep base" and "interrupted" in L[-2]["flags"] and L[-1]["desc"] == "queued crash",
              ([x["desc"] for x in L[-2:]], res.get("b")))
        http_tcp(PORTS["v5"], "POST", "/control/kill")
        th_v.join(30)
        check("kill v5: interrupted ledger line in v5's ledger only",
              ledger(E.root, "v5") and "interrupted" in ledger(E.root, "v5")[-1]["flags"]
              and not bootstrap_procs(E.runs("v5")))
        st = http_tcp(PORTS["base"], "GET", "/control/state")[1]
        check("state: idle, n_runs counted, last run summarized",
              not st["busy"] and st["job"] is None and st["n_runs"] == 6 and st["last"]["desc"] == "queued crash"
              and st["last"]["status"] == "crash", st)

        # ---- runner crash with a run in flight -> the restarted runner reaps it ------------------
        res = {}
        th = submit_bg(E.sock("base"), {"desc": "orphan", "train_py": SLEEP, "head": head}, res, "o")
        wait_for(lambda: bootstrap_procs(E.runs("base")), 60)
        n = len(ledger(E.root, "base"))
        E.procs["base"].kill()
        E.procs["base"].wait()
        orphan = ar_run_procs(E.ledger_path("base"))
        t = time.time()
        E.start("base")
        check("restart: leftover ar_run of the crashed runner is stopped (ledger line written), runner up",
              orphan and not ar_run_procs(E.ledger_path("base")) and not bootstrap_procs(E.runs("base"))
              and len(ledger(E.root, "base")) == n + 1 and "interrupted" in ledger(E.root, "base")[-1]["flags"],
              (orphan, len(ledger(E.root, "base")) - n, time.time() - t))
        th.join(5)

        # ---- SIGTERM with a run in flight ---------------------------------------------------------
        res = {}
        th = submit_bg(E.sock("base"), {"desc": "shutdown", "train_py": SLEEP, "head": head}, res, "s")
        wait_for(lambda: bootstrap_procs(E.runs("base")), 60)
        t = time.time()
        E.procs["base"].send_signal(signal.SIGTERM)
        rc = E.procs["base"].wait(timeout=60)
        th.join(10)
        check("SIGTERM: run killed + recorded, client told to resubmit, runner exits 0, socket removed",
              rc == 0 and time.time() - t < 30 and "interrupted" in ledger(E.root, "base")[-1]["flags"]
              and out_of(res, "s").startswith("status: harness_error")
              and not os.path.exists(E.sock("base")) and not bootstrap_procs(E.runs("base")),
              (rc, time.time() - t, res.get("s")))
        E.start("base")

        # ---- run_client.py end-to-end (through uds_forward, like the sandbox) ----------------------
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        fwd = subprocess.Popen(["/usr/bin/python3", "-I", UDS_FWD, f"{port}:{E.sock('base')}"])
        try:
            wait_for(lambda: port_open(port), 10)
            repo = os.path.join(E.root, "client_repo")
            os.makedirs(repo)
            with open(os.path.join(repo, "train.py"), "w") as f:
                f.write(CRASH)
            g = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)
            g("init", "-q"); g("config", "user.name", "t"); g("config", "user.email", "t@t")
            g("add", "."); g("commit", "-qm", "c")
            repo_head = g("rev-parse", "HEAD").stdout.strip()
            p = subprocess.run(["/usr/bin/python3", "-I", CLIENT, "client", "test"], cwd=repo, capture_output=True,
                               text=True, env={"PATH": "/usr/bin:/bin", "AR_RUNNER_URL": f"http://127.0.0.1:{port}"},
                               timeout=300)
            e = ledger(E.root, "base")[-1]
            check("run_client: prints the runner output, exit 1, sends desc + the repo's git HEAD",
                  p.returncode == 1 and p.stdout.startswith("status: crash") and e["desc"] == "client test"
                  and e["agent_head"] == repo_head, (p.returncode, p.stdout[:200], e.get("agent_head"), repo_head))
            os.makedirs(os.path.join(E.root, "nogit"))
            shutil.copy(os.path.join(repo, "train.py"), os.path.join(E.root, "nogit"))
            p = subprocess.run(["/usr/bin/python3", "-I", CLIENT, "no git"], cwd=os.path.join(E.root, "nogit"),
                               capture_output=True, text=True,
                               env={"PATH": "/usr/bin:/bin", "AR_RUNNER_URL": f"http://127.0.0.1:{port}",
                                    "GIT_CEILING_DIRECTORIES": E.root}, timeout=300)
            check("run_client: outside a git repo -> empty head, run still accepted",
                  p.returncode == 1 and ledger(E.root, "base")[-1]["agent_head"] is None
                  and ledger(E.root, "base")[-1]["desc"] == "no git", (p.returncode, p.stdout[:200]))
        finally:
            fwd.terminate()
            fwd.wait()
    finally:
        E.stop_all()
        leftovers = bootstrap_procs(E.runs("base")) + bootstrap_procs(E.runs("v5"))
        if leftovers:
            subprocess.run(["sudo", "-n", "kill", "-KILL", *map(str, leftovers)])
        if os.environ.get("KEEP"):
            print("kept", E.root)
        else:
            shutil.rmtree(E.root, ignore_errors=True)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
