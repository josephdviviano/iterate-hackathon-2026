#!/usr/bin/env python3
"""End-to-end tests for rltldr/h2h_supervisor.py with the real pi binary (no vLLM, no GPU).

A throw-away RLTLDR_ROOT holds one arm (`base`): a tiny git repo whose run.sh calls the real tools/run_client.py,
and the pi agent template. tests/h2h_supervisor_fakes.py serves a scripted OpenAI-compatible model on a temporary
Unix socket (stand-in for the arm's gateway socket) plus a fake runner (/run socket + /control/state TCP).
H2H_SANDBOX points the supervisor at tests/h2h_supervisor_sandbox.sh, which forwards 127.0.0.1:8100/8200 inside a
private network namespace to those sockets (H2H_TEST_GATEWAY_SOCK / H2H_TEST_RUNNER_SOCK). Timings are shortened
through the supervisor's CLI flags. Checks:
  A  kickoff prompt sent; ./run.sh reaches the runner; nudge after every settle; back-off after 3 settles without
     a run (nudge delayed); events.jsonl filtered (no deltas, usage kept, U+2028 handled); status.json counters;
     a second supervisor for the same arm refuses to start
  V  the guard inside pi (env AR_GUARD_ALLOW_VENV_EXEC=1 from the supervisor): `.venv/bin/python -c ...` runs
     (program.md tells the agent to use it), `ls .venv/bin` is still blocked and logged to guard_blocks.jsonl
  B  back-off signal: runs submitted in a way the command-text heuristic cannot see (`bash -c "./run.sh ..."`) do
     not cause a back-off (the runner's trusted n_runs/last decide); with the runner unreachable the supervisor
     falls back to the command text (undetected form -> back-off; ./run.sh form -> no back-off)
  C  pi killed -> restart after the delay with --continue, restart prompt, history continued, back-off reset
  D  watchdog: silent pi during a long run (runner busy) is left alone; silent pi with an idle runner (stalled
     LLM request) is restarted
  E  SIGTERM: exit 0 within the grace period, status "stopped", no sandbox/pi process left
  G  supervisor SIGKILLed: its pi goes away on its own (stdin EOF); a new supervisor resumes with --continue
  H  orphan reaping without a pid file: a supervisor kills a leftover sandbox of ITS arm, never the other arm's
With --real-sandbox the supervisor uses its default sandbox, tools/h2h_sandbox.sh, with the fake sockets passed
through H2H_GATEWAY_SOCK / H2H_RUNNER_SOCK (arm-neutral layout inside: repo ~/autoresearch).
Needs pi (cfg.pi_bin) and passwordless sudo.

  $RLTLDR_SERVE_PY tests/test_h2h_supervisor.py [--real-sandbox]
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # code root
PY = sys.executable
sys.path.insert(0, ROOT)
from rltldr import h2h_supervisor as S  # noqa: E402

FAILS = []
GIT_ENV = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def jl(path):
    out = []
    try:
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out


def wait_for(cond, timeout, step=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return cond()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def unit_tests():
    yes = ['./run.sh "baseline" > run.log 2>&1', 'cd /x/repo && ./run.sh "a b" > run.log 2>&1; grep val run.log',
           "bash run.sh x", "timeout 900 ./run.sh 'x' > run.log", "FOO=1 ./run.sh x", "git commit -am x && ./run.sh y",
           "/x/rltldr/data/h2h/base/repo/run.sh d", "echo start\n./run.sh x > run.log 2>&1",
           "(./run.sh x > run.log 2>&1)", "./run.sh 'unbalanced > run.log"]
    no = ["cat run.sh", "grep val_bpb run.log", "git add train.py && git commit -m 'run.sh tweak'", "ls",
          "echo ./run.sh", "sed -n 1,5p ./run.sh"]
    check("invokes_run_sh: run.sh executions detected", all(S.invokes_run_sh(c) for c in yes),
          [c for c in yes if not S.invokes_run_sh(c)])
    check("invokes_run_sh: mentions/reads not counted", not any(S.invokes_run_sh(c) for c in no),
          [c for c in no if S.invokes_run_sh(c)])
    big = "x" * 10000 + "é" * 3000
    check("trunc caps at 2 KB (UTF-8 bytes)", len(S.trunc(big).encode()) <= 2048 and S.trunc("ab") == "ab")
    check("filter drops deltas", S.filter_record({"type": "message_update", "assistantMessageEvent": {}}) is None)
    r = S.filter_record({"type": "tool_execution_end", "toolCallId": "c", "toolName": "bash", "isError": False,
                         "result": {"content": [{"type": "text", "text": big}]}})
    check("filter truncates tool results", r["tool"] == "bash" and len(r["result"].encode()) <= 2048)
    check("run_in_flight", S.run_in_flight({"job": {"desc": "x"}}) and not S.run_in_flight({"job": None})
          and S.run_in_flight({"queue": 2}) and not S.run_in_flight({"queue": 0}) and not S.run_in_flight(None))
    m0 = S.run_marker({"n_runs": 3, "last": {"t0": 1.0, "t1": 2.0}})
    check("run_marker", m0 == (3, 2.0) and S.run_marker({"n_runs": 4, "last": {"t1": 2.0}}) != m0
          and S.run_marker({"n_runs": 3, "last": {"t1": 5.0}}) != m0 and S.run_marker({"job": None}) is None
          and S.run_marker(None) is None and S.run_marker({"n_runs": 0, "last": None}) == (0, None), m0)
    check("invokes_run_sh misses `bash -c \"./run.sh ...\"` (why the runner's state decides the back-off)",
          not S.invokes_run_sh(B_CMD))


B_CMD = 'bash -c "./run.sh nudge-run > run.log 2>&1"'      # a run the command-text heuristic cannot see
VENV_OK = "VENV_EXEC_OK"


def make_root(tmp, runner_port):
    for name in ("v5",):           # a second arm (only for the orphan-reaping test)
        os.makedirs(os.path.join(tmp, "data", "h2h", name, "repo"))
        os.makedirs(os.path.join(tmp, "data", "h2h", name, "session", "agent"))
    arm = os.path.join(tmp, "data", "h2h", "base")
    repo = os.path.join(arm, "repo")
    os.makedirs(repo)
    with open(os.path.join(repo, "train.py"), "w") as f:
        f.write("print('train')\n")
    with open(os.path.join(repo, "run.sh"), "w") as f:
        f.write(f'#!/bin/bash\nexec /usr/bin/python3 -I "{ROOT}/tools/run_client.py" "$@"\n')
    os.chmod(os.path.join(repo, "run.sh"), 0o755)
    # stand-in for the repo's venv interpreter (untracked, like the real .venv; read-only in the real sandbox)
    os.makedirs(os.path.join(repo, ".venv", "bin"))
    with open(os.path.join(repo, ".venv", "bin", "python"), "w") as f:
        f.write('#!/bin/sh\nexec /usr/bin/python3 -I "$@"\n')
    os.chmod(os.path.join(repo, ".venv", "bin", "python"), 0o755)
    with open(os.path.join(repo, ".gitignore"), "w") as f:
        f.write(".venv/\nrun.log\n")
    for args in (["init", "-q", "-b", "autoresearch/h2h"], ["config", "user.name", "autoresearch"],
                 ["config", "user.email", "autoresearch@localhost"], ["add", "-A"], ["commit", "-qm", "start"]):
        subprocess.run(["git", *args], cwd=repo, env=GIT_ENV, check=True, capture_output=True)
    shutil.copytree(os.path.join(ROOT, "pi", "agent"), os.path.join(arm, "session", "agent"),
                    ignore=shutil.ignore_patterns("auth.json", "models-store.json", "sessions"))
    from rltldr.h2h_config import H2HConfig
    arms = []
    for name, port in (("base", runner_port), ("v5", free_port())):
        a = H2HConfig().arm(name)
        arms.append({"name": name, "served_model": a.served_model, "adapter_dir": a.adapter_dir,
                     "gpu_uuid": a.gpu_uuid, "gpu_minor": a.gpu_minor, "runner_port": port})
    with open(os.path.join(tmp, "h2h_config.json"), "w") as f:
        json.dump({"arms": arms}, f)
    return arm


def pi_pids(status):
    """The pi process under the supervisor's sandbox root (pi overwrites its argv with the title "pi")."""
    root = json.load(open(status)).get("pi_pid")
    out = []
    for p in S.descendants(root) if root else []:
        try:
            if open(f"/proc/{p}/comm").read().strip() == "pi":
                out.append(p)
        except OSError:
            pass
    return out


def snapshot(status):
    """{pid: start ticks} of the supervisor's whole sandbox tree (sudo root + everything below it)."""
    root = json.load(open(status)).get("pi_pid")
    return {p: S.proc_start_ticks(p) for p in ([root] + S.descendants(root) if root else [])}


def alive(snap):
    """Pids of the snapshot that still run (same start time, not a zombie awaiting its parent's wait())."""
    out = []
    for p, t in snap.items():
        try:
            with open(f"/proc/{p}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
        except OSError:
            continue
        if t is not None and S.proc_start_ticks(p) == t and state != "Z":
            out.append(p)
    return out


def tree_left(sess, tmp, exclude=()):
    """Processes of the test arm still alive (pi, sandbox, forwarder), except the fakes."""
    left = S.pids_with_argv(b"--session-dir", sess.encode())
    left += S.pids_with_env(f"PI_CODING_AGENT_DIR={os.path.dirname(sess)}/agent".encode())
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if tmp.encode() in open(f"/proc/{d}/cmdline", "rb").read():
                    left.append(int(d))
            except OSError:
                pass
    return sorted(set(left) - {os.getpid(), *exclude})


def main():
    real = "--real-sandbox" in sys.argv
    unit_tests()
    tmp = tempfile.mkdtemp(prefix="h2h_sup_test_")
    port = free_port()
    arm = make_root(tmp, port)
    sess = os.path.join(arm, "session", "session")
    sockdir = os.path.join(tmp, "sock")
    os.makedirs(sockdir)
    llm_sock, runner_sock = os.path.join(sockdir, "llm.sock"), os.path.join(sockdir, "runner.sock")
    calls, ctl, argv_log = os.path.join(tmp, "calls.jsonl"), os.path.join(tmp, "ctl.json"), os.path.join(tmp, "argv.jsonl")
    events, status = os.path.join(arm, "events.jsonl"), os.path.join(arm, "status.json")

    def control(**kw):
        with open(ctl + ".tmp", "w") as f:
            json.dump(kw, f)
        os.replace(ctl + ".tmp", ctl)

    control()
    fakes = subprocess.Popen([PY, os.path.join(ROOT, "tests", "h2h_supervisor_fakes.py"), "--llm-sock",
                              llm_sock, "--runner-sock", runner_sock,
                              "--runner-port", str(port), "--log", calls, "--control", ctl],
                             stdout=subprocess.PIPE, stderr=open(os.path.join(tmp, "fakes.log"), "wb"))
    assert fakes.stdout.readline().strip() == b"ready"
    env = {k: v for k, v in os.environ.items() if k != "H2H_SANDBOX"}
    env["RLTLDR_ROOT"] = tmp
    if real:      # the supervisor's default sandbox (tools/h2h_sandbox.sh) with the fake sockets
        env.update(H2H_GATEWAY_SOCK=llm_sock, H2H_RUNNER_SOCK=runner_sock)
    else:
        env.update(H2H_SANDBOX=os.path.join(ROOT, "tests", "h2h_supervisor_sandbox.sh"),
                   H2H_TEST_GATEWAY_SOCK=llm_sock, H2H_TEST_RUNNER_SOCK=runner_sock, H2H_TEST_ARGV_LOG=argv_log)
    print(f"sandbox: {'tools/h2h_sandbox.sh (real)' if real else 'tests/h2h_supervisor_sandbox.sh (stub)'}")

    def pi_starts():
        return [e for e in jl(events) if e.get("event") == "pi_start"]
    flags = ["--idle-settles", "3", "--idle-backoff", "6", "--error-delay", "2", "--restart-delay", "2",
             "--restart-max", "8", "--stable-after", "1000", "--watchdog", "8", "--term-grace", "10"]
    sup_log = open(os.path.join(tmp, "supervisor.log"), "ab")

    def start_sup():
        return subprocess.Popen([PY, os.path.join(ROOT, "rltldr", "h2h_supervisor.py"), "--arm", "base", *flags],
                                env=env, stdout=sup_log, stderr=subprocess.STDOUT)

    def chats():
        return jl(calls)

    def prompts(text):
        return [c for c in chats() if c["last_role"] == "user" and c["last_user"] == text]

    sup = start_sup()
    try:
        # ---------------------------------------------------------------- A: kickoff, nudges, back-off
        ok = wait_for(lambda: len(prompts(S.NUDGE)) >= 5, 60)
        check("A: kickoff prompt is the first request", chats() and chats()[0]["last_user"] == S.KICKOFF,
              chats()[:1])
        check("A: pi requests model `policy` with streaming", chats()[0]["model"] == "policy" and chats()[0]["stream"])
        runs = jl(calls + ".runs")
        check("A: ./run.sh reached the runner through 127.0.0.1:8200", len(runs) == 1 and runs[0]["desc"] == "baseline",
              runs)
        nudges = prompts(S.NUDGE)
        check("A: nudges after settles (>= 5 within 60 s)", ok, len(nudges))
        if len(nudges) >= 5:
            gaps = [round(nudges[i + 1]["ts"] - nudges[i]["ts"], 2) for i in range(4)]
            check("A: first nudges immediate, back-off after 3 idle settles (gap >= 6 s)",
                  gaps[0] < 2 and gaps[1] < 2 and gaps[2] >= 5.5 and gaps[3] >= 5.5, gaps)
        st = json.load(open(status))
        check("A: status counters", st["n_runsh_calls"] == 1 and st["n_nudges"] >= 4 and st["n_backoffs"] >= 1
              and st["n_prompts"] == 1 + st["n_nudges"] and st["n_settles"] >= 4 and st["n_restarts"] == 0
              and st["pid"] == sup.pid and st["pi_pid"] and st["context_tokens"], st)
        sf = st.get("session_file") or ""
        check("A: status session file (pi's path; arm-neutral inside the real sandbox)",
              sf.endswith(".jsonl") and os.path.exists(os.path.join(sess, os.path.basename(sf))), sf)
        ev = jl(events)
        types = {e["type"] for e in ev}
        check("A: events filtered (no deltas)", not types & {"message_update", "tool_execution_update",
                                                             "message_start"}, types)
        check("A: events keep settle/agent/tool/message_end",
              {"agent_start", "agent_end", "agent_settled", "tool_execution_start", "tool_execution_end",
               "message_end", "supervisor"} <= types, types)
        me = [e for e in ev if e["type"] == "message_end" and e.get("role") == "assistant"]
        check("A: message_end keeps usage", me and all(e.get("usage", {}).get("totalTokens") for e in me))
        ts = [e for e in ev if e["type"] == "tool_execution_start"]
        check("A: tool_execution_start has tool + args", ts and ts[0]["tool"] == "bash" and "run.sh" in ts[0]["args"])
        check("A: every event has ts; no unparsable records (U+2028 inside JSON)",
              all("ts" in e for e in ev) and not any(e.get("event") == "bad_record" for e in ev))
        check("A: back-off logged", any(e.get("event") == "backoff" for e in ev))
        check("A: event lines are small", max(len(json.dumps(e)) for e in ev) < 8000)
        check("A: first pi start is a fresh session", pi_starts() and pi_starts()[0]["cont"] is False, pi_starts())
        if not real:
            first_argv = jl(argv_log)[0]
            check("A: pi command line", first_argv[0] == "base" and "--mode" in first_argv and "rpc" in first_argv
                  and "--continue" not in first_argv and "--no-approve" in first_argv
                  and first_argv[first_argv.index("--thinking") + 1] == "medium"
                  and "AR_GUARD_LOG=" + os.path.join(arm, "session", "guard_blocks.jsonl") in first_argv
                  and "AR_GUARD_WRITABLE=train.py,results.tsv" in first_argv
                  and "AR_GUARD_ALLOW_VENV_EXEC=1" in first_argv and "UV_CACHE_DIR=/tmp/uv-cache" in first_argv
                  and "PI_CODING_AGENT_DIR=" + os.path.join(arm, "session", "agent") in first_argv, first_argv)
        dup = start_sup()
        rc = dup.wait(timeout=20)
        check("A: a second supervisor for the same arm refuses to start", rc != 0)

        # ---------------------------------------------------------------- V: the guard lets .venv/bin/python run
        guard_log = os.path.join(arm, "session", "guard_blocks.jsonl")
        t_v = time.time()
        control(nudge_cmd=f".venv/bin/python -c \"print('{VENV_OK}')\"")
        ends = wait_for(lambda: [e for e in jl(events) if e.get("type") == "tool_execution_end" and e["ts"] > t_v], 30)
        check("V: `.venv/bin/python -c ...` passes the guard and runs (as program.md says)",
              ends and ends[0]["is_error"] is False and VENV_OK in ends[0]["result"], ends[:1])
        control(nudge_cmd="ls .venv/bin")
        blocked = wait_for(lambda: [b for b in jl(guard_log) if b.get("rule") == "venv"], 30)
        check("V: inspecting .venv is still blocked (logged to guard_blocks.jsonl)",
              blocked and blocked[0]["input"]["command"] == "ls .venv/bin", blocked)
        check("V: no venv-exec call was blocked", not [b for b in jl(guard_log) if VENV_OK in json.dumps(b)])

        # ---------------------------------------------------------------- B: back-off signal from the runner
        n_runs0 = len(jl(calls + ".runs"))
        t_b = time.time()
        control(nudge_cmd=B_CMD)
        wait_for(lambda: len(jl(calls + ".runs")) >= n_runs0 + 5, 60)
        b_runs = jl(calls + ".runs")[n_runs0:]
        check("B: runs via `bash -c ./run.sh` reached the runner", len(b_runs) >= 5 and
              all(r["desc"] == "nudge-run" for r in b_runs), b_runs)
        b_nudges = [c for c in prompts(S.NUDGE) if c["ts"] > t_b]
        gaps = [round(b_nudges[i + 1]["ts"] - b_nudges[i]["ts"], 2) for i in range(1, min(len(b_nudges), 5) - 1)]
        check("B: no back-off while runs happen (nudges immediate after the first run)",
              len(gaps) >= 3 and max(gaps) < 3, gaps)
        t_b1 = b_nudges[1]["ts"] if len(b_nudges) > 1 else t_b
        check("B: ... no back-off event once runs were seen",
              not [e for e in jl(events) if e.get("event") == "backoff" and e["ts"] > t_b1])
        st = json.load(open(status))
        check("B: decided by the runner's state (run_signal=runner, runner_n_runs tracked, run.sh text unseen)",
              st.get("run_signal") == "runner" and isinstance(st.get("runner_n_runs"), int)
              and 0 <= len(jl(calls + ".runs")) - st["runner_n_runs"] <= 1     # sampled at the last settle
              and st["n_runsh_calls"] == 1, {k: st.get(k) for k in ("run_signal", "runner_n_runs", "n_runsh_calls")})
        # runner unreachable -> command-text fallback: the undetected form now backs off ...
        t_b2 = time.time()
        control(nudge_cmd=B_CMD, state_down=True)
        bo = wait_for(lambda: [e for e in jl(events) if e.get("event") == "backoff" and e["ts"] > t_b2], 40)
        check("B: runner unreachable -> fallback to the command text (undetected runs back off)",
              bo and bo[0].get("signal") == "command", bo[:1])
        # ... and a plain ./run.sh is detected by the fallback (back-off reset, nudges immediate again)
        control(nudge_cmd='./run.sh "fallback-run" > run.log 2>&1', state_down=True)
        n3 = len(jl(calls + ".runs"))
        wait_for(lambda: len([r for r in jl(calls + ".runs")[n3:] if r["desc"] == "fallback-run"]) >= 4, 60)
        fb = [r for r in jl(calls + ".runs")[n3:] if r["desc"] == "fallback-run"]
        t_fb = fb[1]["ts"] if len(fb) > 1 else time.time()
        check("B: fallback detects ./run.sh: back-off reset, no further back-off",
              len(fb) >= 4 and not [e for e in jl(events) if e.get("event") == "backoff" and e["ts"] > t_fb]
              and json.load(open(status)).get("run_signal") == "command", (len(fb), json.load(open(status)).get("run_signal")))
        control()

        # ---------------------------------------------------------------- C: restart with --continue
        control(run_s=20)                       # the restart prompt makes the agent run: a 20 s run
        n_runs_c = len(jl(calls + ".runs"))
        n_before = max(c["n_messages"] for c in chats())
        victims = pi_pids(status)
        check("C: found the pi process", len(victims) == 1, victims)
        t_kill = time.time()
        for p in victims:
            os.kill(p, signal.SIGKILL)
        rp = wait_for(lambda: prompts(S.RESTART), 30)
        check("C: restart prompt after the restart delay", rp and rp[0]["ts"] - t_kill >= 1.8, rp)
        ps = pi_starts()
        check("C: restarted with --continue", len(ps) == 2 and ps[-1]["cont"] is True and
              (real or "--continue" in jl(argv_log)[-1]), ps)
        check("C: session history continued", rp and rp[0]["n_messages"] > n_before and
              rp[0]["first_user"] == S.KICKOFF[:200], (rp[:1], n_before))
        # D1: while the 20 s run is in flight pi is silent > watchdog (8 s), but the runner is busy
        wait_for(lambda: jl(calls + ".runs")[n_runs_c:], 40)
        st = json.load(open(status))
        check("D: no watchdog restart while a run is in flight (silence > 8 s)",
              st["n_watchdog"] == 0 and st["n_restarts"] == 1, st)
        check("D: the watchdog consulted the runner and saw the run", st.get("runner_busy") is True,
              st.get("runner_busy"))
        # back-off reset: before the restart the arm was backing off; the run resets the idle counter, so the
        # first nudge after the restart comes right after the run
        t_rp = rp[0]["ts"] if rp else 0
        after = wait_for(lambda: [c for c in prompts(S.NUDGE) if c["ts"] > t_rp], 15)
        t_run_end = jl(calls + ".runs")[-1]["ts"]
        gap = round(after[0]["ts"] - t_run_end, 2) if after else None
        check("C: back-off reset by the run (next nudge immediate)", gap is not None and 0 <= gap < 3, gap)

        # ---------------------------------------------------------------- D2: watchdog fires on a stalled LLM
        control(stall=True)
        wd = wait_for(lambda: [e for e in jl(events) if e.get("event") == "watchdog"], 60)
        check("D: watchdog restarts a silent pi when no run is in flight", bool(wd), len(jl(events)))
        control()
        rs = wait_for(lambda: [e for e in jl(events) if e.get("event") == "pi_start" and e.get("reason") == "watchdog"], 30)
        check("D: restarted (reason watchdog, --continue)", bool(rs) and rs[0].get("cont") is True, rs)
        wait_for(lambda: len(prompts(S.RESTART)) >= 2, 30)
        st = json.load(open(status))
        check("D: status counts the watchdog restart", st["n_watchdog"] == 1 and st["n_restarts"] == 2, st)

        # ---------------------------------------------------------------- E: SIGTERM
        wait_for(lambda: json.load(open(status))["state"] in ("settled", "backoff", "working"), 20)
        snap = snapshot(status)
        check("E: sandbox tree snapshot taken", len(snap) >= 3, snap)
        t0 = time.time()
        sup.send_signal(signal.SIGTERM)
        rc = sup.wait(timeout=45)
        dt = time.time() - t0
        check("E: SIGTERM -> exit 0 within the grace period", rc == 0 and dt < 15, (rc, round(dt, 1)))
        st = json.load(open(status))
        check("E: status stopped", st["state"] == "stopped" and st["pi_pid"] is None, st["state"])
        left = wait_for(lambda: not alive(snap) and not tree_left(sess, tmp, (fakes.pid,)), 10) or \
            (alive(snap), tree_left(sess, tmp, (fakes.pid,)))
        check("E: no sandbox / pi / forwarder process left", left is True, left)
        check("E: stop recorded", any(e.get("event") == "stopped" for e in jl(events)))

        # ---------------------------------------------------------------- G: SIGKILL of the supervisor
        n_rp = len(prompts(S.RESTART))
        sup = start_sup()
        wait_for(lambda: len(prompts(S.RESTART)) > n_rp, 30)
        st = json.load(open(status))
        check("G: new supervisor resumes the session (--continue, restart prompt, counters kept)",
              len(prompts(S.RESTART)) > n_rp and pi_starts()[-1]["cont"] is True and st["n_restarts"] == 3
              and st["n_runsh_calls"] >= 2, st)
        snap = snapshot(status)
        sup.kill()
        sup.wait()
        left = wait_for(lambda: not alive(snap) and not tree_left(sess, tmp, (fakes.pid,)), 20) or \
            (alive(snap), tree_left(sess, tmp, (fakes.pid,)))
        check("G: after SIGKILL of the supervisor, pi exits by itself (stdin EOF)", left is True, left)

        # ---------------------------------------------------------------- H: orphan reaping is per arm
        sandbox = env.get("H2H_SANDBOX") or os.path.join(ROOT, "tools", "h2h_sandbox.sh")
        strays = {}
        for name in ("base", "v5"):
            sd = os.path.join(tmp, "data", "h2h", name, "session")
            # a leftover sandbox whose command names the arm's session dir (as pi's --session-dir does)
            strays[name] = subprocess.Popen([sandbox, name, "/bin/bash", "-c", "sleep 300", os.path.join(sd, "x")],
                                            cwd=os.path.join(tmp, "data", "h2h", name, "repo"), env=env,
                                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(2)
        snaps = {n: {p.pid: S.proc_start_ticks(p.pid), **{d: S.proc_start_ticks(d) for d in S.descendants(p.pid)}}
                 for n, p in strays.items()}
        check("H: two leftover sandboxes running", all(len(v) >= 3 for v in snaps.values()), snaps)
        if os.path.exists(os.path.join(arm, "supervisor_pi.json")):
            os.unlink(os.path.join(arm, "supervisor_pi.json"))
        sup = start_sup()
        ok = wait_for(lambda: not alive(snaps["base"]), 20)
        check("H: the base supervisor killed base's leftover sandbox", ok, alive(snaps["base"]))
        check("H: ... and left the v5 arm's sandbox alone", len(alive(snaps["v5"])) == len(snaps["v5"]),
              (alive(snaps["v5"]), snaps["v5"]))
        sup.send_signal(signal.SIGTERM)
        sup.wait(timeout=45)
        strays["v5"].stdin.close()
        S.kill_tree(strays["v5"].pid)
        for p in strays.values():
            p.wait(timeout=20)
    finally:
        if sup.poll() is None:
            sup.send_signal(signal.SIGTERM)
            try:
                sup.wait(timeout=45)
            except subprocess.TimeoutExpired:
                sup.kill()
        for p in tree_left(sess, tmp, (fakes.pid,)):
            subprocess.run(["sudo", "-n", "kill", "-9", str(p)], capture_output=True)
        fakes.kill()
        fakes.wait()
        if FAILS:
            print(f"--- supervisor log (kept in {tmp}) ---")
            print(open(os.path.join(tmp, "supervisor.log")).read()[-6000:])
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
