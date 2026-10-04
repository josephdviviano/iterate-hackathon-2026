"""h2h supervisor: ONE continuous pi session (RPC mode) for one arm of the head-to-head run.

  $RLTLDR_SERVE_PY rltldr/h2h_supervisor.py --arm base|v5        (normally started by ./ctl_h2h.sh)

Upstream-autoresearch style: the agent follows program.md in one long session (edit train.py, commit, ./run.sh,
log results.tsv, keep/reset with git, never stop). The supervisor only plays the human:
  * fresh session: sends the kickoff prompt; existing session (pi restarted with --continue): sends the
    "you were interrupted by a restart" prompt;
  * whenever pi reports `agent_settled` (no automatic work left) it sends the upstream "continue" nudge.
    Back-off: after `idle_settles` (3) consecutive settles without a run in between, every further nudge waits
    `idle_backoff` (5 min); reset by the next run. "A run happened since the last settle" comes from the arm's
    runner (trusted `GET /control/state`: its run count / last run changed, or a run is in flight or queued), so
    it does not depend on how the agent invokes ./run.sh (bash -c, a helper script, run_client.py directly);
    the ./run.sh command-text heuristic is only the fallback while the runner is unreachable. A settle right
    after a provider error waits `error_delay` (60 s);
  * pi exits or crashes -> restart with --continue after 30 s (doubling up to 10 min while restarts keep failing
    within `stable_after`);
  * watchdog: no pi record for 45 min while the agent is working and no run is in flight for the arm (runner
    `GET /control/state`) -> restart pi.
pi runs as `<sandbox> <arm> /usr/bin/env -i <env> pi --mode rpc ...` with cwd = arm.repo and host paths in the
arguments (h2h_sandbox.sh maps them to arm-neutral inside paths); pi = cfg.pi_bin, looked up on the inner PATH
unless absolute; HOME inside = this user's $HOME (the tree the sandbox covers). The sandbox is tools/h2h_sandbox.sh
next to this code (override with H2H_SANDBOX, same calling convention, for tests). RPC framing is strict JSONL:
stdout is read as bytes and split on LF only (U+2028/U+2029 inside JSON strings are not boundaries).

Outputs (both arms identical):
  arm.events   filtered pi records + supervisor records, one JSON line each with `ts` (streaming deltas and
               tool/turn updates dropped; message_end summarised with usage; tool args/results <= 2 KB)
  arm.status   json snapshot: state, pids, counters (prompts, nudges, restarts, compactions, run.sh calls, ...),
               last event, context tokens
  arm.dir/pi_stderr.log   pi's stderr
SIGTERM/SIGINT: close pi's stdin (pi shuts down in order), wait 30 s, then kill the sandbox tree with sudo; exit 0.
"""
import argparse
import asyncio
import fcntl
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rltldr.h2h_config import load_h2h_config  # noqa: E402
from rltldr.io_utils import read_json, write_json_atomic  # noqa: E402

log = logging.getLogger("h2h_supervisor")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")          # tools/h2h_sandbox.sh covers the invoking user's $HOME

KICKOFF = ("Hi! Have a look at program.md and let's kick off a new experiment. The setup steps that need the human "
           "are already done (see program.md). Do the rest of the setup and start the experiment loop.")
NUDGE = ("Continue the experiment loop. NEVER STOP: do not ask for confirmation, keep running experiments until you "
         "are interrupted.")
RESTART = "You were interrupted by a restart. Continue the experiment loop."

MAX_FIELD = 2048
DROP_EVENTS = {"message_update", "message_start", "tool_execution_update", "turn_start", "turn_end", "queue_update",
               "bash_execution_update"}
DIALOGS = {"select", "confirm", "input", "editor"}
COUNTERS = ("n_prompts", "n_nudges", "n_restarts", "n_compactions", "n_runsh_calls", "n_settles", "n_tool_calls",
            "n_assistant_messages", "n_errors", "n_watchdog", "n_backoffs", "n_pi_starts")


# ----------------------------------------------------------------------------------------------- helpers
def trunc(s, n: int = MAX_FIELD) -> str:
    """At most n bytes of UTF-8 (a marker says how much was cut)."""
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    b = s.encode("utf-8", "replace")
    if len(b) <= n:
        return s
    return b[:n - 40].decode("utf-8", "ignore") + f"...[truncated {len(b) - n + 40} bytes]"


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text")


def invokes_run_sh(command: str) -> bool:
    """Does a bash command execute run.sh (./run.sh, bash run.sh, /path/run.sh, after cd/&&/;/| ...)?"""
    for line in command.splitlines():
        try:
            lx = shlex.shlex(line, posix=True, punctuation_chars=True)
            lx.whitespace_split = True
            toks = list(lx)
        except ValueError:
            if re.search(r"(?:^|[\s;&|(])(?:\./|/\S*/)run\.sh\b", line):
                return True
            continue
        at_cmd = True
        for t in toks:
            if t and set(t) <= set(";&|()"):
                at_cmd = True
                continue
            if not at_cmd:
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", t) or t in ("time", "exec", "command", "nohup", "env",
                                                                     "bash", "sh", "timeout", "nice", "stdbuf"):
                continue              # env assignments / wrappers: the command word comes later
            if re.fullmatch(r"-\S*|\d+[smh]?", t):
                continue              # wrapper options (timeout 900, bash -x ...)
            if os.path.basename(t) == "run.sh":
                return True
            at_cmd = False
    return False


def summarize_message(m: dict) -> dict:
    role = m.get("role")
    out = {"role": role}
    if role == "assistant":
        content = m.get("content") or []
        out.update(stop_reason=m.get("stopReason"), usage=m.get("usage"),
                   text=trunc(content_text(content)),
                   thinking_chars=sum(len(c.get("thinking") or "") for c in content
                                      if isinstance(c, dict) and c.get("type") == "thinking"),
                   tool_calls=[{"name": c.get("name"), "args": trunc(c.get("arguments"), 512)}
                               for c in content if isinstance(c, dict) and c.get("type") == "toolCall"])
        if m.get("errorMessage"):
            out["error"] = trunc(m["errorMessage"])
    elif role == "user":
        out["text"] = trunc(content_text(m.get("content")))
    elif role == "toolResult":
        out.update(tool=m.get("toolName"), is_error=m.get("isError"))
    else:
        out["chars"] = len(json.dumps(m, ensure_ascii=False))
    return out


def filter_record(ev: dict):
    """The compact form of a pi RPC record for arm.events, or None to drop it."""
    t = ev.get("type")
    if t in DROP_EVENTS:
        return None
    if t == "message_end":
        return {"type": t, **summarize_message(ev.get("message") or {})}
    if t == "agent_end":
        return {"type": t, "will_retry": ev.get("willRetry"), "n_messages": len(ev.get("messages") or [])}
    if t == "tool_execution_start":
        return {"type": t, "id": ev.get("toolCallId"), "tool": ev.get("toolName"), "args": trunc(ev.get("args"))}
    if t == "tool_execution_end":
        res = ev.get("result")
        text = content_text(res.get("content")) if isinstance(res, dict) else res
        return {"type": t, "id": ev.get("toolCallId"), "tool": ev.get("toolName"), "is_error": ev.get("isError"),
                "result": trunc(text if text is not None else "")}
    if t == "response":
        out = {k: ev.get(k) for k in ("type", "id", "command", "success", "error") if k in ev}
        d = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if ev.get("command") == "get_session_stats":
            out["data"] = {k: d.get(k) for k in ("sessionFile", "totalMessages", "tokens", "contextUsage")}
        elif ev.get("command") == "get_state":
            out["data"] = {k: d.get(k) for k in ("sessionFile", "sessionId", "messageCount", "thinkingLevel",
                                                 "autoCompactionEnabled")}
        elif d:
            out["data"] = trunc(d, 512)
        return out
    if t == "compaction_end":
        out = {k: ev.get(k) for k in ("type", "reason", "aborted", "willRetry", "errorMessage") if k in ev}
        r = ev.get("result") or {}
        out.update({k: r.get(k) for k in ("tokensBefore", "estimatedTokensAfter", "usage") if k in r})
        if r.get("summary"):
            out["summary"] = trunc(r["summary"])
        return out
    if t == "extension_ui_request":
        return {"type": t, "method": ev.get("method"), "id": ev.get("id"),
                "text": trunc(ev.get("message") or ev.get("title") or "", 512)}
    s = json.dumps(ev, ensure_ascii=False)          # agent_start/settled, compaction_start, retries, errors, ...
    return ev if len(s.encode()) <= MAX_FIELD else {"type": t, "trunc": trunc(s)}


def run_in_flight(state: dict) -> bool:
    """Interpret the runner's /control/state: a run is in flight or queued for this arm."""
    if not isinstance(state, dict):
        return False
    for k in ("job", "inflight", "in_flight", "running", "busy", "current"):
        if state.get(k):
            return True
    q = state.get("queue", state.get("queued"))
    return bool(q) if not isinstance(q, (int, float)) else q > 0


def run_marker(state):
    """What changes in the runner's /control/state whenever a run finishes: (n_runs, end time of the last run);
    None if the state carries neither (unknown runner / unreachable)."""
    if not isinstance(state, dict):
        return None
    n = state.get("n_runs")
    last = state.get("last") if isinstance(state.get("last"), dict) else {}
    n = n if isinstance(n, int) and not isinstance(n, bool) else None
    t = last.get("t1") or last.get("t0") or last.get("ts")
    if n is None and t is None:
        return None
    return (n, t)


def descendants(root: int) -> list:
    """All live descendant pids of `root` (via /proc; includes processes in child PID namespaces)."""
    children = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    out, stack = [], [root]
    while stack:
        for c in children.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


def proc_start_ticks(pid: int):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def kill_tree(root: int, sig: int = signal.SIGKILL) -> None:
    """Signal a sandbox process tree (sudo/unshare run as root, so use `sudo kill`)."""
    pids = [p for p in descendants(root) + [root] if os.path.exists(f"/proc/{p}")]
    if pids:
        subprocess.run(["sudo", "-n", "kill", f"-{int(sig)}", *map(str, pids)], capture_output=True)


def sandbox_roots(paths) -> list:
    """sudo/unshare sandbox wrapper processes whose arguments name one of `paths` (host paths of ONE arm).
    Inside the sandbox pi's argv/env carry arm-neutral paths (identical in both arms) or just its process title,
    but the wrappers outside it keep the host paths (h2h_sandbox.sh passes them in its expose spec)."""
    pats = [re.compile(re.escape(p.rstrip("/").encode()) + rb"(?![\w.-])") for p in paths]
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        if b"unshare" in argv and any(rx.search(a) for a in argv for rx in pats):
            out.append(int(d))
    return out


def pids_with_argv(*needle: bytes) -> list:
    """Processes whose argv contains the consecutive arguments `needle`."""
    out, n = [], len(needle)
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        if any(tuple(argv[i:i + n]) == needle for i in range(len(argv) - n + 1)):
            out.append(int(d))
    return out


def pids_with_env(entry: bytes) -> list:
    """Processes (readable by us: our uid) whose environment contains `entry` (e.g. b"PI_CODING_AGENT_DIR=...").
    pi overwrites its argv with its process title, so its own command line cannot identify it."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open(f"/proc/{d}/environ", "rb") as f:
                if entry in f.read().split(b"\0"):
                    out.append(int(d))
        except OSError:
            continue
    return out


def session_files(sess: str) -> list:
    """pi session files. The session dir is agent-writable: a symlinked top dir is not walked (os.walk itself
    never enters symlinked subdirs)."""
    out = []
    if os.path.islink(sess):
        return out
    for d, _, files in os.walk(sess):
        out += [os.path.join(d, f) for f in files if f.endswith(".jsonl")]
    return out


# ------------------------------------------------------------------------------------------------- pi rpc
class PiRpc:
    """One pi --mode rpc child: JSONL commands to stdin, records from stdout (split on LF only)."""

    def __init__(self, cmd, cwd, env, stderr_path):
        self.cmd, self.cwd, self.env, self.stderr_path = cmd, cwd, env, stderr_path
        self.proc = None
        self.t_start = None

    async def start(self):
        with open(self.stderr_path, "ab") as err:
            self.proc = await asyncio.create_subprocess_exec(*self.cmd, cwd=self.cwd, env=self.env,
                                                             stdin=asyncio.subprocess.PIPE,
                                                             stdout=asyncio.subprocess.PIPE, stderr=err,
                                                             start_new_session=True)
        self.t_start = time.time()

    @property
    def pid(self):
        return self.proc.pid if self.proc else None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def send(self, obj: dict) -> bool:
        if not self.alive() or self.proc.stdin.is_closing():
            return False
        try:
            self.proc.stdin.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")
            await self.proc.stdin.drain()
            return True
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as e:
            log.warning("could not write to pi: %s", e)
            return False

    def close_stdin(self):
        if self.proc and self.proc.stdin and not self.proc.stdin.is_closing():
            self.proc.stdin.close()

    async def records(self):
        """Yield (record or None for an unparsable line, raw length) until stdout reaches EOF."""
        buf = bytearray()
        while True:
            chunk = await self.proc.stdout.read(1 << 16)
            if not chunk:
                break
            buf += chunk
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                line = bytes(buf[:i])
                del buf[:i + 1]
                if line.endswith(b"\r"):
                    line = line[:-1]
                if not line.strip():
                    continue
                try:
                    yield json.loads(line), len(line)
                except ValueError:
                    yield None, len(line)
        if buf.strip():
            try:
                yield json.loads(bytes(buf)), len(buf)
            except ValueError:
                yield None, len(buf)


# --------------------------------------------------------------------------------------------- supervisor
class Supervisor:
    def __init__(self, cfg, arm, opts):
        self.cfg, self.arm, self.o = cfg, arm, opts
        self.sandbox = os.environ.get("H2H_SANDBOX") or os.path.join(CODE_ROOT, "tools", "h2h_sandbox.sh")
        self.sess_root = arm.session_dir
        self.sess = os.path.join(arm.session_dir, "session")
        self.pidfile = os.path.join(arm.dir, "supervisor_pi.json")
        self.pi = None
        self.stopping = False
        self.stop_event = None
        self.stop_task = None
        self.nudge_task = None
        self.events_f = None
        self.req = 0
        self.failures = 0                       # consecutive quick pi exits (restart back-off)
        self.idle_settles = 0                   # consecutive settles without a run
        self.runsh_since_settle = False         # command-text fallback (runner unreachable)
        self.run_marker = None                  # runner's run_marker() at the last settle (trusted run signal)
        self.agent_busy = False                 # a prompt is being worked on (between prompt and settle)
        self.last_stop_reason = None
        self.open_runsh = set()                 # toolCallIds of ./run.sh calls still executing
        self.first_prompt = None                # kind of the prompt to send once pi answers get_state
        self.restart_reason = None
        self.status_dirty = False
        self.st = self._load_status()

    # ------------------------------------------------------------------------------------ status/events
    def _load_status(self) -> dict:
        now = time.time()
        old = read_json(self.arm.status, {}) if session_files(self.sess) else {}
        st = {"arm": self.arm.name, "state": "starting", "started_at": old.get("started_at") or now,
              "supervisor_started_at": now, "pid": os.getpid(), "pi_pid": None, "pi_started_at": None,
              "session_file": old.get("session_file"), "last_event_ts": None, "last_event_type": None,
              "last_prompt_ts": old.get("last_prompt_ts"), "last_prompt_kind": old.get("last_prompt_kind"),
              "last_runsh_ts": old.get("last_runsh_ts"), "context_tokens": old.get("context_tokens"),
              "context_window": old.get("context_window"), "context_percent": old.get("context_percent"),
              "session_tokens": old.get("session_tokens"), "last_usage": old.get("last_usage"),
              "last_stop_reason": None, "last_error": old.get("last_error"), "idle_settles": 0,
              "next_nudge_ts": None, "runner_busy": None, "runner_n_runs": old.get("runner_n_runs"),
              "run_signal": None, "sandbox": self.sandbox}
        for k in COUNTERS:
            st[k] = int(old.get(k) or 0)
        return st

    def write_status(self, **kw):
        self.st.update(kw)
        self.st["updated_at"] = time.time()
        try:
            write_json_atomic(self.arm.status, self.st)
            self.status_dirty = False
        except OSError as e:
            log.error("could not write %s: %s", self.arm.status, e)

    def record(self, rec: dict):
        rec = {"ts": round(time.time(), 3), **rec}
        try:
            self.events_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.events_f.flush()
        except (OSError, ValueError) as e:
            log.error("could not write %s: %s", self.arm.events, e)

    def sup(self, event: str, **kw):
        self.record({"type": "supervisor", "event": event, **kw})

    # ---------------------------------------------------------------------------------------- process
    def pi_cmd(self, cont: bool) -> list:
        sd = self.sess_root
        inner = {
            "PATH": f"{HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin", "HOME": HOME,
            "LANG": "C.UTF-8", "TERM": "dumb",
            "PI_CODING_AGENT_DIR": os.path.join(sd, "agent"), "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0", "CUDA_VISIBLE_DEVICES": "",
            "AR_GUARD_LOG": os.path.join(sd, "guard_blocks.jsonl"),
            # the guard's default protected dir (pi/agent/extensions/guard.ts): the project root
            "RLTLDR_ROOT": self.cfg.root,
            # program.md has the agent write results.tsv itself (the guard's default only allows train.py)
            "AR_GUARD_WRITABLE": "train.py,results.tsv",
            # program.md: "`.venv/bin/python` has torch (CPU only in your shell) ... for quick checks". The guard
            # blocks every mention of .venv unless this lets the repo's .venv/bin/python run as a command (the venv
            # is a read-only mount in the sandbox; inspecting/modifying .venv stays blocked)
            "AR_GUARD_ALLOW_VENV_EXEC": "1",
            # `uv run python ...` (upstream habit) needs a writable cache: ~/.cache/uv is read-only in the sandbox
            "UV_CACHE_DIR": "/tmp/uv-cache", "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
        }
        pi = [self.cfg.pi_bin, "--mode", "rpc", "--no-approve", "--no-context-files", "--provider", "rl",
              "--model", "policy", "--thinking", self.cfg.agent_thinking, "--session-dir", self.sess]
        if cont:
            pi.append("--continue")
        return [self.sandbox, self.arm.name, "/usr/bin/env", "-i", *[f"{k}={v}" for k, v in inner.items()], *pi]

    def reap_orphans(self):
        """A previous supervisor's pi (killed without cleanup) must not keep working in this arm's repo."""
        old = read_json(self.pidfile, None) if os.path.exists(self.pidfile) else None
        if old and proc_start_ticks(old.get("pid", -1)) == old.get("start_ticks"):
            log.warning("killing the previous supervisor's pi sandbox tree (pid %s)", old["pid"])
            kill_tree(old["pid"])
        # fallback without a pid file: sandbox wrappers naming this arm's repo/session dir; with host-path layouts
        # also pi and its tool processes by their environment (never matches the other arm: different paths)
        strays = sorted(set(sandbox_roots([self.arm.repo, self.sess_root]) +
                            pids_with_env(f"PI_CODING_AGENT_DIR={self.sess_root}/agent".encode())))
        if strays:
            log.warning("killing stray pi process(es) of this arm: %s", strays)
            for p in strays:
                kill_tree(p)
        if os.path.exists(self.pidfile):
            os.unlink(self.pidfile)

    async def send(self, obj: dict) -> bool:
        return await self.pi.send(obj) if self.pi else False

    def next_id(self, kind: str) -> str:
        self.req += 1
        return f"{kind}-{self.st['n_pi_starts']}-{self.req}"

    async def prompt(self, kind: str):
        text = {"kickoff": KICKOFF, "nudge": NUDGE, "restart": RESTART}[kind]
        rid = self.next_id(kind)
        if not await self.send({"id": rid, "type": "prompt", "message": text}):
            return
        self.agent_busy = True
        self.st["n_prompts"] += 1
        if kind == "nudge":
            self.st["n_nudges"] += 1
        self.sup("prompt", kind=kind, id=rid)
        log.info("prompt (%s) #%d sent%s", kind, self.st["n_prompts"],
                 f" [nudge #{self.st['n_nudges']}]" if kind == "nudge" else "")
        self.write_status(state="working", last_prompt_ts=time.time(), last_prompt_kind=kind, next_nudge_ts=None)

    def schedule_nudge(self, delay: float, why: str):
        if self.nudge_task and not self.nudge_task.done():
            self.nudge_task.cancel()
        state = "backoff" if why == "idle back-off" else "settled"
        self.write_status(state=state, next_nudge_ts=time.time() + delay)
        self.nudge_task = asyncio.ensure_future(self._nudge_later(delay))

    async def _nudge_later(self, delay: float):
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if self.stopping or not self.pi or not self.pi.alive() or self.agent_busy:
            return
        await self.prompt("nudge")

    # ------------------------------------------------------------------------------------ records
    async def on_record(self, ev: dict):
        now = time.time()
        t = ev.get("type")
        self.st["last_event_ts"], self.st["last_event_type"] = now, t
        self.status_dirty = True
        rec = filter_record(ev)
        if rec is not None:
            self.record(rec)
        if t == "agent_start":
            self.agent_busy = True
            if self.nudge_task and not self.nudge_task.done():
                self.nudge_task.cancel()
            if self.st["state"] != "working":
                self.write_status(state="working", next_nudge_ts=None)
        elif t == "message_end":
            m = ev.get("message") or {}
            if m.get("role") == "assistant":
                self.st["n_assistant_messages"] += 1
                self.last_stop_reason = m.get("stopReason")
                u = m.get("usage") or {}
                self.st["last_usage"] = u
                self.st["last_stop_reason"] = self.last_stop_reason
                if u.get("totalTokens"):
                    self.st["context_tokens"] = u.get("totalTokens")
                if m.get("stopReason") == "error":
                    self.st["n_errors"] += 1
                    self.st["last_error"] = trunc(m.get("errorMessage") or "", 512)
                    log.warning("pi error: %s", self.st["last_error"])
        elif t == "tool_execution_start":
            self.st["n_tool_calls"] += 1
            args = ev.get("args") or {}
            if ev.get("toolName") == "bash" and invokes_run_sh(str(args.get("command") or "")):
                # informational + fallback signals only: the back-off decision is taken at the settle, from the
                # runner's state when it answers (see ran_since_last_settle)
                self.st["n_runsh_calls"] += 1
                self.st["last_runsh_ts"] = now
                self.runsh_since_settle = True
                self.open_runsh.add(ev.get("toolCallId"))
                log.info("run.sh call #%d: %s", self.st["n_runsh_calls"], trunc(args.get("command"), 200))
        elif t == "tool_execution_end":
            self.open_runsh.discard(ev.get("toolCallId"))
        elif t == "compaction_start":
            log.info("compaction started (%s)", ev.get("reason"))
        elif t == "compaction_end":
            ok = bool(ev.get("result")) and not ev.get("aborted")
            if ok:
                self.st["n_compactions"] += 1
                await self.send({"id": self.next_id("stats"), "type": "get_session_stats"})
            log.info("compaction #%d %s (%s): %s -> %s tokens", self.st["n_compactions"],
                     "done" if ok else "FAILED", ev.get("reason"), (ev.get("result") or {}).get("tokensBefore"),
                     (ev.get("result") or {}).get("estimatedTokensAfter"))
            if not ok:
                self.st["n_errors"] += 1
                self.st["last_error"] = trunc(ev.get("errorMessage") or "compaction aborted", 512)
        elif t in ("auto_retry_start", "auto_retry_end", "extension_error"):
            log.warning("pi %s: %s", t, trunc(json.dumps(ev), 400))
            if t == "extension_error" or (t == "auto_retry_end" and not ev.get("success")):
                self.st["n_errors"] += 1
                self.st["last_error"] = trunc(ev.get("finalError") or ev.get("error") or t, 512)
        elif t == "extension_ui_request" and ev.get("method") in DIALOGS:
            await self.send({"type": "extension_ui_response", "id": ev.get("id"), "cancelled": True})
            log.info("cancelled extension dialog %s: %s", ev.get("method"), trunc(ev.get("title") or "", 200))
        elif t == "response":
            await self.on_response(ev)
        elif t == "agent_settled":
            await self.on_settled()

    async def on_response(self, ev: dict):
        rid, cmd, d = str(ev.get("id") or ""), ev.get("command"), ev.get("data")
        d = d if isinstance(d, dict) else {}
        if not ev.get("success"):
            self.st["n_errors"] += 1
            self.st["last_error"] = trunc(f"{cmd}: {ev.get('error')}", 512)
            log.error("pi rejected %s (%s): %s", cmd, rid, ev.get("error"))
            if cmd == "prompt" and "stream" not in str(ev.get("error")).lower():
                self.agent_busy = False
                self.schedule_nudge(self.o.error_delay, "prompt rejected")
            return
        if cmd == "prompt" and d.get("disposition") == "handled":       # no run started: no settle will follow
            self.agent_busy = False
            self.schedule_nudge(self.o.error_delay, "prompt handled without a run")
        elif cmd == "get_state":
            self.write_status(session_file=d.get("sessionFile"))
            kind, self.first_prompt = self.first_prompt, None
            if kind:
                if kind == "restart" and not d.get("messageCount"):
                    log.warning("--continue opened an empty session (%s); sending the kickoff prompt",
                                d.get("sessionFile"))
                    kind = "kickoff"
                await self.prompt(kind)
        elif cmd == "get_session_stats":
            cu = d.get("contextUsage") or {}
            if cu.get("tokens") is not None:
                self.st["context_tokens"] = cu.get("tokens")
            self.st.update(context_window=cu.get("contextWindow"), context_percent=cu.get("percent"),
                           session_tokens=d.get("tokens"), session_file=d.get("sessionFile")
                           or self.st.get("session_file"))
            self.status_dirty = True

    async def poll_run_marker(self):
        """The runner's state and run marker; the marker becomes the new baseline when known."""
        state = await asyncio.to_thread(self.runner_state, 5)
        marker = run_marker(state)
        if marker is not None:
            self.run_marker = marker
            self.st["runner_n_runs"] = marker[0]
        return state, marker

    async def ran_since_last_settle(self):
        """(a run happened since the previous settle?, signal used). Trusted: the arm runner's run count / last run
        changed or a run is in flight; the ./run.sh command text only when the runner gives no answer."""
        before = self.run_marker
        state, marker = await self.poll_run_marker()
        if state is not None and run_in_flight(state):
            return True, "runner"
        if marker is not None and before is not None:
            return marker != before, "runner"
        return self.runsh_since_settle, "command"

    async def on_settled(self):
        self.agent_busy = False
        self.st["n_settles"] += 1
        ran, signal_src = await self.ran_since_last_settle()
        if ran:
            self.idle_settles = 0
        else:
            self.idle_settles += 1
        self.runsh_since_settle = False
        self.st["idle_settles"], self.st["run_signal"] = self.idle_settles, signal_src
        await self.send({"id": self.next_id("stats"), "type": "get_session_stats"})
        if self.stopping:
            return
        if self.idle_settles >= self.o.idle_settles:
            self.st["n_backoffs"] += 1
            log.warning("back-off: %d consecutive settles without a run (%s signal); next nudge in %.0fs",
                        self.idle_settles, signal_src, self.o.idle_backoff)
            self.sup("backoff", idle_settles=self.idle_settles, delay_s=self.o.idle_backoff, signal=signal_src)
            self.schedule_nudge(self.o.idle_backoff, "idle back-off")
        elif self.last_stop_reason == "error":
            log.info("settled after a provider error; next nudge in %.0fs", self.o.error_delay)
            self.schedule_nudge(self.o.error_delay, "error")
        else:
            self.schedule_nudge(self.o.nudge_delay, "settled")

    # -------------------------------------------------------------------------------------- watchdog
    def runner_state(self, timeout: float = 10):
        """The arm's runner /control/state, or None if unreachable (or it is another arm's runner)."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.arm.runner_port}/control/state",
                                        timeout=timeout) as r:
                st = json.loads(r.read())
        except Exception:
            return None
        if isinstance(st, dict) and st.get("arm") not in (None, self.arm.name):
            log.error("runner on port %d reports arm %r, not %r", self.arm.runner_port, st.get("arm"), self.arm.name)
            return None
        return st

    async def watchdog(self):
        last_stats = time.time()
        while self.pi and self.pi.alive() and not self.stopping:
            await asyncio.sleep(2)
            now = time.time()
            if self.status_dirty or now - self.st.get("updated_at", 0) > 30:
                self.write_status()
            if now - last_stats > 600 and self.agent_busy:
                last_stats = now
                await self.send({"id": self.next_id("stats"), "type": "get_session_stats"})
            silent = now - (self.st["last_event_ts"] or self.pi.t_start)
            if not self.agent_busy or silent < self.o.watchdog:
                continue
            state = await asyncio.to_thread(self.runner_state)
            busy = run_in_flight(state) if state is not None else bool(self.open_runsh)
            self.write_status(runner_busy=busy if state is not None else None)
            if busy:
                continue
            self.st["n_watchdog"] += 1
            log.error("watchdog: no pi record for %.0fs and no run in flight (runner %s); restarting pi",
                      silent, "idle" if state is not None else "unreachable")
            self.sup("watchdog", silent_s=round(silent), runner_reachable=state is not None)
            self.restart_reason = "watchdog"
            await self.stop_pi(grace=10)
            return

    # ------------------------------------------------------------------------------------------ lifecycle
    async def stop_pi(self, grace: float):
        """Orderly: close stdin, wait `grace` s; then kill the whole sandbox tree."""
        if not self.pi or not self.pi.proc:
            return
        self.pi.close_stdin()
        try:
            await asyncio.wait_for(asyncio.shield(self.pi.proc.wait()), grace)
        except asyncio.TimeoutError:
            log.warning("pi did not exit within %.0fs of closing stdin; killing the sandbox tree", grace)
        # once the sandbox root has exited and been reaped its namespace is gone, and its pid may be reused
        if self.pi.alive():
            kill_tree(self.pi.pid)

    async def run_pi_once(self, reason: str) -> int:
        cont = bool(session_files(self.sess))
        self.pi = PiRpc(self.pi_cmd(cont), self.arm.repo, {**os.environ}, os.path.join(self.arm.dir, "pi_stderr.log"))
        await self.pi.start()
        self.st["n_pi_starts"] += 1
        write_json_atomic(self.pidfile, {"pid": self.pi.pid, "start_ticks": proc_start_ticks(self.pi.pid),
                                         "ts": time.time()})
        self.agent_busy, self.runsh_since_settle, self.open_runsh = False, False, set()
        self.last_stop_reason = None
        if self.run_marker is None:
            # baseline of the trusted run signal; only once per supervisor: re-reading it at a pi restart would
            # hide a run that finished between the last settle and the restart
            await self.poll_run_marker()
        log.info("pi started (pid %d, %s, reason: %s)", self.pi.pid, "--continue" if cont else "fresh session", reason)
        self.sup("pi_start", pid=self.pi.pid, cont=cont, reason=reason, n=self.st["n_pi_starts"])
        self.write_status(state="starting", pi_pid=self.pi.pid, pi_started_at=time.time())
        # the first prompt goes out once pi answers get_state (kickoff if the continued session is empty)
        self.first_prompt = "restart" if cont else "kickoff"
        self.agent_busy = True                  # the watchdog also covers a pi that never answers
        await self.send({"id": self.next_id("state"), "type": "get_state"})
        wd = asyncio.ensure_future(self.watchdog())
        n_bad = 0
        try:
            async for ev, n in self.pi.records():
                if ev is None or not isinstance(ev, dict):
                    n_bad += 1
                    self.sup("bad_record", bytes=n)
                    continue
                try:
                    await self.on_record(ev)
                except Exception:                # a handler bug must not take the session down
                    log.exception("error handling pi record %s", ev.get("type"))
        finally:
            wd.cancel()
            if self.nudge_task and not self.nudge_task.done():
                self.nudge_task.cancel()
            if self.pi.alive():                 # stdout closed (or we failed) while the process lives on
                await self.stop_pi(grace=10)
            rc = await self.pi.proc.wait()
            if os.path.exists(self.pidfile):
                os.unlink(self.pidfile)
        up = time.time() - self.pi.t_start
        log.info("pi exited rc=%s after %.0fs (%d unparsable records)", rc, up, n_bad)
        self.sup("pi_exit", rc=rc, uptime_s=round(up), bad_records=n_bad)
        return rc

    async def sleep_or_stop(self, delay: float):
        try:
            await asyncio.wait_for(self.stop_event.wait(), delay)
        except asyncio.TimeoutError:
            pass

    def request_stop(self, signame: str):
        if self.stopping:
            return
        self.stopping = True
        log.warning("%s: stopping (closing pi's stdin, %.0fs grace)", signame, self.o.term_grace)
        self.sup("stop", signal=signame)
        self.write_status(state="stopping")
        self.stop_event.set()
        if self.pi and self.pi.alive():
            self.stop_task = asyncio.ensure_future(self.stop_pi(self.o.term_grace))   # keep a reference

    async def run(self):
        self.stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for s in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(s, self.request_stop, s.name)
        self.reap_orphans()
        reason = "supervisor start"
        if session_files(self.sess):
            self.st["n_restarts"] += 1           # the agent's session is being resumed: it was interrupted
        while not self.stopping:
            rc = await self.run_pi_once(reason)
            if self.stopping:
                break
            up = time.time() - self.pi.t_start
            self.failures = 0 if up >= self.o.stable_after else self.failures
            self.failures += 1
            delay = min(self.o.restart_max, self.o.restart_delay * 2 ** (self.failures - 1))
            self.st["n_restarts"] += 1
            reason, self.restart_reason = self.restart_reason or f"pi exited rc={rc}", None
            log.warning("restart #%d in %.0fs (%s after %.0fs; %d quick failure(s) in a row)",
                        self.st["n_restarts"], delay, reason, up, self.failures)
            self.sup("restart_scheduled", n=self.st["n_restarts"], delay_s=delay, rc=rc, reason=reason)
            self.write_status(state="restarting", pi_pid=None, next_nudge_ts=None)
            await self.sleep_or_stop(delay)
        self.write_status(state="stopped", pi_pid=None, next_nudge_ts=None)
        self.sup("stopped")
        log.info("supervisor stopped")


def main():
    ap = argparse.ArgumentParser(description="h2h supervisor: one continuous pi session for one arm")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--nudge-delay", type=float, default=0.0, help="s between a settle and the nudge")
    ap.add_argument("--idle-settles", type=int, default=3, help="settles without ./run.sh before back-off")
    ap.add_argument("--idle-backoff", type=float, default=300.0, help="s to wait before a nudge while backing off")
    ap.add_argument("--error-delay", type=float, default=60.0, help="s before a nudge after a provider error")
    ap.add_argument("--restart-delay", type=float, default=30.0)
    ap.add_argument("--restart-max", type=float, default=600.0)
    ap.add_argument("--stable-after", type=float, default=600.0, help="pi uptime that resets the restart back-off")
    ap.add_argument("--watchdog", type=float, default=2700.0, help="s without pi records before a restart")
    ap.add_argument("--term-grace", type=float, default=30.0)
    opts = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s %(levelname)s %(name)s[{opts.arm}]: %(message)s")
    cfg = load_h2h_config()
    arm = cfg.arm(opts.arm)
    for p in (arm.repo, os.path.join(arm.session_dir, "agent")):
        if not os.path.isdir(p):
            raise SystemExit(f"{p} missing: run tools/h2h/setup.py first")
    os.makedirs(os.path.join(arm.session_dir, "session"), exist_ok=True)
    lock = open(os.path.join(arm.dir, "supervisor.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another supervisor for arm {arm.name} is running")
    sup = Supervisor(cfg, arm, opts)
    log.info("arm %s: repo %s, session %s, sandbox %s", arm.name, arm.repo, sup.sess, sup.sandbox)
    with open(arm.events, "a") as f:
        sup.events_f = f
        asyncio.run(sup.run())
    sys.exit(0)


if __name__ == "__main__":
    main()
