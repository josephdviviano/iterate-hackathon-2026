"""RLTL;DR autoresearch driver (rollout collection).

One *attempt* (= one rollout) is one fresh pi session, in a sandbox, that performs exactly one autoresearch
experiment: edit train.py -> commit -> ./run.sh (the runner trains it on the agent GPU in a second sandbox and
measures val_bpb with trusted code). The driver, not the agent, applies the keep/discard rule, so the
branch only advances on verified improvements and rewards cannot be self-reported.

Integrity:
  * the canonical branch lives in a driver-only bare repo (canon.git); the agent works in a fresh throwaway
    clone each attempt and the driver never runs git inside the agent's tree (agent-written git config or
    hooks are never executed with harness privileges);
  * results.tsv is harness-owned (data/results.tsv) and copied into each fresh clone;
  * the attempt's outcome is its FIRST valid run (the runner refuses further runs after an ok run);
  * noise (val_bpb std ~0.004) is handled by deciding on unselected evidence: an apparent win is re-run
    `confirm_runs` times with a fresh compile cache, the keep decision uses only those re-runs, and the new
    best is estimated from them alone (no winner's-curse ratchet);
  * infrastructure failures (LLM server / runner down) void and retry the attempt instead of scoring 0.

Every `group_size` (8) sequential attempts form one GRPO group (paper Sec. 3.2):
  * after each failed attempt the current policy writes a one-sentence TL;DR insight in a fresh chat;
  * attempt k is conditioned on all insights of this group (extra user messages after the task, injected by
    the gateway) iff the running success rate of attempts 1..k-1 is <= 50%;
  * insights reset at every group (they are meant to be internalized by the update, App. A.2).
A completed group is handed to the trainer (data/groups/gNNNN/READY) and collection continues; new policy
versions are picked up by the gateway at attempt boundaries.
"""
import argparse
import hashlib
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rltldr.config import dump_config, load_config  # noqa: E402
from rltldr.insight import generate_insight  # noqa: E402
from rltldr.io_utils import append_jsonl, read_json, read_jsonl, write_json_atomic  # noqa: E402
from rltldr.verdict import AGENT_REJECT_FLAGS, FLAG_REASONS, rejecting_flags, valid_run  # noqa: E402

log = logging.getLogger("driver")
RUNNER_URL = "http://127.0.0.1:8200"
STOP = False
GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0"}
HOME = os.path.expanduser("~")      # the invoking user's home; also the (covered) home inside h2h_sandbox.sh


def _sigterm(*_):
    global STOP
    STOP = True
    log.warning("stop requested; finishing the current attempt first")


TASK_TEMPLATE = """You are an autonomous ML researcher taking part in an ongoing "autoresearch" loop. Each session you run exactly ONE experiment; a harness records the result and starts a new session for the next experiment.

## The research problem
The repository in the current directory trains a small GPT language model (cherry-picked from nanochat) for a fixed 5-minute wall-clock training budget on one GPU (NVIDIA RTX PRO 6000 Blackwell, 96 GB, sm_120; FlashAttention-3 is unavailable on this GPU, attention goes through the `attention()` helper in train.py, using FlexAttention for sliding windows and SDPA otherwise). The metric is val_bpb (validation bits per byte) — LOWER is better.

Files:
- `prepare.py` — READ-ONLY. Fixed constants (time budget, sequence length), data loading, tokenizer and `evaluate_bpb`, the ground-truth metric.
- `train.py` — the ONLY file you may edit: model architecture, optimizer, hyperparameters, training loop, batch size, model size... everything is fair game.
- `results.tsv` — log of all experiments so far (maintained by the harness; read it, don't edit it).

Rules:
- Only edit `train.py`. Do not install packages or add dependencies. Do not modify the evaluation.
- The script must run without crashing and finish within the time budget. VRAM is a soft constraint (some increase is fine for meaningful gains, it should not blow up).
- Evaluation contract (checked by the harness): train on the "train" split only; at the end call `evaluate_bpb(model, tokenizer, batch_size)` from prepare.py exactly once; `model(idx)` without targets must return logits of shape [B, T, vocab] (the harness computes the loss from them) and the model must be causal. train.py runs in a sandbox: it cannot write files or use the network, only train and print.
- Simplicity criterion: all else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it; removing code for equal or better results is a win.
- Measurement noise of val_bpb is small (std about {sigma:.4f}). A run counts as an improvement only if it beats the current best by more than {margin:.4f}; an apparent win is re-run {confirm_runs}x with a fresh compile and kept only if the re-run confirms it.

## Current state
Branch `{branch}`, current best commit `{commit}`: val_bpb = {best:.6f}{best_note}
{history}

## Your task: exactly ONE experiment
1. Look at the current `train.py` and the history above. Choose ONE promising idea that has not been tried yet (or a clearly better variant of a near-miss).
2. Implement it by editing `train.py`.
3. Commit it: `git commit -am "<short description of the idea>"`
4. Run the experiment: `./run.sh "<short description of the idea>"` — it takes about 6 minutes, trains on the dedicated GPU and prints `val_bpb: ...` and `peak_vram_mb: ...`, or `status: <error>` with the end of the log if the run failed. Never run training any other way (this sandbox has no GPU).
5. If the run crashed because of a simple bug (typo, shape mismatch, ...), fix it, commit and run again (at most {max_runs} runs in total; only one successful run is recorded per session). If the idea itself is broken, stop.
6. Finish with a short final message: what you tried, the result, and what you learned.
Do not revert, reset, or edit results.tsv yourself: the harness decides keep/discard."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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


def kill_tree(root: int, sig: int) -> None:
    """Signal a sandbox process tree. sudo/unshare run as root, so use `sudo kill` for the whole tree."""
    pids = [p for p in descendants(root) + [root] if os.path.exists(f"/proc/{p}")]
    if pids:
        subprocess.run(["sudo", "-n", "kill", f"-{int(sig)}", *map(str, pids)], capture_output=True)


def orphan_agent_sandboxes(root_dir: str) -> list:
    """sudo/unshare processes of agent sandboxes (tools/sandbox.sh) left over by a previous driver."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if b"unshare" in argv and b"--kill-child" in argv and b"sandbox" in argv and root_dir.encode() in argv:
            out.append(int(d))
    return out


def orphan_arm_sandboxes(repo: str) -> list:
    """sudo/unshare processes of tools/h2h_sandbox.sh sandboxes that expose THIS instance's repo (other
    instances on the machine use other repos and are left alone)."""
    out, key = [], repo.encode()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if any(a.endswith((b"/unshare", b"unshare", b"sudo")) for a in argv[:1]) and any(key in a for a in argv):
            out.append(int(d))
    return out


def _force_rmtree(path: str) -> None:
    """rmtree that also removes agent-made read-only dirs (chmod 000) instead of failing on them."""
    def onerror(func, p, _exc):
        try:
            os.chmod(os.path.dirname(p), 0o700)
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.exists(path):
        shutil.rmtree(path, onerror=onerror)


def clear_dir(path: str) -> None:
    """Remove everything inside `path` (symlinks unlinked, never followed); create it if missing."""
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.islink(p) or not os.path.isdir(p):
            os.unlink(p)
        else:
            _force_rmtree(p)


class Canon:
    """Driver-owned canonical repository (bare canon.git + trusted work clone canon_ws)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.bare = cfg.canon_dir
        self.ws = cfg.canon_ws
        self.branch = cfg.branch

    def git(self, cwd, *args) -> str:
        r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", *args],
                           cwd=cwd, capture_output=True, text=True, env={**os.environ, **GIT_ENV})
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} (in {cwd}) failed: {r.stderr.strip()}")
        return r.stdout.strip()

    def ensure(self, seed_repo: str):
        if not os.path.isdir(self.bare):
            # one-time import from the initial (harness-created) autoresearch checkout
            self.git(self.cfg.root, "clone", "-q", "--bare", "--no-local", "--template=", "--branch", self.branch,
                     seed_repo, self.bare)
        if not os.path.isdir(os.path.join(self.ws, ".git")):
            self.git(self.cfg.root, "clone", "-q", "--template=", "--branch", self.branch, self.bare, self.ws)
            self.git(self.ws, "config", "user.name", "rltldr-driver")
            self.git(self.ws, "config", "user.email", "rltldr@localhost")

    def head(self) -> str:
        return self.git(self.bare, "rev-parse", self.branch)

    def subject(self, commit: str) -> str:
        return self.git(self.bare, "log", "-1", "--format=%s", commit)

    def commit_train(self, parent: str, content: bytes, message: str) -> str:
        self.git(self.ws, "fetch", "-q", "origin")
        self.git(self.ws, "checkout", "-q", "-f", "-B", self.branch, parent)
        with open(os.path.join(self.ws, "train.py"), "wb") as f:
            f.write(content)
        self.git(self.ws, "add", "train.py")
        self.git(self.ws, "commit", "-q", "-m", message)
        self.git(self.ws, "push", "-q", "origin", f"{self.branch}:{self.branch}")
        return self.git(self.ws, "rev-parse", "HEAD")

    def fresh_agent_clone(self, dest: str, results_tsv: str, agent_venv: str):
        """Replace the agent's working tree by a fresh clone of the canonical branch (git internals included)."""
        tmp = dest + ".new"
        shutil.rmtree(tmp, ignore_errors=True)
        self.git(self.cfg.root, "clone", "-q", "--no-local", "--template=", "--branch", self.branch, self.bare, tmp)
        self.git(tmp, "config", "user.name", "researcher")
        self.git(tmp, "config", "user.email", "researcher@localhost")
        os.makedirs(os.path.join(tmp, ".git", "info"), exist_ok=True)
        with open(os.path.join(tmp, ".git", "info", "exclude"), "w") as f:
            f.write("run.log\nresults.tsv\nrun.sh\n.venv\n")
        # tools/run.sh is a template: @RLTLDR_ROOT@ -> this installation's root (where run_client.py lives)
        with open(os.path.join(self.cfg.root, "tools", "run.sh")) as f:
            run_sh = f.read().replace("@RLTLDR_ROOT@", self.cfg.root)
        with open(os.path.join(tmp, "run.sh"), "w") as f:
            f.write(run_sh)
        shutil.copymode(os.path.join(self.cfg.root, "tools", "run.sh"), os.path.join(tmp, "run.sh"))
        with open(os.path.join(tmp, "results.tsv"), "w") as f:
            f.write(results_tsv)
        if os.path.isdir(agent_venv):
            os.symlink(agent_venv, os.path.join(tmp, ".venv"))
        os.chmod(os.path.join(tmp, "prepare.py"), 0o444)
        if self.cfg.sandbox_arm:   # the host path of the bare repo would name the arm; it is hidden in the jail
            self.git(tmp, "remote", "set-url", "origin", os.path.join(HOME, "canon.git"))
        if os.path.lexists(dest):
            old = dest + ".old"
            _force_rmtree(old)
            os.rename(dest, old)
            _force_rmtree(old)
        os.rename(tmp, dest)


class Driver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.canon = Canon(cfg)
        self.state_path = cfg.path("driver_state.json")
        self.results_path = cfg.path("results.tsv")             # authoritative copy (agent gets a copy)
        self.agent_venv = cfg.agent_venv
        global RUNNER_URL
        RUNNER_URL = f"http://127.0.0.1:{cfg.runner_port}"
        os.makedirs(cfg.path("attempts"), exist_ok=True)
        os.makedirs(cfg.path("groups"), exist_ok=True)
        self._migrate_agent_venv()
        self.canon.ensure(cfg.repo)
        self.state = read_json(self.state_path) or self._init_state()
        self._recover()

    # ---------------------------------------------------------------------------------------------- setup
    def _migrate_agent_venv(self):
        """The agent's (read-only) python env lives outside the throwaway clones."""
        v = os.path.join(self.cfg.repo, ".venv")
        if not os.path.exists(self.agent_venv) and os.path.isdir(v) and not os.path.islink(v):
            os.rename(v, self.agent_venv)
            os.symlink(self.agent_venv, v)

    def _init_state(self):
        base = [e for e in read_jsonl(self.cfg.ledger) if e.get("attempt_id") == "baseline" and valid_run(e)]
        if not base:
            raise SystemExit("No valid 'baseline' entry in the ledger. Run the baseline first (see README).")
        b = base[-1]
        sha = b.get("train_sha256")
        evals = [e["val_bpb"] for e in read_jsonl(self.cfg.ledger) + read_jsonl(self.cfg.path("calib_ledger.jsonl"))
                 if valid_run(e) and e.get("train_sha256") == sha]
        mean = sum(evals) / len(evals)
        head = self.canon.head()
        if not os.path.exists(self.results_path):
            with open(self.results_path, "w") as f:
                f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
                f.write(f"{head[:7]}\t{mean:.6f}\t{(b['metrics'].get('peak_vram_mb') or 0) / 1024:.1f}\tkeep\t"
                        f"baseline (mean of {len(evals)} runs)\n")
        st = {"group": 0, "k": 0, "insights": [], "rollouts": [], "inflight": None, "inflight_tries": 0,
              "parent": {"commit": head, "val_bpb": mean, "evals": evals, "train_sha256": sha,
                         "run_id": b.get("run_id")},
              "n_attempts": 0, "n_kept": 0, "n_void": 0}
        write_json_atomic(self.state_path, st)
        return st

    def _recover(self):
        """After a crash/kill: no agent of a previous driver may keep running into the next attempt."""
        orphans = (orphan_arm_sandboxes(self.cfg.repo) if self.cfg.sandbox_arm
                   else orphan_agent_sandboxes(self.cfg.root))
        if orphans:
            log.warning("killing %d orphaned agent sandbox process(es): %s", len(orphans), orphans)
            for p in orphans:
                kill_tree(p, signal.SIGKILL)
        inflight = self.state.get("inflight")
        if inflight:
            log.warning("previous attempt %s was interrupted; it will be redone under a new id", inflight)
            for base in (self.cfg.gateway_url, RUNNER_URL):
                try:
                    requests.post(f"{base}/control/attempt_end", json={"id": inflight}, timeout=60)
                except requests.RequestException:
                    pass
        if self.state["k"] >= self.cfg.group_size:      # crashed between the last attempt and closing the group
            self.close_group()

    def save(self):
        write_json_atomic(self.state_path, self.state)

    # --------------------------------------------------------------------------------------------- prompt
    def results_text(self) -> str:
        return open(self.results_path).read()

    def history_table(self) -> str:
        rows = self.results_text().strip().split("\n")
        header, body = rows[0], rows[1:]
        n = self.cfg.history_rows_in_prompt
        shown = body[-n:]
        note = f"(showing the last {len(shown)} of {len(body)} experiments)\n" if len(body) > n else ""
        return f"Experiment history (results.tsv, oldest first):\n{note}```\n{header}\n" + "\n".join(shown) + "\n```"

    def task_prompt(self) -> str:
        p = self.state["parent"]
        return TASK_TEMPLATE.format(branch=self.cfg.branch, commit=p["commit"][:7], best=p["val_bpb"],
                                    best_note=f'  ("{self.canon.subject(p["commit"])}")',
                                    history=self.history_table(), max_runs=self.cfg.max_runs_per_attempt,
                                    sigma=self.cfg.reward_sigma, confirm_runs=self.cfg.confirm_runs,
                                    margin=self.cfg.keep_margin)

    # ------------------------------------------------------------------------------------------- helpers
    def conditioned_insights(self):
        """Paper App. A.2: insert all insights so far iff running success rate <= threshold."""
        rs = self.state["rollouts"]
        if not self.cfg.insights_enabled or not self.state["insights"] or not rs:
            return []
        succ = sum(r["success"] for r in rs)
        if succ <= self.cfg.insight_success_threshold * len(rs):
            return self.state["insights"][-self.cfg.max_insights_in_context:]
        return []

    @staticmethod
    def post(base, path, payload, tries=30):
        for i in range(tries):
            try:
                r = requests.post(f"{base}{path}", json=payload, timeout=600)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if i == tries - 1:
                    raise
                log.warning("POST %s%s failed (%s); retrying", base, path, e)
                time.sleep(10)

    def run_pi(self, attempt_id: str, adir: str, prompt: str) -> dict:
        """Run one pi session inside the agent sandbox (no GPU, loopback-only network, read-only fs)."""
        cfg = self.cfg
        sess = os.path.join(cfg.pi_sessions_dir, attempt_id)        # writable inside the sandbox
        if cfg.sandbox_arm:
            # h2h_sandbox.sh exposes the WHOLE sessions dir read-write: empty it so nothing an agent wrote there
            # (notes, symlinks) survives into the next attempt -- every attempt starts from a fresh session
            clear_dir(cfg.pi_sessions_dir)
        shutil.rmtree(sess, ignore_errors=True)
        # fresh copy of the pi config per attempt: pi writes state there, nothing leaks between attempts
        shutil.copytree(cfg.pi_agent_dir, os.path.join(sess, "agent"),
                        ignore=shutil.ignore_patterns("auth.json", "models-store.json", "sessions"))
        inner_env = {
            "PATH": f"{HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin", "HOME": HOME,
            "LANG": "C.UTF-8", "TERM": "dumb",
            "PI_CODING_AGENT_DIR": os.path.join(sess, "agent"), "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0",
            "CUDA_VISIBLE_DEVICES": "", "AR_GUARD_LOG": os.path.join(sess, "guard_blocks.jsonl"),
        }
        pi_cmd = [cfg.pi_bin, "--mode", "json", "--no-approve", "--no-context-files", "--provider", "rl",
                  "--model", "policy", "--thinking", cfg.agent_thinking,
                  "--session-dir", os.path.join(sess, "session"), "--", prompt]
        if cfg.sandbox_arm:   # per-instance jail that also hides $HOME (see config.sandbox_arm)
            cmd = [os.path.join(cfg.root, "tools", "h2h_sandbox.sh"), cfg.sandbox_arm, "/usr/bin/env", "-i",
                   *[f"{k}={v}" for k, v in inner_env.items()], *pi_cmd]
        else:
            cmd = [os.path.join(cfg.root, "tools", "sandbox.sh"), sess, "/usr/bin/env", "-i",
                   *[f"{k}={v}" for k, v in inner_env.items()], *pi_cmd]
        with open(os.path.join(adir, "pi_cmd.txt"), "w") as f:
            f.write(" ".join(shlex.quote(c) for c in cmd[:-1]) + " <task.md>\n")
        t0 = time.time()
        timed_out = False
        env = {**os.environ, "AGENT_REPO": cfg.repo}
        with open(os.path.join(adir, "events.jsonl"), "wb") as out, open(os.path.join(adir, "pi.stderr"), "wb") as err:
            p = subprocess.Popen(cmd, cwd=cfg.repo, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 start_new_session=True)
            self.state["inflight_pid"] = p.pid
            self.save()
            try:
                rc = p.wait(timeout=cfg.attempt_timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_tree(p.pid, signal.SIGTERM)
                try:
                    rc = p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    kill_tree(p.pid, signal.SIGKILL)
                    rc = p.wait()
        kill_tree(p.pid, signal.SIGKILL)       # nothing from this attempt may survive it
        for name in ("session",):   # agent-writable: never follow its symlinks
            src = os.path.join(sess, name)
            if os.path.isdir(src) and not os.path.islink(src):
                shutil.copytree(src, os.path.join(adir, name), dirs_exist_ok=True, symlinks=True)
        gb = os.path.join(sess, "guard_blocks.jsonl")
        if os.path.isfile(gb) and not os.path.islink(gb):
            shutil.copy(gb, adir)
        if cfg.sandbox_arm:
            clear_dir(cfg.pi_sessions_dir)
        else:
            shutil.rmtree(sess, ignore_errors=True)
        return {"returncode": rc, "timed_out": timed_out, "wall_s": time.time() - t0}

    def infra_failure(self, calls: list, pi: dict, adir: str):
        """Reasons why an attempt failed for infrastructure (not policy) reasons, or None."""
        agent = [c for c in calls if c.get("kind") == "agent"]
        if not agent:
            return "no LLM calls were recorded (pi could not reach the model)"
        bad = [c["status"] for c in agent if c.get("status") in ("upstream_unavailable", "upstream_stalled")
               or str(c.get("status", "")).startswith("http_5")]
        if bad:
            return f"LLM server errors: {bad[:3]}"
        if not pi["timed_out"] and str(agent[-1].get("status", "")).startswith("aborted"):
            return "the LLM stream was aborted"
        try:
            ev = open(os.path.join(adir, "events.jsonl"), "rb").read()
            if b"status: harness_error" in ev:
                return "the experiment runner was unreachable"
        except OSError:
            pass
        if pi["timed_out"]:
            try:   # a wedged LLM server shows up as a timeout: check it is alive before blaming the policy
                r = requests.post(f"{self.cfg.vllm_url}/v1/chat/completions", timeout=120,
                                  json={"model": self.cfg.base_model_name, "max_tokens": 1,
                                        "messages": [{"role": "user", "content": "ping"}]})
                if r.status_code != 200:
                    return f"LLM server unhealthy after timeout (HTTP {r.status_code})"
            except requests.RequestException as e:
                return f"LLM server unreachable after timeout ({e})"
        return None

    def confirm_runs(self, attempt_id: str, parent_commit: str, snap: str, desc: str):
        """Re-run an apparently winning train.py `confirm_runs` times with fresh compile caches.
        Returns the list of ledger entries, or None if the runner kept failing (infrastructure)."""
        with open(snap) as f:
            src = f.read()
        out = []
        for i in range(self.cfg.confirm_runs):
            cid = f"{attempt_id}-confirm{i + 1}"
            entry = None
            for _ in range(3):
                try:
                    self.post(RUNNER_URL, "/control/attempt", {"id": cid, "parent_commit": parent_commit})
                    r = requests.post(f"{RUNNER_URL}/run", json={"desc": f"{desc} [confirmation re-run]",
                                                                 "train_py": src, "max_runs": 1,
                                                                 "fresh_cache": True}, timeout=7200)
                    log.info("attempt %s confirmation %d: %s", attempt_id, i + 1,
                             r.json().get("output", "")[:200].replace("\n", " | "))
                except requests.RequestException as e:
                    log.error("confirmation run %s failed: %s", cid, e)
                finally:
                    try:
                        self.post(RUNNER_URL, "/control/attempt_end", {"id": cid}, tries=3)
                    except requests.RequestException:
                        pass
                runs = [e for e in read_jsonl(self.cfg.ledger) if e.get("attempt_id") == cid
                        and e.get("status") != "refused"]
                if runs:
                    entry = runs[-1]
                    break
                time.sleep(30)
            if entry is None:
                return None
            out.append(entry)
            if not valid_run(entry):
                break                 # a crashed/invalid re-run already decides the attempt
        return out

    # ------------------------------------------------------------------------------------------- attempt
    def run_attempt(self):
        """Run attempt k of the current group. Returns the rollout, or None if it was voided (retry)."""
        cfg, st = self.cfg, self.state
        g, k = st["group"], st["k"] + 1
        tries = st.get("inflight_tries", 0) + (1 if st.get("inflight") else 0)
        attempt_id = f"g{g:04d}-a{k}" + (f"-r{tries}" if tries else "")
        st["inflight"], st["inflight_tries"] = attempt_id, tries
        self.save()
        adir = cfg.path("attempts", attempt_id)
        os.makedirs(adir, exist_ok=True)
        parent = dict(st["parent"])

        self.canon.fresh_agent_clone(cfg.repo, self.results_text(), self.agent_venv)
        insights = self.conditioned_insights()
        prompt = self.task_prompt()
        with open(os.path.join(adir, "task.md"), "w") as f:
            f.write(prompt)
        self.post(RUNNER_URL, "/control/attempt", {"id": attempt_id, "parent_commit": parent["commit"]})
        gstate = self.post(cfg.gateway_url, "/control/attempt", {"id": attempt_id, "insights": insights})
        policy = gstate["attempt"]["policy"]
        log.info("attempt %s: policy=%s insights=%d best=%.6f", attempt_id, policy["name"], len(insights),
                 parent["val_bpb"])
        try:
            pi = self.run_pi(attempt_id, adir, prompt)
        finally:
            self.post(cfg.gateway_url, "/control/attempt_end", {"id": attempt_id})
            self.post(RUNNER_URL, "/control/attempt_end", {"id": attempt_id})

        calls = read_jsonl(os.path.join(adir, "calls.jsonl"))
        runs = [e for e in read_jsonl(cfg.ledger) if e.get("attempt_id") == attempt_id and e.get("status") != "refused"]
        first_ok = next((e for e in runs if e.get("status") == "ok"), None)
        outcome = first_ok or (runs[-1] if runs else None)
        infra = self.infra_failure(calls, pi, adir)
        if infra and not (outcome and outcome.get("status") == "ok"):
            return self.void_attempt(attempt_id, adir, infra, pi)

        # ---- verifier: keep/discard on unselected evidence ----
        verdict, confirm, success, cmean = [], None, False, None
        ok = valid_run(outcome, AGENT_REJECT_FLAGS)
        snap = (outcome or {}).get("train_copy_path")
        if outcome is None:
            verdict.append("Rollout FAILED: the agent never ran the experiment through ./run.sh, so no result was produced.")
        elif outcome.get("status") != "ok":
            verdict.append(f"Rollout FAILED: the experiment did not produce a result (status: {outcome.get('status')}).")
        elif not ok:
            reasons = "; ".join(FLAG_REASONS.get(f, f) for f in rejecting_flags(outcome))
            verdict.append(f"Rollout FAILED: the run finished but is not a valid experiment: {reasons}.")
        elif outcome["val_bpb"] >= parent["val_bpb"] - max(cfg.screen_margin, cfg.keep_margin):
            verdict.append(f"Rollout FAILED: the experiment ran, but val_bpb = {outcome['val_bpb']:.6f} did not improve "
                           f"on the current best {parent['val_bpb']:.6f} by more than the required "
                           f"{cfg.keep_margin:.4f} (lower is better).")
        elif not (snap and os.path.exists(snap) and sha256_bytes(open(snap, "rb").read()) == outcome.get("train_sha256")):
            log.error("attempt %s: winning run has no matching train.py snapshot", attempt_id)
            verdict.append("Rollout FAILED: the harness could not verify the evaluated code.")
        else:
            confirm = self.confirm_runs(attempt_id, parent["commit"], snap, outcome.get("desc") or "")
            if confirm is None:
                return self.void_attempt(attempt_id, adir, "the confirmation re-run could not be executed", pi)
            if all(valid_run(c) for c in confirm):
                cmean = sum(c["val_bpb"] for c in confirm) / len(confirm)
                success = cmean < parent["val_bpb"] - cfg.keep_margin
                if not success:
                    verdict.append(
                        f"Rollout FAILED: the experiment's run gave val_bpb = {outcome['val_bpb']:.6f}, which looked better "
                        f"than the current best {parent['val_bpb']:.6f}, but the harness's independent re-run(s) of the "
                        f"same code gave {', '.join(f'{c['val_bpb']:.6f}' for c in confirm)}, which did not beat the best "
                        f"by the required margin of {cfg.keep_margin:.4f}: the first result did not reproduce.")
            else:
                bad = confirm[-1]
                verdict.append(f"Rollout FAILED: the harness's re-run of the same code was not valid "
                               f"(status {bad.get('status')}, {', '.join(rejecting_flags(bad)) or 'no flags'}).")

        desc = (outcome or {}).get("desc") or "(no experiment run)"
        if success:
            new_commit = self.canon.commit_train(parent["commit"], open(snap, "rb").read(),
                                                 f"{desc} [val_bpb {cmean:.6f}, {attempt_id}]")
            st["parent"] = {"commit": new_commit, "val_bpb": cmean, "evals": [c["val_bpb"] for c in confirm],
                            "train_sha256": outcome["train_sha256"], "run_id": confirm[0].get("run_id")}
            st["n_kept"] += 1
        status = "keep" if success else ("discard" if outcome and outcome.get("status") == "ok" else "crash")
        mem = ((outcome or {}).get("metrics") or {}).get("peak_vram_mb") or 0.0
        row_commit = st["parent"]["commit"][:7] if success else (outcome or {}).get("train_sha256", "-------")[:7]
        shown = cmean if success else ((outcome or {}).get("val_bpb") or 0.0)
        note = (f" [re-run {', '.join(f'{c['val_bpb']:.6f}' for c in confirm if c.get('val_bpb') is not None)}]"
                if confirm else "")
        flags = rejecting_flags(outcome) if outcome else []
        if flags:
            note += f" [invalid: {', '.join(flags)}]"
        with open(self.results_path, "a") as f:
            f.write(f"{row_commit}\t{shown:.6f}\t{mem / 1024:.1f}\t{status}\t"
                    f"{' '.join(desc.split())[:200]}{note}\n")

        if cfg.reward_mode == "delta":
            val = cmean if cmean is not None else (outcome or {}).get("val_bpb") if ok else None
            reward = max(-3.0, min(3.0, (parent["val_bpb"] - val) / cfg.reward_sigma)) if val is not None else -1.0
        else:
            reward = 1.0 if success else 0.0

        rollout = {
            "attempt_id": attempt_id, "group": g, "k": k, "success": bool(success), "reward": reward,
            "status": status, "val_bpb": (outcome or {}).get("val_bpb"), "confirm_mean": cmean,
            "confirm": confirm, "parent": parent, "policy": policy, "verdict": verdict,
            "conditioned": bool(insights), "insights_in_context": insights,
            "n_runs": len(runs), "outcome": outcome, "pi": pi, "desc": desc,
            "calls_path": os.path.join(adir, "calls.jsonl"), "t_end": time.time(),
        }
        insight = None
        if not success and cfg.insights_enabled:
            diff = ""
            if outcome and outcome.get("diff_path") and os.path.exists(outcome["diff_path"]):
                diff = open(outcome["diff_path"]).read()
            parent_entry = next((e for e in read_jsonl(cfg.ledger) if e.get("run_id") == parent.get("run_id")), None)
            t0 = time.time()
            insight = generate_insight(cfg, attempt_id, prompt, calls, diff, verdict, outcome, parent_entry)
            insight["wall_s"] = time.time() - t0
            if insight.get("hint"):
                st["insights"].append({"idx": k, "hint": insight["hint"]})
            log.info("attempt %s insight (%.0fs): %s", attempt_id, insight["wall_s"],
                     insight.get("hint") or insight.get("error"))
        rollout["insight"] = insight
        write_json_atomic(os.path.join(adir, "rollout.json"), rollout)

        st["rollouts"].append({"attempt_id": attempt_id, "success": bool(success), "reward": reward,
                               "conditioned": bool(insights), "status": status,
                               "val_bpb": (outcome or {}).get("val_bpb")})
        st["k"] = k
        st["n_attempts"] += 1
        st["inflight"], st["inflight_tries"], st["inflight_pid"] = None, 0, None
        self.save()
        log.info("attempt %s done: status=%s val_bpb=%s confirm=%s success=%s pi=%.0fs runs=%d", attempt_id,
                 status, (outcome or {}).get("val_bpb"), cmean, success, pi["wall_s"], len(runs))
        return rollout

    def void_attempt(self, attempt_id, adir, reason, pi):
        """Infrastructure failure: not a policy outcome. Recorded, excluded from the group, retried."""
        log.error("attempt %s VOID (%s); it will be redone", attempt_id, reason)
        write_json_atomic(os.path.join(adir, "rollout.json"), {"attempt_id": attempt_id, "void": True,
                                                              "reason": reason, "pi": pi, "t_end": time.time()})
        self.state["n_void"] = self.state.get("n_void", 0) + 1
        self.state["consecutive_void"] = self.state.get("consecutive_void", 0) + 1
        self.save()          # `inflight` stays set: the next attempt gets a new id (-rN)
        time.sleep(min(600, 30 * 2 ** min(self.state["consecutive_void"], 5)))
        return None

    def close_group(self):
        cfg, st = self.cfg, self.state
        g = st["group"]
        rs = st["rollouts"]
        gdir = cfg.path("groups", f"g{g:04d}")
        os.makedirs(gdir, exist_ok=True)
        no_ins = [r["success"] for r in rs if not r["conditioned"]]
        with_ins = [r["success"] for r in rs if r["conditioned"]]
        metrics = {
            "group": g, "attempts": [r["attempt_id"] for r in rs], "n": len(rs),
            "n_success": sum(r["success"] for r in rs),
            "success_rate": sum(r["success"] for r in rs) / max(1, len(rs)),
            "first_attempt_success": rs[0]["success"] if rs else None,     # deconfounded Pass@1 proxy (App. D)
            "success_rate_no_insight": (sum(no_ins) / len(no_ins)) if no_ins else None,
            "success_rate_with_insight": (sum(with_ins) / len(with_ins)) if with_ins else None,
            "insight_advantage": ((sum(with_ins) / len(with_ins)) - (sum(no_ins) / len(no_ins)))
            if (no_ins and with_ins) else None,
            "n_conditioned": len(with_ins), "n_insights": len(st["insights"]),
            "n_crash": sum(r["status"] == "crash" for r in rs),
            "policy_versions": sorted({(read_json(cfg.path("attempts", r["attempt_id"], "rollout.json"), {})
                                        .get("policy") or {}).get("version") for r in rs} - {None}),
            "best_val_bpb": st["parent"]["val_bpb"], "n_attempts_total": st["n_attempts"],
            "n_void_total": st.get("n_void", 0), "t": time.time(),
        }
        write_json_atomic(os.path.join(gdir, "group.json"), {"group": g, "attempts": metrics["attempts"],
                                                             "metrics": metrics, "insights": st["insights"]})
        open(os.path.join(gdir, "READY"), "w").close()
        append_jsonl(cfg.path("metrics_driver.jsonl"), metrics)
        log.info("group %d closed: %s", g, metrics)
        st["group"], st["k"], st["insights"], st["rollouts"] = g + 1, 0, [], []
        self.save()

    def run(self, max_attempts: int = 0):
        done = 0
        while not STOP and (max_attempts <= 0 or done < max_attempts):
            if self.state["k"] >= self.cfg.group_size:
                self.close_group()
            ro = self.run_attempt()
            if ro is None:
                continue
            self.state["consecutive_void"] = 0
            done += 1
            if self.state["k"] >= self.cfg.group_size:
                self.close_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-attempts", type=int, default=0, help="stop after N attempts (0 = run forever)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    cfg = load_config()
    log.info("config:\n%s", dump_config(cfg))
    Driver(cfg).run(args.max_attempts)


if __name__ == "__main__":
    main()
