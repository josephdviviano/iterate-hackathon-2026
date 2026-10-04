#!/usr/bin/env python3
"""
ar_run.py - trusted experiment runner for autoresearch; the agent's ONLY way to run train.py.

  ar_run.py --repo REPO --ledger LEDGER --gpu GPU_UUID --prepare-sha SHA256 --desc "what changed" [--timeout 660]
            [--full-summary] [--isolate]

The agent calls it through <repo>/run.sh ("./run.sh <description> > run.log 2>&1"). Per invocation:

 1. Refusal checks (no GPU use, still exactly one ledger line, exit code 3):
    - another ar_run already holds the GPU lock (<lock-dir>/<uuid>.lock, default $RLTLDR_ROOT/data/locks),
    - the per-attempt cap: env AR_ATTEMPT_ID set and >= AR_MAX_RUNS (default 3; <= 0 disables)
      non-refused runs of that attempt already in the ledger,
    - train.py changes GPU visibility itself (CUDA_VISIBLE_DEVICES / CUDA_DEVICE_ORDER /
      NVIDIA_VISIBLE_DEVICES / putenv), which could move training off the pinned GPU.
 2. Snapshot BEFORE the run into <runs-dir>/<run_id>/: train.py copy + diff.patch against the parent
    (pi-autoresearch reverts discarded edits after the run, so this is the only reliable record).
    Parent = nearest first-parent ancestor of the candidate whose train.py already has an "ok"
    measurement in the ledger (i.e. the version this experiment branched from); falls back to the
    immediate parent commit.
 3. Re-checks the venv GPU pin (ar_gpu_pin.py) and runs train.py under the trusted bootstrap
    (ar_bootstrap.py: a judge process that evaluates, a child that runs train.py) inside train_sandbox.sh, under
    `timeout --signal=TERM --kill-after=N <timeout>` with CUDA_VISIBLE_DEVICES=<uuid>, in its own session.
    --isolate (the h2h runner): the sandbox also hides $HOME except what the run needs, at arm-neutral paths:
    the repo (ro) at ~/autoresearch, the venv's interpreter (~/.local/share/uv), the data (~/.cache/autoresearch,
    validation shard masked: only the judge reads it, through an fd), the bootstrap at ~/.ar/, and --cache-dir as an
    ephemeral copy-on-write layer at ~/.cache/ar-compile (compiles start warm from it; nothing a run writes there
    reaches the next run). The key and the record file reach the judge only as fds.
    While it runs, nvidia-smi is polled: a descendant on any other GPU is killed (flag gpu_violation); foreign
    processes on our GPU are recorded (flag gpu_shared).
    Inside the sandbox the training is PID 1 of its own PID namespace and ignores SIGTERM, so on the deadline
    or on TERM/INT/HUP to this wrapper the sandbox tree is SIGKILLed via sudo at once (status timeout /
    crash + interrupted) and the ledger line is still written within ~1 s.
 4. Parses the log (ar_parse.py), computes integrity flags, appends exactly ONE JSON line to the
    ledger under flock + fsync, prints a compact summary, exits 0 (ok) / 1 (failed run) / 3 (refused).

stdout (what the agent sees) is ONLY:
    ok:      "val_bpb:          X" and "peak_vram_mb:     Y"   (program.md's grep keeps working);
             with --full-summary the whole upstream block instead ("---", val_bpb, training_seconds, ...,
             depth): trusted val_bpb and training time, the rest from train.py's own summary
    failed:  "status: <status>" followed by a sanitized error tail (<= 4 KB, no '\r' progress spam)
    refused: "status: refused" followed by the reason
The raw log stays in <runs-dir>/<run_id>/run.log and is never echoed.

Ledger line fields (see docs/AR_RUN.md): run_id, attempt_id, ts, wall_s, head, train_sha256,
parent_train_sha256, prepare_sha256, desc, status, returncode, val_bpb, metrics, error_tail, flags,
diff_path, log_path + bookkeeping (seq, branch, mode, parent_rev, parent_val_bpb, delta_vs_parent, ...).
agent_head: env AR_AGENT_HEAD (the submitting agent's own git HEAD, untrusted; <= 40 hex chars, else null).

Trusted training time (over_time_budget, reasons in metrics.timing_problems): t_to_eval - t_step_anchor (end of the
warm-up optimizer steps, ar_bootstrap.py) must be <= budget + STEP_ANCHOR_GRACE, and the warm-up itself must look like
warm-up: steps 2..11 at most WARMUP_STEP_FACTOR x the steady-state step time each (+ WARMUP_GRACE), and no more train
batches before the anchor / the first step than the steady-state batches per step allow (WARMUP_BATCH_*). Without an
anchor (no torch.optim.Optimizer step seen): t_to_eval - t_train2 (2nd train batch request) <= budget +
TRAIN_TIME_GRACE + the estimated time of the 11 warm-up steps; without either, only the outer bound. Always (outer
bound, on the judge's clock alone): t_to_eval <= budget + EVAL_TIME_ALLOWANCE. metrics.t_train_trusted /
t_train_source record which rule applied.
Only the Python standard library is used (run it with a trusted interpreter in isolated mode, e.g. /usr/bin/python3 -I
or $RLTLDR_TRUSTED_PY -I). Paths: the project root is $RLTLDR_ROOT, else the directory above tools/; $HOME is the
passwd home of the calling user.
"""
import argparse
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_gpu_pin import ensure_pin                      # noqa: E402
from ar_parse import format_last_step, parse_run_log  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SANDBOX = os.path.join(HERE, "train_sandbox.sh")   # GPU-only, no network, read-only home
BOOTSTRAP = os.path.join(HERE, "ar_bootstrap.py")         # trusted evaluation (signed record)
CAUSAL_TOL = 1e-3            # max |delta logit| at positions <= t when tokens after t are perturbed
STEP_ANCHOR_GRACE = 10.0     # trusted training time (eval start - end of the warm-up optimizer steps, the same
                             # steps train.py excludes from its own count) may exceed the budget by this [s]
TRAIN_TIME_GRACE = 20.0      # fallback without an optimizer anchor: eval start - 2nd train batch (this also
                             # counts the optimizer compile; the 11 warm-up steps are estimated and added) [s]
EVAL_TIME_ALLOWANCE = 300.0  # outer bound for every run (the only rule when train.py uses neither a torch optimizer
                             # nor prepare's train loader): start-up + compile before the budget [s]
WARMUP_STEP_FACTOR = 2.0     # optimizer steps 2..11 may take this many steady-state step times each ...
WARMUP_GRACE = 30.0          # ... plus this (e.g. a recompile in the warm-up) [s]
WARMUP_BATCH_FACTOR = 2.0    # train batches drawn before the anchor (before step 1) <= this x 11 (x 1) x the
WARMUP_BATCH_SLACK = 8       # steady-state train batches per step + this (prefetching) [batches]
STARTUP_NOTE = 120.0         # informational flag startup_long: the warm-up ended later than this after the start [s]
NO_OPT_MAX_BPS = 16          # fallback rule: cap on the batches-per-step estimate (from train.py's own num_steps)
WARMUP_OPT_STEPS = 11        # = ar_bootstrap.WARMUP_OPT_STEPS (train.py excludes steps 0..10 from its own time)
# --isolate: where things appear inside the train sandbox (relative to $HOME; the same for every caller and arm,
# so tracebacks, warnings and cache paths name no host directory). ~/autoresearch is also where the h2h agent sees
# its own repo.
IN_REPO, IN_VENV, IN_BOOTSTRAP, IN_CACHE = "autoresearch", ".venv", ".ar/ar_bootstrap.py", ".cache/ar-compile"
IN_RO = (".local/share/uv", ".cache/autoresearch")       # the venvs' base interpreter; data shards + tokenizer
VAL_SHARD = ".cache/autoresearch/data/shard_06542.parquet"   # prepare.VAL_FILENAME of the pinned prepare.py
KEY_FD, RECORD_FD, VAL_FD = 7, 8, 9
SANDBOX_KILL_AFTER = 5       # timeout(1) --kill-after in the sandbox, where its TERM cannot reach the training [s]
AGENT_HEAD_RE = re.compile(r"[0-9a-f]{1,40}")
# upstream train.py's final summary block: (key, format); keys are padded to 18 columns like upstream
SUMMARY_LINES = (("val_bpb", "{:.6f}"), ("training_seconds", "{:.1f}"), ("total_seconds", "{:.1f}"),
                 ("peak_vram_mb", "{:.1f}"), ("mfu_percent", "{:.2f}"), ("total_tokens_M", "{:.1f}"),
                 ("num_steps", "{:d}"), ("num_params_M", "{:.1f}"), ("depth", "{:d}"))
# Flags that make a measurement invalid (the runner/driver reject these runs).
INVALID_REASONS = {
    "no_trusted_eval": "the metric must come from calling prepare.evaluate_bpb(model, tokenizer, batch_size) "
                       "exactly once at the end of training (no evaluation was recorded)",
    "multiple_evals": "prepare.evaluate_bpb was called more than once (selecting on validation data is not allowed)",
    "val_leak": "validation data was accessed outside the final evaluation (training/selection on val is not allowed)",
    "non_causal": "the model is not causal: its outputs at a position depend on later tokens (information leak)",
    "eval_unverifiable": "model(idx) without targets must return logits of shape [B, T, vocab]; the harness "
                         "computes the loss from them",
    "over_time_budget": "training ran longer than the fixed 5-minute budget",
    "prepare_py_modified": "prepare.py was modified",
}
TRUSTED_FATAL = ("no_trusted_eval", "multiple_evals", "val_leak", "non_causal", "eval_unverifiable",
                 "over_time_budget")

ROOT = os.environ.get("RLTLDR_ROOT") or os.path.dirname(HERE)      # project root (contains tools/, data/)
HOME = pwd.getpwuid(os.getuid()).pw_dir     # from passwd, not $HOME; prepare.py reads $HOME/.cache/autoresearch
SAFE_PATH = os.path.join(HOME, ".local", "bin") + ":/usr/local/bin:/usr/bin:/bin"
DEFAULT_LOCK_DIR = os.path.join(ROOT, "data", "locks")
DEFAULT_CACHE_DIR = os.path.join(ROOT, "data", "cache")
# Environment variables that could redirect the interpreter, the GPU or the loader; never forwarded.
ENV_DENY_PREFIXES = ("PYTHON", "CUDA_", "NVIDIA_", "LD_", "UV_", "VIRTUAL_ENV", "CONDA", "TORCHINDUCTOR_",
                     "TRITON_", "HIP_", "ROCR_", "GPU_DEVICE_ORDINAL")
# Device-visibility changes inside train.py run after the venv pin and BEFORE CUDA initialises, so
# they are the one way train.py itself could move training to another GPU. (cuda:<n> / set_device
# cannot escape: only the pinned GPU is visible.) Obfuscated variants are caught by the nvidia-smi
# poll during the run (flag gpu_violation).
GPU_TAMPER_RE = re.compile(r"CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|NVIDIA_VISIBLE_DEVICES|\bputenv\b")
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 3
PR_SET_PDEATHSIG = 1


def ast_sha256(src: bytes):
    """Hash of the module AST: identical for edits that only touch comments or formatting."""
    import ast
    try:
        return hashlib.sha256(ast.dump(ast.parse(src.decode("utf-8", "replace"))).encode()).hexdigest()
    except (SyntaxError, ValueError):
        return None


def gpu_minor(uuid_: str) -> int:
    out = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=30).stdout
    cur = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("GPU UUID"):
            cur = line.split(":", 1)[1].strip()
        elif line.startswith("Minor Number") and cur == uuid_:
            return int(line.split(":", 1)[1])
    raise RuntimeError(f"minor number of {uuid_} not found")


def sudo_kill_tree(root: int) -> None:
    children = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    children.setdefault(int(f.read().rsplit(")", 1)[1].split()[1]), []).append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    pids, stack = [], [root]
    while stack:
        for c in children.get(stack.pop(), []):
            pids.append(c)
            stack.append(c)
    if pids:
        subprocess.run(["sudo", "-n", "kill", "-KILL", *map(str, pids)], capture_output=True)


def read_trusted(tdir: str, key: str) -> dict:
    """Verify and collect the bootstrap's signed records (forged or unsigned lines are ignored)."""
    import hmac
    out = {"valid": False, "evals": [], "val_leak": [], "bad_lines": 0, "start": None, "end": None}
    try:
        lines = open(os.path.join(tdir, "record.jsonl")).read().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            r = json.loads(line)
            want = hmac.new(key.encode(), r["payload"].encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(want, r["sig"]):
                raise ValueError("bad signature")
            d = json.loads(r["payload"])
        except (ValueError, KeyError, TypeError):
            out["bad_lines"] += 1
            continue
        ev = d.get("event")
        if ev == "start":
            out["start"], out["valid"] = d, True
        elif ev == "eval":
            out["evals"].append(d)
        elif ev == "val_leak":
            out["val_leak"].append(d.get("what"))
        elif ev == "end":
            out["end"] = d
    for e in out["evals"]:
        out["val_leak"] += e.get("val_leak") or []
    out["val_leak"] = sorted(set(out["val_leak"]))
    return out


def die_with_parent(parent_pid):
    """preexec_fn for timeout(1): if the wrapper dies (even by SIGKILL), the kernel sends timeout(1) a
    SIGTERM, which it forwards to the training process group (and SIGKILLs it after --kill-after).
    Without this an orphaned run would keep the GPU (and its lock) busy for up to --timeout seconds
    with no ledger line ever written for it."""
    def fn():
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        if os.getppid() != parent_pid:     # the wrapper already died between fork() and prctl()
            os._exit(1)
    return fn


# ----------------------------------------------------------------------------------------------- utils
def git(repo, *args, binary=False):
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True)
    return r.returncode, (r.stdout if binary else r.stdout.decode("utf-8", "replace").strip())


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    try:
        with open(path, "rb") as f:
            return sha256_bytes(f.read())
    except OSError:
        return None


def now_iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds")


def read_ledger(path):
    """All parseable entries; a torn/partial line (e.g. disk full) is skipped, not fatal."""
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return out


def append_ledger(path, entry):
    """Append exactly one line; returns its 0-based sequence number (computed under the lock)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            entry["seq"] = sum(1 for line in f if line.strip())
            f.seek(0, os.SEEK_END)
            f.write(json.dumps(entry, sort_keys=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return entry["seq"]


def agent_head_from_env():
    """The submitting agent's own `git rev-parse HEAD` (forwarded by the runner; untrusted, display/matching only)."""
    h = (os.environ.get("AR_AGENT_HEAD") or "").strip().lower()
    return h if AGENT_HEAD_RE.fullmatch(h) else None


def trusted_train_time(e0, budget, num_steps=None):
    """(training time or None, rule, limit) for the bootstrap's eval record e0; see the module docstring.
    num_steps (train.py's own count, untrusted) only sizes the warm-up allowance of the no-optimizer fallback."""
    if e0.get("t_step_anchor") is not None:
        return e0["t_to_eval"] - e0["t_step_anchor"], "step_anchor", budget + STEP_ANCHOR_GRACE
    if e0.get("t_train2"):
        t = e0["t_to_eval"] - e0["t_train2"]
        warm = 0.0
        tb = e0.get("train_batches")
        if isinstance(num_steps, int) and num_steps > WARMUP_OPT_STEPS and isinstance(tb, int) and tb > 2:
            bps = min(max(tb / num_steps, 1.0), NO_OPT_MAX_BPS)          # train batches per optimizer step
            warm = WARMUP_OPT_STEPS * bps * t / (tb - 2)                 # 11 steps at the average batch time
        return t, "train_batch2", budget + TRAIN_TIME_GRACE + warm
    return None, "t_to_eval", budget + EVAL_TIME_ALLOWANCE


def timing_problems(e0, budget, t_train, limit):
    """Reasons why the run trained longer than the budget (empty: it did not); see the module docstring."""
    out = []
    if t_train is not None and t_train > limit:
        out.append(f"training took {t_train:.1f} s (limit {limit:.1f} s)")
    if e0["t_to_eval"] > budget + EVAL_TIME_ALLOWANCE:
        out.append(f"the evaluation started {e0['t_to_eval']:.0f} s after the start "
                   f"(limit {budget + EVAL_TIME_ALLOWANCE:.0f} s)")
    anchor, steps = e0.get("t_step_anchor"), e0.get("opt_steps")
    if anchor is None or not isinstance(steps, int) or steps <= WARMUP_OPT_STEPS:
        return out
    n_after = steps - WARMUP_OPT_STEPS
    steady = (e0["t_to_eval"] - anchor) / n_after                       # seconds per step after the warm-up
    if e0.get("t_step1") is not None:
        warm, lim = anchor - e0["t_step1"], WARMUP_STEP_FACTOR * (WARMUP_OPT_STEPS - 1) * steady + WARMUP_GRACE
        if warm > lim:
            out.append(f"warm-up steps 2-{WARMUP_OPT_STEPS} took {warm:.1f} s (limit {lim:.1f} s at "
                       f"{steady:.2f} s per later step)")
    tb, ba, b1 = e0.get("train_batches"), e0.get("train_batches_at_anchor"), e0.get("train_batches_at_step1")
    if isinstance(tb, int) and isinstance(ba, int):
        bps = max(1.0, (tb - ba) / n_after)                              # train batches per step after the warm-up
        lim = WARMUP_BATCH_FACTOR * WARMUP_OPT_STEPS * bps + WARMUP_BATCH_SLACK
        if ba > lim:
            out.append(f"{ba} train batches before the end of warm-up step {WARMUP_OPT_STEPS} (limit {lim:.0f} at "
                       f"{bps:.1f} per later step)")
        lim1 = WARMUP_BATCH_FACTOR * bps + WARMUP_BATCH_SLACK
        if isinstance(b1, int) and b1 > lim1:
            out.append(f"{b1} train batches before the first optimizer step (limit {lim1:.0f})")
    return out


def format_summary(m, val_bpb):
    """Upstream's final summary block: trusted val_bpb + training time, train.py's own values for the rest.
    Lines whose value is unavailable are omitted."""
    vals = dict(m)
    vals["val_bpb"] = val_bpb
    if m.get("t_train_trusted") is not None:
        vals["training_seconds"] = m["t_train_trusted"]
    lines = ["---"]
    for key, fmt in SUMMARY_LINES:
        v = vals.get(key)
        if v is None:
            continue
        try:
            txt = fmt.format(int(v) if fmt == "{:d}" else float(v))
        except (TypeError, ValueError, OverflowError):
            continue
        lines.append(f"{key + ':':<18}{txt}")
    return "\n".join(lines)


def max_runs_from_env():
    try:
        return int(os.environ.get("AR_MAX_RUNS", "3"))
    except ValueError:
        return 3


# --------------------------------------------------------------------------------------- nvidia-smi
def gpu_apps():
    """[(pid, gpu_uuid, used_mib)] for all compute processes (nvidia-smi reports container PIDs here)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    apps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            apps.append((int(parts[0]), parts[1], int(parts[2]) if parts[2].isdigit() else -1))
    return apps


def gpu_exists(gpu_uuid):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return gpu_uuid in {l.strip() for l in out.splitlines()}


def descends_from(pid, root):
    for _ in range(512):
        if pid == root:
            return True
        if pid <= 1:
            return False
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])   # field 4 = ppid
        except (OSError, ValueError, IndexError):
            return False
    return False


# ------------------------------------------------------------------------------------- git snapshot
def find_parent(repo, dirty, ledger):
    """(parent_rev, parent_train_bytes): nearest first-parent ancestor of the candidate whose train.py
    has an ok measurement in the ledger, else the candidate's immediate parent (HEAD if the working
    tree is dirty, HEAD^ if the experiment was committed)."""
    measured = {e.get("train_sha256") for e in ledger if e.get("status") == "ok"}
    start = "HEAD" if dirty else "HEAD^"
    rc, revs = git(repo, "rev-list", "--first-parent", "--max-count=200", start)
    revs = revs.split() if rc == 0 else []
    fallback = None
    for rev in revs:
        rc, blob = git(repo, "show", f"{rev}:train.py", binary=True)
        if rc != 0:
            continue
        if fallback is None:
            fallback = (rev, blob)
        if sha256_bytes(blob) in measured:
            return rev, blob
    return fallback if fallback else (None, b"")


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Trusted autoresearch run wrapper (see module docstring).")
    ap.add_argument("--repo", required=True, help="autoresearch checkout (contains train.py, prepare.py)")
    ap.add_argument("--ledger", required=True, help="append-only JSONL ledger (outside the repo)")
    ap.add_argument("--gpu", required=True, help="physical GPU UUID (GPU-..., see nvidia-smi -L)")
    ap.add_argument("--prepare-sha", required=True, help="expected sha256 of prepare.py")
    ap.add_argument("--desc", default="", help="short description of the experiment")
    ap.add_argument("--timeout", type=int, default=660, help="hard wall-clock limit for uv run train.py [s]")
    ap.add_argument("--budget", type=float, default=300.0, help="prepare.TIME_BUDGET [s]")
    ap.add_argument("--min-steps", type=int, default=20, help="flag runs with fewer optimizer steps")
    ap.add_argument("--max-overhead", type=float, default=330.0,
                    help="flag runs whose wall time exceeds budget + this (compile + eval + startup) [s]")
    ap.add_argument("--runs-dir", default=None, help="default: <ledger dir>/runs")
    ap.add_argument("--venv", default=None, help="venv used by uv run (default <repo>/.venv)")
    ap.add_argument("--lock-dir", default=DEFAULT_LOCK_DIR,
                    help="per-GPU flock files (default $RLTLDR_ROOT/data/locks)")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                    help="TorchInductor/Triton caches for runs (default $RLTLDR_ROOT/data/cache)")
    ap.add_argument("--poll", type=float, default=5.0, help="nvidia-smi polling interval during the run [s]")
    ap.add_argument("--gpu-minor", type=int, default=None, help="/dev/nvidia<minor> of --gpu (default: query)")
    ap.add_argument("--no-autotune", action="store_true", help="disable Inductor runtime autotuning (less noise)")
    ap.add_argument("--unsandboxed", action="store_true", help="tests only: run train.py without the sandbox")
    ap.add_argument("--full-summary", action="store_true",
                    help="on success print upstream's whole summary block (default: val_bpb + peak_vram_mb only)")
    ap.add_argument("--isolate", action="store_true",
                    help="hide $HOME in the train sandbox except what the run needs, at neutral paths; ephemeral "
                         "compile cache; val shard for the judge only (see the module docstring)")
    a = ap.parse_args()
    if a.isolate and a.unsandboxed:
        ap.error("--isolate needs the sandbox")

    repo = os.path.abspath(a.repo)
    ledger_path = os.path.abspath(a.ledger)
    venv = os.path.abspath(a.venv or os.path.join(repo, ".venv"))
    runs_dir = os.path.abspath(a.runs_dir or os.path.join(os.path.dirname(ledger_path), "runs"))
    attempt_id = os.environ.get("AR_ATTEMPT_ID") or None
    max_runs = max_runs_from_env()
    t0 = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime(t0)) + "-" + uuid.uuid4().hex[:6]

    # Fields known before anything runs; refused entries carry these too.
    train_path = os.path.join(repo, "train.py")
    try:
        with open(train_path, "rb") as f:
            train_bytes = f.read()
    except OSError:
        train_bytes = b""
    _, head = git(repo, "rev-parse", "HEAD")
    _, branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    entry = {
        "run_id": run_id, "attempt_id": attempt_id, "ts": now_iso(t0), "wall_s": 0.0,
        "head": head or None, "branch": branch or None,
        "train_sha256": sha256_bytes(train_bytes), "train_ast_sha256": ast_sha256(train_bytes),
        "parent_train_sha256": None, "parent_rev": None,
        "prepare_sha256": sha256_file(os.path.join(repo, "prepare.py")), "desc": a.desc,
        "status": None, "returncode": None, "val_bpb": None, "metrics": {}, "error_tail": None,
        "flags": [], "diff_path": None, "log_path": None,
        "gpu": a.gpu, "max_runs": max_runs, "timeout_s": a.timeout, "agent_head": agent_head_from_env(),
    }

    def refuse(reason, flag):
        entry.update(status="refused", flags=entry["flags"] + [flag], error_tail=reason,
                     wall_s=round(time.time() - t0, 1), ts_end=now_iso(time.time()))
        append_ledger(ledger_path, entry)
        print("status: refused")
        print(reason)
        return EXIT_REFUSED

    # ---- 1. refusal checks (no GPU use) -------------------------------------------------------
    os.makedirs(a.lock_dir, exist_ok=True)
    lock_f = open(os.path.join(a.lock_dir, f"{a.gpu}.lock"), "a+")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return refuse("Another experiment is still running on this GPU. Wait for it to finish; "
                      "run experiments one at a time, in the foreground.", "gpu_locked")
    if not gpu_exists(a.gpu):
        return refuse(f"GPU {a.gpu} not found by nvidia-smi.", "gpu_missing")

    ledger = read_ledger(ledger_path)
    if attempt_id is not None and max_runs > 0:
        used = sum(1 for e in ledger if e.get("attempt_id") == attempt_id and e.get("status") != "refused")
        if used >= max_runs:
            return refuse(f"Run cap reached: this attempt has already used {used}/{max_runs} experiment runs. "
                          f"No GPU time was used. Do not run more experiments in this session; end with your "
                          f"final summary message. The harness records the result and decides keep/discard.",
                          "run_cap")
        # one measured result per attempt: re-running variants until noise produces a "win" is not allowed
        if any(e.get("attempt_id") == attempt_id and e.get("status") == "ok" for e in ledger):
            return refuse("This session already has a successful experiment run; only one result is recorded "
                          "per session. Do not run more experiments; end with your final summary message.",
                          "attempt_has_ok_run")
    tamper = sorted({m.group(0) for m in GPU_TAMPER_RE.finditer(train_bytes.decode("utf-8", "replace"))})
    if tamper:
        return refuse("train.py must not change GPU visibility (found: " + ", ".join(tamper)[:200] +
                      "). The GPU is assigned by the run wrapper; use torch.device('cuda').", "gpu_tamper")

    # ---- 2. snapshot the candidate BEFORE running ---------------------------------------------
    rdir = os.path.join(runs_dir, run_id)
    os.makedirs(rdir, exist_ok=True)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "train.py"], cwd=repo).returncode != 0
    # Detached HEAD = rltldr/runner.py's workspace: HEAD is always the attempt's parent commit and the
    # submitted train.py is the candidate, even when it is byte-identical to HEAD (a no-op submission
    # must get parent_rev=HEAD and the empty_diff flag, not be compared against HEAD^).
    parent_rev, parent_bytes = find_parent(repo, dirty or branch == "HEAD", ledger)
    diff = ""
    if parent_rev:
        _, diff = git(repo, "diff", parent_rev, "--", "train.py")   # parent commit vs working tree
    with open(os.path.join(rdir, "train.py"), "wb") as f:
        f.write(train_bytes)
    with open(os.path.join(rdir, "diff.patch"), "w") as f:
        f.write(diff + ("\n" if diff else ""))
    _, deps = git(repo, "hash-object", "pyproject.toml", "uv.lock")
    entry.update(parent_rev=parent_rev, parent_train_sha256=sha256_bytes(parent_bytes) if parent_rev else None,
                 parent_ast_sha256=ast_sha256(parent_bytes) if parent_rev else None,
                 mode="dirty" if dirty else "committed", diff_path=os.path.join(rdir, "diff.patch"),
                 train_copy_path=os.path.join(rdir, "train.py"), deps_git_hashes=deps.split(),
                 diff_lines=sum(1 for l in diff.splitlines()
                                if l[:1] in "+-" and not l.startswith(("+++", "---"))))

    # ---- 3. run, pinned to one GPU -------------------------------------------------------------
    flags = entry["flags"]
    try:
        if ensure_pin(venv, a.gpu):
            flags.append("gpu_pin_repaired")
    except FileNotFoundError:
        flags.append("gpu_pin_missing_venv")
    foreign_start = [(p, m) for p, g, m in gpu_apps() if g == a.gpu]
    if foreign_start:
        flags.append("gpu_shared_at_start")
        entry["gpu_foreign_at_start"] = foreign_start

    home = HOME                                 # prepare.py reads data from $HOME/.cache/autoresearch
    cache_in = os.path.join(home, IN_CACHE) if a.isolate else a.cache_dir
    env = {k: v for k, v in os.environ.items() if not k.startswith(ENV_DENY_PREFIXES)}
    env.update({
        "PATH": SAFE_PATH, "HOME": home,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": a.gpu,
        "UV_PROJECT_ENVIRONMENT": venv, "UV_FROZEN": "1", "UV_OFFLINE": "1", "UV_NO_PROGRESS": "1",
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(cache_in, "inductor"),
        "TRITON_CACHE_DIR": os.path.join(cache_in, "triton"),
        "PYTHONUNBUFFERED": "1", "AR_RUN_ID": run_id,
    })
    if a.no_autotune:
        env["AR_NO_AUTOTUNE"] = "1"
    env["CUDA_CACHE_PATH"] = os.path.join(cache_in, "nv")
    if a.isolate:
        env.pop("UV_PROJECT_ENVIRONMENT")       # a host path; uv is not used inside
    log_path = os.path.join(rdir, "run.log")
    entry["log_path"] = log_path
    # trusted channel: a one-time HMAC key only the judge process of ar_bootstrap.py gets (it reads and deletes the
    # file before train.py starts; with --isolate the sandbox hands it over as an fd and deletes the file)
    tdir = os.path.join(rdir, "trusted")
    os.makedirs(tdir, exist_ok=True)
    key = os.urandom(32).hex()
    kfd = os.open(os.path.join(tdir, "key"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(kfd, key.encode())
    os.close(kfd)
    os.makedirs(a.cache_dir, exist_ok=True)
    if a.isolate:
        if not tdir.startswith(home + "/"):
            raise SystemExit(f"--isolate: the runs dir must lie under {home} (it is hidden in the sandbox)")
        open(os.path.join(tdir, "record.jsonl"), "a").close()
        repo_in = os.path.join(home, IN_REPO)
        opts = ["--hide-home", home, "--expose", f"ro:{repo}:{repo_in}"]
        if venv == os.path.join(repo, ".venv"):
            py = os.path.join(repo_in, ".venv", "bin", "python")
        else:
            opts += ["--expose", f"ro:{venv}:{os.path.join(home, IN_VENV)}"]
            py = os.path.join(home, IN_VENV, "bin", "python")
        boot_in = os.path.join(home, IN_BOOTSTRAP)
        opts += ["--expose", f"ro:{BOOTSTRAP}:{boot_in}"]
        for d in IN_RO:
            if os.path.exists(os.path.join(home, d)):
                opts += ["--expose", f"ro:{os.path.join(home, d)}"]
        opts += ["--overlay", f"{os.path.abspath(a.cache_dir)}:{cache_in}",
                 "--fd", f"{KEY_FD}:consume:{tdir}/key", "--fd", f"{RECORD_FD}:append:{tdir}/record.jsonl"]
        boot_args = [repo_in, "-", "--key-fd", str(KEY_FD), "--record-fd", str(RECORD_FD)]
        val = os.path.join(home, VAL_SHARD)
        if os.path.exists(val):
            opts += ["--fd", f"{VAL_FD}:read:{val}", "--mask", val]
            boot_args += ["--val-fd", str(VAL_FD)]
        opts += ["--chdir", repo_in]
        inner = ["/usr/bin/env", "-i", *[f"{k}={v}" for k, v in sorted(env.items())], py, "-I", boot_in, *boot_args]
    else:
        opts = []
        py = os.path.join(venv, "bin", "python")
        inner = ["/usr/bin/env", "-i", *[f"{k}={v}" for k, v in sorted(env.items())],
                 py, "-I", BOOTSTRAP, repo, tdir]
    if a.unsandboxed:
        cmd = ["timeout", "--signal=TERM", "--kill-after=30", str(a.timeout), *inner]
    else:
        minor = a.gpu_minor if a.gpu_minor is not None else gpu_minor(a.gpu)
        rw = tdir if a.isolate else f"{a.cache_dir}:{tdir}"
        cmd = ["timeout", "--signal=TERM", f"--kill-after={SANDBOX_KILL_AFTER}", str(a.timeout), TRAIN_SANDBOX,
               str(minor), rw, *opts, "--", *inner]

    interrupted, timed_out = [], []
    child = None

    def stop_child():
        """TERM the training process group. In the sandbox that is not enough: the training is PID 1 of its own
        PID namespace (no TERM handler -> TERM is dropped) and unshare blocks TERM, so the whole sandbox tree
        is SIGKILLed via sudo while it is still intact (killing the namespace's init kills every process in it)."""
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except OSError:
            pass
        if not a.unsandboxed:
            sudo_kill_tree(child.pid)

    def on_signal(signum, _frame):
        # Stop the training right away instead of waiting for the next poll tick: callers such as
        # rltldr/runner.py SIGKILL the wrapper a few seconds after their TERM, and the ledger line must be
        # written before that.
        interrupted.append(signum)
        if child is not None:
            stop_child()

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)

    violations, foreign_seen = [], {}
    t_run = time.time()
    with open(log_path, "wb") as log:
        # The GPU lock fd is inherited by the whole training process tree, so an orphaned run
        # (wrapper SIGKILLed) keeps the GPU locked until it actually exits; die_with_parent makes
        # that exit happen within seconds.
        child = subprocess.Popen(cmd, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_f.fileno(),),
                                 preexec_fn=die_with_parent(os.getpid()))
        if interrupted:   # signal arrived while the child was being spawned
            stop_child()
        # own deadline (backstop for timeout(1), whose TERM the sandboxed training ignores)
        deadline = t_run + a.timeout + 1
        while True:
            try:
                rc = child.wait(timeout=max(0.1, min(a.poll, deadline - time.time())))
                break
            except subprocess.TimeoutExpired:
                pass
            if interrupted or time.time() >= deadline:
                if not interrupted:
                    timed_out.append(time.time() - t_run)
                stop_child()
                try:
                    rc = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    sudo_kill_tree(child.pid)
                    rc = child.wait()
                break
            for pid, g, mib in gpu_apps():
                ours = descends_from(pid, child.pid)
                if ours and g != a.gpu:
                    violations.append({"pid": pid, "gpu_uuid": g, "used_mib": mib})
                elif not ours and g == a.gpu:
                    foreign_seen[pid] = max(mib, foreign_seen.get(pid, 0))
            if violations:
                sudo_kill_tree(child.pid)
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except OSError:
                    pass
                rc = child.wait()
                break
    try:   # reap stragglers that ignored the TERM forwarded by timeout(1)
        os.killpg(child.pid, signal.SIGKILL)
    except OSError:
        pass
    sudo_kill_tree(child.pid)   # sudo/unshare run as root: make sure nothing of this run survives
    wall = time.time() - t_run
    try:                        # normally already consumed (the judge never started, e.g. a sandbox failure)
        os.unlink(os.path.join(tdir, "key"))
    except FileNotFoundError:
        pass
    trusted = read_trusted(tdir, key)
    entry["trusted"] = trusted

    # ---- 4. parse, flag, record ----------------------------------------------------------------
    with open(log_path, "rb") as f:
        text = f.read().decode("utf-8", "replace")
    res = parse_run_log(text)
    status, m = res["status"], res["metrics"]
    # timeout(1) exits 124 (TERM sufficed) or 137 (needed --kill-after), or is itself SIGKILLed (-9) together
    # with its process group by --kill-after; 137 is also "child SIGKILLed" (e.g. host OOM killer), so require
    # the wall clock to have actually reached the limit. timed_out: this wrapper's own deadline fired.
    if timed_out or (rc in (124, 137, -9) and wall >= a.timeout - 2):
        status = "timeout"       # program.md: a run over the limit is a failure, even if it printed metrics
    if violations:
        status = "crash"
        flags.append("gpu_violation")
        entry["gpu_violations"] = violations
    if interrupted:
        status = "crash" if status == "ok" else status
        flags.append("interrupted")
    if status == "ok" and rc != 0:
        flags.append("nonzero_exit")
    if foreign_seen:
        flags.append("gpu_shared")
        entry["gpu_foreign_max_mib"] = foreign_seen

    # Trusted evaluation (ar_bootstrap.py): the metric that counts. train.py's own summary is display-only.
    ev = trusted.get("evals") or []
    if status == "ok":
        if not trusted.get("valid") or not ev or ev[0].get("val_bpb") is None:
            flags.append("no_trusted_eval")
            m["val_bpb_stdout"] = m.get("val_bpb")
            m["val_bpb"] = None                        # nothing trustworthy was measured
        else:
            e0 = ev[0]
            if len(ev) > 1:
                flags.append("multiple_evals")
            if trusted.get("val_leak"):
                flags.append("val_leak")
            if "check_error" in e0 or "causal_max_abs" not in e0:
                flags.append("eval_unverifiable")
            elif e0["causal_max_abs"] > CAUSAL_TOL:
                flags.append("non_causal")
            t_train, source, limit = trusted_train_time(e0, a.budget, m.get("num_steps"))
            t_b2 = (e0["t_to_eval"] - e0["t_train2"]) if e0.get("t_train2") else None
            m["t_train_trusted"] = round(t_train, 1) if t_train is not None else None
            m["t_train_source"] = source
            m["t_train_limit"] = round(limit, 1)
            m["t_train_batch2"] = round(t_b2, 1) if t_b2 is not None else None   # old rule, for comparison
            for k in ("t_step_anchor", "t_step1"):
                if e0.get(k) is not None:
                    m[k] = round(e0[k], 1)
            for k in ("opt_steps", "train_batches", "train_batches_at_step1", "train_batches_at_anchor"):
                if e0.get(k) is not None:
                    m[k] = e0[k]                                 # trusted counts (optimizer steps, train batches)
            problems = timing_problems(e0, a.budget, t_train, limit)
            if problems:
                flags.append("over_time_budget")
                m["timing_problems"] = problems
            warm_end = e0.get("t_step_anchor") if e0.get("t_step_anchor") is not None else e0.get("t_train2")
            if warm_end is not None and warm_end > STARTUP_NOTE:
                flags.append("startup_long")                     # informational: slow compile or start-up
            if m.get("val_bpb") is not None and abs(m["val_bpb"] - e0["val_bpb"]) > 1e-5:
                flags.append("stdout_mismatch")      # informational: printed summary differs from the trusted value
            m["val_bpb_stdout"] = m.get("val_bpb")
            m["val_bpb"] = e0["val_bpb"]
            m["t_to_eval"] = round(e0["t_to_eval"], 1)

    # Integrity flags: detection, not prevention (cf. karpathy/autoresearch#599).
    if entry["prepare_sha256"] != a.prepare_sha:
        flags.append("prepare_py_modified")
    if status == "ok":
        if m.get("training_seconds", 0.0) > a.budget + 5 and "over_time_budget" not in flags:
            flags.append("over_time_budget")
        if m.get("num_steps", 0) < a.min_steps:
            flags.append("too_few_steps")
        if wall > a.budget + a.max_overhead:
            flags.append("wall_clock_excessive")
    # cosmetic edits (comments/whitespace) do not make a new experiment: compare the AST as well
    ast_sha = entry["train_ast_sha256"]
    if entry["diff_lines"] == 0 or (ast_sha and ast_sha == entry.get("parent_ast_sha256")):
        flags.append("empty_diff")
    if any(e.get("status") == "ok" and (e.get("train_sha256") == entry["train_sha256"]
                                        or (ast_sha and e.get("train_ast_sha256") == ast_sha)) for e in ledger):
        flags.append("reeval")   # this train.py (up to comments/formatting) was already measured

    error_tail = None
    if status != "ok":
        parts = [res.get("error_tail") or ""]
        if status == "timeout":
            parts.append(f"killed: exceeded the {a.timeout}s wall-clock limit")
        if violations:
            parts.append("killed: the run used a GPU other than the one assigned to it")
        if interrupted:
            parts.append(f"killed: wrapper received signal {interrupted[0]}")
        ls = format_last_step(res["last_step"])
        if ls:
            parts.append(ls)
        error_tail = "\n".join(p for p in parts if p).strip("\n") or "(no output)"
        b = error_tail.encode("utf-8")
        if len(b) > 4096:
            error_tail = b[-4096:].decode("utf-8", "ignore")

    val = m.get("val_bpb") if status == "ok" else None
    invalid = [f for f in flags if f in TRUSTED_FATAL or f == "prepare_py_modified"]
    parent_vals = [e["val_bpb"] for e in ledger
                   if e.get("status") == "ok" and e.get("val_bpb") is not None
                   and e.get("train_sha256") == entry["parent_train_sha256"]]
    parent_val = sum(parent_vals) / len(parent_vals) if parent_vals else None
    entry.update(
        wall_s=round(wall, 1), ts_end=now_iso(time.time()), status=status, returncode=rc, val_bpb=val,
        metrics=m if status == "ok" else {}, error_tail=error_tail, last_step=res["last_step"],
        parent_val_bpb=parent_val, parent_n=len(parent_vals),
        delta_vs_parent=(parent_val - val) if (val is not None and parent_val is not None) else None,
    )
    append_ledger(ledger_path, entry)
    lock_f.close()

    # ---- compact, agent-facing stdout ----------------------------------------------------------
    if status == "ok" and invalid:
        print("status: invalid")
        print("The run finished but its result is INVALID and will not count:")
        for f in invalid:
            detail = "; ".join(m.get("timing_problems") or []) if f == "over_time_budget" else ""
            print(f"- {INVALID_REASONS.get(f, f)}" + (f" ({detail})" if detail else ""))
        return EXIT_FAILED
    if status == "ok":
        if a.full_summary:
            print(format_summary(m, val))
        else:
            print(f"val_bpb:          {val:.6f}")
            print(f"peak_vram_mb:     {m['peak_vram_mb']:.1f}")
        return EXIT_OK
    print(f"status: {status}")
    print(error_tail)
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
