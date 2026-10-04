#!/usr/bin/env python3
"""Head-to-head (h2h) setup: the per-arm agent repos and pi session dirs (stdlib only).

  python3 tools/h2h/setup.py [--arm base|v5 ...] [--check] [--force] [--create-start]

1. Verifies canon.git branch `h2h/start`: one commit on top of the baseline (cfg.baseline_commit) whose
   program.md is byte-identical to tools/h2h/program.md and which carries an executable run.sh. The branch is
   never modified; with --create-start a missing branch is first created (the only write to canon.git): the
   baseline tree with program.md = tools/h2h/program.md and run.sh = tools/h2h/run.sh (a template: its
   @RLTLDR_ROOT@ token becomes the project root, where the sandboxed agent finds tools/run_client.py).
2. Per arm, `arm.repo` = the agent's git repo, built as a fresh clone of only that branch:
     git clone --no-local --template= --single-branch --branch h2h/start --no-tags canon.git
     -> branch renamed to `autoresearch/h2h`, `origin` removed, reflogs expired, `git gc --prune=now`,
     local identity `autoresearch <autoresearch@localhost>`, `.venv` via `uv sync --frozen`.
   `--no-local` makes git transfer only the objects reachable from h2h/start (a local clone would hardlink
   canon.git's whole object store, including branch autoresearch/oct3 = the old run's best solutions). The
   clone is then audited: every ref, every commit of `git log --all` and every object of
   `git cat-file --batch-all-objects` must be reachable from the start commit, and no object reachable only from
   another canon.git branch may exist. The audit runs on every invocation (also for repos built earlier).
   The venv is built with UV_LINK_MODE=copy so that it shares no inodes with the uv cache (and hence with the
   runners' venvs, which uv hardlinks from that cache); H2H_UV_LINK_MODE overrides it (tests use hardlink).
3. Per arm, `arm.session_dir/agent` = copy of pi/agent without auth.json / models-store.json / sessions, and an
   empty `arm.session_dir/session/` (pi's session files; the supervisor runs pi with --session-dir there).
Runner workspaces are created by h2h_runner itself.

Idempotent: a complete, valid arm is left as is. Nothing is archived or rebuilt while the arm's supervisor or runner
holds its lock (data/h2h/<arm>/{supervisor,runner}.lock). An arm that is in use (ledger lines, runs, a pi session or a
results.tsv with rows) is never rebuilt unless --force; --force first moves the arm's agent and result state
(repo, session, ledger, runs, events, status, calls) to data/h2h/archive/<timestamp>/<arm>/ (runner_ws and the
compile cache stay). --check only verifies and reports. Exit code 0 iff every selected arm is ready.

The agent's repo and session dir are writable inside its sandbox, so this script treats their content as hostile
(it runs on the host as the invoking user, who has sudo): no agent-controlled code is ever executed (the venv import check runs
`python -I` from `/`, so a planted torch.py/numpy.py in the repo cannot shadow the real modules; `uv sync` only runs
on a project whose pyproject.toml / uv.lock / .python-version are the start commit's), git runs with the settings
that make read-only commands execute programs overridden (an agent-written .git/config could set
log.showSignature + gpg.program) and with a timeout, and agent-writable files are read only as regular files,
without following symlinks, without blocking on FIFOs and up to a size cap.
"""
import argparse
import fcntl
import filecmp
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rltldr.h2h_config import H2H, load_h2h_config  # noqa: E402

log = logging.getLogger("h2h_setup")
GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0",
           "GIT_ASKPASS": "/bin/false", "LC_ALL": "C", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_PAGER": "cat",
           "PAGER": "cat"}
# Command-line config beats the repo's own (agent-writable) .git/config. Besides hooks/fsmonitor these are the
# settings through which the read-only commands used here could run a program (git log + log.showSignature runs
# gpg.program on a signed commit; verified).
GIT_SAFE = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c", "log.showSignature=false",
            "-c", "gpg.program=/bin/false", "-c", "gpg.ssh.program=/bin/false", "-c", "gpg.x509.program=/bin/false")
GIT_TIMEOUT = 600                 # s; an agent-planted FIFO (e.g. .git/HEAD) must not hang setup forever
READ_CAP = 64 << 20               # max bytes read from one agent-writable file
PROJECT_FILES = ("pyproject.toml", "uv.lock", ".python-version")    # what `uv sync` reads from the project
UV = os.environ.get("UV_BIN") or "uv"
AGENT_IGNORE = ("auth.json", "models-store.json", "sessions")
IDENTITY = ("autoresearch", "autoresearch@localhost")
# per-arm state moved away by --force (agent + results); runner_ws / cache are harness infrastructure
ARM_STATE = ("repo", "session", "ledger.jsonl", "runs", "events.jsonl", "status.json", "calls.jsonl",
             "runner_state.json", "pi_stderr.log")
DAEMON_LOCKS = ("supervisor.lock", "runner.lock")     # flock'ed by h2h_supervisor / h2h_runner while they run
START_IDENTITY = ("autoresearch harness", "harness@localhost")         # author/committer of the h2h/start commit
START_MESSAGE = "harness: program.md for continuous sessions (./run.sh on a dedicated GPU)"
ROOT_TOKEN = b"@RLTLDR_ROOT@"


class SetupError(RuntimeError):
    pass


def _git_run(cwd: str, args, inp=None):
    try:
        return subprocess.run(["git", *GIT_SAFE, *args], cwd=cwd, capture_output=True, input=inp,
                              env={**os.environ, **GIT_ENV}, timeout=GIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise SetupError(f"git {' '.join(args)} (in {cwd}) did not finish within {GIT_TIMEOUT} s")


def git(cwd: str, *args, check: bool = True, binary: bool = False, inp=None):
    """git with harness-safe settings: no hooks, no fsmonitor, no signature programs, no global/system config."""
    r = _git_run(cwd, args, inp)
    if check and r.returncode != 0:
        raise SetupError(f"git {' '.join(args)} (in {cwd}) failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode(errors="replace").strip()


def git_ok(cwd: str, *args) -> bool:
    return _git_run(cwd, args).returncode == 0


def lines(s: str) -> list:
    return [l for l in s.splitlines() if l.strip()]


def read_untrusted(path: str, cap: int = READ_CAP):
    """Content of an agent-writable file, or None if it does not exist. Opened without following a symlink and
    without blocking (a FIFO), it must be a regular file of at most `cap` bytes; otherwise SetupError."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    except OSError as e:                         # ELOOP: a symlink
        raise SetupError(f"{path}: cannot be read safely ({e.strerror}; a symlink?)")
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SetupError(f"{path}: not a regular file")
        out, n = [], 0
        while True:                              # st_size may lie (/proc-like files): count what is read
            b = os.read(fd, min(1 << 20, cap + 1 - n))
            if not b:
                break
            out.append(b)
            n += len(b)
            if n > cap:
                raise SetupError(f"{path}: larger than {cap} bytes")
        return b"".join(out)
    finally:
        os.close(fd)


# ------------------------------------------------------------------------------------------ canon (read-only)
def verify_start(cfg) -> dict:
    """Check canon.git's h2h/start (never modified here). Returns {"start": sha, ...}."""
    canon = cfg.canon_dir
    start = git(canon, "rev-parse", "--verify", f"refs/heads/{cfg.start_ref}^{{commit}}", check=False)
    if not start:
        raise SetupError(f"canon.git has no branch {cfg.start_ref}; it must be created before setup (not by setup)")
    parents = git(canon, "rev-list", "--parents", "-n", "1", start).split()[1:]
    if parents != [git(canon, "rev-parse", "--verify", f"{cfg.baseline_commit}^{{commit}}")]:
        raise SetupError(f"{cfg.start_ref} ({start[:7]}) must have exactly one parent, the baseline "
                         f"{cfg.baseline_commit[:7]}; has {[p[:7] for p in parents]}")
    with open(os.path.join(cfg.root, "tools", "h2h", "program.md"), "rb") as f:
        want = f.read()
    if git(canon, "show", f"{start}:program.md", binary=True) != want:
        raise SetupError(f"program.md on {cfg.start_ref} differs from tools/h2h/program.md")
    entry = git(canon, "ls-tree", start, "run.sh").split()
    if not entry or entry[0] != "100755":
        raise SetupError(f"run.sh on {cfg.start_ref} is missing or not executable: {entry}")
    if b"tools/run_client.py" not in git(canon, "show", f"{start}:run.sh", binary=True):
        raise SetupError("run.sh on h2h/start does not call tools/run_client.py")
    changed = sorted(lines(git(canon, "diff", "--name-only", cfg.baseline_commit, start)))
    return {"start": start, "parent": parents[0], "changed_vs_baseline": changed}


def create_start(cfg) -> str:
    """Create canon.git branch cfg.start_ref if it does not exist yet: one commit on top of the baseline with
    program.md = tools/h2h/program.md and an executable run.sh rendered from tools/h2h/run.sh. Built with a
    temporary index (no work tree, nothing else in canon.git changes). Returns the new commit."""
    canon = cfg.canon_dir
    if git(canon, "rev-parse", "--verify", "-q", f"refs/heads/{cfg.start_ref}", check=False):
        raise SetupError(f"canon.git already has branch {cfg.start_ref}; it is never modified by setup")
    base = git(canon, "rev-parse", "--verify", f"{cfg.baseline_commit}^{{commit}}")
    h2h_dir = os.path.join(cfg.root, "tools", "h2h")
    with open(os.path.join(h2h_dir, "program.md"), "rb") as f:
        program = f.read()
    with open(os.path.join(h2h_dir, "run.sh"), "rb") as f:
        run_sh = f.read()
    if ROOT_TOKEN not in run_sh:
        raise SetupError(f"{h2h_dir}/run.sh has no {ROOT_TOKEN.decode()} token")
    # physical path: tools/h2h_sandbox.sh re-exposes tools/ at its symlink-free path (pwd -P)
    run_sh = run_sh.replace(ROOT_TOKEN, os.path.realpath(cfg.root).encode())
    with tempfile.TemporaryDirectory(prefix="h2h_start_") as td:
        env = {**os.environ, **GIT_ENV, "GIT_INDEX_FILE": os.path.join(td, "index"),
               "GIT_AUTHOR_NAME": START_IDENTITY[0], "GIT_AUTHOR_EMAIL": START_IDENTITY[1],
               "GIT_COMMITTER_NAME": START_IDENTITY[0], "GIT_COMMITTER_EMAIL": START_IDENTITY[1]}

        def g(*args, inp=None):
            r = subprocess.run(["git", *GIT_SAFE, *args], cwd=canon, input=inp, capture_output=True, env=env,
                               timeout=GIT_TIMEOUT)
            if r.returncode != 0:
                raise SetupError(f"git {' '.join(args)} (in {canon}) failed: {r.stderr.decode(errors='replace')}")
            return r.stdout.decode().strip()
        g("read-tree", base)
        for name, data, mode in (("program.md", program, "100644"), ("run.sh", run_sh, "100755")):
            oid = g("hash-object", "-w", "--stdin", inp=data)
            g("update-index", "--add", "--cacheinfo", f"{mode},{oid},{name}")
        commit = g("commit-tree", g("write-tree"), "-p", base, "-m", START_MESSAGE)
        g("update-ref", f"refs/heads/{cfg.start_ref}", commit, "0" * 40)      # create only, never overwrite
    log.info("created canon.git branch %s = %s on top of %s", cfg.start_ref, commit[:7], base[:7])
    return commit


def foreign_objects(cfg, start: str) -> set:
    """Objects reachable from any canon.git ref but not from the start commit (e.g. autoresearch/oct3)."""
    canon = cfg.canon_dir
    refs = [r for r in lines(git(canon, "for-each-ref", "--format=%(refname)")) if r != f"refs/heads/{cfg.start_ref}"]
    if not refs:
        return set()
    out = git(canon, "rev-list", "--objects", *refs, "--not", start)
    return {l.split()[0] for l in lines(out)}


# ------------------------------------------------------------------------------------------------ the repos
def audit_repo(cfg, repo: str, start: str, foreign: set) -> dict:
    """Information isolation + shape of an agent repo. Raises SetupError on any violation."""
    gitdir = os.path.join(repo, ".git")
    if os.path.islink(gitdir) or not os.path.isdir(gitdir):
        raise SetupError(f"{repo}: .git is not a plain directory")
    for f in ("objects/info/alternates", "objects/info/http-alternates", "commondir"):
        if os.path.lexists(os.path.join(gitdir, f)):
            raise SetupError(f"{repo}: has .git/{f} (would make git read another repository's objects/refs)")
    refs = lines(git(repo, "for-each-ref", "--format=%(refname) %(objectname)"))
    want_ref = f"refs/heads/{cfg.agent_branch}"
    if [r.split()[0] for r in refs] != [want_ref]:
        raise SetupError(f"{repo}: refs must be exactly [{want_ref}], found {refs}")
    if git(repo, "symbolic-ref", "HEAD") != want_ref:
        raise SetupError(f"{repo}: HEAD is not {want_ref}")
    if lines(git(repo, "remote")):
        raise SetupError(f"{repo}: has remotes {lines(git(repo, 'remote'))}")
    for f in ("FETCH_HEAD", "ORIG_HEAD", "MERGE_HEAD", "shallow"):
        if os.path.lexists(os.path.join(gitdir, f)):
            raise SetupError(f"{repo}: unexpected .git/{f}")
    if not git_ok(repo, "cat-file", "-e", f"{start}^{{commit}}"):
        raise SetupError(f"{repo}: the start commit {start[:7]} is missing")
    history = set(lines(git(repo, "rev-list", start)))
    logged = set(lines(git(repo, "log", "--no-show-signature", "--all", "--reflog", "--format=%H")))
    if not history <= logged:
        raise SetupError(f"{repo}: git log --all does not contain the start commit's history")
    # commits beyond the start commit can only be the agent's own work, on top of the start commit
    agent_commits = logged - history
    bad = [c for c in agent_commits if not git_ok(repo, "merge-base", "--is-ancestor", start, c)]
    if bad:
        raise SetupError(f"{repo}: git log --all shows {len(bad)} commit(s) outside the start commit's "
                         f"history: {sorted(bad)[:5]}")
    reachable = {l.split()[0] for l in lines(git(repo, "rev-list", "--objects", start))}
    present = set(lines(git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)")))
    if not agent_commits and present - reachable:      # fresh repo: exactly the start commit's objects
        raise SetupError(f"{repo}: {len(present - reachable)} object(s) not reachable from the start commit")
    if present & foreign:
        raise SetupError(f"{repo}: contains {len(present & foreign)} object(s) of other canon.git branches")
    # text metadata (config, packed-refs, logs, ...) must not name other branches either. os.walk does not enter
    # symlinked dirs; read_untrusted rejects symlinks, FIFOs, devices and oversized files (git creates none here).
    for d, _, files in os.walk(gitdir):
        if os.sep + "objects" in d:
            continue
        for fn in files:
            if b"oct3" in (read_untrusted(os.path.join(d, fn)) or b""):
                raise SetupError(f"{repo}: {os.path.join(d, fn)} mentions oct3")
    tip = git(repo, "rev-parse", "HEAD")
    ident = (git(repo, "config", "--local", "user.name", check=False),
             git(repo, "config", "--local", "user.email", check=False))
    if ident != IDENTITY:
        raise SetupError(f"{repo}: git identity is {ident}, want {IDENTITY}")
    return {"head": tip, "n_commits": len(history), "n_agent_commits": len(agent_commits),
            "n_objects": len(present), "refs": [want_ref]}


def build_repo(cfg, arm, start: str) -> None:
    tmp = arm.repo + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(arm.dir, exist_ok=True)
    git(arm.dir, "clone", "-q", "--no-local", "--template=", "--single-branch", "--branch", cfg.start_ref,
        "--no-tags", cfg.canon_dir, tmp)
    git(tmp, "branch", "-m", cfg.start_ref, cfg.agent_branch)
    git(tmp, "remote", "remove", "origin")
    git(tmp, "config", "user.name", IDENTITY[0])
    git(tmp, "config", "user.email", IDENTITY[1])
    git(tmp, "reflog", "expire", "--expire=now", "--expire-unreachable=now", "--all")
    git(tmp, "gc", "-q", "--prune=now")
    if git(tmp, "rev-parse", "HEAD") != start:
        raise SetupError(f"fresh clone HEAD is not {start}")
    if git(tmp, "status", "--porcelain"):
        raise SetupError(f"fresh clone is not clean: {git(tmp, 'status', '--porcelain')}")
    os.rename(tmp, arm.repo)       # venv after the move: uv writes absolute paths into it


def project_drift(cfg, repo: str, start: str) -> list:
    """Project files that differ from the start commit (uv sync on an agent-edited uv.lock/pyproject.toml could
    build and run agent-chosen code), plus a uv.toml that the start commit does not have."""
    bad = []
    for f in PROJECT_FILES:
        want = git(cfg.canon_dir, "show", f"{start}:{f}", binary=True) \
            if git_ok(cfg.canon_dir, "cat-file", "-e", f"{start}:{f}") else None
        try:
            have = read_untrusted(os.path.join(repo, f))
        except SetupError:
            have = b"\0(not a regular file)"
        if have != want:
            bad.append(f)
    if os.path.lexists(os.path.join(repo, "uv.toml")):
        bad.append("uv.toml")
    return bad


def ensure_venv(cfg, repo: str, start: str) -> dict:
    py = os.path.join(repo, ".venv", "bin", "python")
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")}
    env.update(UV_LINK_MODE=os.environ.get("H2H_UV_LINK_MODE", "copy"), UV_PYTHON_DOWNLOADS="never")
    if not os.path.exists(py):
        drift = project_drift(cfg, repo, start)
        if drift:
            raise SetupError(f"{repo}: .venv is missing and {drift} differ from {cfg.start_ref}; refusing to run "
                             f"uv sync on agent-edited project files (rebuild the arm with --force)")
        log.info("uv sync --frozen in %s (link mode %s; takes a few minutes)", repo, env["UV_LINK_MODE"])
        r = subprocess.run([UV, "sync", "--frozen", "--no-progress"], cwd=repo, env=env, capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise SetupError(f"uv sync failed in {repo}: {r.stderr.strip()[-2000:]}")
    # CPU-only import check (no CUDA context is created by importing torch). The venv is read-only inside the
    # sandbox, the repo is not: -I keeps the cwd/script dir and PYTHON* variables off sys.path, and cwd=/ is
    # neutral, so an agent-planted torch.py / numpy.py / torch/ in the repo cannot run here on the host.
    try:
        r = subprocess.run([py, "-I", "-c", "import sys, torch, numpy; print(sys.version.split()[0], "
                            "torch.__version__)"], cwd="/", capture_output=True, text=True,
                           env={**env, "CUDA_VISIBLE_DEVICES": ""}, timeout=300)
    except subprocess.TimeoutExpired:
        raise SetupError(f"{repo}/.venv: importing torch took more than 300 s")
    if r.returncode != 0:
        raise SetupError(f"{repo}/.venv is broken: {r.stderr.strip()[-1000:]}")
    return {"venv": r.stdout.strip()}


# --------------------------------------------------------------------------------------------- session dirs
def agent_template_diff(src: str, dst: str) -> list:
    """Template files (relative paths) that are missing from or differ in dst."""
    out = []
    for d, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if x not in AGENT_IGNORE]
        for fn in files:
            if fn in AGENT_IGNORE:
                continue
            rel = os.path.relpath(os.path.join(d, fn), src)
            t = os.path.join(dst, rel)
            if not os.path.isfile(t) or not filecmp.cmp(os.path.join(d, fn), t, shallow=False):
                out.append(rel)
    return out


def ensure_session(cfg, arm, may_write: bool) -> dict:
    agent = os.path.join(arm.session_dir, "agent")
    sess = os.path.join(arm.session_dir, "session")
    for p in (arm.session_dir, agent, sess):          # agent-writable: a symlink would redirect rmtree/copytree
        if os.path.islink(p):
            raise SetupError(f"{p} is a symlink")
    if may_write:
        if os.path.isdir(agent) and agent_template_diff(cfg.pi_agent_template, agent):
            shutil.rmtree(agent)
        if not os.path.isdir(agent):
            shutil.copytree(cfg.pi_agent_template, agent, ignore=shutil.ignore_patterns(*AGENT_IGNORE))
        os.makedirs(sess, exist_ok=True)
    diff = agent_template_diff(cfg.pi_agent_template, agent) if os.path.isdir(agent) else ["(missing)"]
    leaked = [f for f in AGENT_IGNORE if os.path.exists(os.path.join(agent, f))]
    if diff:
        raise SetupError(f"{agent} differs from {cfg.pi_agent_template}: {diff}")
    if leaked:
        raise SetupError(f"{agent} contains {leaked} (must not be copied)")
    if not os.path.isdir(sess):
        raise SetupError(f"{sess} missing")
    return {"session_files": len(session_files(sess))}


def session_files(sess: str) -> list:
    """pi session files (os.walk never enters symlinked dirs; a symlinked top dir is not walked either)."""
    out = []
    if os.path.islink(sess):
        return out
    for d, _, files in os.walk(sess):
        out += [os.path.join(d, f) for f in files if f.endswith(".jsonl")]
    return out


# ------------------------------------------------------------------------------------------------- per arm
def in_use(arm) -> list:
    """Reasons why this arm's state must not be overwritten without --force (agent-written files: read safely)."""
    why = []
    if os.path.exists(arm.ledger) and os.path.getsize(arm.ledger) > 0:
        why.append("ledger has runs")
    if os.path.isdir(arm.runs_dir) and os.listdir(arm.runs_dir):
        why.append("runs/ is not empty")
    sess = os.path.join(arm.session_dir, "session")
    if os.path.islink(sess) or os.path.islink(arm.session_dir):
        why.append("the session dir is a symlink")
    elif session_files(sess):
        why.append("a pi session exists")
    tsv = os.path.join(arm.repo, "results.tsv")
    try:
        data = read_untrusted(tsv)
    except SetupError as e:
        why.append(f"results.tsv is unusual ({e})")
    else:
        if data is not None and len([l for l in data.splitlines() if l.strip()]) > 1:
            why.append("results.tsv has rows")
    return why


def live_daemons(arm) -> list:
    """Lock files of this arm currently held by a running supervisor/runner."""
    held = []
    for name in DAEMON_LOCKS:
        p = os.path.join(arm.dir, name)
        if not os.path.exists(p):
            continue
        with open(p, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except BlockingIOError:
                held.append(name)
    return held


def archive_arm(arm) -> str:
    dest = os.path.join(H2H, "archive", time.strftime("%Y%m%d_%H%M%S"), arm.name)
    os.makedirs(dest, exist_ok=True)
    for name in ARM_STATE:
        p = os.path.join(arm.dir, name)
        if os.path.lexists(p):
            shutil.move(p, os.path.join(dest, name))
    return dest


def setup_arm(cfg, arm, info: dict, foreign: set, check_only: bool, force: bool) -> dict:
    start = info["start"]
    res = {"arm": arm.name, "repo": arm.repo, "session_dir": arm.session_dir}
    busy = in_use(arm)
    res["in_use"] = busy
    valid = None
    if os.path.isdir(os.path.join(arm.repo, ".git")):
        try:
            res.update(audit_repo(cfg, arm.repo, start, foreign))
            valid = True
        except SetupError as e:
            valid = False
            res["audit_error"] = str(e)
    held = live_daemons(arm)

    def refuse_if_live(action):
        if held:
            raise SetupError(f"{arm.name}: cannot {action}: {', '.join(held)} held (stop the arm's supervisor/runner "
                             f"first)")
    if check_only:
        if not valid:
            raise SetupError(res.get("audit_error") or f"{arm.repo} does not exist")
        res.update(ensure_venv(cfg, arm.repo, start))
        res.update(ensure_session(cfg, arm, may_write=False))
        return res
    if force and (busy or os.path.lexists(arm.repo) or os.path.lexists(arm.session_dir)):
        refuse_if_live("archive the arm")
        res["archived_to"] = archive_arm(arm)
        log.warning("%s: previous state moved to %s", arm.name, res["archived_to"])
        busy, valid = [], None
    if valid is False or (valid is None and os.path.lexists(arm.repo)):
        if busy:
            raise SetupError(f"{arm.name}: repo is invalid ({res.get('audit_error')}) but the arm is in use "
                             f"({', '.join(busy)}); refusing to rebuild without --force")
        refuse_if_live("rebuild the repo")
        log.warning("%s: rebuilding invalid/incomplete repo (%s)", arm.name, res.get("audit_error"))
        shutil.rmtree(arm.repo)
        valid = None
    if valid is None:
        log.info("%s: cloning %s (%s) into %s", arm.name, cfg.start_ref, start[:7], arm.repo)
        build_repo(cfg, arm, start)
        res.pop("audit_error", None)
        res.update(audit_repo(cfg, arm.repo, start, foreign))
        res["created"] = True
    res.update(ensure_venv(cfg, arm.repo, start))
    res.update(audit_repo(cfg, arm.repo, start, foreign))       # uv must not have touched git state
    res.update(ensure_session(cfg, arm, may_write=not busy and not held))
    os.makedirs(arm.sock_dir, mode=0o755, exist_ok=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", help="arm(s) to set up (default: all)")
    ap.add_argument("--check", action="store_true", help="verify only, change nothing")
    ap.add_argument("--force", action="store_true", help="archive and rebuild arms that are already in use")
    ap.add_argument("--create-start", action="store_true",
                    help="first create canon.git branch h2h/start if it is missing (from tools/h2h/{program.md,run.sh})")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_h2h_config()
    arms = [cfg.arm(n) for n in args.arm] if args.arm else cfg.arms
    if args.create_start and not args.check and not git(cfg.canon_dir, "rev-parse", "--verify", "-q",
                                                         f"refs/heads/{cfg.start_ref}", check=False):
        create_start(cfg)
    info = verify_start(cfg)
    log.info("canon %s = %s (parent %s); changed vs baseline: %s", cfg.start_ref, info["start"][:7],
             info["parent"][:7], info["changed_vs_baseline"])
    foreign = foreign_objects(cfg, info["start"])
    log.info("%d object(s) in canon.git are reachable only from other branches (must never reach an arm)",
             len(foreign))
    out, ok = {"start": info, "arms": []}, True
    for arm in arms:
        try:
            r = setup_arm(cfg, arm, info, foreign, args.check, args.force)
            log.info("%s: ready %s", arm.name, {k: v for k, v in r.items() if k not in ("repo", "session_dir")})
        except SetupError as e:
            ok = False
            r = {"arm": arm.name, "error": str(e)}
            log.error("%s: %s", arm.name, e)
        out["arms"].append(r)
    heads = {r.get("head") for r in out["arms"] if "head" in r}
    if len(heads) > 1:
        log.warning("arms are at different commits: %s", heads)
    print(json.dumps(out, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
