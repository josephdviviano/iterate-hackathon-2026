#!/usr/bin/env python3
"""Tests for tools/h2h/setup.py (no GPU, no vLLM; canon.git is only read).

Runs setup.py against a throw-away RLTLDR_ROOT whose canon.git / program.md / pi agent template are symlinks to
the real ones, and checks:
  * a fresh arm: clone audited clean (git log --all == start history, cat-file --batch-all-objects == objects
    reachable from the start commit, nothing of autoresearch/oct3), venv imports torch, agent dir without
    auth/models-store, empty session dir; a second run is a no-op (idempotent);
  * the audit rejects leaky repos: a plain local clone (hardlinked object store incl. oct3), an extra ref,
    a remote;
  * an arm in use is never rebuilt without --force; --force archives it first; --check changes nothing;
  * hostile agent content (the repo is writable inside the sandbox, setup runs on the host): a planted torch.py /
    numpy/ package, a signed commit + .git/config log.showSignature/gpg.program, a FIFO or symlinked results.tsv,
    a .git/x -> /dev/zero symlink, a FIFO or commondir in .git: nothing runs, hangs or is read unboundedly; uv sync
    is refused on agent-edited project files.

Needs canon.git (with h2h/start) and pi/agent under the project root ($RLTLDR_ROOT, default: the repo root) and
uv. The --create-start checks run first on a synthetic canon.git (need only git; --create-start-only stops there).

  python3 tests/test_h2h_setup.py [--create-start-only]
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("RLTLDR_ROOT") or CODE
SETUP = os.path.join(CODE, "tools", "h2h", "setup.py")
GIT_ENV = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def git(cwd, *args):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=cwd, capture_output=True, text=True,
                          env=GIT_ENV).stdout.strip()


def make_root(tmp):
    os.makedirs(os.path.join(tmp, "tools", "h2h"))
    os.makedirs(os.path.join(tmp, "pi"))
    os.symlink(os.path.join(ROOT, "canon.git"), os.path.join(tmp, "canon.git"))
    os.symlink(os.path.join(CODE, "tools", "h2h", "program.md"), os.path.join(tmp, "tools", "h2h", "program.md"))
    os.symlink(os.path.join(CODE, "pi", "agent"), os.path.join(tmp, "pi", "agent"))


def setup(tmp, *args, timeout=1800):
    env = {**os.environ, "RLTLDR_ROOT": tmp, "H2H_CONFIG": os.path.join(tmp, "none.json"),
           "H2H_UV_LINK_MODE": "hardlink"}            # fast in tests; production uses copy
    try:
        r = subprocess.run([sys.executable, SETUP, *args], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", {}, ""

    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        out = {}
    return r.returncode, out, r.stderr


def audit_in_process(tmp, repo, timeout=600):
    """Run setup.audit_repo on an arbitrary repo (in a subprocess with the temp root, address space capped at
    2 GB so that an unbounded read fails fast instead of eating the host's memory)."""
    code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "import resource; resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))\n"
            "import setup as s\n"
            "cfg = s.load_h2h_config(); info = s.verify_start(cfg)\n"
            "try:\n    print(json.dumps(s.audit_repo(cfg, %r, info['start'], s.foreign_objects(cfg, info['start']))))\n"
            "except s.SetupError as e:\n    print('AUDIT_ERROR', e)\n") % (CODE, os.path.dirname(SETUP), repo)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "RLTLDR_ROOT": tmp, "H2H_CONFIG": os.path.join(tmp, "none.json")})
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    return r.stdout.strip() + r.stderr.strip()[-500:]


def hostile_tests(tmp, repo, start):
    """Agent-written content must neither run on the host nor hang / exhaust setup."""
    marker = os.path.join(tmp, "PWNED")
    plant = "import os\nopen(%r, 'a').write('ran\\n')\n" % marker
    with open(os.path.join(repo, "torch.py"), "w") as f:
        f.write(plant)
    os.makedirs(os.path.join(repo, "numpy"))
    with open(os.path.join(repo, "numpy", "__init__.py"), "w") as f:
        f.write(plant)
    rc, out, err = setup(tmp, "--arm", "base", "--check")
    a = (out.get("arms") or [{}])[0]
    check("planted torch.py / numpy/ in the repo: not imported by --check (venv check is python -I from /)",
          rc == 0 and not os.path.exists(marker) and "+cu" in a.get("venv", ""), (rc, a, err[-500:]))
    rc, out, err = setup(tmp, "--arm", "base")
    check("... nor by a plain re-run", rc == 0 and not os.path.exists(marker), (rc, err[-500:]))
    # sanity: the same python without -I from the repo would have run them
    subprocess.run([os.path.join(repo, ".venv", "bin", "python"), "-c", "import torch"], cwd=repo,
                   capture_output=True)
    check("    (sanity: without -I, from the repo, the planted torch.py does run)", os.path.exists(marker))
    os.unlink(marker)
    os.unlink(os.path.join(repo, "torch.py"))
    shutil.rmtree(os.path.join(repo, "numpy"))

    # a signed agent commit + .git/config that makes `git log` run a program
    gpg = os.path.join(tmp, "evilgpg")
    with open(gpg, "w") as f:
        f.write("#!/bin/sh\necho ran >> %s\nexit 1\n" % marker)
    os.chmod(gpg, 0o755)
    obj = (f"tree {git(repo, 'rev-parse', 'HEAD^{tree}')}\nparent {start}\nauthor a <a@b> 1 +0000\n"
           "committer a <a@b> 1 +0000\ngpgsig -----BEGIN PGP SIGNATURE-----\n \n x\n -----END PGP SIGNATURE-----\n"
           "\nsigned\n")
    signed = subprocess.run(["git", "hash-object", "-t", "commit", "-w", "--stdin"], cwd=repo, input=obj, text=True,
                            capture_output=True, env=GIT_ENV).stdout.strip()
    git(repo, "update-ref", "refs/heads/autoresearch/h2h", signed)
    git(repo, "config", "log.showSignature", "true")
    git(repo, "config", "gpg.program", gpg)
    res = audit_in_process(tmp, repo)
    check(".git/config log.showSignature + gpg.program: the audit's git log runs nothing",
          not res.startswith("AUDIT_ERROR") and not os.path.exists(marker), (res, os.path.exists(marker)))
    git(repo, "log", "--all", "--format=%H")
    check("    (sanity: a plain git log in that repo does run gpg.program)", os.path.exists(marker))
    os.unlink(marker)
    git(repo, "update-ref", "refs/heads/autoresearch/h2h", start)
    git(repo, "config", "--unset", "log.showSignature")
    git(repo, "config", "--unset", "gpg.program")

    # FIFO / symlinked results.tsv: in_use must not block on it or follow it
    tsv = os.path.join(repo, "results.tsv")
    os.mkfifo(tsv)
    t0 = time.time()
    rc, out, err = setup(tmp, "--arm", "base", "--check", timeout=120)
    a = (out.get("arms") or [{}])[0]
    check("a FIFO results.tsv does not hang setup (reported as in use)",
          rc == 0 and any("results.tsv is unusual" in w for w in a.get("in_use", [])), (rc, round(time.time() - t0), a))
    os.unlink(tsv)
    os.symlink("/etc/passwd", tsv)
    rc, out, err = setup(tmp, "--arm", "base", "--check", timeout=120)
    a = (out.get("arms") or [{}])[0]
    check("a symlinked results.tsv is not followed (reported as in use)",
          rc == 0 and any("symlink" in w for w in a.get("in_use", [])), a)
    os.unlink(tsv)

    # .git content the audit must refuse without reading it unboundedly or blocking
    for name, make, want in (
            ("zero", lambda p: os.symlink("/dev/zero", p), "cannot be read safely"),
            ("fifo", os.mkfifo, "not a regular file"),
            ("commondir", lambda p: open(p, "w").write(os.path.join(ROOT, "canon.git") + "\n"), ".git/commondir")):
        p = os.path.join(repo, ".git", name)
        make(p)
        t0 = time.time()
        res = audit_in_process(tmp, repo, timeout=120)
        check(f"audit rejects .git/{name} quickly ({want})", res.startswith("AUDIT_ERROR") and want in res
              and time.time() - t0 < 60, (res[:300], round(time.time() - t0)))
        os.unlink(p)
    res = audit_in_process(tmp, repo)
    check("... and passes again once removed", not res.startswith("AUDIT_ERROR"), res)

    # uv sync only on the start commit's project files
    venv = os.path.join(repo, ".venv")
    os.rename(venv, venv + ".off")
    with open(os.path.join(repo, "uv.lock"), "a") as f:
        f.write("\n# agent edit\n")
    rc, out, err = setup(tmp, "--arm", "base")
    check("missing .venv + agent-edited uv.lock: uv sync refused", rc != 0 and
          "refusing to run uv sync" in json.dumps(out) and not os.path.exists(venv), (rc, out))
    git(repo, "checkout", "--", "uv.lock")
    os.rename(venv + ".off", venv)
    rc, out, err = setup(tmp, "--arm", "base", "--check")
    check("arm ready again after the hostile tests", rc == 0, err[-1000:])


def create_start_tests():
    """setup.py --create-start on a synthetic canon.git (git only): the branch is created once on top of the baseline
    with program.md = tools/h2h/program.md and an executable run.sh pointing at this root; never overwritten."""
    tmp = tempfile.mkdtemp(prefix="h2h_create_start_test_")
    try:
        os.makedirs(os.path.join(tmp, "tools", "h2h"))
        for f in ("program.md", "run.sh"):
            os.symlink(os.path.join(CODE, "tools", "h2h", f), os.path.join(tmp, "tools", "h2h", f))
        w, canon = os.path.join(tmp, "w"), os.path.join(tmp, "canon.git")
        env = {**GIT_ENV, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@t"}
        for cmd in (["git", "init", "-q", "-b", "autoresearch/oct3", w], ["git", "init", "-q", "--bare", canon]):
            subprocess.run(cmd, check=True, env=env)
        for name, text in (("program.md", "upstream program\n"), ("train.py", "print(1)\n")):
            with open(os.path.join(w, name), "w") as f:
                f.write(text)
        subprocess.run(["git", "add", "-A"], cwd=w, check=True, env=env)
        subprocess.run(["git", "commit", "-qm", "baseline"], cwd=w, check=True, env=env)
        base = git(w, "rev-parse", "HEAD")
        subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "later"], cwd=w, check=True, env=env)
        subprocess.run(["git", "push", "-q", canon, "autoresearch/oct3"], cwd=w, check=True, env=env)
        cfg_path = os.path.join(tmp, "h2h_config.json")
        with open(cfg_path, "w") as f:
            json.dump({"baseline_commit": base}, f)
        code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
                "import setup as s\n"
                "cfg = s.load_h2h_config(); c = s.create_start(cfg); info = s.verify_start(cfg)\n"
                "try:\n    s.create_start(cfg); again = 'created twice'\n"
                "except s.SetupError as e:\n    again = str(e)\n"
                "print(json.dumps({'commit': c, 'info': info, 'again': again}))\n") % (CODE, os.path.dirname(SETUP))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "RLTLDR_ROOT": tmp, "H2H_CONFIG": cfg_path})
        try:
            out = json.loads(r.stdout)
        except json.JSONDecodeError:
            out = {}
        check("--create-start: h2h/start created on the baseline and passes verify_start",
              out.get("info", {}).get("parent") == base and out["info"]["start"] == out["commit"]
              and sorted(out["info"]["changed_vs_baseline"]) == ["program.md", "run.sh"], (out, r.stderr[-800:]))
        run_sh = git(canon, "show", "h2h/start:run.sh")
        check("--create-start: run.sh executable, @RLTLDR_ROOT@ replaced by the root",
              git(canon, "ls-tree", "h2h/start", "run.sh").startswith("100755") and
              f'"{tmp}/tools/run_client.py"' in run_sh and "@RLTLDR_ROOT@" not in run_sh, run_sh)
        check("--create-start: an existing branch is never overwritten", "never modified" in out.get("again", ""),
              out.get("again"))
        check("--create-start: other branches untouched", git(canon, "rev-parse", "autoresearch/oct3") ==
              git(w, "rev-parse", "HEAD"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    create_start_tests()
    if "--create-start-only" in sys.argv:
        print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
        sys.exit(1 if FAILS else 0)
    tmp = tempfile.mkdtemp(prefix="h2h_setup_test_")
    try:
        make_root(tmp)
        canon = os.path.join(ROOT, "canon.git")
        start = git(canon, "rev-parse", "h2h/start")
        oct3_only = set(git(canon, "rev-list", "--objects", "autoresearch/oct3", "--not", start).split("\n"))
        oct3_only = {l.split()[0] for l in oct3_only if l.strip()}
        check("canon has objects only reachable from autoresearch/oct3", len(oct3_only) > 0, len(oct3_only))

        rc, out, err = setup(tmp, "--arm", "base")
        check("setup (fresh arm) exits 0", rc == 0, err[-2000:])
        repo = os.path.join(tmp, "data", "h2h", "base", "repo")
        a = (out.get("arms") or [{}])[0]
        check("fresh arm created and audited", a.get("created") and a.get("head") == start, a)
        check("branch is autoresearch/h2h", git(repo, "symbolic-ref", "--short", "HEAD") == "autoresearch/h2h")
        check("no remotes", git(repo, "remote") == "")
        check("refs: only the agent branch", git(repo, "for-each-ref", "--format=%(refname)") ==
              "refs/heads/autoresearch/h2h")
        logged = set(git(repo, "log", "--all", "--reflog", "--format=%H").split())
        check("git log --all == start history", logged == set(git(canon, "rev-list", start).split()))
        objs = set(git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)").split())
        reach = {l.split()[0] for l in git(canon, "rev-list", "--objects", start).split("\n") if l.strip()}
        check("cat-file --batch-all-objects == objects of the start commit", objs == reach,
              f"{len(objs)} vs {len(reach)}")
        check("no object of autoresearch/oct3", not (objs & oct3_only))
        check("identity autoresearch <autoresearch@localhost>",
              (git(repo, "config", "user.name"), git(repo, "config", "user.email")) ==
              ("autoresearch", "autoresearch@localhost"))
        check("work tree clean", git(repo, "status", "--porcelain") == "")
        check("venv imports torch", "venv" in a and len(a["venv"].split()) == 2, a.get("venv"))
        agent = os.path.join(tmp, "data", "h2h", "base", "session", "agent")
        check("agent dir copied without auth/models-store",
              os.path.isfile(os.path.join(agent, "models.json")) and
              os.path.isfile(os.path.join(agent, "extensions", "guard.ts")) and
              not os.path.exists(os.path.join(agent, "auth.json")) and
              not os.path.exists(os.path.join(agent, "models-store.json")))
        check("empty session dir", os.listdir(os.path.join(tmp, "data", "h2h", "base", "session", "session")) == [])

        rc, out2, err = setup(tmp, "--arm", "base")
        check("second run is a no-op", rc == 0 and not out2["arms"][0].get("created"), err[-1000:])
        rc, out3, err = setup(tmp, "--arm", "base", "--check")
        check("--check passes", rc == 0, err[-1000:])

        hostile_tests(tmp, repo, start)

        # --- the audit must reject leaky repos -------------------------------------------------------------
        leaky = os.path.join(tmp, "leaky")
        subprocess.run(["git", "clone", "-q", "--single-branch", "--branch", "h2h/start", "--no-tags", canon, leaky],
                       env=GIT_ENV, check=True, capture_output=True)
        git(leaky, "branch", "-m", "h2h/start", "autoresearch/h2h")
        git(leaky, "remote", "remove", "origin")
        git(leaky, "config", "user.name", "autoresearch")
        git(leaky, "config", "user.email", "autoresearch@localhost")
        leaked = set(git(leaky, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)").split())
        print(f"       (a plain local clone holds {len(leaked & oct3_only)} oct3-only objects)")
        res = audit_in_process(tmp, leaky)
        check("audit rejects a plain local clone (hardlinked object store)", res.startswith("AUDIT_ERROR"), res)
        git(leaky, "reflog", "expire", "--expire=now", "--all")
        git(leaky, "gc", "-q", "--prune=now")
        res = audit_in_process(tmp, leaky)
        check("... the same clone after reflog expire + gc --prune=now passes", not res.startswith("AUDIT_ERROR"),
              res)
        git(leaky, "fetch", "-q", canon, "autoresearch/oct3:refs/remotes/x/oct3")
        res = audit_in_process(tmp, leaky)
        check("audit rejects an extra ref (oct3 fetched)", res.startswith("AUDIT_ERROR"), res)
        git(leaky, "update-ref", "-d", "refs/remotes/x/oct3")
        res = audit_in_process(tmp, leaky)
        check("audit rejects unreferenced oct3 objects left in the store", res.startswith("AUDIT_ERROR"), res)
        shutil.rmtree(leaky)

        # agent work on top of the start commit is fine
        with open(os.path.join(repo, "train.py"), "a") as f:
            f.write("\n# agent edit\n")
        git(repo, "commit", "-qam", "agent experiment")
        res = audit_in_process(tmp, repo)
        check("audit accepts agent commits on top of the start commit", not res.startswith("AUDIT_ERROR"), res)

        # --- in-use protection --------------------------------------------------------------------------------
        arm_dir = os.path.join(tmp, "data", "h2h", "base")
        with open(os.path.join(arm_dir, "ledger.jsonl"), "w") as f:
            f.write('{"run_id": "x"}\n')
        git(repo, "remote", "add", "evil", canon)              # make the repo invalid
        rc, out4, err = setup(tmp, "--arm", "base")
        check("in-use arm with an invalid repo: refused without --force", rc != 0 and "in use" in
              json.dumps(out4), out4)
        check("... and left untouched", git(repo, "remote") == "evil")
        import fcntl
        with open(os.path.join(arm_dir, "supervisor.lock"), "w") as lk:      # a "running supervisor"
            fcntl.flock(lk, fcntl.LOCK_EX)
            rc, out_l, err = setup(tmp, "--arm", "base", "--force")
        check("--force refused while the arm's supervisor holds its lock", rc != 0 and "supervisor.lock held" in
              json.dumps(out_l), out_l)
        with open(os.path.join(arm_dir, "runner.lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            rc, out_l, err = setup(tmp, "--arm", "base", "--check")
        check("--check still works while a daemon holds a lock", rc != 0 and "evil" in json.dumps(out_l), out_l)
        rc, out5, err = setup(tmp, "--arm", "base", "--force")
        a5 = out5["arms"][0] if out5 else {}
        check("--force archives and rebuilds", rc == 0 and a5.get("created") and a5.get("archived_to"), err[-1000:])
        arch = a5.get("archived_to") or ""
        check("archive holds the old repo + ledger", os.path.isfile(os.path.join(arch, "ledger.jsonl")) and
              os.path.isdir(os.path.join(arch, "repo", ".git")))
        check("rebuilt repo is clean", git(repo, "remote") == "" and git(repo, "rev-parse", "HEAD") == start)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
