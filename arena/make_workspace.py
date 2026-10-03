"""Build one arm's workspace: the task's repository + the task spec + one framework.

    python make_workspace.py <dest> --framework baseline|hypothesis --task <tasks/NAME>
        --repo <git url or path> [--commit <ref>] [--env CUDA_VISIBLE_DEVICES=0] [--cpus 8-11] [--prepare <script>]

Two arms built from the same --task, --repo and --commit get byte-identical task files and the
identical runner (ar.py); they differ only in the framework files (program.md, research.py).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK = os.path.join(os.path.dirname(HERE), "framework")


def sh(*cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dest")
    p.add_argument("--framework", required=True, choices=["baseline", "hypothesis"])
    p.add_argument("--task", required=True, help="directory with task.json and task.md")
    p.add_argument("--repo", required=True, help="the task's repository (git url/path, or a plain dir)")
    p.add_argument("--commit", default="", help="ref to check out (git repos)")
    p.add_argument("--env", action="append", default=[], help="NAME=value for the run command")
    p.add_argument("--cpus", default=None)
    p.add_argument("--prepare", default="", help="script run in the workspace afterwards (env setup)")
    a = p.parse_args()
    if os.path.exists(a.dest):
        sys.exit(f"{a.dest} already exists")
    if os.path.isdir(os.path.join(a.repo, ".git")) or a.repo.endswith(".git") or "://" in a.repo:
        sh("git", "clone", "-q", a.repo, a.dest)
        sh("git", "checkout", "-q", "-B", "task-base", a.commit or "HEAD", cwd=a.dest)
        sh("git", "remote", "remove", "origin", cwd=a.dest)
        # keep only the task's commit: no other branches, tags or unreachable objects from the source
        out = subprocess.run(["git", "for-each-ref", "--format=%(refname)"], cwd=a.dest,
                             capture_output=True, text=True, check=True).stdout.split()  # fmt: skip
        for ref in out:
            if ref != "refs/heads/task-base":
                sh("git", "update-ref", "-d", ref, cwd=a.dest)
        sh("git", "reflog", "expire", "--expire=now", "--all", cwd=a.dest)
        sh("git", "gc", "-q", "--prune=now", cwd=a.dest)
    else:  # a plain directory: make it a repository
        shutil.copytree(a.repo, a.dest)
        sh("git", "init", "-q", cwd=a.dest)
        sh("git", "add", "-A", cwd=a.dest)
        sh("git", "-c", "user.email=harness@local", "-c", "user.name=harness", "commit", "-q",
           "-m", "task repository", cwd=a.dest)  # fmt: skip
    files = {
        "task.json": os.path.join(a.task, "task.json"),
        "task.md": os.path.join(a.task, "task.md"),
        "ar.py": os.path.join(FRAMEWORK, "core", "ar.py"),
        "program.md": os.path.join(FRAMEWORK, a.framework, "program.md"),
    }
    if a.framework == "hypothesis":
        files["research.py"] = os.path.join(FRAMEWORK, "hypothesis", "research.py")
    for name, src in files.items():
        shutil.copyfile(src, os.path.join(a.dest, name))
    ws = {"env": dict(kv.split("=", 1) for kv in a.env)}
    if a.cpus:
        ws["cpus"] = a.cpus
    with open(os.path.join(a.dest, "workspace.json"), "w") as f:
        json.dump(ws, f)
    with open(os.path.join(a.dest, ".git", "info", "exclude"), "a") as f:
        f.write("".join(f"/{n}\n" for n in [*files, "workspace.json"]))
    if a.prepare:
        sh("bash", a.prepare, cwd=a.dest)
    print(f"{a.dest}: framework {a.framework}, task {json.load(open(files['task.json']))['name']}, {ws}")


if __name__ == "__main__":
    main()
