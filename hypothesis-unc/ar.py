"""
ar.py: the task-agnostic experiment runner shared by every research framework.

It knows nothing about any particular problem. Everything task-specific comes from
task.json in the workspace root:

    {
      "name": "...",
      "editable": ["path/", ...],          # the only paths an experiment may change
      "setup": ["shell command", ...],     # run once by `init` (e.g. create the editable folder)
      "run": "shell command",              # one experiment; its output is the run log
      "timeout_s": 1800,
      "metrics": {"name": {"regex": "..."}},   # first capture group of the last match in the log
      "objective": {"metric": "name", "direction": "minimize" | "maximize"},
      "constraints": [{"metric": "name", "op": ">=" | "<=" | "==" | ">" | "<", "value": ...}],
      "infeasible_order": {"metric": "name", "direction": "..."}   # ranks runs that miss a constraint
    }

and the machine placement from workspace.json ({"env": {"NAME": "value"}, "cpus": "8-11"}, optional:
extra environment variables for the run command, and the CPU cores it is pinned to).

Every framework measures experiments only through this file, so two frameworks working on the
same task are judged identically.

    python ar.py init                      # task setup + empty results.tsv
    python ar.py run  [--tag T] [--description D]     # start the committed experiment and wait
    python ar.py start [--tag T] [--description D]    # start it in the background
    python ar.py wait [--max 540]          # wait for the run (exit 3 if still running at --max)
    python ar.py tail [-n 30]              # last lines of the running/last run's log
    python ar.py abort --reason "..."      # stop the run
    python ar.py log --status keep|discard|crash [--description D] [--force]
    python ar.py status
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TASK_FILE = os.path.join(ROOT, "task.json")
WORKSPACE_FILE = os.path.join(ROOT, "workspace.json")
RESULTS_FILE = os.path.join(ROOT, "results.tsv")
RUNS_DIR = os.path.join(ROOT, "runs")
STATE_DIR = os.path.join(ROOT, ".ar")
INFLIGHT_FILE = os.path.join(STATE_DIR, "inflight.json")
BASE_FILE = os.path.join(STATE_DIR, "task_commit")
# files the framework puts in the workspace; they are never part of an experiment
UNTRACKED = ["results.tsv", "runs/", ".ar/", "task.json", "task.md", "workspace.json",
             "program.md", "ar.py", "research.py", "__pycache__/"]  # fmt: skip
STATUSES = ["keep", "discard", "crash"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def die(msg, code=1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_text(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def git(*args):
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def head():
    code, out = git("rev-parse", "--short=7", "HEAD")
    return out if code == 0 else "-"


def task():
    t = read_json(TASK_FILE)
    if t is None:
        die("task.json not found in the workspace")
    return t


def metric_names(t):
    return list(t["metrics"])


def result_cols(t):
    cols = ["exp", "commit", *metric_names(t), "feasible", "improved", "status", "tag"]
    return cols + ["description"]


def clean(value):
    return re.sub(r"[\t\r\n]+", " ", str(value)).strip()


def read_results(t):
    if not os.path.exists(RESULTS_FILE):
        die("results.tsv not found; run `python ar.py init` first")
    lines = [line for line in read_text(RESULTS_FILE).splitlines() if line.strip()]
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        rows.append(dict(zip(header, parts + [""] * (len(header) - len(parts)))))
    return rows


def write_results(t, rows):
    cols = result_cols(t)
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(clean(r.get(c, "")) for c in cols) + "\n")
    os.replace(tmp, RESULTS_FILE)


def next_exp(rows):
    nums = [int(r["exp"][1:]) for r in rows if r["exp"][1:].isdigit()]
    return f"E{max(nums, default=-1) + 1:03d}"


# ---------------------------------------------------------------------------
# metrics and the objective
# ---------------------------------------------------------------------------


def parse_value(s):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(s.lower(), s)


def extract_metrics(t, log_text):
    """Each metric: the first capture group of the LAST match of its regex in the run log."""
    out = {}
    for name, spec in t["metrics"].items():
        matches = re.findall(spec["regex"], log_text, re.MULTILINE)
        if matches:
            m = matches[-1]
            out[name] = parse_value(m[0] if isinstance(m, tuple) else m)
    return out


OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


def feasible(t, m):
    for c in t.get("constraints", []):
        v = m.get(c["metric"])
        if v is None or not OPS[c["op"]](v, c["value"]):
            return False
    return True


def valid(t, m):
    needed = {t["objective"]["metric"]} | {c["metric"] for c in t.get("constraints", [])}
    if t.get("infeasible_order"):
        needed.add(t["infeasible_order"]["metric"])
    return all(name in m for name in needed)


def better(direction, a, b):
    return a < b if direction == "minimize" else a > b


def improves(t, cand, tip):
    """Does a measured run beat the branch tip under the task's objective? (bool, reason)"""
    if not valid(t, cand):
        return False, "metrics missing from the run log"
    if tip is None:
        return True, "no branch tip yet"
    obj, order = t["objective"], t.get("infeasible_order") or t["objective"]
    cf, tf = feasible(t, cand), feasible(t, tip)
    if cf and not tf:
        return True, "first run that meets the constraints"
    if tf and not cf:
        return False, "misses a constraint the tip meets"
    spec = obj if cf else order
    a, b = cand[spec["metric"]], tip.get(spec["metric"])
    if b is None:
        return True, "the tip has no value to compare"
    ok = better(spec["direction"], a, b)
    where = "" if cf else " (no run meets the constraints yet)"
    return ok, f"{spec['metric']} {a} vs tip {b}{where}"


def row_metrics(t, row):
    return {k: parse_value(row[k]) for k in metric_names(t) if row.get(k, "") != ""}


def tip_row(rows):
    for r in reversed(rows):
        if r["status"] == "keep":
            return r
    return None


# ---------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------


def proc_start_ticks(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return fields[0], int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def alive(fl):
    try:
        os.kill(fl["pid"], 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    st = proc_start_ticks(fl["pid"])
    if st is None:
        return True
    return st[0] != "Z" and (fl.get("start_ticks") is None or st[1] == fl["start_ticks"])


def descendants(pid):
    children = {}
    for name in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                children.setdefault(ppid, []).append(int(name))
            except (OSError, IndexError, ValueError):
                pass
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def inflight():
    return read_json(INFLIGHT_FILE)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def exclude_untracked():
    path = os.path.join(ROOT, ".git", "info", "exclude")
    have = read_text(path).splitlines()
    with open(path, "a", encoding="utf-8") as f:
        for p in UNTRACKED:
            if "/" + p not in have:
                f.write("/" + p + "\n")


def cmd_init(args):
    t = task()
    if os.path.exists(RESULTS_FILE) and not args.force:
        die("already initialized (results.tsv exists)")
    exclude_untracked()
    code, dirty = git("status", "--porcelain", "--untracked-files=no")
    if code != 0:
        die("the workspace is not a git repository")
    if dirty:
        die(f"uncommitted changes:\n{dirty}")
    missing = [p for p in t["editable"] if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        for cmd in t.get("setup", []):
            print(f"$ {cmd}")
            if subprocess.run(cmd, shell=True, cwd=ROOT).returncode != 0:
                die(f"task setup command failed: {cmd}")
        git("add", "-A", "--", *t["editable"])
        git("commit", "-q", "-m", f"{t['name']}: task setup", "--", *t["editable"])
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(BASE_FILE, "w") as f:
        f.write(head() + "\n")
    os.makedirs(RUNS_DIR, exist_ok=True)
    write_results(t, [])
    print(f"initialized task '{t['name']}' at commit {head()}")
    print(f"editable: {', '.join(t['editable'])}")


def check_workspace(t):
    code, dirty = git("status", "--porcelain", "--untracked-files=all", "--", *t["editable"])
    if dirty:
        die(f"uncommitted changes in the editable paths (commit the experiment first):\n{dirty}")
    code, dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        die(f"uncommitted changes outside the experiment:\n{dirty}")
    base = read_text(BASE_FILE).strip()
    if base:
        _, changed = git("diff", "--name-only", base, "HEAD")
        outside = [
            p for p in changed.splitlines()
            if not any(p == e.rstrip("/") or p.startswith(e.rstrip("/") + "/") for e in t["editable"])
        ]  # fmt: skip
        if outside:
            die(f"committed changes outside the editable paths: {', '.join(outside)}")


def cmd_start(args, quiet=False):
    t = task()
    rows = read_results(t)
    fl = inflight()
    if fl:
        state = "still running" if alive(fl) else "finished but not logged"
        die(f"{fl['exp']} is {state}; `wait` and `log` it first")
    check_workspace(t)
    ws = read_json(WORKSPACE_FILE, {})
    cmd = ["bash", "-c", t["run"]]
    if ws.get("cpus"):
        cmd = ["taskset", "-c", str(ws["cpus"])] + cmd
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **{k: str(v) for k, v in ws.get("env", {}).items()}}
    exp = next_exp(rows)
    os.makedirs(RUNS_DIR, exist_ok=True)
    log_path = os.path.join(RUNS_DIR, f"{exp}.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, start_new_session=True)  # fmt: skip
    st = proc_start_ticks(proc.pid)
    fl = {
        "exp": exp,
        "commit": head(),
        "pid": proc.pid,
        "start_ticks": st[1] if st else None,
        "started": time.time(),
        "tag": args.tag or "",
        "description": args.description or "",
        "log": log_path,
        "timeout_s": t.get("timeout_s"),
    }
    write_json(INFLIGHT_FILE, fl)
    if not quiet:
        print(f"started {exp} at commit {fl['commit']}; log: runs/{exp}.log")
    return fl


def enforce_timeout(fl):
    limit = fl.get("timeout_s")
    if limit and alive(fl) and time.time() - fl["started"] > limit:
        kill(fl)
        fl["timed_out"] = True
        write_json(INFLIGHT_FILE, fl)


def kill(fl, grace=30):
    workers = descendants(fl["pid"])
    try:
        os.killpg(fl["pid"], signal.SIGINT)
    except ProcessLookupError:
        pass
    end = time.time() + grace
    while alive(fl) and time.time() < end:
        time.sleep(0.2)
    for pid in [fl["pid"]] + workers:
        try:
            os.killpg(pid, signal.SIGKILL) if pid == fl["pid"] else os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def show_metrics(t, fl):
    m = extract_metrics(t, read_text(fl["log"]))
    if m:
        print("metrics: " + ", ".join(f"{k}={v}" for k, v in m.items()))
    else:
        print("no metrics found in the run log; last lines:")
        print("\n".join(read_text(fl["log"]).splitlines()[-15:]))
    return m


def cmd_wait(args):
    t = task()
    fl = inflight()
    if not fl:
        die("no run in flight")
    end = time.time() + args.max
    while alive(fl) and time.time() < end:
        enforce_timeout(fl)
        time.sleep(args.poll)
    if alive(fl):
        print(f"{fl['exp']} still running after {args.max:.0f}s "
              f"({time.time() - fl['started']:.0f}s total); call `wait` again")  # fmt: skip
        sys.exit(3)
    fl = inflight()
    took = time.time() - fl["started"]
    note = " (killed: exceeded the task's timeout)" if fl.get("timed_out") else ""
    print(f"{fl['exp']} finished after {took:.0f}s{note}")
    show_metrics(t, fl)
    print("next: `python ar.py log --status keep|discard|crash`")


def cmd_run(args):
    cmd_start(args)
    cmd_wait(args)


def cmd_tail(args):
    fl = inflight()
    rows = read_results(task())
    path = fl["log"] if fl else (os.path.join(RUNS_DIR, f"{rows[-1]['exp']}.log") if rows else "")
    print("\n".join(read_text(path).replace("\r", "\n").splitlines()[-args.n :]))


def cmd_abort(args):
    fl = inflight()
    if not fl:
        die("no run in flight")
    if not alive(fl):
        die("the run already finished; `log` it")
    kill(fl)
    fl["aborted"] = args.reason
    write_json(INFLIGHT_FILE, fl)
    print(f"aborted {fl['exp']}: {args.reason}; `log` it")


def cmd_log(args):
    t = task()
    rows = read_results(t)
    fl = inflight()
    if not fl:
        die("no run in flight; `start` one first")
    if alive(fl):
        die(f"{fl['exp']} is still running; `wait` (or `abort`) first")
    m = extract_metrics(t, read_text(fl["log"]))
    status = args.status
    if not valid(t, m) and status != "crash":
        warn("the run log lacks the task's metrics: recording a crash")
        status = "crash"
    tip = tip_row(rows)
    ok, why = improves(t, m, row_metrics(t, tip) if tip else None)
    if status == "keep" and not ok and not args.force:
        die(f"does not improve on the tip: {why}; log it as discard (or --force for a simplification)")
    if fl.get("aborted") or fl.get("timed_out"):
        if status == "keep":
            die("an aborted or timed-out run cannot be kept")
    row = {
        "exp": fl["exp"],
        "commit": fl["commit"],
        **{k: m.get(k, "") for k in metric_names(t)},
        "feasible": "yes" if valid(t, m) and feasible(t, m) else "no",
        "improved": "yes" if ok else "no",
        "status": status,
        "tag": fl.get("tag", ""),
        "description": (args.description or fl.get("description") or "")
        + (f" [aborted: {fl['aborted']}]" if fl.get("aborted") else "")
        + (" [timed out]" if fl.get("timed_out") else ""),
    }
    rows.append(row)
    write_results(t, rows)
    write_json(os.path.join(RUNS_DIR, f"{fl['exp']}.json"), {**fl, "metrics": m, "row": row})
    os.remove(INFLIGHT_FILE)
    print(f"{fl['exp']} logged: {status}; {why}")
    if status != "keep" and tip:
        if fl["commit"] == tip["commit"]:
            print(f"git: HEAD is already the tip {tip['commit']}; nothing to reset")
        else:
            print(f"git: return to the tip with `git reset --hard {tip['commit']}`")


def cmd_status(args):
    t = task()
    rows = read_results(t)
    tip = tip_row(rows)
    n = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    print(f"task {t['name']}: {len(rows)} experiments "
          f"(keep {n['keep']}, discard {n['discard']}, crash {n['crash']})")  # fmt: skip
    obj = t["objective"]
    print(f"objective: {obj['direction']} {obj['metric']}"
          + "".join(f"; {c['metric']} {c['op']} {c['value']}" for c in t.get("constraints", [])))  # fmt: skip
    if tip:
        vals = ", ".join(f"{k}={tip[k]}" for k in metric_names(t))
        print(f"branch tip {tip['exp']} @ {tip['commit']}: {vals} (feasible: {tip['feasible']})")
    else:
        print("branch tip: none yet (run the unmodified baseline first)")
    for r in rows[-8:]:
        vals = " ".join(f"{k}={r[k]}" for k in metric_names(t))
        print(f"  {r['exp']} {r['commit']} {vals} {r['status']:<7} {r['tag']} {r['description']}")
    fl = inflight()
    if fl:
        state = "running" if alive(fl) else "finished, not logged"
        print(f"in flight: {fl['exp']} @ {fl['commit']} {state}, "
              f"{time.time() - fl['started']:.0f}s since start")  # fmt: skip


def build_parser():
    p = argparse.ArgumentParser(description="Task-agnostic experiment runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)
    for name, fn in (("start", cmd_start), ("run", cmd_run)):
        s = sub.add_parser(name)
        s.add_argument("--tag", default="")
        s.add_argument("--description", default="")
        if name == "run":
            s.add_argument("--max", type=float, default=540)
            s.add_argument("--poll", type=float, default=2)
        s.set_defaults(fn=fn)
    s = sub.add_parser("wait")
    s.add_argument("--max", type=float, default=540)
    s.add_argument("--poll", type=float, default=2)
    s.set_defaults(fn=cmd_wait)
    s = sub.add_parser("tail")
    s.add_argument("-n", type=int, default=30)
    s.set_defaults(fn=cmd_tail)
    s = sub.add_parser("abort")
    s.add_argument("--reason", required=True)
    s.set_defaults(fn=cmd_abort)
    s = sub.add_parser("log")
    s.add_argument("--status", required=True, choices=STATUSES)
    s.add_argument("--description", default="")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_log)
    s = sub.add_parser("status")
    s.set_defaults(fn=cmd_status)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
