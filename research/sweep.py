"""Run, inspect and collate declarative sweeps of recipe configurations through the harness.

    python research/sweep.py run SWEEP.toml [--retry-failed] [--resnapshot] [--dry-run]
    python research/sweep.py status research/sweeps/p1-frontier.toml
    python research/sweep.py collate research/sweeps/p1-frontier.toml [--frontier epochs=0.753]

A sweep file expands ``[base]`` with every ``[[grid]]`` block (cartesian product over list
values) and every ``[[configs]]`` entry. A configuration's identity hashes its parameters,
seeds and the sweep's source snapshot: the first ``run`` copies the submission folder to
``<results>/source`` and every run, status and collation uses that snapshot, so editing the
live recipe neither disturbs a running sweep nor reuses stale results (``--resnapshot``
replaces the snapshot, which makes every configuration pending again). Each
configuration runs as one harness invocation over all seeds on one device slot. Slot locks
live in a host-wide directory, so concurrent sweeps from several agents share GPUs safely.
An interrupted run leaves no record and is rerun on resume; a failed run is recorded and
skipped unless ``--retry-failed`` is given. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import queue
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOWED_DEVICES_FILE = REPO / "research" / "allowed-devices"
INTERRUPTED = {130, 143, -signal.SIGINT, -signal.SIGTERM}
SPEC_KEYS = {
    "name",
    "submission",
    "python",
    "seeds",
    "devices",
    "slots_per_device",
    "harness_args",
    "results_root",
    "base",
    "grid",
    "configs",
}
METRICS = ("n", "mean_acc", "sd_acc", "se_acc", "mean_time_local", "error")


@dataclass(frozen=True)
class Sweep:
    name: str
    path: Path
    submission: Path
    python: str
    seeds: list[int]
    devices: list[int]
    slots_per_device: int
    harness_args: list[str]
    root: Path
    configs: list[dict]

    @classmethod
    def load(cls, path: Path) -> Sweep:
        spec = tomllib.loads(path.read_text())
        unknown = sorted(set(spec) - SPEC_KEYS)
        if unknown:
            raise ValueError(f"{path}: unknown sweep keys {unknown}")
        seeds = spec.get("seeds")
        if not seeds or len(set(seeds)) != len(seeds) or not all(type(s) is int for s in seeds):
            raise ValueError(f"{path}: seeds must be a non-empty list of distinct integers")
        devices = list(spec.get("devices", [0]))
        allowed = allowed_devices()
        if allowed is not None and not set(devices) <= allowed:
            raise ValueError(f"{path}: devices {devices} outside allowed {sorted(allowed)}")
        configs = expand(spec.get("base", {}), spec.get("grid", []), spec.get("configs", []))
        if not configs:
            raise ValueError(f"{path}: the sweep expands to no configurations")
        name = spec.get("name", path.stem)
        return cls(
            name=name,
            path=path,
            submission=(REPO / spec.get("submission", "submissions/team_segal")).resolve(),
            python=spec.get("python", sys.executable),
            seeds=seeds,
            devices=devices,
            slots_per_device=int(spec.get("slots_per_device", 1)),
            harness_args=list(spec.get("harness_args", [])),
            root=(REPO / spec.get("results_root", "results/sweeps") / name).resolve(),
            configs=configs,
        )

    def slots(self) -> list[str]:
        devices = [str(d) for d in self.devices] or ["cpu"]
        return [f"{d}-{k}" for d in devices for k in range(self.slots_per_device)]

    @property
    def snapshot(self) -> Path:
        return self.root / "source"

    @property
    def source(self) -> Path:
        """The frozen snapshot once it exists, otherwise the live submission folder."""
        return self.snapshot if self.snapshot.is_dir() else self.submission

    def take_snapshot(self, replace: bool = False) -> None:
        if self.snapshot.is_dir() and not replace:
            return
        shutil.rmtree(self.snapshot, ignore_errors=True)
        ignore = shutil.ignore_patterns("__pycache__")
        shutil.copytree(self.submission, self.snapshot, ignore=ignore)

    def config_id(self, params: dict) -> str:
        key = json.dumps([params, self.seeds, source_hash(self.source)], sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def run_dir(self, params: dict) -> Path:
        return self.root / "runs" / self.config_id(params)


def allowed_devices() -> set[int] | None:
    """Device indices from research/allowed-devices, or None when the file is absent."""
    if not ALLOWED_DEVICES_FILE.exists():
        return None
    lines = ALLOWED_DEVICES_FILE.read_text().splitlines()
    return {int(line) for line in lines if line.strip() and not line.startswith("#")}


def contention() -> dict:
    """GPU utilisation and host load at launch, to flag runs perturbed by other workloads."""
    query = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        rows = subprocess.run(query, capture_output=True, text=True, timeout=5).stdout
        gpus = {}
        for row in rows.strip().splitlines():
            index, *values = row.split(",")
            gpus[index.strip()] = [int(v) for v in values]
    except (OSError, subprocess.SubprocessError, ValueError):
        gpus = {}
    return {"gpu_util_mem": gpus, "load1": round(os.getloadavg()[0], 2)}


def expand(base: dict, grids: list[dict], explicit: list[dict]) -> list[dict]:
    """Merge base with each grid's cartesian product and each explicit config; dedupe."""
    seen: dict[str, dict] = {}
    for grid in grids:
        keys = sorted(grid)
        axes = [grid[k] if _is_axis(k, grid[k]) else [grid[k]] for k in keys]
        for combo in itertools.product(*axes):
            params = base | dict(zip(keys, combo, strict=True))
            seen.setdefault(json.dumps(params, sort_keys=True), params)
    for config in explicit:
        params = base | config
        seen.setdefault(json.dumps(params, sort_keys=True), params)
    return list(seen.values())


def _is_axis(key: str, value: object) -> bool:
    """Lists are grid axes, except ``widths = [64, 256, 256]``, which is one value."""
    if not isinstance(value, list):
        return False
    return not (key == "widths" and all(isinstance(v, int) for v in value))


def source_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def latest_harness_dir(run_dir: Path) -> Path | None:
    summaries = sorted((run_dir / "harness").glob("*/*/summary.json"))
    return summaries[-1].parent if summaries else None


def status_of(run_dir: Path) -> str:
    record = read_json(run_dir / "record.json")
    return record["status"] if record else "pending"


@contextmanager
def try_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Runner:
    def __init__(self, sweep: Sweep, retry_failed: bool) -> None:
        self.sweep = sweep
        self.retry_failed = retry_failed
        self.slot_dir = Path(os.environ.get("C100_SLOT_DIR", "/tmp/c100-speedrun-slots"))
        self.stopping = threading.Event()
        self.children: set[subprocess.Popen] = set()
        self.lock = threading.Lock()
        self.log_path = sweep.root / "sweep.log"

    def log(self, event: str, **fields: object) -> None:
        line = json.dumps({"time": datetime.now(UTC).isoformat(), "event": event, **fields})
        with self.lock, self.log_path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def pending(self) -> list[dict]:
        wanted = {"pending"} | ({"failed"} if self.retry_failed else set())
        return [p for p in self.sweep.configs if status_of(self.sweep.run_dir(p)) in wanted]

    def run(self) -> int:
        self.sweep.root.mkdir(parents=True, exist_ok=True)
        self.sweep.take_snapshot()
        if source_hash(self.sweep.source) != source_hash(self.sweep.submission):
            self.log("snapshot-differs-from-live", snapshot=str(self.sweep.source))
        work: queue.Queue[dict] = queue.Queue()
        todo = self.pending()
        for params in todo:
            work.put(params)
        self.log(
            "start",
            sweep=str(self.sweep.path),
            configs=len(self.sweep.configs),
            pending=len(todo),
            slots=self.sweep.slots(),
        )
        threads = [threading.Thread(target=self.worker, args=(work,)) for _ in self.sweep.slots()]
        previous = {s: signal.signal(s, self.interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            for thread in threads:
                thread.start()
            while any(t.is_alive() for t in threads):
                time.sleep(0.5)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        remaining = len(self.pending())
        self.log("stop", interrupted=self.stopping.is_set(), remaining=remaining)
        return 130 if self.stopping.is_set() else 0

    def interrupt(self, signum, frame) -> None:
        self.stopping.set()
        with self.lock:
            for child in self.children:
                child.send_signal(signal.SIGINT)

    def worker(self, work: queue.Queue[dict]) -> None:
        while not self.stopping.is_set():
            try:
                params = work.get_nowait()
            except queue.Empty:
                return
            run_dir = self.sweep.run_dir(params)
            with try_lock(run_dir / ".lock") as owned:
                if not owned or status_of(run_dir) == "complete":
                    continue
                with self.acquire_slot() as slot:
                    if slot is not None:
                        self.execute(params, run_dir, slot)

    @contextmanager
    def acquire_slot(self):
        while not self.stopping.is_set():
            for slot in self.sweep.slots():
                with try_lock(self.slot_dir / f"{slot}.lock") as owned:
                    if owned:
                        yield slot
                        return
            time.sleep(2)
        yield None

    def execute(self, params: dict, run_dir: Path, slot: str) -> None:
        sweep = self.sweep
        cid = run_dir.name
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "config_id": cid,
            "params": params,
            "seeds": sweep.seeds,
            "submission": str(sweep.submission),
            "source": str(sweep.source),
            "source_hash": source_hash(sweep.source),
        }
        (run_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        (run_dir / "seeds.json").write_text(json.dumps(sweep.seeds))
        (run_dir / "record.json").unlink(missing_ok=True)
        device = slot.rsplit("-", 1)[0]
        command = [
            sweep.python,
            "-m",
            "benchmark.run",
            "--submission-path",
            str(sweep.source),
            "--n",
            str(len(sweep.seeds)),
            "--seed-file",
            str(run_dir / "seeds.json"),
            "--no-accuracy-target",
            "--params",
            json.dumps(params),
            "--results-root",
            str(run_dir / "harness"),
            *sweep.harness_args,
        ]
        env = os.environ | {"CUDA_VISIBLE_DEVICES": "" if device == "cpu" else device}
        started = datetime.now(UTC).isoformat()
        self.log("launch", config_id=cid, slot=slot, params=params, contention=contention())
        with (run_dir / "harness.log").open("w") as output:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=output, stderr=output)
            with self.lock:
                self.children.add(child)
            code = child.wait()
            with self.lock:
                self.children.discard(child)
        if code in INTERRUPTED or self.stopping.is_set():
            self.log("interrupted", config_id=cid, exit_code=code)
            return
        harness_dir = latest_harness_dir(run_dir)
        summary = read_json(harness_dir / "summary.json") if harness_dir else None
        status = "complete" if summary and summary.get("complete") else "failed"
        record = {
            "config_id": cid,
            "status": status,
            "exit_code": code,
            "slot": slot,
            "started": started,
            "finished": datetime.now(UTC).isoformat(),
            "harness_dir": str(harness_dir) if harness_dir else None,
            "summary": summary,
        }
        (run_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        accuracy = (summary or {}).get("mean_accuracy")
        self.log("finish", config_id=cid, status=status, exit_code=code, mean_accuracy=accuracy)


def failure_reason(run_dir: Path) -> str:
    harness_dir = latest_harness_dir(run_dir)
    if harness_dir is not None:
        for line in (harness_dir / "trials.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("status") != "ok":
                return str(row.get("failure_reason", row["status"]))
        summary = read_json(harness_dir / "summary.json") or {}
        if summary.get("run_error"):
            return str(summary["run_error"])
    log = run_dir / "harness.log"
    lines = log.read_text().strip().splitlines() if log.exists() else []
    return lines[-1] if lines else "no harness output"


def collate(sweep: Sweep) -> list[dict]:
    """One row per configuration with mean, sd and standard error over successful trials."""
    keys = {k for params in sweep.configs for k in params}
    varying = sorted(
        k for k in keys if len({json.dumps(p.get(k)) for p in sweep.configs}) > 1
    )
    rows = []
    for params in sweep.configs:
        run_dir = sweep.run_dir(params)
        status = status_of(run_dir)
        row = {"config_id": run_dir.name, "status": status}
        for k in varying:
            value = params.get(k)
            row[k] = json.dumps(value) if isinstance(value, list) else value
        harness_dir = latest_harness_dir(run_dir) if status != "pending" else None
        trials = []
        if harness_dir is not None:
            lines = (harness_dir / "trials.jsonl").read_text().splitlines()
            trials = [json.loads(line) for line in lines]
        accs = [t["accuracy"] for t in trials if t.get("status") == "ok"]
        times = [t["total_timed_time"] for t in trials if t.get("status") == "ok"]
        sd = statistics.stdev(accs) if len(accs) > 1 else None
        row |= {
            "n": len(accs),
            "mean_acc": round(statistics.mean(accs), 5) if accs else None,
            "sd_acc": round(sd, 5) if sd is not None else None,
            "se_acc": round(sd / math.sqrt(len(accs)), 5) if sd is not None else None,
            "mean_time_local": round(statistics.mean(times), 3) if times else None,
            "error": failure_reason(run_dir) if status == "failed" else None,
        }
        rows.append(row)
    return rows


def frontier(rows: list[dict], x: str, target: float) -> list[dict]:
    """Per group of all other varying params, linearly interpolate where mean_acc hits target."""
    groups: dict[str, list[dict]] = {}
    fixed = [k for k in rows[0] if k not in {"config_id", "status", x, *METRICS}]
    for row in rows:
        if row["status"] == "complete" and row["mean_acc"] is not None and row.get(x) is not None:
            groups.setdefault(json.dumps({k: row[k] for k in fixed}), []).append(row)
    out = []
    for key, members in groups.items():
        members.sort(key=lambda r: r[x])
        crossing = time_at = None
        for lo, hi in itertools.pairwise(members):
            if lo["mean_acc"] < target <= hi["mean_acc"]:
                frac = (target - lo["mean_acc"]) / (hi["mean_acc"] - lo["mean_acc"])
                crossing = round(lo[x] + frac * (hi[x] - lo[x]), 3)
                if lo["mean_time_local"] is not None and hi["mean_time_local"] is not None:
                    span = hi["mean_time_local"] - lo["mean_time_local"]
                    time_at = round(lo["mean_time_local"] + frac * span, 3)
                break
        if crossing is None and members and members[0]["mean_acc"] >= target:
            crossing = f"<= {members[0][x]}"
            time_at = f"<= {members[0]['mean_time_local']}"
        best = max(m["mean_acc"] for m in members)
        out.append(
            json.loads(key)
            | {f"{x}_at_target": crossing, "time_local_at_target": time_at, "best_mean_acc": best}
        )
    return out


def write_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        lines.append("| " + " | ".join("" if row[k] is None else str(row[k]) for k in keys) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run, inspect and collate recipe sweeps.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "collate"):
        command = sub.add_parser(name)
        command.add_argument("sweep", type=Path)
        if name == "run":
            command.add_argument("--retry-failed", action="store_true")
            command.add_argument("--resnapshot", action="store_true")
            command.add_argument("--dry-run", action="store_true")
        if name == "collate":
            command.add_argument("--frontier", help="X=TARGET, e.g. epochs=0.753")
    args = parser.parse_args()
    sweep = Sweep.load(args.sweep)
    if args.command == "run":
        if args.resnapshot and not args.dry_run:
            sweep.take_snapshot(replace=True)
        runner = Runner(sweep, retry_failed=args.retry_failed)
        if args.dry_run:
            for params in runner.pending():
                print(sweep.config_id(params), json.dumps(params, sort_keys=True))
            return
        raise SystemExit(runner.run())
    if args.command == "status":
        counts: dict[str, int] = {}
        for params in sweep.configs:
            status = status_of(sweep.run_dir(params))
            counts[status] = counts.get(status, 0) + 1
        print(json.dumps({"sweep": sweep.name, "configs": len(sweep.configs), **counts}))
        return
    rows = collate(sweep)
    write_table(rows, sweep.root / "table.csv")
    print(markdown(rows))
    if args.frontier:
        x, target = args.frontier.split("=")
        result = frontier(rows, x, float(target))
        if result:
            write_table(result, sweep.root / f"frontier-{x}.csv")
            print("\n" + markdown(result))
    print(f"\nWrote {sweep.root / 'table.csv'}")


if __name__ == "__main__":
    main()
