"""Run sweep configurations on Modal A100s in the pinned harness environment.

    .venv-modal/bin/modal run research/modal_a100.py::download_data          # once
    .venv-modal/bin/modal run research/modal_a100.py --sweep research/sweeps/m1-a100.toml

The image replicates the repository Dockerfile (CUDA 12.4.1 cuDNN devel on Ubuntu 22.04,
uv 0.10.8, Python 3.12.10, ``uv sync --frozen`` from uv.lock, so PyTorch 2.4.0+cu124) but
copies only the harness. Each configuration ships its sweep's frozen source snapshot at call
time, runs once through ``benchmark.run`` over all seeds in a fresh container
with 4 CPUs and networking blocked, and its results are written into the sweep's normal run
directory, so ``research/sweep.py status|collate`` work unchanged.

Modal's A100s are not guaranteed to be the official "NVIDIA A100 80GB PCIe": every record keeps
the reported GPU name, power limit and clocks, and timing claims must state the variant.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
GPU = os.environ.get("C100_MODAL_GPU", "A100-80GB")
APP_DIR = "/app"
HARNESS_PYTHON = f"{APP_DIR}/.venv/bin/python"

app = modal.App("c100-speedrun-a100")
data_volume = modal.Volume.from_name("c100-speedrun-data", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("ca-certificates", "build-essential", "git")
    .pip_install("uv==0.10.8")
    .env(
        {
            "UV_PYTHON_INSTALL_DIR": "/opt/python",
            "UV_LINK_MODE": "copy",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
        }
    )
    .add_local_file(REPO / "pyproject.toml", f"{APP_DIR}/pyproject.toml", copy=True)
    .add_local_file(REPO / "uv.lock", f"{APP_DIR}/uv.lock", copy=True)
    .add_local_file(REPO / ".python-version", f"{APP_DIR}/.python-version", copy=True)
    .add_local_file(REPO / "README.md", f"{APP_DIR}/README.md", copy=True)
    .add_local_file(REPO / "LICENSE", f"{APP_DIR}/LICENSE", copy=True)
    .add_local_dir(REPO / "benchmark", f"{APP_DIR}/benchmark", copy=True, ignore=["__pycache__"])
    .run_commands(f"cd {APP_DIR} && uv python install 3.12.10 && uv sync --frozen --no-dev")
)


def _tar_directory(path: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(path, arcname=".", filter=lambda m: None if "__pycache__" in m.name else m)
    return buffer.getvalue()


def _gpu_report() -> str:
    query = "name,power.limit,clocks.max.sm,memory.total,driver_version"
    command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]
    return subprocess.run(command, capture_output=True, text=True).stdout.strip()


@app.function(image=image, volumes={"/data": data_volume}, timeout=1800)
def download_data() -> str:
    command = [HARNESS_PYTHON, "-m", "benchmark.data", "--root", "/data"]
    result = subprocess.run(command, cwd=APP_DIR, capture_output=True, text=True, check=True)
    data_volume.commit()
    return result.stdout.strip()


@app.function(
    image=image,
    gpu=GPU,
    cpu=4.0,
    volumes={"/data": data_volume},
    timeout=3600,
    block_network=True,
    single_use_containers=True,
    max_containers=4,
)
def run_config(source: bytes, name: str, params: dict, seeds: list[int]) -> dict:
    """Run one configuration over all seeds; return the harness results as a tarball."""
    work = Path("/tmp/work")
    submission = work / name
    submission.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as archive:
        archive.extractall(submission, filter="data")
    seed_file = work / "seeds.json"
    seed_file.write_text(json.dumps(seeds))
    results = work / "results"
    command = [
        HARNESS_PYTHON, "-m", "benchmark.run",
        "--submission-path", str(submission),
        "--n", str(len(seeds)),
        "--seed-file", str(seed_file),
        "--no-accuracy-target",
        "--params", json.dumps(params),
        "--data-root", "/data",
        "--results-root", str(results),
    ]  # fmt: skip
    started = time.time()
    completed = subprocess.run(command, cwd=APP_DIR, capture_output=True, text=True)
    return {
        "exit_code": completed.returncode,
        "log": completed.stdout + completed.stderr,
        "results": _tar_directory(results) if results.exists() else None,
        "gpu": _gpu_report(),
        "wall_seconds": round(time.time() - started, 1),
    }


@app.function(
    image=image,
    gpu=GPU,
    cpu=4.0,
    volumes={"/data": data_volume},
    timeout=4 * 3600,
    block_network=True,
    single_use_containers=True,
    max_containers=4,
)
def run_interleaved(source: bytes, name: str, arms: list[dict], blocks: list[list[int]]) -> dict:
    """Run every arm on one host, block by block, reversing the arm order in alternate blocks
    (ABC, CBA, ...), so arm differences are paired within a host and drift averages out."""
    work = Path("/tmp/work")
    submission = work / name
    submission.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as archive:
        archive.extractall(submission, filter="data")
    runs = []
    for block, seeds in enumerate(blocks):
        order = list(range(len(arms)))
        for arm in order if block % 2 == 0 else order[::-1]:
            seed_file = work / "seeds.json"
            seed_file.write_text(json.dumps(seeds))
            results = work / f"results-b{block}-a{arm}"
            command = [
                HARNESS_PYTHON, "-m", "benchmark.run",
                "--submission-path", str(submission),
                "--n", str(len(seeds)),
                "--seed-file", str(seed_file),
                "--no-accuracy-target",
                "--params", json.dumps(arms[arm]),
                "--data-root", "/data",
                "--results-root", str(results),
            ]  # fmt: skip
            completed = subprocess.run(command, cwd=APP_DIR, capture_output=True, text=True)
            runs.append(
                {
                    "block": block,
                    "arm": arm,
                    "exit_code": completed.returncode,
                    "log": (completed.stdout + completed.stderr)[-20000:],
                    "results": _tar_directory(results) if results.exists() else None,
                }
            )
    return {"gpu": _gpu_report(), "runs": runs}


@app.function(
    image=image, gpu=GPU, cpu=4.0, volumes={"/data": data_volume}, timeout=3600, block_network=True
)
def run_profile(
    source: bytes, name: str, script: str, params: dict, extra: list[str] | None = None
) -> str:
    """Run ``research/profile_step.py`` against a submission snapshot; return its report."""
    work = Path("/tmp/work")
    submission = work / name
    submission.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as archive:
        archive.extractall(submission, filter="data")
    (work / "profile_step.py").write_text(script)
    command = [
        HARNESS_PYTHON, str(work / "profile_step.py"),
        "--submission-path", str(submission),
        "--data-root", "/data",
        "--params", json.dumps(params),
        *(extra or []),
    ]  # fmt: skip
    env = os.environ | {"PYTHONPATH": APP_DIR}
    completed = subprocess.run(command, cwd=APP_DIR, capture_output=True, text=True, env=env)
    return _gpu_report() + "\n" + completed.stdout + completed.stderr[-200000:]


def _load_sweep_module():
    sys.path.insert(0, str(REPO / "research"))
    import sweep as sweeplib  # noqa: PLC0415

    return sweeplib


@app.local_entrypoint()
def main(sweep: str, retry_failed: bool = False) -> None:
    sweeplib = _load_sweep_module()
    spec = sweeplib.Sweep.load(Path(sweep).resolve())
    spec.root.mkdir(parents=True, exist_ok=True)
    spec.take_snapshot()
    wanted = {"pending"} | ({"failed"} if retry_failed else set())
    todo = [p for p in spec.configs if sweeplib.status_of(spec.run_dir(p)) in wanted]
    print(f"{spec.name}: {len(spec.configs)} configs, {len(todo)} to run on Modal {GPU}")
    if not todo:
        return
    source = _tar_directory(spec.source)
    source_hash = sweeplib.source_hash(spec.source)
    log_path = spec.root / "sweep.log"
    calls = [(source, spec.submission.name, params, spec.seeds) for params in todo]
    for params, outcome in zip(todo, run_config.starmap(calls), strict=True):
        run_dir = spec.run_dir(params)
        run_dir.mkdir(parents=True, exist_ok=True)
        spec_record = {
            "config_id": run_dir.name,
            "params": params,
            "seeds": spec.seeds,
            "submission": str(spec.submission),
            "source": str(spec.source),
            "source_hash": source_hash,
            "backend": f"modal:{GPU}",
        }
        (run_dir / "spec.json").write_text(json.dumps(spec_record, indent=2) + "\n")
        (run_dir / "harness.log").write_text(outcome["log"])
        if outcome["results"]:
            with tarfile.open(fileobj=io.BytesIO(outcome["results"]), mode="r:gz") as archive:
                archive.extractall(run_dir / "harness", filter="data")
        harness_dir = sweeplib.latest_harness_dir(run_dir)
        summary = sweeplib.read_json(harness_dir / "summary.json") if harness_dir else None
        status = "complete" if summary and summary.get("complete") else "failed"
        record = {
            "config_id": run_dir.name,
            "status": status,
            "exit_code": outcome["exit_code"],
            "slot": f"modal:{GPU}",
            "gpu": outcome["gpu"],
            "wall_seconds": outcome["wall_seconds"],
            "finished": datetime.now(UTC).isoformat(),
            "harness_dir": str(harness_dir) if harness_dir else None,
            "summary": summary,
        }
        (run_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        event = {"time": record["finished"], "event": "finish", "config_id": run_dir.name,
                 "status": status, "gpu": outcome["gpu"],
                 "mean_accuracy": (summary or {}).get("mean_accuracy")}  # fmt: skip
        with log_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event))


@app.local_entrypoint()
def interleave(sweep: str, hosts: int = 2, blocks: int = 2) -> None:
    """Same-host paired timing: each of ``hosts`` containers runs every config of the sweep in
    ``blocks`` alternating-order blocks; the sweep's seeds are split across hosts and blocks.

        .venv-modal/bin/modal run research/modal_a100.py::interleave --sweep <toml>
    """
    sweeplib = _load_sweep_module()
    spec = sweeplib.Sweep.load(Path(sweep).resolve())
    spec.root.mkdir(parents=True, exist_ok=True)
    spec.take_snapshot()
    chunks = hosts * blocks
    if len(spec.seeds) % chunks:
        raise SystemExit(f"{len(spec.seeds)} seeds do not split into {chunks} host-blocks")
    size = len(spec.seeds) // chunks
    seed_blocks = [spec.seeds[i * size : (i + 1) * size] for i in range(chunks)]
    calls = [
        (_tar_directory(spec.source), spec.submission.name, spec.configs,
         seed_blocks[h * blocks : (h + 1) * blocks])
        for h in range(hosts)
    ]  # fmt: skip
    print(f"{spec.name}: {len(spec.configs)} arms x {hosts} hosts x {blocks} blocks on {GPU}")
    rows = []
    for host, outcome in enumerate(run_interleaved.starmap(calls)):
        for run in outcome["runs"]:
            params = spec.configs[run["arm"]]
            out = spec.root / "interleaved" / f"h{host}-b{run['block']}-{spec.run_dir(params).name}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "harness.log").write_text(run["log"])
            summary = None
            if run["results"]:
                with tarfile.open(fileobj=io.BytesIO(run["results"]), mode="r:gz") as archive:
                    archive.extractall(out / "harness", filter="data")
                harness_dir = sweeplib.latest_harness_dir(out)
                summary = sweeplib.read_json(harness_dir / "summary.json") if harness_dir else None
            row = {
                "host": host,
                "gpu": outcome["gpu"],
                "block": run["block"],
                "config_id": spec.run_dir(params).name,
                "params": params,
                "exit_code": run["exit_code"],
                "summary": summary,
            }
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "summary"}))
    (spec.root / "interleaved.json").write_text(json.dumps(rows, indent=2) + "\n")


@app.local_entrypoint()
def profile_step(
    params: str = "{}", submission: str = "research/lab_recipe", bandwidth: bool = False
) -> None:
    """Kernel profile of one trial on the Modal GPU (see research/profile_step.py)."""
    path = (REPO / submission).resolve()
    script = (REPO / "research" / "profile_step.py").read_text()
    extra = ["--bandwidth"] if bandwidth else []
    print(run_profile.remote(_tar_directory(path), path.name, script, json.loads(params), extra))
