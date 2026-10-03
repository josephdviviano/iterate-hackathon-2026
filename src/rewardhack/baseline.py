"""The E5 baseline grid: every model family, the 2x2, three level kinds, launched on Modal in parallel.

Every run is one call to the deployed `rewardhack-synth` function
(`uv run modal deploy -m rewardhack.modal_synth` once), submitted from a
thread pool, so no app is created per cell and the local process only
waits. Cells whose records already exist are skipped, so a rerun resumes.

    uv run python -m rewardhack.baseline --backends devin modal:openai/gpt-oss-120b modal:openai/gpt-oss-20b
    uv run python -m rewardhack.baseline --backends claude:opus claude:sonnet claude:haiku   # needs claude-auth
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from .experiment import ARTIFACTS

LEVELS = [("tr87", 1, 0.6), ("ls20", 3, 0.6), ("re86", 5, 0.6)]   # decided rows; hidden geometry; moving content
CONDITIONS = [(False, False), (False, True), (True, False), (True, True)]   # (contradiction, abstain)
RUNS = 3
ROUNDS = {"modal": 12, "claude": 40, "devin": 40}
CONTEXT = {"openai/gpt-oss-120b": 20000}   # one H100 holds about 19k of KV cache for the 120b; others take 32k


def commands(backend: str, runs: int = RUNS) -> list[list[str]]:
    kind, _, model = backend.partition(":")
    out = []
    for game, level, frac in LEVELS:
        for contradiction, abstain in CONDITIONS:
            cmd = [sys.executable, "-m", "modal", "run", "-m", "rewardhack.modal_synth", "--game", game, "--level",
                   str(level), "--train-frac", str(frac), "--runs", str(runs), "--frame-out", "--backend", kind,
                   "--max-turns", str(ROUNDS[kind]), "--timeout", "1800"]
            if model:
                cmd += ["--model", model, "--max-model-len", str(CONTEXT.get(model, 32768))]
            if contradiction:
                cmd.append("--contradiction")
            if abstain:
                cmd.append("--abstain")
            out.append(cmd)
    return out


def cells(backend: str) -> list[tuple[str, dict]]:
    kind, _, model = backend.partition(":")
    model = "devin" if kind == "devin" else model
    out = []
    for game, level, frac in LEVELS:
        for contradiction, abstain in CONDITIONS:
            flags = {"model": model, "max_turns": ROUNDS[kind], "timeout_s": 1800.0, "contradiction": contradiction,
                     "abstain": abstain, "frame": False, "frame_out": True, "backend": kind, "label": "baseline",
                     "max_model_len": CONTEXT.get(model, 32768)}
            out.append((f"{game} L{level} {'contradiction' if contradiction else 'intact'}{'_abstain' if abstain else ''} {model.split('/')[-1]}",
                        {"game": game, "level": level, "train_frac": frac, "flags": flags}))
    return out


def launch(backends: list[str], runs: int, parallel: int) -> list[dict]:
    """Submit every run of every cell to the deployed `rewardhack-synth` function and store results as they land."""
    import modal

    from .modal_synth import payload_for, store

    fn = modal.Function.from_name("rewardhack-synth", "run_remote")
    jobs = []
    for b in backends:
        for tag, c in cells(b):
            payload, base = payload_for(c["game"], c["level"], c["train_frac"], c["flags"])
            for k in range(runs):
                if (base / f"run{k}" / "meta.json").exists():
                    continue
                jobs.append((tag, base, k, payload))
    print(f"{len(jobs)} runs to submit", flush=True)
    results = []

    def one(job):
        tag, base, k, payload = job
        try:
            rec = fn.remote(payload)
        except Exception as exc:
            print(f"{tag} run{k}: FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            return {"tag": tag, "run": k, "error": str(exc)[:300]}
        return {"tag": tag, "run": k, "outcome": store(base, k, rec, tag)["outcome"]}

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        results = list(pool.map(one, jobs))
    (ARTIFACTS / "baseline_manifest.json").write_text(json.dumps(results, indent=1))
    return results


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--backends", nargs="+", required=True, help="devin, claude:<model>, modal:<hf model>")
    p.add_argument("--parallel", type=int, default=12)
    p.add_argument("--runs", type=int, default=RUNS)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.dry_run:
        for b in a.backends:
            for tag, _ in cells(b):
                print(tag)
        return
    results = launch(a.backends, a.runs, a.parallel)
    bad = [r for r in results if "error" in r]
    print(f"{len(results)} runs, {len(bad)} failed; manifest at {ARTIFACTS / 'baseline_manifest.json'}")


if __name__ == "__main__":
    main()
