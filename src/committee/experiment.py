"""Run synthesis conditions on a game level and store every artifact.

Layout: artifacts/<game>/L<level>_f<train_frac>/<condition>/run<k>/
  program.py   the synthesized program
  meta.json    synthesis metadata, seed, verdict (train pass, test pass, test accuracy)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .loader import ROOT, Transition, build_buffer, temporal_split
from .seeds import make_seeds
from .synth import SynthResult
from .synth_api import synthesize_any
from .verify import Verdict, run_program


def add_backend_args(parser) -> None:
    parser.add_argument("--backend", choices=["claude", "api", "devin"], default="api")
    parser.add_argument("--model", default=None, help="claude alias (opus) or served model name (llm)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint for --backend api")
    parser.add_argument("--max-turns", type=int, default=40, help="claude agent turns")
    parser.add_argument("--max-rounds", type=int, default=8, help="api repair rounds")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--reasoning-effort", default=None, help="for reasoning models served by vLLM")


def backend_cfg(args) -> dict:
    return {"backend": args.backend, "model": args.model or ("opus" if args.backend == "claude" else "llm"),
            "base_url": args.base_url, "max_turns": args.max_turns, "max_rounds": args.max_rounds,
            "timeout_s": args.timeout, "reasoning_effort": args.reasoning_effort}

ARTIFACTS = ROOT / "artifacts"


def condition_dir(game: str, level: int, train_frac: float, condition: str,
                  test_level: int | None = None, train_n: int | None = None, test_n: int | None = None) -> Path:
    split = f"L{level}_n{train_n}" if train_n else f"L{level}_f{int(round(train_frac * 100))}"
    split += (f"_T{test_level}" if test_level else "") + (f"_t{test_n}" if test_n else "")
    return ARTIFACTS / game / split / condition


def pack_run(result: SynthResult, verdict: Verdict) -> dict:
    """Everything a run produces, as one JSON-serialisable record. Shared by the local
    and the Modal paths so artifacts have one layout."""
    return {
        "source": result.source,
        "test_preds": verdict.test_preds,
        "meta": {
            "seed": result.seed,
            "synth": result.meta,
            "check_output": result.check_output,
            "train_pass": verdict.train_pass,
            "test_pass": verdict.test_pass,
            "consistent": verdict.consistent,
            "test_accuracy": verdict.test_accuracy,
            "error": verdict.error,
            "source_bytes": len(result.source.encode()),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }


def write_run(out: Path, record: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "program.py").write_text(record["source"])
    (out / "meta.json").write_text(json.dumps(record["meta"], indent=1))
    (out / "test_preds.json").write_text(json.dumps(record["test_preds"]))
    return record["meta"]


def describe(meta: dict, label: str) -> str:
    s = meta["synth"]
    return (f"{label}: consistent={meta['consistent']} "
            f"train={sum(meta['train_pass'])}/{len(meta['train_pass'])} "
            f"test_acc={meta['test_accuracy']} turns={s.get('num_turns')} wall={s.get('wall_s')}s "
            f"err={meta['error']}")


def run_one(train: list[Transition], test: list[Transition], out: Path, seed: str | None,
            cfg: dict) -> dict:
    result = synthesize_any(train, seed, cfg)
    verdict = run_program(result.source, train, test)
    return write_run(out, pack_run(result, verdict))


def run_condition(game: str, level: int, train_frac: float, condition: str, seeds: list[str | None],
                  cfg: dict, start_index: int = 0, parallel: int = 1,
                  test_level: int | None = None, train_n: int | None = None, test_n: int | None = None) -> list[dict]:
    transitions = build_buffer(game)
    train, test = temporal_split(transitions, level, train_frac, test_level, train_n, test_n)
    base = condition_dir(game, level, train_frac, condition, test_level, train_n, test_n)

    def job(k: int, seed: str | None) -> dict:
        meta = run_one(train, test, base / f"run{k}", seed, cfg)
        print(describe(meta, f"{game} L{level} {condition} run{k}"), flush=True)
        return meta

    jobs = list(enumerate(seeds, start=start_index))
    if parallel <= 1:
        return [job(k, seed) for k, seed in jobs]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(lambda ks: job(*ks), jobs))


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Synthesize programs for one game level.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--start", type=int, default=0, help="first run index, to add runs later")
    add_backend_args(parser)
    parser.add_argument("--seeded", action="store_true", help="one data-driven seed hypothesis per run")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--test-level", type=int, default=None, help="test on all of another level")
    parser.add_argument("--train-n", type=int, default=None, help="fixed-count split: first N transitions train")
    parser.add_argument("--test-n", type=int, default=None, help="fixed-count split: at most N test transitions")
    parser.add_argument("--seed-file", default=None, help="one seed text for every run (e.g. a mechanism library)")
    args = parser.parse_args(argv)
    seeds: list[str | None] = [None] * args.runs
    if args.seeded:
        train, _ = temporal_split(build_buffer(args.game), args.level, args.train_frac, args.test_level,
                                  args.train_n, args.test_n)
        seeds = list(make_seeds(train, args.runs))
    if args.seed_file:
        seeds = [Path(args.seed_file).read_text()] * args.runs
    run_condition(args.game, args.level, args.train_frac, args.condition, seeds, backend_cfg(args),
                  start_index=args.start, parallel=args.parallel, test_level=args.test_level,
                  train_n=args.train_n, test_n=args.test_n)


if __name__ == "__main__":
    main()
