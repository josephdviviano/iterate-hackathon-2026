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

from .env import engine_source, mode_name
from .loader import ROOT, Transition, build_buffer, temporal_split
from .seeds import make_seeds
from .synth import SynthResult
from .synth_api import synthesize_any
from .verify import Verdict, run_program


def add_mode_args(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--frame", action="store_true",
                       help="the program also gets the 64x64 before frame, as OPINE-World's rule does")
    group.add_argument("--frame-out", action="store_true",
                       help="the program returns the next frame, as OPINE-World's rule does; admission is frame equality")


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
            "timeout_s": args.timeout, "reasoning_effort": args.reasoning_effort,
            "mode": mode_name(getattr(args, "frame", False), getattr(args, "frame_out", False))}

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
        "test_objs": verdict.test_objs,
        "meta": {
            "seed": result.seed,
            "mode": verdict.mode,
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
    if record.get("test_objs") is not None:
        (out / "test_objs.json").write_text(json.dumps(record["test_objs"]))
    return record["meta"]


def run_mode(cond: Path) -> str | None:
    """The mode of the runs stored under a condition; None when there are none."""
    modes = {json.loads(p.read_text()).get("mode", "objects") for p in cond.glob("run*/meta.json")}
    if len(modes) > 1:
        raise ValueError(f"{cond} mixes modes {sorted(modes)}")
    return next(iter(modes), None)


def require_objects(cond: Path, what: str) -> None:
    """The studies that build hypothetical object states have no frame to go with them."""
    stored = run_mode(cond)
    if stored not in (None, "objects"):
        raise SystemExit(f"{what} is defined for the objects mode; {cond} holds {stored} runs")


def check_mode(cond: Path, mode: str) -> None:
    stored = run_mode(cond)
    if stored is not None and stored != mode:
        raise SystemExit(f"{cond} holds {stored} runs; a {mode} run needs another --condition")


def describe(meta: dict, label: str) -> str:
    s = meta["synth"]
    return (f"{label}: consistent={meta['consistent']} "
            f"train={sum(meta['train_pass'])}/{len(meta['train_pass'])} "
            f"test_acc={meta['test_accuracy']} turns={s.get('num_turns')} wall={s.get('wall_s')}s "
            f"err={meta['error']}")


def run_one(train: list[Transition], test: list[Transition], out: Path, seed: str | None,
            cfg: dict, engine_src: str | None = None) -> dict:
    result = synthesize_any(train, seed, cfg)
    verdict = run_program(result.source, train, test, mode=cfg.get("mode", "objects"), engine_src=engine_src)
    return write_run(out, pack_run(result, verdict))


def run_condition(game: str, level: int, train_frac: float, condition: str, seeds: list[str | None],
                  cfg: dict, start_index: int = 0, parallel: int = 1,
                  test_level: int | None = None, train_n: int | None = None, test_n: int | None = None) -> list[dict]:
    transitions = build_buffer(game)
    train, test = temporal_split(transitions, level, train_frac, test_level, train_n, test_n)
    base = condition_dir(game, level, train_frac, condition, test_level, train_n, test_n)
    return run_split(train, test, base, seeds, cfg, f"{game} L{level} {condition}", start_index, parallel,
                     engine_source(game, cfg.get("mode", "objects")))


def run_split(train: list[Transition], test: list[Transition], base: Path, seeds: list[str | None],
              cfg: dict, label: str, start_index: int = 0, parallel: int = 1,
              engine_src: str | None = None) -> list[dict]:
    check_mode(base, cfg.get("mode", "objects"))

    def job(k: int, seed: str | None) -> dict | None:
        try:
            meta = run_one(train, test, base / f"run{k}", seed, cfg, engine_src)
        except Exception as e:  # one lost run must not end the other runs of the condition
            print(f"{label} run{k}: FAILED {e!r}", flush=True)
            return None
        print(describe(meta, f"{label} run{k}"), flush=True)
        return meta

    jobs = list(enumerate(seeds, start=start_index))
    if parallel <= 1:
        return [job(k, seed) for k, seed in jobs]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(lambda ks: job(*ks), jobs))


def seeds_for(train: list[Transition], runs: int, start: int = 0) -> list[str]:
    """Seeds of runs start to start+runs-1 of one seeded batch, so added runs continue that batch."""
    return list(make_seeds(train, start + runs))[start:]


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Synthesize programs for one game level.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--start", type=int, default=0,
                        help="first run index, to add runs later; seeds are those of a batch of start+runs")
    add_backend_args(parser)
    add_mode_args(parser)
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
        seeds = seeds_for(train, args.runs, args.start)
    if args.seed_file:
        seeds = [Path(args.seed_file).read_text()] * args.runs
    run_condition(args.game, args.level, args.train_frac, args.condition, seeds, backend_cfg(args),
                  start_index=args.start, parallel=args.parallel, test_level=args.test_level,
                  train_n=args.train_n, test_n=args.test_n)


if __name__ == "__main__":
    main()
