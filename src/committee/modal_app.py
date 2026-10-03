"""Synthesis fan-out on Modal.

One container per committee member: it runs the claude CLI in an isolated
workspace, verifies the program by exact replay, and returns the run record.
The local entrypoint builds the split, maps the seeds over containers and
writes artifacts in the same layout as the local path, so evaluate, explore
and demo do not change.

Credential: a Modal Secret named `claude-auth` holding ANTHROPIC_API_KEY or
CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`).

    uv run modal run -m committee.modal_app --game ar25 --level 3 --train-frac 0.4 --runs 3
    uv run modal run -m committee.modal_app --game ar25 --level 3 --train-frac 0.4 \\
        --runs 8 --seeded --condition committee
"""

from __future__ import annotations

from dataclasses import asdict

import modal

app = modal.App("committee-synth")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "gnupg", "git")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code",
        "claude --version",
    )
    .env({"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
    .uv_pip_install("openai>=1.76", "httpx")
    .add_local_python_source("committee")
)


def _light(t, keep_grids: bool = False) -> dict:
    """A transition without its grids unless the mode uses them."""
    d = asdict(t)
    if not keep_grids:
        d["before_grid"] = []
        d["after_grid"] = []
    return d


@app.function(image=image, timeout=120)
def claude_version() -> str:
    import subprocess

    return subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()


@app.function(image=image, secrets=[modal.Secret.from_name("claude-auth"), modal.Secret.from_name("vllm-auth"),
                                    modal.Secret.from_name("devin-auth")],
              timeout=1800, max_containers=24, cpu=1.0, memory=2048)
def synth_remote(payload: dict) -> dict:
    from committee.experiment import pack_run
    from committee.loader import Transition
    from committee.synth_api import synthesize_any
    from committee.verify import run_program

    def restore(rows):
        return [Transition(**{**r, "click": tuple(r["click"]) if r["click"] else None}) for r in rows]

    train, test = restore(payload["train"]), restore(payload["test"])
    result = synthesize_any(train, payload["seed"], payload["cfg"])
    verdict = run_program(result.source, train, test, mode=payload["cfg"].get("mode", "objects"),
                          engine_src=payload.get("engine_src"))
    record = pack_run(result, verdict)
    record["meta"]["synth"]["host"] = "modal"
    return record


@app.local_entrypoint()
def main(game: str, level: int, train_frac: float = 0.6, test_level: int | None = None,
         condition: str = "baseline", runs: int = 1, seeded: bool = False, backend: str = "api",
         model: str | None = None, base_url: str | None = None, max_turns: int = 40, max_rounds: int = 8,
         timeout: int = 900, start: int = 0, reasoning_effort: str | None = None,
         frame: bool = False, frame_out: bool = False):
    from committee.env import engine_source, mode_name, takes_frame
    from committee.experiment import check_mode, condition_dir, describe, write_run
    from committee.loader import build_buffer, temporal_split
    from committee.seeds import make_seeds

    mode = mode_name(frame, frame_out)
    train, test = temporal_split(build_buffer(game), level, train_frac, test_level)
    seeds = make_seeds(train, runs) if seeded else [None] * runs
    base = condition_dir(game, level, train_frac, condition, test_level)
    check_mode(base, mode)
    light_train = [_light(t, takes_frame(mode)) for t in train]
    light_test = [_light(t, takes_frame(mode)) for t in test]
    cfg = {"backend": backend, "model": model or ("opus" if backend == "claude" else "llm"),
           "base_url": base_url, "max_turns": max_turns, "max_rounds": max_rounds, "timeout_s": timeout,
           "reasoning_effort": reasoning_effort, "mode": mode}
    engine = engine_source(game, mode)
    payloads = [{"train": light_train, "test": light_test, "seed": s, "cfg": cfg, "engine_src": engine} for s in seeds]
    print(f"{game} L{level} f{train_frac}" + (f" test=L{test_level}" if test_level else "")
          + f" {condition}: {len(train)} train, {len(test)} test, {runs} runs on Modal", flush=True)
    k = start
    for record in synth_remote.map(payloads, order_outputs=False):
        meta = write_run(base / f"run{k}", record)
        print(describe(meta, f"{game} L{level} {condition} run{k}"), flush=True)
        k += 1
