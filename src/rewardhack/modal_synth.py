"""Synthesis runs on Modal, one container per run, so sweeps do not load the local machine.

The container holds the claude CLI and this repository's two packages. It
receives the split, the flags and, for frame output, the game's engine
source, and returns the program and its record. The local entrypoint
writes artifacts in the same layout as `rewardhack.experiment`.

    uv run modal run -m rewardhack.modal_synth --game re86 --level 5 --runs 3 --frame-out
    uv run modal run -m rewardhack.modal_synth --game tr87 --level 1 --runs 3 --contradiction --model haiku
    uv run modal run -m rewardhack.modal_synth --game re86 --level 5 --runs 3 --frame-out --backend devin
    uv run modal run -m rewardhack.modal_synth --game re86 --level 5 --runs 3 --frame-out --backend modal \
        --model openai/gpt-oss-120b --max-turns 12       # chat loop in the container, vLLM engine on a GPU

Credentials: the Modal Secret `claude-auth` with CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY for the
claude backend; `devin-auth` with DEVIN_API_KEY for the devin backend. Neither is needed locally.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import modal

app = modal.App("rewardhack-synth")

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
    .uv_pip_install("httpx")
    .add_local_python_source("committee", "rewardhack")
)


@app.function(image=image, secrets=[modal.Secret.from_name("claude-auth"), modal.Secret.from_name("devin-auth")],
              timeout=2400, max_containers=12, cpu=2.0, memory=4096)
def run_remote(payload: dict) -> dict:
    import tempfile
    from pathlib import Path

    from committee.loader import Transition
    from rewardhack.experiment import make_record, make_runner, make_synth

    def restore(rows):
        return [Transition(**{**r, "click": tuple(r["click"]) if r["click"] else None}) for r in rows]

    train, test = restore(payload["train"]), restore(payload["test"])
    f = payload["flags"]
    synth_fn = make_synth(f.get("backend", "claude"), f["model"], f["max_turns"], f["timeout_s"], frame=f["frame"],
                          frame_out=f["frame_out"], max_model_len=f.get("max_model_len", 32768))
    runner = make_runner(f["frame"], f["frame_out"], payload.get("engine_src", ""))
    out = Path(tempfile.mkdtemp())
    source, abstain, meta = make_record(train, test, out, contradiction=f["contradiction"], abstain=f["abstain"],
                                        synth_fn=synth_fn, frame=f["frame"] or f["frame_out"], runner=runner)
    meta["synth"]["host"] = "modal"
    return {"source": source, "abstain": abstain, "meta": meta}


def payload_for(game: str, level: int, train_frac: float, flags: dict) -> tuple[dict, "Path"]:
    """The remote payload for one cell and the local directory its runs are written to."""
    from committee.loader import build_buffer, final_engine_source, load_bundle, temporal_split
    from rewardhack.experiment import condition_dir, condition_name

    train, test = temporal_split(build_buffer(game), level, train_frac)
    payload = {"train": [asdict(t) for t in train], "test": [asdict(t) for t in test], "flags": flags,
               "engine_src": final_engine_source(load_bundle(game)) if flags["frame_out"] else ""}
    cond = condition_name(flags["contradiction"], flags["abstain"], flags["frame"], flags["frame_out"])
    model = flags["model"]
    base = condition_dir(game, level, train_frac, cond, "" if model == "opus" else model.split("/")[-1])
    return payload, base


def store(base, k: int, rec: dict, tag: str = "") -> dict:
    from rewardhack.experiment import write_record

    write_record(base / f"run{k}", rec["source"], rec["abstain"], rec["meta"])
    m = rec["meta"]; s = m["synth"]
    print(f"{tag} run{k}: outcome={m['outcome']} train={sum(m['train_pass'])}/{len(m['train_pass'])} "
          f"test_acc={m['test_accuracy']} turns={s.get('num_turns') or s.get('rounds')} wall={s.get('wall_s')}s "
          f"usd={s.get('total_cost_usd')} err={(s.get('backend_error') or '')[:60]!r}", flush=True)
    return m


@app.function(image=image, secrets=[modal.Secret.from_name("claude-auth")], timeout=600, max_containers=16,
              cpu=1.0, memory=1024)
def complete(payload: dict) -> str:
    """One tool-free completion through `claude -p`, for the bio tasks. System messages become the system prompt."""
    import json
    import os
    import subprocess

    msgs = payload["messages"]
    system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in msgs if m["role"] == "user")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CLAUDE") or k in ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")}
    cmd = ["claude", "-p", user, "--model", payload["model"], "--system-prompt", system, "--output-format", "json",
           "--max-turns", "1", "--no-session-persistence", "--strict-mcp-config", "--tools", ""]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    try:
        j = json.loads(proc.stdout)
        if isinstance(j, list):
            j = next((m for m in j if m.get("type") == "result"), {})
        return j.get("result") or ""
    except json.JSONDecodeError:
        return "BACKEND ERROR: " + (proc.stdout or proc.stderr)[-200:]


@app.local_entrypoint()
def main(game: str, level: int, train_frac: float = 0.6, runs: int = 3, start: int = 0, model: str = "opus",
         max_turns: int = 40, timeout: int = 900, contradiction: bool = False, abstain: bool = False,
         frame: bool = False, frame_out: bool = False, backend: str = "claude", label: str = "",
         max_model_len: int = 32768):
    model = "devin" if backend == "devin" else model
    flags = {"model": model, "max_turns": max_turns, "timeout_s": float(timeout), "contradiction": contradiction,
             "abstain": abstain, "frame": frame, "frame_out": frame_out, "backend": backend, "label": label,
             "max_model_len": max_model_len}
    payload, base = payload_for(game, level, train_frac, flags)
    ks = list(range(start, start + runs))
    for k, rec in zip(ks, list(run_remote.map([payload] * len(ks)))):
        store(base, k, rec, f"{game} L{level} {base.parent.name if base.parent.name.startswith(('intact', 'contradiction')) else base.name}")
