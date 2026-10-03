"""The SciGym arms on Modal CPU containers, one system per container, so sweeps never load the local machine.

    uv run modal run -m scigym.modal_app --arm committee_probe --model gptoss --n-systems 30

The model servers are the vLLM apps; the container reads their URLs and key
from the `vllm-auth` secret and from the arguments. Results land in the same
layout as `scigym.loop`.
"""

from __future__ import annotations

import modal

app = modal.App("scigym-arms")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "scipy", "pandas", "pyarrow", "httpx", "openai", "libroadrunner", "python-libsbml")
    .add_local_python_source("scigym", "bioprot", "committee")
)


@app.function(image=image, secrets=[modal.Secret.from_name("vllm-auth")], timeout=5400, max_containers=40, cpu=2.0, memory=2048)
def run_remote(payload: dict) -> dict:
    import os

    from scigym.data import load_systems
    from scigym.loop import _job

    os.environ.update(payload["env"])
    system = next(s for s in load_systems() if s.id == payload["system_id"])
    return _job((system, payload["arm"], payload["model"], payload["k"], payload["budget"], payload["rounds"], payload["max_calls"]))


@app.local_entrypoint()
def main(arm: str, model: str = "gptoss", n_systems: int = 30, k: int = 4, budget: int = 5, rounds: int = 3,
         max_calls: int = 60, only: str = ""):
    import json
    import os
    import time

    from committee.synth_api import load_env_file
    from scigym.data import load_systems
    from scigym.loop import ART

    load_env_file()
    env = {k_: os.environ[k_] for k_ in ("OPENAI_BASE_URL", "BIOPROT_GPTOSS_URL", "BIOPROT_MISTRAL_URL", "BIOPROT_QWEN_URL") if k_ in os.environ}
    out = ART / f"{arm}_{model}"
    out.mkdir(parents=True, exist_ok=True)
    systems = load_systems(limit=n_systems)
    if only:
        systems = [s for s in load_systems() if s.id in only.split(",")]
    todo = [s for s in systems if not (out / f"{s.id}.json").exists() or "error" in json.loads((out / f"{s.id}.json").read_text())]
    print(f"{arm} {model}: {len(systems) - len(todo)} stored, {len(todo)} to run", flush=True)
    payloads = [{"system_id": s.id, "arm": arm, "model": model, "k": k, "budget": budget, "rounds": rounds,
                 "max_calls": max_calls, "env": env} for s in todo]
    t0 = time.time()
    for n, r in enumerate(run_remote.map(payloads, order_outputs=False), 1):
        (out / f"{r['system']}.json").write_text(json.dumps(r, indent=1, default=float))
        if "error" in r:
            print(f"{r['system']}: ERROR {r['error'][:160]}", flush=True)
        else:
            print(f"{r['system']}: admitted {r['n_admitted']}/{r['n_members']} rms_f1 medoid "
                  f"{(r['rms_medoid'] or {}).get('f1', float('nan')):.2f} majority {r['rms_majority']['f1']:.2f} "
                  f"ste {r['ste_medoid']:.3f} (partial {r['ste_partial']:.3f}) calls {r['calls']} {r['wall_s']}s "
                  f"[{n}/{len(payloads)} in {time.time() - t0:.0f}s]", flush=True)
