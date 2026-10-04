"""The SciGym arms on Modal CPU containers, one system per container, so sweeps never load the local machine.

    uv run modal run -m scigym.modal_app --arm committee_probe --model gptoss --n-systems 30

The model servers are the vLLM apps; the container reads their URLs and key
from the `vllm-auth` secret and from the arguments. Results land in the same
layout as `scigym.loop`.
"""

from __future__ import annotations

import modal

app = modal.App("scigym-arms")
results = modal.Volume.from_name("scigym-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "scipy", "pandas", "pyarrow", "httpx", "openai", "libroadrunner", "python-libsbml")
    .add_local_python_source("scigym", "bioprot", "committee")
)


@app.function(image=image, secrets=[modal.Secret.from_name("vllm-auth")], timeout=5400, max_containers=60, cpu=4.0, memory=2048,
              volumes={"/results": results})
def run_remote(payload: dict) -> dict:
    import json
    import os
    from pathlib import Path

    os.environ.update(payload["env"])  # before the scigym imports: the admission tolerance is read at import

    from scigym.data import load_systems
    from scigym.loop import _job

    system = next(s for s in load_systems() if s.id == payload["system_id"])
    out = _job((system, payload["arm"], payload["model"], payload["k"], payload["budget"], payload["rounds"], payload["max_calls"]))
    d = Path("/results") / f"{payload['arm']}_{payload['model']}{payload.get('tag', '')}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{system.id}.json").write_text(json.dumps(out, default=float))  # survives a local disconnect
    results.commit()
    return out


@app.local_entrypoint()
def main(arm: str = "committee_probe,committee_fixed,single_fixed", model: str = "gptoss", n_systems: int = 30, k: int = 4,
         budget: int = 5, rounds: int = 3, max_calls: int = 60, only: str = "", collect: bool = False, eps: float = 0.15):
    import json
    import os
    import time

    from committee.synth_api import load_env_file
    from scigym.data import load_systems
    from scigym.loop import ART

    load_env_file()
    env = {k_: os.environ[k_] for k_ in ("OPENAI_BASE_URL", "BIOPROT_GPTOSS_URL", "BIOPROT_MISTRAL_URL", "BIOPROT_QWEN_URL") if k_ in os.environ}
    tag = "" if abs(eps - 0.15) < 1e-9 else f"_eps{int(round(eps * 100))}"  # the tolerance arms keep their own directories
    env["SCIGYM_EPS"] = str(eps)
    arms = arm.split(",")
    systems = load_systems(limit=n_systems)
    if only:
        systems = [s for s in load_systems() if s.id in only.split(",")]
    if collect:  # pull what the containers stored, for a run whose local client dropped
        for a in arms:
            d = ART / f"{a}_{model}{tag}"
            d.mkdir(parents=True, exist_ok=True)
            n = 0
            for entry in results.listdir(f"{a}_{model}{tag}"):
                name = entry.path.split("/")[-1]
                if not (d / name).exists():
                    (d / name).write_bytes(b"".join(results.read_file(entry.path)))
                    n += 1
            print(f"{a}_{model}{tag}: collected {n} new files")
        return
    payloads = []
    for a in arms:
        out = ART / f"{a}_{model}{tag}"
        out.mkdir(parents=True, exist_ok=True)
        todo = [s for s in systems if not (out / f"{s.id}.json").exists() or "error" in json.loads((out / f"{s.id}.json").read_text())]
        print(f"{a} {model}: {len(systems) - len(todo)} stored, {len(todo)} to run", flush=True)
        payloads += [{"system_id": s.id, "arm": a, "model": model, "k": k, "budget": budget, "rounds": rounds,
                      "max_calls": max_calls, "env": env, "tag": tag} for s in todo]
    t0 = time.time()
    for n, r in enumerate(run_remote.map(payloads, order_outputs=False), 1):
        (ART / f"{r['arm']}_{model}{tag}" / f"{r['system']}.json").write_text(json.dumps(r, indent=1, default=float))
        if "error" in r:
            print(f"{r['arm']} {r['system']}: ERROR {r['error'][:160]}", flush=True)
        else:
            print(f"{r['arm']} {r['system']}: admitted {r['n_admitted']}/{r['n_members']} rms_f1 medoid "
                  f"{(r['rms_medoid'] or {}).get('f1', float('nan')):.2f} majority {r['rms_majority']['f1']:.2f} "
                  f"ste {r['ste_medoid']:.3f} (partial {r['ste_partial']:.3f}) calls {r['calls']} {r['wall_s']}s "
                  f"[{n}/{len(payloads)} in {time.time() - t0:.0f}s]", flush=True)
