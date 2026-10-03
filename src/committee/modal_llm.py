"""An OpenAI-compatible model server on Modal for the API synthesis loop.

Serves an open-weights coder model with vLLM on one GPU. Weights and vLLM
compile artifacts are cached in Volumes so later cold starts are short. The
endpoint requires the key in the `vllm-auth` secret.

    uv run modal deploy -m committee.modal_llm
    uv run modal run -m committee.modal_llm            # health check and one completion
"""

from __future__ import annotations

import modal

import os

PRESETS = {
    "qwen": {"model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "gpu": "H100", "max_len": 131072, "extra": []},
    "gptoss": {"model": "openai/gpt-oss-120b", "gpu": "H100", "max_len": 65536,
               "extra": ["--reasoning-parser", "openai_gptoss", "--async-scheduling"]},
}
PRESET = os.environ.get("COMMITTEE_LLM", "qwen")
MODEL_NAME = PRESETS[PRESET]["model"]
SERVED_NAME = "llm"
GPU = PRESETS[PRESET]["gpu"]
PORT = 8000
MINUTES = 60
MAX_MODEL_LEN = PRESETS[PRESET]["max_len"]
APP_NAME = "committee-llm" if PRESET == "qwen" else f"committee-llm-{PRESET}"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "30"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.server(
    image=image,
    gpu=GPU,
    secrets=[modal.Secret.from_name("vllm-auth")],
    scaledown_window=15 * MINUTES,
    startup_timeout=20 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    port=PORT,
    target_concurrency=32,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        import os
        import subprocess

        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--served-model-name", SERVED_NAME, MODEL_NAME,
            "--host", "0.0.0.0", "--port", str(PORT),
            "--api-key", os.environ["VLLM_API_KEY"],
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", "0.92",
            "--tensor-parallel-size", "1",
            "--no-enforce-eager",
            "--uvicorn-log-level=info",
        ] + PRESETS[PRESET]["extra"]
        print(*cmd[:-1])
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def main(prompt: str = "Reply with exactly: OK"):
    import os
    import time

    from openai import OpenAI

    from committee.synth_api import load_env_file

    load_env_file()
    base_url = os.environ["OPENAI_BASE_URL"]  # <server url>/v1, printed by modal deploy, stored in .env
    print("server:", base_url)
    client = OpenAI(base_url=base_url, api_key=os.environ["VLLM_API_KEY"], timeout=1200)
    t0 = time.time()
    for attempt in range(40):
        try:
            resp = client.chat.completions.create(model=SERVED_NAME, max_tokens=20,
                                                  messages=[{"role": "user", "content": prompt}])
            print(f"after {time.time() - t0:.0f}s:", resp.choices[0].message.content)
            break
        except Exception as e:
            print(f"waiting for server ({attempt}): {type(e).__name__}", flush=True)
            time.sleep(30)
