"""An open-weight chat model on Modal, served by vLLM, for the synthesizer sweep.

Deploy once per model, then call it from `oss_synth`:

    uv run modal deploy src/rewardhack/modal_app.py
    uv run python -m rewardhack.experiment tr87 --level 1 --backend modal --model Qwen/Qwen2.5-Coder-7B-Instruct ...

One container holds one model. Inputs queue on it, so the engine is never
called from two threads at once.
"""


import modal

APP = "rewardhack-vllm"
CACHE = "/root/.cache/huggingface"

app = modal.App(APP)
# The official vLLM image carries prebuilt CUDA wheels; a pip install of vllm
# on a plain image resolves to a source build that needs CUDA_HOME.
image = (
    modal.Image.from_registry("vllm/vllm-openai:latest")
    .entrypoint([])
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")
)
weights = modal.Volume.from_name("rewardhack-hf-cache", create_if_missing=True)


@app.cls(image=image, gpu="H100", volumes={CACHE: weights}, timeout=3600, scaledown_window=600,
         max_containers=1)
class Engine:
    model: str = modal.parameter(default="Qwen/Qwen2.5-Coder-7B-Instruct")
    max_model_len: int = modal.parameter(default=32768)

    @modal.enter()
    def load(self) -> None:
        from vllm import LLM

        self.llm = LLM(self.model, max_model_len=self.max_model_len, gpu_memory_utilization=0.9,
                       trust_remote_code=True)
        weights.commit()

    @modal.method()
    def chat(self, messages: list[dict], max_tokens: int = 6000, temperature: float = 0.2) -> str:
        from vllm import SamplingParams

        out = self.llm.chat([messages], SamplingParams(max_tokens=max_tokens, temperature=temperature),
                            use_tqdm=False)
        return out[0].outputs[0].text
