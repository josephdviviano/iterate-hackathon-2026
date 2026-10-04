"""Central configuration for the RLTL;DR autoresearch loop.

Values can be overridden with a JSON file pointed to by $RLTLDR_CONFIG (default $RLTLDR_ROOT/config.json, see
config.example.json); keys there replace the defaults below one-to-one. String values that start with "~" are
expanded to the user's home directory. Machine-specific settings (GPU UUIDs, model paths) belong in that file.

Shell scripts read resolved values with
    python -m rltldr.config get <key>        (prints one value; `dump` prints the whole config as JSON)
"""
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, fields

# project root: $RLTLDR_ROOT, else the directory that contains the rltldr/ package
ROOT = os.environ.get("RLTLDR_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # --- topology -----------------------------------------------------------------------------------
    vllm_url: str = "http://127.0.0.1:8000"
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 8100
    base_model_name: str = "qwen3.8-27b-fp8"          # vLLM --served-model-name (version 0 = no adapter)
    # GPUs by UUID (`nvidia-smi -L`); no defaults: set them in config.json (code that needs one fails clearly)
    agent_gpu_uuid: str = ""             # experiment (nanochat training) GPU; physical GPU 2 on our machine
    agent_gpu_minor: int = 2             # /dev/nvidia<minor> of agent_gpu_uuid
    no_autotune: bool = True             # disable Inductor runtime autotuning in training runs (less noise)
    trainer_gpu_uuid: str = ""           # policy-update (LoRA trainer) GPU; physical GPU 3 on our machine

    run_name: str = "oct3"               # prefix of adapter names in vLLM (<run_name>-v<N>); unique per run
    # --- paths ----------------------------------------------------------------------------------------
    root: str = ROOT
    repo: str = f"{ROOT}/autoresearch"         # the agent's working tree (fresh clone per attempt)
    canon_dir: str = f"{ROOT}/canon.git"        # driver-owned canonical repository (bare)
    canon_ws: str = f"{ROOT}/canon_ws"          # driver's trusted work clone of canon_dir
    branch: str = "autoresearch/oct3"
    data: str = f"{ROOT}/data"
    ledger: str = f"{ROOT}/data/ledger.jsonl"
    pi_agent_dir: str = f"{ROOT}/pi/agent"
    pi_bin: str = "pi"                          # pi coding agent executable (PATH lookup inside the sandbox)
    run_dir: str = f"{ROOT}/run"                # unix sockets: gateway.sock, runner.sock
    runner_port: int = 8200                     # runner's trusted TCP control port
    runner_ws: str = f"{ROOT}/runner_ws"        # runner's pristine workspace (own clone + venv)
    cache_dir: str = f"{ROOT}/data/cache"       # compile cache for (non-confirmation) training runs
    pi_sessions_dir: str = f"{ROOT}/pi_sessions"
    agent_venv: str = f"{ROOT}/agent_venv"      # symlinked into each agent clone as .venv (if it exists)
    # Several independent instances on one machine (e.g. frozen-policy reruns): run pi in
    # tools/h2h_sandbox.sh <sandbox_arm> (hides everything under $HOME except the arm's repo, session dir
    # and sockets) instead of tools/sandbox.sh, and train with ar_run --isolate (train.py cannot read $HOME).
    sandbox_arm: str = ""
    runner_isolate: bool = False
    model_dir: str = "~/models/Qwen3.8-27B-FP8"       # served checkpoint (serve.sh; source of tools/dequant_fp8.py)
    tokenizer_dir: str = "~/models/Qwen3.8-27B-FP8"
    # sha256 of the pristine autoresearch prepare.py (fixed eval/data); the runner refuses any other version
    prepare_sha256: str = "4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150"
    trainer_base_dir: str = "~/models/Qwen3.8-27B-FP8-dequant-bf16"   # bf16 dequant of model_dir (trainer base)

    # --- RLTL;DR rollout collection (paper Sec. 3.2, App. A.2) ---------------------------------------
    group_size: int = 8                  # K sequential attempts = one GRPO group = one update phase
    insight_success_threshold: float = 0.5   # insert insights while running success rate <= this
    max_insights_in_context: int = 16    # n_fb; most recent kept
    insights_enabled: bool = True        # False: no insight generation, nothing injected (ablation)
    keep_margin: float = 0.0005          # keep iff confirmed val_bpb < best - keep_margin (~5 sigma)
    reward_mode: str = "binary"          # "binary" (paper) | "delta" (clip(delta/sigma, -3, 3))
    reward_sigma: float = 0.0002         # val_bpb noise std: 0.0001 measured with no_autotune (0.0039 with autotuning)
    screen_margin: float = 0.0           # agent's run must beat best by this to trigger confirmation re-runs
    confirm_runs: int = 1                # independent fresh-compile re-runs of an apparent win
    # keep iff mean(confirmation re-runs) < best - keep_margin; the new best = mean of those re-runs only
    # (unselected evidence -> no winner's-curse ratchet)
    attempt_timeout_s: int = 3600        # hard wall-clock cap for one pi attempt
    max_runs_per_attempt: int = 3        # ./run.sh invocations allowed per attempt (crash-fix reruns)
    agent_thinking: str = "medium"       # pi thinking level for the research agent
    history_rows_in_prompt: int = 40     # most recent results.tsv rows shown in the task prompt

    # --- insight generation (paper App. A.1 / B) ------------------------------------------------------
    insight_thinking_budget: int = 4096  # think tokens for the judge call
    insight_max_tokens: int = 8192       # think + answer
    insight_max_chars: int = 400         # hard cap on the stored one-sentence hint
    insight_tool_output_chars: int = 2000  # truncation of each tool output inside the judge prompt

    # --- trainer (paper Table 4 / App. A.3, adapted to LoRA on one GPU) --------------------------------
    lora_r: int = 32
    lora_alpha: int = 32
    lr: float = 2e-5                     # ~10x the paper's full-FT 3e-6 (LoRA rule of thumb)
    adam_betas: tuple = (0.9, 0.99)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ppo_epochs: int = 2
    num_minibatches: int = 2             # optimizer steps per epoch
    clip_eps: float = 0.2
    sft_lambda: float = 0.5              # insight internalization strength
    positive_ratio: float = 0.75         # positive-ratio filtering (App. A.3.2); <=0 disables
    tis_cap: float = 2.0                 # truncated IS weight cap for behaviour/proximal mismatch
    max_train_seq_len: int = 64000       # longer segments are skipped for GRPO (logged); measured safe max 72k w/ offload
    offload_threshold: int = 8192        # activation offload above this many tokens (measured: no time cost)
    seed: int = 0

    def __post_init__(self):
        _expand_home(self)

    # derived ------------------------------------------------------------------------------------------
    def path(self, *parts: str) -> str:
        return os.path.join(self.data, *parts)

    @property
    def gateway_url(self) -> str:
        return f"http://{self.gateway_host}:{self.gateway_port}"


GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F-]+")


def _expand_home(cfg: "Config") -> None:
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, str) and v.startswith("~"):
            setattr(cfg, f.name, os.path.expanduser(v))


def require_gpu_uuid(cfg: "Config", key: str) -> str:
    """The GPU UUID in config field `key`; exits with a clear message if it is unset or not a UUID."""
    v = getattr(cfg, key)
    if not GPU_UUID_RE.fullmatch(v or ""):
        raise SystemExit(f"config: {key}={v!r} is not a GPU UUID. Set it in config.json "
                         f"(see config.example.json; list the UUIDs with `nvidia-smi -L`).")
    return v


def load_config() -> Config:
    cfg = Config()
    p = os.environ.get("RLTLDR_CONFIG", os.path.join(ROOT, "config.json"))
    if os.path.exists(p):
        with open(p) as f:
            over = json.load(f)
        known = {f.name for f in fields(Config)}
        unknown = set(over) - known
        if unknown:
            raise ValueError(f"unknown config keys in {p}: {sorted(unknown)}")
        for k, v in over.items():
            setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
    _expand_home(cfg)
    return cfg


def dump_config(cfg: Config) -> str:
    return json.dumps(asdict(cfg), indent=1)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "get":
        cfg = load_config()
        key = argv[1]
        if key.startswith("_") or not hasattr(cfg, key) or callable(getattr(cfg, key)):
            print(f"unknown config key: {key}", file=sys.stderr)
            return 2
        v = getattr(cfg, key)
        print(v if isinstance(v, str) else json.dumps(v))
        return 0
    if argv == ["dump"]:
        print(dump_config(load_config()))
        return 0
    print("usage: python -m rltldr.config get <key> | dump", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
