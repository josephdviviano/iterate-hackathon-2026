"""Head-to-head (h2h) configuration: two continuous autoresearch sessions, base model vs RLTL;DR policy v5.

Both arms share one vLLM server (FP8 base; started by ./ctl.sh) and are served through LoRA adapters so that both
run the identical kernel path and decode at the same speed:
  * base: an all-zero adapter (rank 32, same target modules as v5) -> outputs are exactly the base model's,
  * v5:   the RLTL;DR v5 adapter, unmerged (merging into bf16 keeps only ~38% of the update: most entries are
          below bf16 rounding), i.e. exactly the policy that produced the training rollouts.
Each arm has its own nanochat GPU, runner, gateway socket, sandbox, pi session and git repo. Nothing is shared
between arms except the vLLM server and the (read-only) baseline commit they both start from.

Machine-specific values (the arms' GPU UUIDs and minors, ports, paths) come from $H2H_CONFIG, default
<RLTLDR_ROOT>/h2h_config.json (template: h2h_config.example.json); keys there replace the defaults below. Path
values may start with ~ (expanded) or be relative (resolved against RLTLDR_ROOT). The arms' GPU UUIDs have no
default: the runner refuses to start an arm without one (Arm.check_gpu).
"""
import json
import os
from dataclasses import asdict, dataclass, field

ROOT = os.environ.get("RLTLDR_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H2H = os.path.join(ROOT, "data", "h2h")          # all h2h state lives here
RUN = os.path.join(ROOT, "run", "h2h")           # per-arm socket dirs: run/h2h/<arm>/{gateway,runner}.sock


@dataclass
class Arm:
    name: str                 # "base" | "v5"
    served_model: str         # LoRA adapter name registered in vLLM
    adapter_dir: str          # adapter files (frozen copies under data/h2h/adapters/)
    gpu_uuid: str             # nanochat GPU of this arm (`nvidia-smi -L`); "" = not configured
    gpu_minor: int            # /dev/nvidia<minor> of that GPU; -1 = not configured
    runner_port: int          # trusted TCP control port of this arm's runner (127.0.0.1)

    def check_gpu(self) -> None:
        """Raise a clear error unless this arm's GPU is configured (needed by everything that trains)."""
        if not self.gpu_uuid or int(self.gpu_minor) < 0:
            raise SystemExit(f"arm {self.name!r}: gpu_uuid/gpu_minor not configured (got {self.gpu_uuid!r}, "
                             f"{self.gpu_minor}); set them in h2h_config.json (see h2h_config.example.json)")

    @property
    def dir(self) -> str:                 # data/h2h/<arm>
        return os.path.join(H2H, self.name)

    @property
    def repo(self) -> str:                # the agent's git repo (rw inside its sandbox)
        return os.path.join(self.dir, "repo")

    @property
    def session_dir(self) -> str:         # pi agent dir + pi session files (rw inside its sandbox)
        return os.path.join(self.dir, "session")

    @property
    def runner_ws(self) -> str:           # pristine runner workspace (own clone + venv pinned to gpu_uuid)
        return os.path.join(self.dir, "runner_ws")

    @property
    def ledger(self) -> str:              # trusted ledger: one line per run (written by tools/ar_run.py)
        return os.path.join(self.dir, "ledger.jsonl")

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.dir, "runs")

    @property
    def cache_dir(self) -> str:           # persistent TorchInductor/Triton cache for this arm's runs
        return os.path.join(self.dir, "cache")

    @property
    def calls(self) -> str:               # gateway call records of this arm
        return os.path.join(self.dir, "calls.jsonl")

    @property
    def events(self) -> str:              # filtered pi RPC events (supervisor)
        return os.path.join(self.dir, "events.jsonl")

    @property
    def status(self) -> str:              # supervisor status snapshot (json)
        return os.path.join(self.dir, "status.json")

    @property
    def sock_dir(self) -> str:            # only this dir is bound into the arm's sandbox
        return os.path.join(RUN, self.name)

    @property
    def gateway_sock(self) -> str:
        return os.path.join(self.sock_dir, "gateway.sock")

    @property
    def runner_sock(self) -> str:
        return os.path.join(self.sock_dir, "runner.sock")


@dataclass
class H2HConfig:
    root: str = ROOT
    canon_dir: str = os.path.join(ROOT, "canon.git")
    baseline_commit: str = "c7666de87837da189c70f11541486c9c42240ceb"   # "sm120 attention patch (baseline)"
    start_ref: str = "h2h/start"            # canon.git branch: baseline + continuous-mode program.md + run.sh
    agent_branch: str = "autoresearch/h2h"  # same name in both arms (the agent cannot tell which arm it is)
    vllm_url: str = "http://127.0.0.1:8000"
    gateway_port: int = 8110                # trusted TCP side of the h2h gateway (control/state/health)
    prepare_sha256: str = "4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150"
    no_autotune: bool = True                # harness default (train.py may still opt in explicitly)
    pi_bin: str = "pi"                      # PATH lookup (inside the sandbox: ~/.local/bin, /usr/local/bin, ...)
    pi_agent_template: str = os.path.join(ROOT, "pi", "agent")
    agent_thinking: str = "medium"
    zero_adapter: str = os.path.join(H2H, "adapters", "base0")
    v5_adapter_src: str = os.path.join(ROOT, "data", "adapters", "policy-v5")
    arms: list = field(default_factory=lambda: [
        Arm("base", "h2h-base0", os.path.join(H2H, "adapters", "base0"), "", -1, 8210),
        Arm("v5", "h2h-v5", os.path.join(H2H, "adapters", "v5"), "", -1, 8220),
    ])

    def arm(self, name: str) -> Arm:
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(f"unknown arm {name!r} (have {[a.name for a in self.arms]})")


PATH_KEYS = ("root", "canon_dir", "pi_agent_template", "zero_adapter", "v5_adapter_src")   # + Arm.adapter_dir


def resolve_path(v):
    """~ expanded; a relative path is taken relative to ROOT (config files carry no absolute paths)."""
    if not isinstance(v, str) or not v:
        return v
    v = os.path.expanduser(v)
    return v if os.path.isabs(v) else os.path.join(ROOT, v)


def load_h2h_config() -> H2HConfig:
    cfg = H2HConfig()
    p = os.environ.get("H2H_CONFIG") or os.path.join(ROOT, "h2h_config.json")
    if os.path.exists(p):
        with open(p) as f:
            over = json.load(f)
        for k, v in over.items():
            if k == "arms":           # keys starting with "_" are comments
                cfg.arms = [Arm(**{**{ak: av for ak, av in a.items() if not ak.startswith("_")},
                                   "adapter_dir": resolve_path(a["adapter_dir"])}) for a in v]
            elif k in PATH_KEYS:
                setattr(cfg, k, resolve_path(v))
            elif k == "pi_bin":
                setattr(cfg, k, os.path.expanduser(v))
            elif not k.startswith("_") and hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    c = load_h2h_config()
    print(json.dumps({**{k: v for k, v in asdict(c).items() if k != "arms"},
                      "arms": [{**asdict(a), "dir": a.dir, "gateway_sock": a.gateway_sock,
                                "runner_sock": a.runner_sock} for a in c.arms]}, indent=1))
