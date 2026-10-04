"""Create a PEFT-format LoRA adapter for Qwen3.8-27B with zero (identity) or random B matrices.
    python tools/make_test_adapter.py <out_dir> zero|random [rank=16]
Layer types are read from the config's model_dir checkpoint (config.json text_config)."""
import json, os, sys, torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rltldr.config import load_config  # noqa: E402
out, mode, r = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16
H, I = 5120, 17408
SHAPES = {  # name: (in, out)
    "self_attn.q_proj": (H, 12288), "self_attn.k_proj": (H, 1024), "self_attn.v_proj": (H, 1024),
    "self_attn.o_proj": (6144, H), "linear_attn.in_proj_qkv": (H, 10240), "linear_attn.in_proj_z": (H, 6144),
    "linear_attn.out_proj": (6144, H), "mlp.gate_proj": (H, I), "mlp.up_proj": (H, I), "mlp.down_proj": (I, H)}
cfg = json.load(open(os.path.join(load_config().model_dir, "config.json")))["text_config"]
g = torch.Generator().manual_seed(0)
sd = {}
for i, lt in enumerate(cfg["layer_types"]):
    for name, (fin, fout) in SHAPES.items():
        if name.startswith("self_attn") and lt != "full_attention": continue
        if name.startswith("linear_attn") and lt != "linear_attention": continue
        p = f"base_model.model.model.language_model.layers.{i}.{name}"
        sd[p + ".lora_A.weight"] = (torch.randn(r, fin, generator=g) / fin**0.5).to(torch.bfloat16)
        sd[p + ".lora_B.weight"] = (torch.zeros(fout, r) if mode == "zero" else torch.randn(fout, r, generator=g) * 0.02).to(torch.bfloat16)
os.makedirs(out, exist_ok=True)
save_file(sd, os.path.join(out, "adapter_model.safetensors"))
json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": 2 * r, "lora_dropout": 0.0,
           "bias": "none", "fan_in_fan_out": False, "base_model_name_or_path": "Qwen/Qwen3.8-27B",
           "target_modules": sorted({n.split(".")[-1] for n in SHAPES})}, open(os.path.join(out, "adapter_config.json"), "w"), indent=1)
print(out, len(sd), "tensors")
