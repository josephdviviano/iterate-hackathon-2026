"""One-time: dequantize the official block-FP8 checkpoint to bf16 so the trainer's frozen base == served weights.
W_bf16 = fp8.float() * scale_inv (expanded over 128x128 blocks). Text model + MTP kept; config drops quantization_config.
Paths come from the config: model_dir (source) -> trainer_base_dir (output). Run with the train env:
    ~/envs/train/bin/python tools/dequant_fp8.py"""
import json, os, shutil, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rltldr.config import load_config  # noqa: E402
_cfg = load_config()
SRC, DST = _cfg.model_dir, _cfg.trainer_base_dir
os.makedirs(DST, exist_ok=True)
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
def deq(w, s, B=128):
    S = s.float().repeat_interleave(B, 0)[: w.shape[0]].repeat_interleave(B, 1)[:, : w.shape[1]]
    return (w.float() * S).to(torch.bfloat16)
new = {}
for shard in sorted(set(wm.values())):
    out = {}
    with safe_open(f"{SRC}/{shard}", "pt") as f:
        for k in f.keys():
            if k.endswith(".weight_scale_inv"): continue
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                sk = k + "_scale_inv"
                with safe_open(f"{SRC}/{wm[sk]}", "pt") as g:
                    t = deq(t, g.get_tensor(sk))
            out[k] = t.contiguous()
    save_file(out, f"{DST}/{shard}", metadata={"format": "pt"}); new.update({k: shard for k in out})
    print(shard, len(out), flush=True)
json.dump({"metadata": {}, "weight_map": new}, open(f"{DST}/model.safetensors.index.json", "w"))
cfg = json.load(open(f"{SRC}/config.json")); cfg.pop("quantization_config", None)
cfg.get("text_config", {}).pop("quantization_config", None)
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
for fn in os.listdir(SRC):
    if fn.endswith((".json", ".jinja", ".txt")) and fn not in ("config.json", "model.safetensors.index.json"):
        shutil.copy(f"{SRC}/{fn}", DST)
print("DONE")
