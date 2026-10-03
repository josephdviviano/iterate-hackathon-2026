"""transformers 5.18 qwen3_5: call fla's chunk_gated_delta_rule in native GVA mode.

fla's gated-delta BACKWARD faults on B200 ("misaligned address") when q/k are
expanded to 48 heads (HK == HV); fla natively supports grouped q/k (HK=16 < HV=48),
which is numerically equivalent (ceselder/EasyNLA utils/patch_transformers_qwen35_gva.py,
rel_err ~7e-3 = bf16 noise). Port of that patch to the 5.18 module layout, where the
kernels are module-level functions. Only the CHUNK path (training / prefill) skips the
expansion; single-token decode keeps the expanded layout. Idempotent.
"""
import importlib.util
import sys
from pathlib import Path

OLD = """        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)"""
NEW = """        # NLA PATCH: native-GVA fla chunk kernel (see patch_qwen35_gva_518.py)
        _nla_chunk_fla = _NLA_HAS_FLA and not (use_precomputed_states and seq_len == 1)
        if self.num_v_heads // self.num_k_heads > 1 and not _nla_chunk_fla:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)"""
HEADER = "\nimport importlib.util as _nla_iu\n_NLA_HAS_FLA = _nla_iu.find_spec('fla') is not None\n"

spec = importlib.util.find_spec("transformers.models.qwen3_5.modeling_qwen3_5")
path = Path(spec.origin)
src = path.read_text()
if "NLA PATCH" in src:
    print("already patched", path)
    sys.exit(0)
assert src.count(OLD) == 1, "pattern not found / ambiguous — transformers version drift"
anchor = "\nclass Qwen3_5GatedDeltaNet"
i = src.index("@use_kernel_forward_from_hub(\"Qwen3_5GatedDeltaNet\")")
src = src[:i] + HEADER + "\n\n" + src[i:]
src = src.replace(OLD, NEW, 1)
path.write_text(src)
print("patched", path)
