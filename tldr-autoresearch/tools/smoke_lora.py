"""Smoke test of runtime LoRA loading in the live vLLM server: a zero-B adapter must reproduce the base model.
    python tools/smoke_lora.py <zero_adapter_dir> <random_adapter_dir>     (made by tools/make_test_adapter.py)"""
import sys, time, requests
B = "http://127.0.0.1:8000"
if len(sys.argv) != 3:
    sys.exit(__doc__)
for name, path in [("policy-v0", sys.argv[1]), ("policy-test", sys.argv[2])]:
    t = time.time(); r = requests.post(f"{B}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path}, timeout=600)
    print("load", name, r.status_code, r.text[:120], f"{time.time()-t:.2f}s")
msgs = [{"role": "user", "content": "Write a haiku about gradient descent."}]
def gen(model, n=256, effort="none"):
    t = time.time()
    d = requests.post(f"{B}/v1/chat/completions", json=dict(model=model, messages=msgs, max_tokens=n, temperature=0.0,
        logprobs=True, top_logprobs=0, return_token_ids=True, reasoning_effort=effort, ignore_eos=True), timeout=600).json()
    c = d["choices"][0]; dt = time.time() - t
    return c["token_ids"], [x["logprob"] for x in c["logprobs"]["content"]], c["message"]["content"], dt
base = gen("qwen3.8-27b-fp8"); z = gen("policy-v0"); rnd = gen("policy-test")
print("base text:", base[2][:80].replace("\n", " | "))
print("zero==base ids:", z[0] == base[0], "| rand==base ids:", rnd[0] == base[0])
print("rand text:", rnd[2][:80].replace("\n", " | "))
import statistics
print("mean|dlogp| zero-base (first 20):", statistics.mean(abs(a - b) for a, b in zip(z[1][:20], base[1][:20])))
for m in ["qwen3.8-27b-fp8", "policy-v0"]:
    _, _, _, dt = gen(m, n=1024); print(f"decode {m}: {1024/dt:.1f} tok/s")
