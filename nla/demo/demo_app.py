"""Interactive NLA demo for Qwen3.8-27B (layer 42): chat with the model, click a token, explain it.

Separate Modal app + volume from the training app (`qwen38-nla`) so it never touches training state.

    modal run demo/demo_app.py::prefetch      # once, CPU only: base model -> demo volume
    modal deploy demo/demo_app.py             # prints the URL; open it with ?key=<demo/.demo_key>

A CPU `web` function serves the page + API (so the page loads instantly) and calls the GPU class
`Demo`: one B200 holds base/AV (~54 GB) + AR (~36 GB); it scales to zero after SCALEDOWN s idle.
Checkpoints come from the HF repo; the UI offers a hot reload when a newer revision is pushed.
"""

import os
import secrets as _secrets
from pathlib import Path

import modal

APP_NAME = "qwen38-nla-demo"
REPO = "gereon/qwen3.8-27b-nla-L42"
BASE = "Qwen/Qwen3.8-27B"
LAYER = 42
HF_CACHE = "/cache/hf"
SCALEDOWN = 5 * 60
HERE = Path(__file__).parent

_key_file = HERE / ".demo_key"
if not _key_file.exists() and modal.is_local():
    _key_file.write_text(_secrets.token_urlsafe(18))
DEMO_KEY = _key_file.read_text().strip() if _key_file.exists() else ""

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.0", "transformers==5.18.0", "accelerate", "peft==0.21.2",
        "flash-linear-attention==0.5.2", "kernels", "safetensors", "numpy",
        "huggingface_hub[hf_xet]", "sentencepiece", "protobuf", "fastapi[standard]",
    )
    .pip_install("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
    .env({"HF_HOME": HF_CACHE, "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE.parent / "nla_qwen38.py", "/root/nla_qwen38.py")
)
web_image = image.add_local_file(HERE / "index.html", "/root/index.html")   # UI edits don't restart the GPU

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("qwen38-nla-demo", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
key_secret = modal.Secret.from_dict({"DEMO_KEY": DEMO_KEY})


def parse_expl(raw):
    """(explanation, truncated) — falls back to an unclosed <explanation> block."""
    import re
    m = re.search(r"<explanation>(.*?)</explanation>", raw, re.S)
    if m:
        return m.group(1).strip(), False
    if "<explanation>" in raw:
        rest = raw.split("<explanation>", 1)[1].strip()
        return (rest, True) if len(rest) > 40 else (None, False)
    return None, False


@app.function(volumes={"/cache": vol}, secrets=[hf_secret], cpu=8.0, memory=16384, timeout=3600)
def prefetch():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "merges*", "vocab*"])
    snapshot_download(REPO)
    vol.commit()
    print("cached", BASE, REPO)


@app.cls(gpu="B200", volumes={"/cache": vol}, secrets=[hf_secret], timeout=3600,
         scaledown_window=SCALEDOWN, max_containers=1)
@modal.concurrent(max_inputs=16)
class Demo:
    @modal.enter()
    def load(self):
        import sys
        import threading
        import time
        sys.path.insert(0, "/root")
        from huggingface_hub import HfApi, snapshot_download
        from nla_qwen38 import NLA

        t0 = time.time()
        self.lock = threading.Lock()
        self.api = HfApi()
        self.sha = self.api.model_info(REPO).sha
        self._latest = (self.sha, time.time())
        d = snapshot_download(REPO, revision=self.sha)
        self.nla = NLA(d, stage="rl")
        self.an = {"sft": "sft", "rl": "rl"}          # stage -> current PEFT adapter name
        self._load_ar_states(d)
        self.tok = self.nla.tok
        vol.commit()
        print(f"loaded {REPO}@{self.sha[:8]} stages={self.stages} in {time.time()-t0:.0f}s", flush=True)
        self._warmup()
        print(f"warm in {time.time()-t0:.0f}s", flush=True)

    def _warmup(self):
        """First generate() pays ~35 s of kernel compile/autotune; pay it before serving."""
        import torch
        ids = self._render([{"role": "user", "content": "Hi!"}], False)
        with self.lock, torch.no_grad():
            x = torch.tensor([ids], device=self.nla.device)
            with self.nla.av.disable_adapter():
                self.nla.av.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=8,
                                     do_sample=False, pad_token_id=self.tok.eos_token_id)
            self._use_stage(self.stages[0])
            acts, _ = self._acts_and_preds(ids, [len(ids) - 1])
            self.nla.explain(acts, max_new_tokens=8, return_raw=True)
            self.nla.reconstruct(["warmup"])

    # ---- checkpoint management ------------------------------------------------------
    def _load_ar_states(self, d):
        import torch
        from safetensors.torch import load_file
        self.ar_states = {}
        for stage, p in (("sft", f"{d}/ar_sft_critic/ar_lora_value_head.safetensors"),
                         ("rl", f"{d}/rl_critic/ar_lora_value_head.safetensors")):
            if os.path.exists(p):
                self.ar_states[stage] = {k: v.to(self.nla.ar_device) for k, v in load_file(p).items()}
        self.has_rl = os.path.exists(f"{d}/av_rl_lora/adapter_model.safetensors") and "rl" in self.ar_states
        self.stages = ["rl", "sft"] if self.has_rl else ["sft"]
        self.ar_stage = None
        torch.cuda.empty_cache()

    def _use_stage(self, stage):
        """Activate AV adapters + AR weights for `stage` (caller holds the lock)."""
        stage = stage if stage in self.stages else self.stages[0]
        av = self.nla.av
        want = [self.an["sft"], self.an["rl"]] if stage == "rl" else [self.an["sft"]]
        if list(av.base_model.active_adapters) != want:
            av.base_model.set_adapter(want)
        if self.ar_stage != stage:
            self.nla.ar.load_state_dict(self.ar_states[stage], strict=False)
            self.nla.ar.value_head.float()
            self.ar_stage = stage
        return stage

    def _reload(self):
        import json
        from huggingface_hub import snapshot_download
        latest = self.api.model_info(REPO).sha
        if latest == self.sha:
            return False
        d = snapshot_download(REPO, revision=latest)
        av = self.nla.av
        old = list(av.peft_config)
        tag = latest[:8]
        an = {"sft": f"sft_{tag}", "rl": f"rl_{tag}"}
        av.load_adapter(f"{d}/av_sft_lora", adapter_name=an["sft"])
        if os.path.exists(f"{d}/av_rl_lora/adapter_model.safetensors"):
            av.load_adapter(f"{d}/av_rl_lora", adapter_name=an["rl"])
        av.base_model.set_adapter([an["sft"]])
        for name in old:
            av.delete_adapter(name)
        self.an = an
        self.nla.meta = json.load(open(f"{d}/nla_config.json"))
        self.nla.dir = d
        self._load_ar_states(d)
        self.sha = latest
        vol.commit()
        return True

    def _latest_sha(self):
        import time
        sha, t = self._latest
        if time.time() - t > 60:
            try:
                sha = self.api.model_info(REPO).sha
            except Exception:
                pass
            self._latest = (sha, time.time())
        return sha

    # ---- model ops ------------------------------------------------------------------
    def _tokens(self, ids):
        special = set(self.tok.all_special_ids)
        out = []
        for i in ids:
            t = self.tok.decode([i])
            out.append({"id": i, "text": t, "special": i in special or t in ("<think>", "</think>")})
        return out

    def _render(self, messages, thinking, add_gen=True):
        return self.tok.apply_chat_template(messages, add_generation_prompt=add_gen, tokenize=True,
                                            enable_thinking=thinking, return_dict=False)

    def _acts_and_preds(self, ids, positions, topk=5):
        """Layer-42 residual at `positions` of `ids` (adapters off) + the model's top next-token preds."""
        import torch
        av = self.nla.av
        model = av.base_model.model
        grab = {}
        h = self.nla._layers[LAYER].register_forward_hook(
            lambda _m, _i, o: grab.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        try:
            with av.disable_adapter(), torch.no_grad():
                x = torch.tensor([ids[:max(positions) + 1]], device=self.nla.device)
                hid = model.model(input_ids=x, use_cache=False).last_hidden_state[0, positions]
                logits = model.lm_head(hid).float()
        finally:
            h.remove()
        acts = grab["h"][0, positions].float().cpu()
        p = logits.softmax(-1).topk(topk)
        preds = [[{"text": self.tok.decode([int(t)]), "p": float(q)} for t, q in zip(ti, pi)]
                 for ti, pi in zip(p.indices.tolist(), p.values.tolist())]
        return acts, preds

    # ---- methods called by the CPU front (generators stream event dicts) ---------------
    @modal.method()
    def status(self):
        return {"repo": REPO, "sha": self.sha, "latest": self._latest_sha(), "stages": self.stages, "layer": LAYER}

    @modal.method()
    def reload(self):
        with self.lock:
            changed = self._reload()
        return {"changed": changed, "sha": self.sha, "stages": self.stages}

    @modal.method()
    def tokenize(self, b):
        if b.get("raw") is not None:
            ids = self.tok.encode(b["raw"], add_special_tokens=False)
        else:
            ids = self._render(b["messages"], b.get("thinking", False), add_gen=False)
        return {"tokens": self._tokens(ids)}

    def _stream(self, work, streamer):
        """Run `work(out)` in a thread holding the GPU lock; yield streamer text as deltas."""
        import threading
        out = {}

        def run():
            try:
                with self.lock:
                    work(out)
            except Exception as e:  # surface to the client instead of hanging the stream
                out["error"] = repr(e)
                out.setdefault("_ready", threading.Event()).set()
                streamer.end()

        out["_ready"] = threading.Event()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        out["_ready"].wait()
        if "meta" in out:
            yield {"meta": out["meta"]}
        if "error" not in out:
            for piece in streamer:
                if piece:
                    yield {"delta": piece}
        th.join()
        if "error" in out:
            yield {"error": out["error"]}
        else:
            yield {"done": True, **out["final"]}

    @modal.method()
    def chat(self, b):
        import torch
        from transformers import TextIteratorStreamer
        ids = self._render(b["messages"], b.get("thinking", False))
        temp = float(b.get("temperature", 0.7))
        max_new = int(min(b.get("max_new_tokens", 512), 4096))
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=False)

        def work(out):
            out["_ready"].set()
            with self.nla.av.disable_adapter(), torch.no_grad():
                x = torch.tensor([ids], device=self.nla.device)
                kw = dict(do_sample=True, temperature=temp, top_p=0.95) if temp > 0 else dict(do_sample=False)
                g = self.nla.av.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                                         streamer=streamer, pad_token_id=self.tok.eos_token_id,
                                         eos_token_id=[self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")],
                                         **kw)
            full = g[0].tolist()
            out["final"] = {"text": self.tok.decode(full[len(ids):], skip_special_tokens=True),
                            "tokens": self._tokens(full)}

        yield from self._stream(work, streamer)

    @modal.method()
    def explain_stream(self, b):
        """One token, one sample: meta (preds) -> explanation deltas -> final recon score."""
        import torch
        from transformers import TextIteratorStreamer
        ids = [int(i) for i in b["ids"]]
        p = int(b["position"])
        if not 0 <= p < len(ids):
            yield {"error": "bad position"}
            return
        temp = float(b.get("temperature", 0.0))
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)

        def work(out):
            stage = self._use_stage(b.get("stage", self.stages[0]))
            acts, preds = self._acts_and_preds(ids, [p])
            out["meta"] = {"stage": stage, "sha": self.sha, "position": p, "token": self.tok.decode([ids[p]]),
                           "norm": float(acts[0].norm()), "preds": preds[0]}
            out["_ready"].set()
            stop = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|endoftext|>")]
            pt = torch.tensor([self.nla.prompt_ids], device=self.nla.device)
            kw = dict(do_sample=True, temperature=temp, top_p=1.0, top_k=0) if temp > 0 else dict(do_sample=False)
            self.nla._vec = acts
            try:
                with torch.no_grad():
                    g = self.nla.av.generate(input_ids=pt, attention_mask=torch.ones_like(pt), max_new_tokens=448,
                                             streamer=streamer, eos_token_id=stop, pad_token_id=stop[0], **kw)
            finally:
                self.nla._vec = None
            raw = self.tok.decode(g[0, pt.shape[1]:], skip_special_tokens=True)
            expl, trunc = parse_expl(raw)
            cos = None
            if expl:
                rec = self.nla.reconstruct([expl])
                cos = round(float(torch.nn.functional.cosine_similarity(rec, acts, dim=-1)[0]), 4)
            out["final"] = {"sample": {"explanation": expl, "raw": None if expl else raw, "recon_cos": cos,
                                       "truncated": trunc}}

        yield from self._stream(work, streamer)

    @modal.method()
    def explain(self, b):
        import torch
        ids = [int(i) for i in b["ids"]]
        positions = sorted({int(p) for p in b["positions"]})[:32]
        if not positions or positions[-1] >= len(ids) or positions[0] < 0:
            raise ValueError("bad positions")
        temp = float(b.get("temperature", 0.0))
        n = max(1, min(int(b.get("n_samples", 1)), 8)) if temp > 0 else 1
        with self.lock:
            stage = self._use_stage(b.get("stage", self.stages[0]))
            acts, preds = self._acts_and_preds(ids, positions)
            rep = acts.repeat_interleave(n, 0)
            raws = self.nla.explain(rep, temperature=temp, max_new_tokens=448, return_raw=True)
            parsed = [parse_expl(r) for r in raws]
            expl = [e for e, _ in parsed]
            rec = self.nla.reconstruct(expl)
        cos = torch.nn.functional.cosine_similarity(rec, rep, dim=-1).tolist()
        res = []
        for j, p in enumerate(positions):
            samples = [{"explanation": expl[k], "raw": raws[k] if expl[k] is None else None,
                        "recon_cos": None if expl[k] is None else round(cos[k], 4), "truncated": parsed[k][1]}
                       for k in range(j * n, (j + 1) * n)]
            res.append({"position": p, "token": self.tok.decode([ids[p]]), "norm": float(acts[j].norm()),
                        "preds": preds[j], "samples": samples})
        return {"stage": stage, "sha": self.sha, "results": res}


# ---- CPU front: serves the page instantly (even while the GPU cold-starts) and proxies the API --
@app.function(image=web_image, secrets=[key_secret], cpu=1.0, memory=1024, scaledown_window=300, timeout=900)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import json

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, StreamingResponse

    api = FastAPI()
    html = Path("/root/index.html").read_text()
    key = os.environ.get("DEMO_KEY", "")
    gpu = Demo()

    def auth(req: Request):
        if key and req.headers.get("x-demo-key") != key:
            raise HTTPException(401, "bad or missing key")

    def sse(gen):
        async def it():
            try:
                async for ev in gen:
                    yield f"data: {json.dumps(ev)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': repr(e)})}\n\n"
        return StreamingResponse(it(), media_type="text/event-stream", headers={"cache-control": "no-cache"})

    async def call(fn, *a):
        try:
            return await fn.remote.aio(*a)
        except Exception as e:
            raise HTTPException(500, repr(e))

    @api.get("/", response_class=HTMLResponse)
    def index():
        return html

    @api.get("/api/status")
    async def status(req: Request):
        auth(req)
        return await call(gpu.status)

    @api.post("/api/reload")
    async def reload(req: Request):
        auth(req)
        return await call(gpu.reload)

    @api.post("/api/tokenize")
    async def tokenize(req: Request):
        auth(req)
        return await call(gpu.tokenize, await req.json())

    @api.post("/api/explain")
    async def explain(req: Request):
        auth(req)
        return await call(gpu.explain, await req.json())

    @api.post("/api/chat")
    async def chat(req: Request):
        auth(req)
        return sse(gpu.chat.remote_gen.aio(await req.json()))

    @api.post("/api/explain_stream")
    async def explain_stream(req: Request):
        auth(req)
        return sse(gpu.explain_stream.remote_gen.aio(await req.json()))

    return api
