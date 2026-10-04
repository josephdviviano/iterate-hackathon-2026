#!/usr/bin/env python3
"""test_h2h_gateway.py - live tests of the h2h adapters + gateway against the running vLLM (./ctl.sh).

Part A (direct to vLLM, no gateway): greedy (temperature 0), one request in flight at a time, every request with a
unique `cache_salt` so each one is a full fresh prefill:
  * control: base vs base again (run-to-run determinism of the server),
  * base0 (all-zero LoRA) vs plain base: identical prompt logprobs, completion tokens, chosen-token and top-5
    logprobs (|diff| < 1e-6, i.e. bitwise in practice),
  * v5 vs base: differs on at least one prompt.
Part B (through the gateway, started here as a subprocess on a temporary root: own sockets, own TCP port, own
calls.jsonl, the real adapter dirs/names; stopped at the end): sockets/permissions, untrusted surface, request
sanitising, forced sampling, model rewrite + scrubbed responses, call records, pause/resume, adapter
re-registration after an unload, client hang-up, decode-speed parity of the two arms running concurrently, and
load isolation (one arm flooding with parallel requests does not slow the other arm).
Part C (fake vLLM, no GPU): outages/restarts, upstream death, drains, shutdown, and the per-arm concurrency cap
(flood -> 429 / one generation per arm, abandoned waiters never sent, pause vs slot, body size limit).
  --real-paths  only a smoke test of the gateway on its production sockets/port (no chat calls, nothing recorded).

  $RLTLDR_SERVE_PY tests/test_h2h_gateway.py [--part a|b|c] [--real-paths] [--keep]
"""
import argparse
import asyncio
import collections
import json
import os
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

import aiohttp
import aiohttp.web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rltldr.h2h_config import load_h2h_config  # noqa: E402

PY = os.environ.get("RLTLDR_SERVE_PY") or os.path.expanduser("~/envs/serve/bin/python")
GATEWAY = os.path.join(ROOT, "rltldr", "h2h_gateway.py")
TEST_PORT = 18110
VLLM = "http://127.0.0.1:8000"
PROMPTS = [
    "Explain in two sentences what a learning-rate warmup does.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "In train.py of a small GPT, which hyperparameter would you tune first to lower validation loss and why?",
    "List three differences between AdamW and Muon optimizers.",
]
LONG_PROMPT = ("Write out every integer from 1 to 3000 in increasing order, separated by single spaces. "
               "No commentary, no code, just the numbers.")
BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command in the current working directory.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Command"}},
                   "required": ["command"]}}}
RESULTS = []


def check(name: str, ok: bool, detail="") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    return ok


# ------------------------------------------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------------------------------------------
def session(sock: str = None) -> aiohttp.ClientSession:
    conn = aiohttp.UnixConnector(path=sock) if sock else aiohttp.TCPConnector(limit=0)
    return aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=1800))


async def post_json(url: str, body, sock: str = None, raw: bytes = None):
    async with session(sock) as s:
        kw = {"data": raw, "headers": {"Content-Type": "application/json"}} if raw is not None else {"json": body}
        async with s.post(url, **kw) as r:
            txt = await r.text()
            try:
                return r.status, json.loads(txt)
            except json.JSONDecodeError:
                return r.status, txt


async def get_json(url: str, sock: str = None):
    async with session(sock) as s:
        async with s.get(url) as r:
            txt = await r.text()
            try:
                return r.status, json.loads(txt)
            except json.JSONDecodeError:
                return r.status, txt


async def stream_chat(url: str, body: dict, sock: str = None, hang_up_after: int = None) -> dict:
    """Streaming chat request; returns raw bytes, parsed events and client-side token timing."""
    out = {"raw": b"", "events": [], "t0": time.time(), "t_first": None, "t_last": None, "n_tok": 0, "n_first": 0}
    async with session(sock) as s:
        async with s.post(url, json=body) as r:
            out["status"] = r.status
            if r.status != 200:
                out["error"] = await r.text()
                return out
            buf = b""
            async for chunk in r.content.iter_any():
                now = time.time()
                out["raw"] += chunk
                buf += chunk
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    if not ev.startswith(b"data: ") or ev[6:].strip() == b"[DONE]":
                        continue
                    d = json.loads(ev[6:])
                    out["events"].append(d)
                    for c in d.get("choices", []):
                        n = len(c.get("token_ids") or [])
                        if n:
                            if out["t_first"] is None:
                                out["t_first"], out["n_first"] = now, n
                            out["t_last"] = now
                            out["n_tok"] += n
                if hang_up_after is not None and out["n_tok"] >= hang_up_after:
                    break               # leaving the context closes the connection mid-stream
    if out["t_first"] and out["t_last"] > out["t_first"]:
        out["tok_s"] = (out["n_tok"] - out["n_first"]) / (out["t_last"] - out["t_first"])
    return out


async def vllm_models() -> set:
    _, d = await get_json(f"{VLLM}/v1/models")
    return {m["id"] for m in d["data"]}


async def vllm_running():
    """vLLM's num_requests_running gauge (all clients), or None."""
    try:
        async with session() as s:
            async with s.get(f"{VLLM}/metrics") as r:
                txt = await r.text()
        return sum(float(ln.rsplit(" ", 1)[1]) for ln in txt.splitlines()
                   if ln.startswith("vllm:num_requests_running"))
    except (aiohttp.ClientError, ValueError, IndexError):
        return None


async def vllm_load(name: str, path: str) -> None:
    st, d = await post_json(f"{VLLM}/v1/load_lora_adapter", {"lora_name": name, "lora_path": path})
    assert st == 200 or "already" in str(d).lower(), (st, d)


# ------------------------------------------------------------------------------------------------------
# Part A: exactness of base0, v5 differs (direct to vLLM)
# ------------------------------------------------------------------------------------------------------
async def greedy(model: str, prompt: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64,
            "temperature": 0, "logprobs": True, "top_logprobs": 5, "prompt_logprobs": 0, "return_token_ids": True,
            "cache_salt": uuid.uuid4().hex}
    st, d = await post_json(f"{VLLM}/v1/chat/completions", body)
    assert st == 200, (st, d)
    c = d["choices"][0]
    content = c["logprobs"]["content"]
    plp = [None if x is None else next(iter(x.values()))["logprob"] for x in d["prompt_logprobs"]]
    return {"prompt_ids": d["prompt_token_ids"], "ids": c["token_ids"], "lp": [x["logprob"] for x in content],
            "top": [[(t["token"], t["logprob"]) for t in x["top_logprobs"]] for x in content], "plp": plp}


def compare(a: dict, b: dict) -> dict:
    """Token equality and max |logprob diff| over the common prefix (chosen, top-5, prompt logprobs)."""
    n = 0
    while n < min(len(a["ids"]), len(b["ids"])) and a["ids"][n] == b["ids"][n]:
        n += 1
    d_lp = max((abs(x - y) for x, y in zip(a["lp"][:n], b["lp"][:n])), default=0.0)
    d_top, top_same = 0.0, True
    for ta, tb in zip(a["top"][:n], b["top"][:n]):
        if [t for t, _ in ta] != [t for t, _ in tb]:
            top_same = False
        else:
            d_top = max([d_top] + [abs(x - y) for (_, x), (_, y) in zip(ta, tb)])
    d_plp = max((abs(x - y) for x, y in zip(a["plp"], b["plp"]) if x is not None and y is not None), default=0.0)
    return {"same_prompt": a["prompt_ids"] == b["prompt_ids"], "same_tokens": a["ids"] == b["ids"],
            "common_prefix": n, "len": (len(a["ids"]), len(b["ids"])), "max_d_lp": d_lp, "top5_same": top_same,
            "max_d_top5": d_top, "max_d_prompt_lp": d_plp}


async def part_a(cfg) -> dict:
    base_name = (await get_json(f"{VLLM}/v1/models"))[1]["data"][0]["id"]
    for arm in cfg.arms:
        await vllm_load(arm.served_model, arm.adapter_dir)
    b0, v5 = cfg.arm("base").served_model, cfg.arm("v5").served_model
    rows = []
    for p in PROMPTS:
        r_base = await greedy(base_name, p)
        r_b0 = await greedy(b0, p)
        r_base2 = await greedy(base_name, p)
        r_v5 = await greedy(v5, p)
        rows.append({"prompt": p[:40], "base_vs_base": compare(r_base, r_base2),
                     "base0_vs_base": compare(r_base, r_b0), "v5_vs_base": compare(r_base, r_v5)})
    for r in rows:
        print(json.dumps(r))
    ctl = [r["base_vs_base"] for r in rows]
    check("A0 control: plain base is deterministic run-to-run (greedy, fresh prefill)",
          all(c["same_tokens"] and c["max_d_lp"] < 1e-6 and c["max_d_prompt_lp"] < 1e-6 for c in ctl),
          f"max|dlp|={max(c['max_d_lp'] for c in ctl):.3g}")
    z = [r["base0_vs_base"] for r in rows]
    worst = {k: max(c[k] for c in z) for k in ("max_d_lp", "max_d_top5", "max_d_prompt_lp")}
    check("A1 base0 == base: identical completion tokens on all prompts", all(c["same_tokens"] for c in z),
          f"{sum(c['same_tokens'] for c in z)}/{len(z)} identical, lengths {[c['len'][0] for c in z]}")
    check("A2 base0 == base: chosen/top-5/prompt logprobs |diff| < 1e-6",
          all(c["top5_same"] for c in z) and all(v < 1e-6 for v in worst.values()), json.dumps(worst))
    v = [r["v5_vs_base"] for r in rows]
    check("A3 v5 differs from base on at least one prompt",
          any(not c["same_tokens"] or c["max_d_lp"] > 1e-3 for c in v),
          f"tokens differ on {sum(not c['same_tokens'] for c in v)}/{len(v)} prompts; first divergence at "
          f"{[c['common_prefix'] for c in v]}; max|dlp| on common prefix "
          f"{max(c['max_d_lp'] for c in v):.4f}; max|d prompt lp| {max(c['max_d_prompt_lp'] for c in v):.4f}")
    return {"rows": rows}


# ------------------------------------------------------------------------------------------------------
# Part B: the gateway
# ------------------------------------------------------------------------------------------------------
class Gateway:
    """The gateway as a subprocess with an isolated root (or the production config with real_paths)."""

    def __init__(self, real_paths: bool = False, port: int = TEST_PORT, extra: dict = None, env: dict = None):
        self.real = real_paths
        self.tmp = tempfile.mkdtemp(prefix="h2h_gw_test_")
        self.env = {**os.environ, **(env or {})}
        prod = load_h2h_config()
        if real_paths:
            self.port = prod.gateway_port
        else:
            self.port = port
            cfg_path = os.path.join(self.tmp, "h2h_config.json")
            with open(cfg_path, "w") as f:
                json.dump({**(extra or {}), "gateway_port": self.port, "arms": [
                    {"name": a.name, "served_model": a.served_model, "adapter_dir": a.adapter_dir,
                     "gpu_uuid": a.gpu_uuid, "gpu_minor": a.gpu_minor, "runner_port": a.runner_port}
                    for a in prod.arms]}, f)
            self.env.update(RLTLDR_ROOT=self.tmp, H2H_CONFIG=cfg_path)
        # the test process sees the same config the gateway sees
        out = subprocess.run([PY, os.path.join(ROOT, "rltldr", "h2h_config.py")], env=self.env,
                             capture_output=True, text=True, check=True).stdout
        self.cfg = json.loads(out)
        self.arms = {a["name"]: a for a in self.cfg["arms"]}
        self.log_path = os.path.join(self.tmp, "gateway.log")
        self.proc = None

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def calls(self, arm: str) -> list:
        p = os.path.join(self.arms[arm]["dir"], "calls.jsonl")
        if self.real or not os.path.exists(p):
            return []
        with open(p) as f:
            return [json.loads(x) for x in f if x.strip()]

    async def new_calls(self, arm: str, since: int, n: int, timeout: float = 30) -> list:
        """Records after index `since`, waiting until at least n exist (a record lands just after the stream ends)."""
        t = time.time()
        while len(self.calls(arm)) < since + n and time.time() - t < timeout:
            await asyncio.sleep(0.1)
        return self.calls(arm)[since:]

    def log(self) -> str:
        with open(self.log_path) as f:
            return f.read()

    async def start(self, wait_adapters: bool = True):
        self.proc = subprocess.Popen([PY, GATEWAY], env=self.env, stdout=open(self.log_path, "w"),
                                     stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(120):
            await asyncio.sleep(0.5)
            if self.proc.poll() is not None:
                raise RuntimeError(f"gateway exited early:\n{self.log()}")
            try:
                st, d = await get_json(self.url("/control/state"))
                if st == 200 and (not wait_adapters or all(a["adapter_loaded"] for a in d["arms"].values())):
                    return d
            except aiohttp.ClientError:
                pass
        raise RuntimeError(f"gateway did not become ready:\n{self.log()}")

    def stop(self) -> int:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                return self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                return self.proc.wait()
        return self.proc.returncode if self.proc else None

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def chat_body(prompt: str, **kw) -> dict:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], **kw}


async def part_b_smoke(gw: Gateway):
    d = await gw.start()
    check("R1 production gateway up, both adapters registered", all(a["adapter_loaded"] for a in d["arms"].values()),
          json.dumps({k: v["served_model"] for k, v in d["arms"].items()}))
    for name, a in gw.arms.items():
        s = os.stat(a["gateway_sock"])
        dmode = stat.S_IMODE(os.stat(os.path.dirname(a["gateway_sock"])).st_mode)
        st, m = await get_json("http://gw/v1/models", sock=a["gateway_sock"])
        check(f"R2 {name}: {a['gateway_sock']} mode {oct(stat.S_IMODE(s.st_mode))}, dir {oct(dmode)}, /v1/models",
              stat.S_ISSOCK(s.st_mode) and stat.S_IMODE(s.st_mode) == 0o666 and dmode == 0o755 and st == 200
              and [x["id"] for x in m["data"]] == ["policy"])
    rc = gw.stop()
    check("R3 SIGTERM: clean exit, sockets removed",
          rc == 0 and not any(os.path.exists(a["gateway_sock"]) for a in gw.arms.values()), f"rc={rc}")


async def part_b(gw: Gateway) -> dict:
    stale_before = sorted(m for m in await vllm_models() if m.startswith("oct3-"))
    t0 = time.time()
    d = await gw.start()
    check("B0 gateway up; both adapters registered", all(a["adapter_loaded"] for a in d["arms"].values()),
          f"ready in {time.time() - t0:.1f}s; state={json.dumps(d['arms'])[:200]}...")
    served = await vllm_models()
    check("B1 stale oct3-* adapters unloaded at startup",
          not any(m.startswith("oct3-") for m in served), f"before={stale_before} now={sorted(served)}")
    socks = {n: a["gateway_sock"] for n, a in gw.arms.items()}
    models = {n: a["served_model"] for n, a in gw.arms.items()}

    # --- sockets and untrusted surface
    for n, sock in socks.items():
        s = os.stat(sock)
        dmode = stat.S_IMODE(os.stat(os.path.dirname(sock)).st_mode)
        check(f"B2 {n}: socket mode 0666, dir 0755", stat.S_IMODE(s.st_mode) == 0o666 and dmode == 0o755,
              f"{oct(stat.S_IMODE(s.st_mode))} {oct(dmode)}")
        st, m = await get_json("http://gw/v1/models", sock=sock)
        check(f"B3 {n}: /v1/models lists only 'policy', no adapter name",
              st == 200 and [x["id"] for x in m["data"]] == ["policy"] and "h2h" not in json.dumps(m), json.dumps(m))
        bad = []
        for meth, path in (("GET", "/control/state"), ("POST", "/control/pause"), ("POST", "/control/resume"),
                           ("GET", "/health"), ("POST", "/v1/completions"), ("POST", "/v1/load_lora_adapter")):
            async with session(sock) as s_:
                async with s_.request(meth, f"http://gw{path}", json={}) as r:
                    if r.status not in (404, 405):
                        bad.append((meth, path, r.status))
        check(f"B4 {n}: control/other endpoints not exposed on the sandbox socket", not bad, str(bad))
        st, _ = await post_json("http://gw/v1/chat/completions", None, sock=sock, raw=b"{not json")
        st2, _ = await post_json("http://gw/v1/chat/completions", {"model": "x"}, sock=sock)
        st3, _ = await post_json("http://gw/v1/chat/completions", chat_body("hi", max_tokens="lots"), sock=sock)
        check(f"B5 {n}: malformed requests -> 400", (st, st2, st3) == (400, 400, 400), str((st, st2, st3)))

    # --- concurrent streaming through both sockets with hostile sampling params
    hostile = dict(stream=True, max_tokens=200, temperature=0.0, top_k=1, top_p=0.1, seed=7, n=3,
                   logit_bias={"15": 100}, presence_penalty=2.0, chat_template_kwargs={"enable_thinking": False},
                   prompt_logprobs=5, extra_body={"x": 1}, cache_salt="evil")
    n_calls = {n: len(gw.calls(n)) for n in socks}
    res = await asyncio.gather(*[stream_chat("http://gw/v1/chat/completions", chat_body(PROMPTS[1], **hostile),
                                             sock=socks[n]) for n in socks for _ in range(2)])
    by_arm = {n: res[2 * i: 2 * i + 2] for i, n in enumerate(socks)}
    for n, rs in by_arm.items():
        ok = all(r["status"] == 200 for r in rs)
        evmodels = {e.get("model") for r in rs for e in r["events"]}
        leak = any(b"h2h" in r["raw"] for r in rs)
        check(f"B6 {n}: two requests at once succeed (served in turn: 1 slot per arm); model scrubbed to 'policy'",
              ok and evmodels == {"policy"} and not leak, f"models={evmodels} leak={leak}")
        recs = await gw.new_calls(n, n_calls[n], 2)
        p = recs[-1]["params"] if recs else {}
        check(f"B7 {n}: sampling forced (temperature 0/top_k 1 overridden; two identical requests diverge)",
              len(recs) == 2 and all(r["params"]["temperature"] == 1.0 and r["params"]["top_k"] == 20
                                     and r["params"]["n"] == 1 and r["params"]["top_p"] == 0.95
                                     and r["params"]["presence_penalty"] == 0.0 for r in recs)
              and recs[0]["completion_ids"] != recs[1]["completion_ids"], f"params={p}")
        check(f"B8 {n}: model rewritten to {models[n]} and call recorded with tokens/logprobs",
              len(recs) == 2 and all(r["served_model"] == models[n] and r["arm"] == n and r["status"] == "ok"
                                     and r["prompt_ids"] and len(r["completion_ids"]) == len(r["logprobs"])
                                     == r["n_completion"] > 0 and r["n_prompt"] == len(r["prompt_ids"])
                                     and r["ttft_s"] is not None and r["decode_tok_s"] for r in recs),
              "; ".join(f"{r['status']} fin={r['finish_reason']} n={r['n_prompt']}/{r['n_completion']} "
                        f"ttft={r['ttft_s']} tok/s={r['decode_tok_s']}" for r in recs))
        r0 = recs[0] if recs else {}
        first_keys = list(r0)[:10]
        check(f"B9 {n}: record schema (scalars first, token lists last)",
              first_keys[:4] == ["call_id", "arm", "served_model", "ts_start"]
              and list(r0)[-3:] == ["prompt_ids", "completion_ids", "logprobs"]
              and all(k in r0 for k in ("ts_end", "status", "finish_reason", "n_prompt", "n_completion",
                                        "ttft_s", "decode_tok_s")), str(list(r0)))

    # --- pi-like request with tools: streamed tool call parsed by vLLM, relayed and recorded (both arms)
    tool_body = chat_body("List the files in the current directory. Use the bash tool.", stream=True,
                          stream_options={"include_usage": True}, max_tokens=4096, reasoning_effort="low",
                          tools=[BASH_TOOL], tool_choice="auto", parallel_tool_calls=False)
    n_calls = {n: len(gw.calls(n)) for n in socks}
    res = await asyncio.gather(*[stream_chat("http://gw/v1/chat/completions", tool_body, sock=socks[n])
                                 for n in socks])
    for (n, sock), r in zip(socks.items(), res):
        recs = await gw.new_calls(n, n_calls[n], 1)
        tc = (recs[0]["response"].get("tool_calls") or []) if recs else []
        streamed = [t for e in r["events"] for c in e.get("choices", []) for t in
                    ((c.get("delta") or {}).get("tool_calls") or [])]
        check(f"B10 {n}: streamed tool call relayed and recorded",
              r["status"] == 200 and recs and recs[0]["status"] == "ok" and recs[0]["finish_reason"] == "tool_calls"
              and tc and tc[0]["function"]["name"] == "bash" and streamed
              and recs[0]["params"]["reasoning_effort"] == "low",
              f"fin={recs[0]['finish_reason'] if recs else None} tool_calls={json.dumps(tc)[:160]}")

    # --- non-streaming request, max_tokens cap, model in body
    st, d = await post_json("http://gw/v1/chat/completions",
                            chat_body("Reply with just the word OK.", max_tokens=10 ** 9), sock=socks["v5"])
    rec = gw.calls("v5")[-1]
    check("B10b non-stream: ok, model 'policy' in response, max_tokens capped at 32768",
          st == 200 and d.get("model") == "policy" and rec["params"]["max_tokens"] == 32768
          and rec["status"] == "ok" and len(rec["completion_ids"]) == len(rec["logprobs"]) > 0,
          f"status={st} model={d.get('model') if isinstance(d, dict) else d} max_tokens={rec['params']['max_tokens']}")

    # --- pause / resume one arm
    st, s1 = await post_json(gw.url("/control/pause"), {"arm": "base"})
    t_req = time.time()
    tb = asyncio.create_task(post_json("http://gw/v1/chat/completions", chat_body("Say hi.", max_tokens=64),
                                       sock=socks["base"]))
    await asyncio.sleep(1.0)
    stv, _ = await post_json("http://gw/v1/chat/completions", chat_body("Say hi.", max_tokens=64), sock=socks["v5"])
    _, s2 = await get_json(gw.url("/control/state"))
    queued_ok = not tb.done() and s2["arms"]["base"]["queued"] == 1 and s2["arms"]["base"]["paused"] \
        and not s2["arms"]["v5"]["paused"]
    await asyncio.sleep(1.0)
    await post_json(gw.url("/control/resume?arm=base"), None)
    stb, _ = await tb
    rec = gw.calls("base")[-1]
    check("B11 pause(base) queues base requests while v5 is served; resume releases them",
          st == 200 and queued_ok and stv == 200 and stb == 200 and rec["queued_s"] >= 1.9,
          f"queued_s={rec['queued_s']} waited={time.time() - t_req:.1f}s")
    _, s3 = await post_json(gw.url("/control/pause"), None)
    _, s4 = await post_json(gw.url("/control/resume"), {})
    st5, _ = await post_json(gw.url("/control/pause"), {"arm": "nope"})
    check("B12 pause/resume all arms; unknown arm -> 404",
          all(a["paused"] for a in s3["arms"].values()) and not any(a["paused"] for a in s4["arms"].values())
          and st5 == 404)

    # --- adapter lost (as after a vLLM restart): unload behind the gateway's back, next call re-registers it
    async with session() as s_:
        async with s_.post(f"{VLLM}/v1/unload_lora_adapter", json={"lora_name": models["base"]}) as r:
            unloaded = r.status == 200
    gone = models["base"] not in await vllm_models()
    st, d = await post_json("http://gw/v1/chat/completions", chat_body("Say hi.", max_tokens=32), sock=socks["base"])
    back = models["base"] in await vllm_models()
    check("B13 404 for a lost adapter -> re-registered, request retried and succeeds",
          unloaded and gone and st == 200 and back and "re-registering adapter" in gw.log(),
          f"unloaded={unloaded} gone={gone} status={st} back={back}")

    # --- client hang-up mid-stream: upstream drained, record complete
    n0 = len(gw.calls("v5"))
    r = await stream_chat("http://gw/v1/chat/completions", chat_body(LONG_PROMPT, stream=True, max_tokens=300),
                          sock=socks["v5"], hang_up_after=20)
    for _ in range(120):
        if len(gw.calls("v5")) > n0:
            break
        await asyncio.sleep(0.5)
    recs = gw.calls("v5")[n0:]
    check("B14 client hang-up mid-stream: upstream drained, full record written",
          len(recs) == 1 and recs[0]["status"] == "ok" and recs[0]["finish_reason"] == "length"
          and recs[0]["n_completion"] == 300 and r["n_tok"] < 300,
          f"client got {r['n_tok']} tokens; record={recs[0]['status'] if recs else None} "
          f"n={recs[0]['n_completion'] if recs else None}")

    # --- decode speed: each arm alone, then both arms concurrently (the fairness criterion)
    speed = {"alone": {}, "concurrent": {n: [] for n in socks}, "client_concurrent": {n: [] for n in socks}}
    body = chat_body(LONG_PROMPT, stream=True, max_tokens=1024)
    base_name = (await get_json(f"{VLLM}/v1/models"))[1]["data"][0]["id"]
    direct = await stream_chat(f"{VLLM}/v1/chat/completions",
                               {**body, "model": base_name, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                                "return_token_ids": True, "cache_salt": uuid.uuid4().hex})
    speed["alone"]["plain_base_direct"] = round(direct.get("tok_s", 0), 2)
    for n in socks:
        k0 = len(gw.calls(n))
        await stream_chat("http://gw/v1/chat/completions", body, sock=socks[n])
        speed["alone"][n] = (await gw.new_calls(n, k0, 1))[0]["decode_tok_s"]
    for rnd in range(3):
        k0 = {n: len(gw.calls(n)) for n in socks}
        outs = await asyncio.gather(*[stream_chat("http://gw/v1/chat/completions", body, sock=socks[n])
                                      for _ in range(2) for n in socks])
        for i, n in enumerate(list(socks) * 2):
            speed["client_concurrent"][n].append(round(outs[i].get("tok_s", 0), 2))
        for n in socks:
            for rec in await gw.new_calls(n, k0[n], 2):
                speed["concurrent"][n].append(rec["decode_tok_s"])
    mean = {n: statistics.mean(v) for n, v in speed["concurrent"].items()}
    ratio = mean["v5"] / mean["base"]
    speed["concurrent_mean"] = {n: round(v, 2) for n, v in mean.items()}
    speed["ratio_v5_over_base"] = round(ratio, 4)
    print(json.dumps(speed, indent=1))
    check("B15 decode tok/s of the two arms within 10% when run concurrently (3 rounds x 2 requests per arm; "
          "the per-arm cap serves them in turn, i.e. 1+1 streams in vLLM as in production)",
          abs(ratio - 1) < 0.10 and all(len(v) == 6 for v in speed["concurrent"].values()),
          f"base={mean['base']:.2f} v5={mean['v5']:.2f} ratio={ratio:.4f}; alone: {speed['alone']}")

    # --- load isolation on the live vLLM: v5 floods with 6 parallel streams while base runs one stream
    flood_body = chat_body(LONG_PROMPT, stream=True, max_tokens=300)
    k0 = {n: len(gw.calls(n)) for n in socks}
    samples = []

    async def sampler():
        while True:
            try:
                _, s_ = await get_json(gw.url("/control/state"))
                samples.append((s_["arms"]["v5"]["active"], s_["arms"]["v5"]["waiting_slot"], await vllm_running()))
            except (aiohttp.ClientError, KeyError, ValueError):
                pass
            await asyncio.sleep(0.5)

    smp = asyncio.create_task(sampler())
    flood = [asyncio.create_task(stream_chat("http://gw/v1/chat/completions", flood_body, sock=socks["v5"]))
             for _ in range(6)]
    await asyncio.sleep(1.0)
    rb = await stream_chat("http://gw/v1/chat/completions", flood_body, sock=socks["base"])
    rs = await asyncio.gather(*flood)
    smp.cancel()
    rec_b = (await gw.new_calls("base", k0["base"], 1))[-1]
    recs_v = await gw.new_calls("v5", k0["v5"], 5)
    codes = sorted(r["status"] for r in rs)
    ref = speed["concurrent_mean"]["base"]
    ratio_f = (rec_b["decode_tok_s"] or 0) / ref
    speed["flood"] = {"base_tok_s": rec_b["decode_tok_s"], "ratio_to_B15_base": round(ratio_f, 4),
                      "v5_codes": codes, "max_v5_active": max((x[0] for x in samples), default=None),
                      "max_vllm_running": max((x[2] for x in samples if x[2] is not None), default=None)}
    check("B16a live flood: 6 parallel v5 streams -> 5 served one at a time + 1 rejected (429), never more than "
          "1 v5 generation in vLLM",
          codes == [200] * 5 + [429] and samples and max(x[0] for x in samples) == 1
          and max(x[1] for x in samples) >= 3 and len(recs_v) == 5 and all(r["status"] == "ok" for r in recs_v),
          f"codes={codes}; sampled v5 active max={speed['flood']['max_v5_active']}, waiting max="
          f"{max((x[1] for x in samples), default=None)}; vLLM num_requests_running max (all clients, "
          f"informational)={speed['flood']['max_vllm_running']}")
    check("B16b live flood: base decodes at its normal two-arm speed (within 10% of B15's concurrent mean)",
          rb["status"] == 200 and rec_b["status"] == "ok" and abs(ratio_f - 1) < 0.10,
          f"base {rec_b['decode_tok_s']} tok/s during the flood vs {ref} in B15 (ratio {ratio_f:.3f})")

    await asyncio.sleep(1)
    _, st = await get_json(gw.url("/control/state"))
    a = st["arms"]
    check("B16 /control/state aggregates match the call records",
          all(a[n]["calls"] == len(gw.calls(n)) and a[n]["in_flight"] == 0
              and a[n]["completion_tokens"] == sum(r["n_completion"] or 0 for r in gw.calls(n))
              and a[n]["decode_tok_s_mean"] and a[n]["max_active"] == 1
              and a[n]["active"] == a[n]["waiting_slot"] == a[n]["draining"] == a[n]["queued"] == 0
              and a[n]["n_overlap"] > 0 and a[n]["alert"] for n in socks)       # B6/B15 overlap on purpose
          and a["v5"]["n_rejected_busy"] == 1 and a["base"]["n_rejected_busy"] == 0,
          json.dumps({n: {k: a[n][k] for k in ("calls", "calls_ok", "completion_tokens", "decode_tok_s_mean",
                                               "decode_tok_s_recent")} for n in socks}))
    rc = gw.stop()
    check("B17 SIGTERM: clean exit, sockets removed",
          rc == 0 and not any(os.path.exists(s) for s in socks.values()), f"rc={rc}")
    errs = [ln for ln in gw.log().splitlines() if " ERROR " in ln or "Traceback" in ln]
    check("B18 no errors/tracebacks in the gateway log", not errs, "\n".join(errs[:5]))
    return speed


# ------------------------------------------------------------------------------------------------------
# Part C: failure paths against a fake vLLM (no GPU): down at startup / restart, upstream death mid-stream
# ------------------------------------------------------------------------------------------------------
FAKE_PORT, FAKE_GW_PORT = 18999, 18111


class FakeVLLM:
    """Minimal OpenAI/vLLM look-alike: adapters registry, 404 for unknown models, scripted streams."""

    def __init__(self):
        self.adapters, self.mode, self.runner, self.bodies = {}, "ok", None, []
        self.disconnected_at = None
        self.active, self.peak, self.started = collections.Counter(), collections.Counter(), []

    async def models(self, _):
        return aiohttp.web.json_response({"data": [{"id": m} for m in ["fake-base", *self.adapters]]})

    async def load(self, req):
        d = await req.json()
        if d["lora_name"] in self.adapters:
            return aiohttp.web.Response(status=400, text=f"The lora adapter '{d['lora_name']}' has already been loaded.")
        self.adapters[d["lora_name"]] = d["lora_path"]
        return aiohttp.web.Response(text="Success")

    async def unload(self, req):
        self.adapters.pop((await req.json())["lora_name"], None)
        return aiohttp.web.Response(text="Success")

    async def chat(self, req):
        body = await req.json()
        self.bodies.append(body)
        if body["model"] not in self.adapters:
            return aiohttp.web.json_response({"error": {"message": f"The model `{body['model']}` does not exist."}},
                                             status=404)
        m = body["model"]
        self.active[m] += 1                      # generations of this adapter running at once
        self.peak[m] = max(self.peak[m], self.active[m])
        self.started.append((time.time(), m))
        try:
            return await self.generate(req, body)
        finally:
            self.active[m] -= 1

    async def generate(self, req, body):
        resp = aiohttp.web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        ev = lambda d: b"data: " + json.dumps(d, separators=(",", ":")).encode() + b"\n\n"
        await resp.write(ev({"model": body["model"], "prompt_token_ids": [1, 2, 3],
                             "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
        if self.mode == "slow":                  # long generation; notice when the gateway drops us
            try:
                for i in range(1000):
                    await asyncio.sleep(0.1)
                    await resp.write(ev({"model": body["model"], "choices": [{"index": 0, "delta": {"content": "x"},
                                         "token_ids": [7], "logprobs": {"content": [{"logprob": -1.0}]}}]}))
            except (ConnectionResetError, aiohttp.ClientConnectionError):
                self.disconnected_at = time.time()
            return resp
        n, dt = (15, 0.1) if self.mode == "steady" else (6, 0.05)      # steady: a 1.5 s generation
        for i in range(n):
            if self.mode == "die" and i == 2:
                req.transport.abort()            # upstream dies mid-stream
                return resp
            await asyncio.sleep(dt)
            await resp.write(ev({"model": body["model"], "choices": [{"index": 0, "delta": {"content": f"t{i} "},
                                 "token_ids": [100 + i], "logprobs": {"content": [{"logprob": -0.5}]},
                                 "finish_reason": "length" if i == n - 1 else None}]}))
        await resp.write(ev({"model": body["model"], "choices": [],
                             "usage": {"prompt_tokens": 3, "completion_tokens": n}}))
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def start(self):
        app = aiohttp.web.Application()
        app.add_routes([aiohttp.web.get("/v1/models", self.models),
                        aiohttp.web.post("/v1/load_lora_adapter", self.load),
                        aiohttp.web.post("/v1/unload_lora_adapter", self.unload),
                        aiohttp.web.post("/v1/chat/completions", self.chat)])
        self.runner = aiohttp.web.AppRunner(app)
        await self.runner.setup()
        await aiohttp.web.TCPSite(self.runner, "127.0.0.1", FAKE_PORT).start()

    async def stop(self):
        await self.runner.cleanup()


async def part_c(gw: Gateway):
    fake = FakeVLLM()
    await gw.start(wait_adapters=False)
    socks = {n: a["gateway_sock"] for n, a in gw.arms.items()}
    body = chat_body("hi", stream=True, max_tokens=50)
    try:
        # vLLM down when the request arrives; it comes up 4 s later without our adapters (fresh restart)
        t0 = time.time()
        task = asyncio.create_task(stream_chat("http://gw/v1/chat/completions", body, sock=socks["v5"]))
        await asyncio.sleep(4)
        await fake.start()
        r = await task
        rec = (await gw.new_calls("v5", 0, 1))[-1]
        check("C1 request during a vLLM outage waits, then succeeds (adapter re-registered after the restart)",
              r["status"] == 200 and r["n_tok"] == 6 and rec["status"] == "ok" and "h2h-v5" in fake.adapters
              and time.time() - t0 >= 4 and b"h2h" not in r["raw"],
              f"waited {time.time() - t0:.1f}s; record {rec['status']} n={rec['n_completion']}")
        for _ in range(40):
            _, st = await get_json(gw.url("/control/state"))
            if all(a["adapter_loaded"] for a in st["arms"].values()):
                break
            await asyncio.sleep(0.5)
        check("C2 keeper registers both adapters once vLLM is back",
              all(a["adapter_loaded"] for a in st["arms"].values()) and set(fake.adapters) >= {"h2h-base0", "h2h-v5"}
              and st["vllm_up"], f"fake adapters={sorted(fake.adapters)}")
        sent = fake.bodies[-1]
        check("C3 upstream request carries the gateway extras (token ids, logprobs, usage)",
              sent["return_token_ids"] is True and sent["logprobs"] is True and sent["top_logprobs"] == 0
              and sent["stream_options"]["include_usage"] is True and sent["model"] == "h2h-v5")
        # upstream dies mid-stream: truncated stream to the client, aborted record, gateway keeps serving
        fake.mode = "die"
        k0 = len(gw.calls("base"))
        r = await stream_chat("http://gw/v1/chat/completions", body, sock=socks["base"])
        rec = (await gw.new_calls("base", k0, 1))[-1]
        fake.mode = "ok"
        r2 = await stream_chat("http://gw/v1/chat/completions", body, sock=socks["base"])
        rec2 = (await gw.new_calls("base", k0, 2))[-1]
        check("C4 upstream dies mid-stream -> aborted record, client stream ends, next call fine",
              rec["status"].startswith("aborted:") and rec["finish_reason"] is None and r["n_tok"] == 2
              and r2["n_tok"] == 6 and rec2["status"] == "ok",
              f"record status={rec['status']} client got {r['n_tok']} tokens; next={rec2['status']}")
        # vLLM restarts again (adapters lost): the next request re-registers via the 404 path
        fake.adapters.clear()
        r = await stream_chat("http://gw/v1/chat/completions", body, sock=socks["base"])
        check("C5 adapters lost in a restart -> 404 -> re-register -> retried once, succeeds",
              r["status"] == 200 and r["n_tok"] == 6 and "h2h-base0" in fake.adapters)
        # client abandons a long generation: drained for H2H_GW_DRAIN_S (3 s here), then upstream is dropped
        fake.mode = "slow"
        k0 = len(gw.calls("v5"))
        r = await stream_chat("http://gw/v1/chat/completions", body, sock=socks["v5"], hang_up_after=5)
        t_hang = time.time()
        rec = (await gw.new_calls("v5", k0, 1))[-1]
        for _ in range(50):
            if fake.disconnected_at:
                break
            await asyncio.sleep(0.1)
        fake.mode = "ok"
        dt = (fake.disconnected_at or 0) - t_hang
        check("C5b abandoned generation: drained ~3 s after the hang-up, then upstream dropped; aborted record",
              rec["status"] == "aborted:client_disconnected" and 2.5 < dt < 6 and 20 < rec["n_completion"] < 100,
              f"upstream dropped {dt:.1f}s after hang-up; record {rec['status']} n={rec['n_completion']}")
        await part_c_isolation(gw, fake, socks, body)
        # SIGTERM with a stream in flight: bounded shutdown, the in-flight call is still recorded
        fake.mode = "slow"
        k0 = len(gw.calls("base"))
        task = asyncio.create_task(stream_chat("http://gw/v1/chat/completions", body, sock=socks["base"]))
        await asyncio.sleep(1.5)
        t0 = time.time()
        rc = await asyncio.get_running_loop().run_in_executor(None, gw.stop)
        t_stop = time.time() - t0
        try:
            client = f"client got {(await task)['n_tok']} tokens"
        except aiohttp.ClientError as e:      # the cut stream ends without its terminating chunk
            client = f"client stream cut ({type(e).__name__})"
        recs = gw.calls("base")[k0:]
        check("C6 SIGTERM with a stream in flight: clean exit within the grace, call recorded as aborted",
              rc == 0 and t_stop < 15 and len(recs) == 1 and recs[0]["status"].startswith("aborted"),
              f"rc={rc} stop took {t_stop:.1f}s; record {recs[0]['status'] if recs else None}; {client}")
    finally:
        rc = gw.stop()
        await fake.stop() if fake.runner else None
    errs = [ln for ln in gw.log().splitlines() if "Traceback" in ln]
    check("C7 no tracebacks in the gateway log", not errs, "\n".join(errs[:5]))


async def part_c_isolation(gw: Gateway, fake: FakeVLLM, socks: dict, body: dict):
    """Per-arm load isolation (MAX_ACTIVE=1, MAX_WAITING=4): flood, abandoned waiters, pause vs slot, body limit."""
    url = "http://gw/v1/chat/completions"
    # one arm floods: 9 parallel requests; base sends one request meanwhile
    fake.mode = "steady"
    fake.peak.clear()
    k0 = {n: len(gw.calls(n)) for n in socks}
    flood = [asyncio.create_task(stream_chat(url, body, sock=socks["v5"])) for _ in range(9)]
    await asyncio.sleep(0.3)
    t_b = time.time()
    tb = asyncio.create_task(stream_chat(url, body, sock=socks["base"]))
    await asyncio.sleep(0.7)
    _, mid = await get_json(gw.url("/control/state"))
    rb = await tb
    dt_b = time.time() - t_b
    rs = await asyncio.gather(*flood)
    await asyncio.sleep(0.3)
    _, after = await get_json(gw.url("/control/state"))
    codes = [r["status"] for r in rs]
    recs = await gw.new_calls("v5", k0["v5"], 5)
    rec_b = (await gw.new_calls("base", k0["base"], 1))[-1]
    mv, av, ab = mid["arms"]["v5"], after["arms"]["v5"], after["arms"]["base"]
    check("C8 flood: 9 parallel v5 requests -> 5 admitted (1 active + 4 waiting), 4 rejected with 429",
          codes.count(200) == 5 and codes.count(429) == 4 and mv["active"] == 1 and mv["waiting_slot"] == 4
          and mv["in_flight"] == 5 and av["n_rejected_busy"] == 4 and av["peak_in_flight"] == 5,
          f"codes={sorted(codes)} mid: active={mv['active']} waiting={mv['waiting_slot']} in_flight={mv['in_flight']}")
    qs = sorted(r["queued_s"] for r in recs)
    check("C9 flooding arm: never more than 1 generation in vLLM; admitted requests served one after another",
          fake.peak["h2h-v5"] == 1 and len(recs) == 5 and all(r["status"] == "ok" for r in recs) and qs[-1] > 5.0,
          f"peak v5 generations in vLLM={fake.peak['h2h-v5']}; queued_s={qs}")
    check("C10 other arm unaffected by the flood: base served at once, full speed",
          rb["status"] == 200 and rb["n_tok"] == 15 and rec_b["status"] == "ok" and rec_b["queued_s"] < 0.3
          and dt_b < 2.5 and fake.peak["h2h-base0"] == 1,
          f"base took {dt_b:.2f}s (alone: 1.5 s), queued_s={rec_b['queued_s']}")
    idle = all(after["arms"][n][k] == 0 for n in socks for k in ("in_flight", "active", "waiting_slot", "draining",
                                                                 "queued"))
    check("C11 /control/state: alert for the flooding arm only; all counters back to idle",
          set(after["alerts"]) == {"v5"} and av["alert"] and av["n_overlap"] >= 4 and ab["alert"] is None
          and ab["n_overlap"] == 0 and ab["n_rejected_busy"] == 0 and av["max_active"] == 1 and idle
          and any("[v5]" in ln and "so far)" in ln for ln in gw.log().splitlines())
          and not any("[base]" in ln and "so far)" in ln for ln in gw.log().splitlines()),
          f"alerts={after['alerts']} idle={idle}")

    # a waiter whose client leaves is dropped by the poll (before the slot frees) and never sent to vLLM;
    # the slot holder hangs up mid-stream and is drained (3 s) while still holding the slot
    fake.mode = "slow"
    fake.disconnected_at = None
    n_bodies, k0 = len(fake.bodies), len(gw.calls("v5"))
    ta = asyncio.create_task(stream_chat(url, body, sock=socks["v5"], hang_up_after=30))
    await asyncio.sleep(0.2)
    try:
        await asyncio.wait_for(stream_chat(url, body, sock=socks["v5"]), 0.5)
    except asyncio.TimeoutError:
        pass
    await asyncio.sleep(3.5)
    _, mid = await get_json(gw.url("/control/state"))
    await ta
    recs = await gw.new_calls("v5", k0, 2, timeout=15)
    fake.mode = "ok"
    st_ = {r["status"]: r for r in recs}
    gone, held = st_.get("aborted:client_gone_before_send"), st_.get("aborted:client_disconnected")
    mv = mid["arms"]["v5"]
    check("C12 waiter whose client left is dropped while the slot is still held and never sent upstream",
          gone and held and gone["ts_end"] < held["ts_end"] - 1 and len(fake.bodies) == n_bodies + 1
          and mv["n_gone_before_send"] == 1 and mv["draining"] == 1 and mv["active"] == 1 and mv["in_flight"] == 1,
          f"records={sorted(st_)} sent upstream={len(fake.bodies) - n_bodies} "
          f"mid: draining={mv['draining']} in_flight={mv['in_flight']}")

    # paused while waiting for the slot: back to the gate, sent only after resume
    fake.mode = "steady"
    ta = asyncio.create_task(stream_chat(url, body, sock=socks["v5"]))
    await asyncio.sleep(0.2)
    tb = asyncio.create_task(stream_chat(url, body, sock=socks["v5"]))
    await asyncio.sleep(0.3)
    await post_json(gw.url("/control/pause"), {"arm": "v5"})
    await asyncio.sleep(2.0)
    _, mid = await get_json(gw.url("/control/state"))
    t_res = time.time()
    await post_json(gw.url("/control/resume"), {"arm": "v5"})
    ra, rb = await ta, await tb
    fake.mode = "ok"
    mv = mid["arms"]["v5"]
    check("C13 pause while waiting for the slot: request returns to the pause gate, sent only after resume",
          ra["status"] == rb["status"] == 200 and mv["queued"] == 1 and mv["active"] == 0 and mv["paused"]
          and fake.started[-1][0] >= t_res,
          f"mid: queued={mv['queued']} active={mv['active']}; "
          f"B started {fake.started[-1][0] - t_res:+.2f}s after resume")

    big = b'{"messages": [{"role": "user", "content": "' + b"x" * (65 << 20) + b'"}]}'
    try:
        st, _ = await post_json(url, None, sock=socks["base"], raw=big)
    except aiohttp.ClientError as e:
        st = type(e).__name__
    _, s = await get_json(gw.url("/control/state"))
    check("C14 request body over 64 MiB rejected (413); arm back to idle",
          st == 413 and s["arms"]["base"]["in_flight"] == 0, f"status={st}")


async def main_async(a) -> int:
    cfg = load_h2h_config()
    if a.part in (None, "a") and not a.real_paths:
        await part_a(cfg)
    if a.part in (None, "b") or a.real_paths:
        gw = Gateway(real_paths=a.real_paths)
        try:
            if a.real_paths:
                await part_b_smoke(gw)
            else:
                await part_b(gw)
        except Exception as e:
            check("B gateway test run", False, f"{type(e).__name__}: {e}")
            raise
        finally:
            gw.stop()
            if a.keep:
                print("kept", gw.tmp)
            else:
                gw.cleanup()
    if a.part in (None, "c") and not a.real_paths:
        gw = Gateway(port=FAKE_GW_PORT, extra={"vllm_url": f"http://127.0.0.1:{FAKE_PORT}"},
                     env={"H2H_GW_DRAIN_S": "3"})
        try:
            await part_c(gw)
        finally:
            gw.stop()
            print("kept", gw.tmp) if a.keep else gw.cleanup()
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("a", "b", "c"))
    ap.add_argument("--real-paths", action="store_true", help="smoke-test the gateway on its production sockets/port")
    ap.add_argument("--keep", action="store_true", help="keep the temporary root (calls.jsonl, gateway.log)")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
