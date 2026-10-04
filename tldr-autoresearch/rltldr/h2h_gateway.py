"""h2h gateway: OpenAI-compatible proxy between the two head-to-head pi sessions and the shared vLLM server.

One process serves both arms (rltldr/h2h_config.py):
  * per arm an UNTRUSTED Unix socket `arm.gateway_sock` (bound into that arm's sandbox, forwarded to 127.0.0.1:8100
    there): only `POST /v1/chat/completions` and `GET /v1/models` (one model, `policy`). Requests are treated
    identically for both arms: fields are whitelisted (SANDBOX_FIELDS), sampling is forced (SANDBOX_SAMPLING),
    `max_tokens` is capped and `model` is rewritten to the arm's fixed LoRA adapter (`h2h-base0` = all-zero adapter
    = exactly the base model, `h2h-v5` = the RLTL;DR v5 policy), so both arms run the same LoRA kernel path. The
    served adapter name is replaced by `policy` in everything sent back, so an agent cannot tell which arm it is.
  * a TRUSTED TCP listener 127.0.0.1:`cfg.gateway_port` (never reachable from a sandbox): `GET /health`,
    `GET /control/state` (per arm: in-flight/queued, calls, tokens, decode tok/s, last call, adapter loaded),
    `POST /control/pause` and `/control/resume` (body or query `arm=<name>`; no arm = all arms; a paused arm's
    requests wait at a gate).
  * one JSON line per call to `arm.calls` (scalar fields first, then prompt/completion token ids and completion
    logprobs, obtained with the same request extras as rltldr/gateway.py: return_token_ids/logprobs/top_logprobs=0).
  * robustness (as rltldr/gateway.py): requests wait up to UPSTREAM_WAIT_S for a down/restarting vLLM, a stream
    with no bytes for STALL_S is aborted, upstream is drained after the client hangs up (bounded by
    DRAIN_AFTER_HANGUP_S so an abandoned generation does not keep loading the server shared with the other arm),
    a 404 / "does not exist" for the adapter re-registers it and retries once. A keeper task registers both adapters
    at startup and after vLLM restarts and unloads stale adapters of the old run (STALE_PREFIXES).
  * per-arm load isolation: the arms share one vLLM, and an agent can reach its socket from its own shell, so a
    flood of parallel requests from one arm would slow the other arm's decoding. pi sends one request at a time
    (compaction summaries included), so each arm gets MAX_ACTIVE (1) upstream slot: further requests wait FIFO
    (MAX_WAITING of them; beyond that, 429 before the body is even read). The slot is held until the upstream
    request is over (including the drain after a hang-up), so an arm never has more than MAX_ACTIVE generations in
    vLLM. A request whose client left while it waited is never sent. Requests that arrive while another request
    of the same arm is still attached are counted and raise a per-arm `alert` in /control/state.
No insights, attempts or policy versions: the arm -> adapter mapping is fixed for the whole run.

  $RLTLDR_SERVE_PY -m rltldr.h2h_gateway        (cwd = the project root; or run the file; normally ./ctl_h2h.sh)
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from collections import deque

import aiohttp
from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rltldr.h2h_config import load_h2h_config  # noqa: E402
from rltldr.io_utils import append_jsonl  # noqa: E402

CFG = load_h2h_config()
UPSTREAM_WAIT_S = 1800          # how long a request waits for a dead/restarting vLLM before failing
STALL_S = 900                   # a stream with no bytes for this long is treated as a wedged server
# keep reading upstream this long after the client hung up (to complete the record); env override for tests
DRAIN_AFTER_HANGUP_S = float(os.environ.get("H2H_GW_DRAIN_S", 60))
KEEPER_PERIOD_S = 60            # adapter registration check period
STALE_PREFIXES = ("oct3-",)     # adapters of the old RLTL;DR run: unloaded at startup
PUBLIC_MODEL = "policy"
# Same treatment as rltldr/gateway.py for sandbox requests: only the fields pi sends, fixed sampling.
SANDBOX_FIELDS = {"model", "messages", "tools", "tool_choice", "stream", "stream_options", "max_tokens",
                  "reasoning_effort", "parallel_tool_calls"}
SANDBOX_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                    "repetition_penalty": 1.0, "n": 1}
SANDBOX_MAX_TOKENS = 32768
RECENT_N = 20                   # window for the recent decode-speed figure in /control/state
SHUTDOWN_S = 5                  # grace for in-flight requests on SIGTERM
MIN_DECODE_TOKENS = 16          # calls shorter than this do not enter the decode-speed statistics
# per-arm load isolation (env overrides for tests): concurrent upstream generations per arm, requests that may wait
# for one, and the body size limit (a 131k-token context is ~1 MB of JSON; bounds the gateway's memory under a flood)
MAX_ACTIVE = max(1, int(os.environ.get("H2H_GW_MAX_ACTIVE", 1)))
MAX_WAITING = max(0, int(os.environ.get("H2H_GW_MAX_WAITING", 4)))
MAX_BODY_BYTES = 64 << 20
WAIT_POLL_S = 2.0               # waiting requests re-check that their client is still connected this often
ALERT_LOG_S = 60                # at most one overlap warning per arm per this many seconds
log = logging.getLogger("h2h_gateway")
SESSION: aiohttp.ClientSession = None


class ArmState:
    def __init__(self, arm):
        self.arm = arm
        self.gate = asyncio.Event()
        self.gate.set()
        self.reg_lock = asyncio.Lock()
        self.adapter_loaded = False
        self.slots = asyncio.Semaphore(MAX_ACTIVE)   # upstream generations of this arm (FIFO)
        self.in_flight = 0          # requests being handled (including queued ones)
        self.queued = 0             # requests waiting at the pause gate
        self.waiting_slot = 0       # requests waiting for an upstream slot
        self.active = 0             # requests holding an upstream slot
        self.draining = 0           # active requests whose client already hung up
        self.peak_in_flight = 0
        self.n_overlap = 0          # requests that arrived while another request of this arm was attached
        self.n_slot_waits = 0       # requests that had to wait for an upstream slot
        self.n_rejected = 0         # 429: too many pending requests
        self.n_gone_before_send = 0  # client left while waiting; never sent upstream
        self.last_overlap_ts = None
        self._last_warn = 0.0
        self.calls = self.calls_ok = self.calls_err = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.decode_tokens, self.decode_s = 0, 0.0
        self.recent = deque(maxlen=RECENT_N)   # (decode tokens, decode seconds) of recent calls
        self.last_call_ts = None
        self.last_status = None

    @property
    def paused(self) -> bool:
        return not self.gate.is_set()

    def account(self, rec: dict) -> None:
        self.calls += 1
        ok = rec["status"].startswith("ok")
        self.calls_ok += ok
        self.calls_err += not ok
        self.prompt_tokens += rec.get("n_prompt") or 0
        self.completion_tokens += rec.get("n_completion") or 0
        if rec.get("decode_tok_s") and (rec.get("n_completion") or 0) >= MIN_DECODE_TOKENS:
            n, s = rec["_decode_n"], rec["_decode_s"]
            self.decode_tokens += n
            self.decode_s += s
            self.recent.append((n, s))
        self.last_call_ts = rec["ts_end"]
        self.last_status = rec["status"]

    def note_overlap(self, what: str) -> None:
        """More than one request of this arm at once: not what a single pi session does (alert + rate-limited log)."""
        now = time.time()
        self.last_overlap_ts = now
        if now - self._last_warn >= ALERT_LOG_S:
            self._last_warn = now
            log.warning("[%s] %s: in_flight=%d active=%d waiting=%d draining=%d (overlapping=%d rejected=%d so far)",
                        self.arm.name, what, self.in_flight, self.active, self.waiting_slot, self.draining,
                        self.n_overlap, self.n_rejected)

    def alert(self):
        if not (self.n_overlap or self.n_rejected):
            return None
        return (f"{self.n_overlap} overlapping requests, {self.n_rejected} rejected (429); a single pi session "
                f"sends one request at a time; last at {time.strftime('%F %T', time.localtime(self.last_overlap_ts))}")

    def snapshot(self) -> dict:
        rn, rs = sum(n for n, _ in self.recent), sum(s for _, s in self.recent)
        return {"served_model": self.arm.served_model, "adapter_dir": self.arm.adapter_dir,
                "adapter_loaded": self.adapter_loaded, "paused": self.paused, "in_flight": self.in_flight,
                "queued": self.queued, "calls": self.calls, "calls_ok": self.calls_ok,
                "calls_err": self.calls_err, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "decode_tok_s_mean": round(self.decode_tokens / self.decode_s, 2) if self.decode_s else None,
                "decode_tok_s_recent": round(rn / rs, 2) if rs else None,
                "last_call_ts": self.last_call_ts, "last_status": self.last_status,
                "max_active": MAX_ACTIVE, "max_pending": MAX_ACTIVE + MAX_WAITING, "active": self.active,
                "waiting_slot": self.waiting_slot, "draining": self.draining, "peak_in_flight": self.peak_in_flight,
                "n_overlap": self.n_overlap, "n_slot_waits": self.n_slot_waits, "n_rejected_busy": self.n_rejected,
                "n_gone_before_send": self.n_gone_before_send, "last_overlap_ts": self.last_overlap_ts,
                "alert": self.alert()}


ARMS: dict = {}          # name -> ArmState (created inside the event loop)
STARTED_AT = time.time()
VLLM_UP = None           # last known vLLM reachability (keeper)


# ------------------------------------------------------------------------------------------------------
# vLLM helpers
# ------------------------------------------------------------------------------------------------------
async def vllm_served_models() -> set:
    async with SESSION.get(f"{CFG.vllm_url}/v1/models", timeout=aiohttp.ClientTimeout(total=10)) as r:
        d = await r.json()
    return {m["id"] for m in d.get("data", [])}


async def vllm_load_adapter(name: str, path: str) -> bool:
    try:
        async with SESSION.post(f"{CFG.vllm_url}/v1/load_lora_adapter",
                                json={"lora_name": name, "lora_path": path},
                                timeout=aiohttp.ClientTimeout(total=600)) as r:
            txt = await r.text()
            if r.status == 200 or "already" in txt.lower():
                return True
            log.error("load_lora_adapter %s failed: %s %s", name, r.status, txt[:300])
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.error("load_lora_adapter %s error: %s", name, e)
    return False


async def vllm_unload_adapter(name: str) -> bool:
    try:
        async with SESSION.post(f"{CFG.vllm_url}/v1/unload_lora_adapter", json={"lora_name": name},
                                timeout=aiohttp.ClientTimeout(total=120)) as r:
            txt = await r.text()
            if r.status != 200:
                log.warning("unload_lora_adapter %s: %s %s", name, r.status, txt[:200])
            return r.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.warning("unload_lora_adapter %s error: %s", name, e)
        return False


async def register_adapter(st: ArmState, force: bool = False) -> bool:
    """Make sure the arm's adapter is registered in vLLM (force: re-register without asking first)."""
    async with st.reg_lock:
        if not force:
            try:
                if st.arm.served_model in await vllm_served_models():
                    st.adapter_loaded = True
                    return True
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                st.adapter_loaded = False
                return False
        ok = await vllm_load_adapter(st.arm.served_model, st.arm.adapter_dir)
        st.adapter_loaded = ok
        if ok:
            log.info("adapter %s registered from %s", st.arm.served_model, st.arm.adapter_dir)
        return ok


async def keeper() -> None:
    """Register both adapters (startup, vLLM restarts) and unload stale adapters of the old run."""
    global VLLM_UP
    while True:
        try:
            served = await vllm_served_models()
            if not VLLM_UP:
                log.info("vLLM reachable; served models: %s", sorted(served))
            VLLM_UP = True
            for name in sorted(served):
                if name.startswith(STALE_PREFIXES):
                    ok = await vllm_unload_adapter(name)
                    log.info("unloaded stale adapter %s: %s", name, ok)
            for st in ARMS.values():
                if st.arm.served_model in served:
                    st.adapter_loaded = True
                else:
                    st.adapter_loaded = False
                    await register_adapter(st, force=True)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            if VLLM_UP is not False:
                log.warning("vLLM unreachable (%s); keeper will retry", e)
            VLLM_UP = False
            for st in ARMS.values():
                st.adapter_loaded = False
        await asyncio.sleep(KEEPER_PERIOD_S if VLLM_UP else 10)


# ------------------------------------------------------------------------------------------------------
# request rewriting / response scrubbing
# ------------------------------------------------------------------------------------------------------
class BadRequest(Exception):
    pass


def sanitize(body, arm) -> dict:
    """Untrusted request -> what is sent to vLLM (identical rules for both arms)."""
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    body = {k: v for k, v in body.items() if k in SANDBOX_FIELDS}
    if not isinstance(body.get("messages"), list):
        raise BadRequest("messages must be a list")
    body.update(SANDBOX_SAMPLING)
    try:
        mt = int(body.get("max_tokens") or SANDBOX_MAX_TOKENS)
    except (TypeError, ValueError):
        raise BadRequest("max_tokens must be an integer")
    body["max_tokens"] = min(mt, SANDBOX_MAX_TOKENS)
    body["model"] = arm.served_model
    body["return_token_ids"] = True
    body["logprobs"] = True
    body["top_logprobs"] = 0
    if body.get("tools") == []:            # vLLM >= 0.28 rejects an empty tools list
        body.pop("tools")
    if body.get("stream"):
        so = body.get("stream_options")
        body["stream_options"] = {**(so if isinstance(so, dict) else {}), "include_usage": True}
    return body


def scrub_bytes(data: bytes, arm) -> bytes:
    """Replace the arm's adapter name by the public model name in an upstream JSON payload."""
    return data.replace(b'"model":"' + arm.served_model.encode() + b'"', b'"model":"' + PUBLIC_MODEL.encode() + b'"')


def scrub_text(txt: str, arm) -> str:
    return txt.replace(arm.served_model, PUBLIC_MODEL)


def merge_tool_call_deltas(acc: dict, deltas: list) -> None:
    for d in deltas or []:
        i = d.get("index", 0)
        tc = acc.setdefault(i, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        if d.get("id"):
            tc["id"] = d["id"]
        f = d.get("function") or {}
        if f.get("name"):
            tc["function"]["name"] += f["name"]
        if f.get("arguments"):
            tc["function"]["arguments"] += f["arguments"]


# ------------------------------------------------------------------------------------------------------
# main proxy endpoint (untrusted, one app per arm)
# ------------------------------------------------------------------------------------------------------
def client_gone(request: web.Request) -> bool:
    t = request.transport
    return t is None or t.is_closing()


async def wait_attached(request: web.Request, aw_factory, undo=None) -> bool:
    """Await aw_factory() while the client stays connected, checking it every WAIT_POLL_S (aiohttp does not cancel
    a handler whose client left). True: it completed; False: the client left first (the wait is abandoned). The
    awaitable keeps its place in line while we poll. `undo` gives back what it obtained if we are cancelled
    just as it completed."""
    t = asyncio.ensure_future(aw_factory())
    try:
        while True:
            await asyncio.wait({t}, timeout=WAIT_POLL_S)
            if t.done():
                t.result()
                return True
            if client_gone(request):
                t.cancel()              # a cancelled Semaphore.acquire gives back a slot it was just handed
                return False
    except BaseException:
        if not t.done():
            t.cancel()
        elif undo is not None and not t.cancelled() and t.exception() is None:
            undo()
        raise


async def take_slot(request: web.Request, st: ArmState) -> bool:
    """Pass the arm's pause gate, then take one of its MAX_ACTIVE upstream slots (FIFO). False: the client left
    while waiting (no slot held). The caller releases the slot with release_slot()."""
    counted = False
    while True:
        if st.paused:
            st.queued += 1
            try:
                if not await wait_attached(request, st.gate.wait):
                    return False
            finally:
                st.queued -= 1
        if st.slots.locked() and not counted:
            st.n_slot_waits += 1
            counted = True
        st.waiting_slot += 1
        try:
            if not await wait_attached(request, st.slots.acquire, undo=st.slots.release):
                return False
        finally:
            st.waiting_slot -= 1
        if not st.paused:
            break
        st.slots.release()              # paused while waiting for the slot: back to the gate
    if client_gone(request):
        st.slots.release()
        return False
    st.active += 1
    return True


def release_slot(st: ArmState) -> None:
    st.active -= 1
    st.slots.release()


def new_record(st: ArmState, body: dict, ts_start: float) -> dict:
    return {"call_id": uuid.uuid4().hex[:12], "arm": st.arm.name, "served_model": st.arm.served_model,
            "ts_start": ts_start, "ts_end": None, "status": None, "finish_reason": None, "n_prompt": None,
            "n_completion": None, "ttft_s": None, "decode_tok_s": None,
            "queued_s": round(time.time() - ts_start, 3), "stream": bool(body.get("stream")),
            "params": {k: body.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens",
                                                "reasoning_effort", "presence_penalty", "repetition_penalty", "n")}}


async def chat_completions(request: web.Request) -> web.StreamResponse:
    st: ArmState = request.app["arm_state"]
    ts_start = time.time()
    if st.in_flight >= MAX_ACTIVE + MAX_WAITING:     # before the body is read: bounded memory under a flood
        st.n_rejected += 1
        st.note_overlap("request rejected (429)")
        return web.json_response({"error": {"message": "too many concurrent requests; send one request at a time",
                                            "type": "RateLimitError", "code": 429}},
                                 status=429, headers={"Retry-After": "5"})
    st.in_flight += 1
    st.peak_in_flight = max(st.peak_in_flight, st.in_flight)
    try:
        try:
            body = sanitize(await request.json(), st.arm)
        except (BadRequest, json.JSONDecodeError, UnicodeDecodeError) as e:
            return web.json_response({"error": {"message": f"bad request: {e}", "type": "BadRequestError",
                                                "code": 400}}, status=400)
        if st.in_flight - st.draining > 1:          # another request of this arm still has its client attached
            st.n_overlap += 1
            st.note_overlap("overlapping request")
        if not await take_slot(request, st):
            st.n_gone_before_send += 1
            rec = new_record(st, body, ts_start)
            rec.update(status="aborted:client_gone_before_send")
            finish(st, rec)
            return web.Response(status=499, text="client closed request")
        rec = new_record(st, body, ts_start)
        try:
            return await proxy(request, st, body, rec)
        finally:
            if rec.pop("_draining", False):
                st.draining -= 1
            release_slot(st)
    finally:
        st.in_flight -= 1


async def proxy(request: web.Request, st: ArmState, body: dict, rec: dict) -> web.StreamResponse:
    """Forward one sanitised request to vLLM (the caller holds an upstream slot of the arm) and record it."""
    arm = st.arm
    stream = rec["stream"]
    t_send = time.time()
    upstream = await open_upstream(body, st)
    if isinstance(upstream, web.Response):      # gave up waiting for vLLM
        rec.update(status="upstream_unavailable")
        finish(st, rec)
        return upstream
    try:
        if upstream.status != 200:
            txt = scrub_text(await upstream.text(), arm)
            rec.update(status=f"http_{upstream.status}", error=txt[:2000])
            finish(st, rec)
            return web.Response(status=upstream.status, text=txt, content_type="application/json")
        if not stream:
            try:
                d = await upstream.json()
                c = d["choices"][0]
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, IndexError) as e:
                rec.update(status=f"aborted:{type(e).__name__}", error=str(e)[:2000])
                finish(st, rec)
                return web.json_response({"error": f"upstream failed: {type(e).__name__}"}, status=502)
            lps = [x["logprob"] for x in ((c.get("logprobs") or {}).get("content") or [])]
            rec.update(status="ok", finish_reason=c.get("finish_reason"), usage=d.get("usage"),
                       response=c.get("message"), prompt_ids=d.get("prompt_token_ids"),
                       completion_ids=c.get("token_ids"), logprobs=lps)
            finish(st, rec)
            d["model"] = PUBLIC_MODEL
            return web.json_response(d)
        return await relay_stream(request, upstream, st, rec, t_send)
    finally:
        upstream.release()


async def open_upstream(body: dict, st: ArmState):
    """POST to vLLM, waiting out restarts and re-registering a missing adapter once."""
    deadline = time.time() + UPSTREAM_WAIT_S
    delay, reloaded = 2.0, False
    while True:
        try:
            r = await SESSION.post(f"{CFG.vllm_url}/v1/chat/completions", json=body,
                                   timeout=aiohttp.ClientTimeout(total=None, sock_connect=10))
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            if time.time() > deadline:
                return web.json_response({"error": f"vLLM unavailable: {e}"}, status=503)
            log.warning("[%s] vLLM unreachable (%s); retrying in %.0fs", st.arm.name, e, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 30)
            continue
        if r.status in (400, 404) and not reloaded:
            txt = await r.text()
            low = txt.lower()
            if r.status == 404 or (st.arm.served_model in txt and ("does not exist" in low or "not found" in low)):
                r.release()
                log.warning("[%s] %s for %s (%s); re-registering adapter", st.arm.name, r.status,
                            st.arm.served_model, txt[:200])
                await register_adapter(st, force=True)
                reloaded = True
                continue
            return _Replayed(r.status, txt)
        return r


class _Replayed:
    """An already-read upstream error response (looks enough like aiohttp's ClientResponse for the caller)."""
    def __init__(self, status: int, txt: str):
        self.status, self._txt = status, txt

    async def text(self) -> str:
        return self._txt

    def release(self) -> None:
        pass


async def relay_stream(request, upstream, st: ArmState, rec: dict, t_send: float):
    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                   "Cache-Control": "no-cache"})
    client_gone_at = None
    try:
        await resp.prepare(request)
    except (ConnectionResetError, aiohttp.ClientConnectionError):
        client_gone_at = time.time()        # left before the headers went out: drained like any hang-up
        mark_draining(st, rec)
    arm = st.arm
    prompt_ids, ids, lps, fin, usage = None, [], [], None, None
    content, reasoning, tool_acc = [], [], {}
    t_first = t_last = None
    n_first = 0
    buf = b""
    status = "ok"
    eof = False
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(upstream.content.readany(), timeout=STALL_S)
            except asyncio.TimeoutError:
                status = "upstream_stalled"
                log.error("[%s] stream for call %s stalled for %ds; aborting", arm.name, rec["call_id"], STALL_S)
                break
            if not chunk:
                eof = True
                break
            now = time.time()
            buf += chunk
            # forward complete SSE events only (so the adapter name can be scrubbed reliably)
            cut = buf.rfind(b"\n\n")
            if cut < 0:
                continue
            events, buf = buf[:cut + 2], buf[cut + 2:]
            if client_gone_at is None:
                try:
                    await resp.write(scrub_bytes(events, arm))
                except (ConnectionResetError, aiohttp.ClientConnectionError):
                    # pi hangs up right after the final event; keep draining so the record is complete (a genuine
                    # abort is detected below by the missing finish_reason)
                    client_gone_at = now
                    mark_draining(st, rec)
            for ev in events.split(b"\n\n"):
                for line in ev.split(b"\n"):
                    if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                        continue
                    try:
                        d = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    prompt_ids = prompt_ids or d.get("prompt_token_ids")
                    usage = d.get("usage") or usage
                    for c in d.get("choices", []):
                        tids = c.get("token_ids") or []
                        if tids:
                            if t_first is None:
                                t_first, n_first = now, len(tids)
                            t_last = now
                        ids += tids
                        lps += [x["logprob"] for x in ((c.get("logprobs") or {}).get("content") or [])]
                        fin = c.get("finish_reason") or fin
                        dl = c.get("delta") or {}
                        if dl.get("content"):
                            content.append(dl["content"])
                        r = dl.get("reasoning") or dl.get("reasoning_content")
                        if r:
                            reasoning.append(r)
                        merge_tool_call_deltas(tool_acc, dl.get("tool_calls"))
            if client_gone_at is not None:
                if fin is not None and usage is not None:
                    break                # everything needed is recorded
                if now - client_gone_at > DRAIN_AFTER_HANGUP_S:
                    break                # abandoned generation: stop loading the shared server
        if client_gone_at is None and status == "ok":
            try:
                if buf:
                    await resp.write(scrub_bytes(buf, arm))
                await resp.write_eof()
            except (ConnectionResetError, aiohttp.ClientConnectionError):
                pass
        if client_gone_at is not None and fin is None:
            status = "aborted:client_disconnected"
    except (ConnectionResetError, aiohttp.ClientError, asyncio.CancelledError) as e:
        # upstream died (connection lost / payload error), or the request task was cancelled (shutdown)
        status = f"aborted:{type(e).__name__}"
    if not eof:
        upstream.close()        # drop the connection so vLLM aborts the (stalled/abandoned) generation
    if t_first is not None:
        rec["ttft_s"] = round(t_first - t_send, 4)
    if t_first is not None and t_last > t_first and len(ids) - n_first > 0:
        rec["_decode_n"], rec["_decode_s"] = len(ids) - n_first, t_last - t_first
        rec["decode_tok_s"] = round(rec["_decode_n"] / rec["_decode_s"], 3)
    rec.update(status=status, finish_reason=fin, usage=usage,
               response={"role": "assistant", "content": "".join(content) or None,
                         "reasoning": "".join(reasoning) or None,
                         "tool_calls": [tool_acc[i] for i in sorted(tool_acc)] or None},
               prompt_ids=prompt_ids, completion_ids=ids, logprobs=lps)
    if status == "ok" and len(ids) != len(lps):
        rec["status"] = "ok_logprob_mismatch"
    finish(st, rec)
    return resp


def mark_draining(st: ArmState, rec: dict) -> None:
    """The client hung up but the request still holds its upstream slot (not an attached client any more)."""
    if not rec.get("_draining"):
        rec["_draining"] = True
        st.draining += 1


def finish(st: ArmState, rec: dict) -> None:
    """Complete, account and append one call record (scalar fields first, bulky token lists last)."""
    rec["ts_end"] = time.time()
    usage = rec.get("usage") or {}
    if rec.get("prompt_ids") is not None or usage:
        rec["n_prompt"] = len(rec["prompt_ids"]) if rec.get("prompt_ids") else usage.get("prompt_tokens")
    if rec.get("completion_ids") is not None or usage:
        rec["n_completion"] = len(rec["completion_ids"]) if rec.get("completion_ids") else \
            usage.get("completion_tokens")
    st.account(rec)
    out = {k: v for k, v in rec.items() if not k.startswith("_")}
    for k in ("prompt_ids", "completion_ids", "logprobs"):     # keep the bulky fields at the end of the line
        if k in out:
            out[k] = out.pop(k)
    try:
        append_jsonl(st.arm.calls, out)
    except OSError as e:
        log.error("[%s] cannot record call %s: %s", st.arm.name, rec["call_id"], e)
    log.info("[%s] call %s %s fin=%s prompt=%s completion=%s ttft=%s tok/s=%s", st.arm.name, rec["call_id"],
             rec["status"], rec.get("finish_reason"), rec.get("n_prompt"), rec.get("n_completion"),
             rec.get("ttft_s"), rec.get("decode_tok_s"))


async def models(_):
    return web.json_response({"object": "list", "data": [
        {"id": PUBLIC_MODEL, "object": "model", "owned_by": "local", "root": PUBLIC_MODEL}]})


# ------------------------------------------------------------------------------------------------------
# trusted control endpoints
# ------------------------------------------------------------------------------------------------------
async def health(_):
    return web.json_response({"ok": True})


def snapshot() -> dict:
    alerts = {name: a for name, a in ((n, st.alert()) for n, st in ARMS.items()) if a}
    return {"started_at": STARTED_AT, "pid": os.getpid(), "vllm_url": CFG.vllm_url, "vllm_up": VLLM_UP,
            "alerts": alerts, "arms": {name: st.snapshot() for name, st in ARMS.items()}}


async def ctl_state(_):
    return web.json_response(snapshot())


async def _target_arms(request) -> list:
    name = request.query.get("arm")
    if name is None and request.can_read_body:
        try:
            d = await request.json()
            name = d.get("arm") if isinstance(d, dict) else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    if name in (None, "", "all"):
        return list(ARMS.values())
    if name not in ARMS:
        raise web.HTTPNotFound(text=json.dumps({"error": f"unknown arm {name!r}"}), content_type="application/json")
    return [ARMS[name]]


async def ctl_pause(request):
    for st in await _target_arms(request):
        st.gate.clear()
        log.info("[%s] paused", st.arm.name)
    return web.json_response(snapshot())


async def ctl_resume(request):
    for st in await _target_arms(request):
        st.gate.set()
        log.info("[%s] resumed", st.arm.name)
    return web.json_response(snapshot())


# ------------------------------------------------------------------------------------------------------
# server
# ------------------------------------------------------------------------------------------------------
def make_sandbox_app(st: ArmState) -> web.Application:
    app = web.Application(client_max_size=MAX_BODY_BYTES)
    app["arm_state"] = st
    app.add_routes([web.post("/v1/chat/completions", chat_completions), web.get("/v1/models", models)])
    return app


def make_control_app() -> web.Application:
    app = web.Application()
    app.add_routes([web.get("/health", health), web.get("/control/state", ctl_state),
                    web.post("/control/pause", ctl_pause), web.post("/control/resume", ctl_resume)])
    return app


async def serve():
    global SESSION
    SESSION = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
    for arm in CFG.arms:
        ARMS[arm.name] = ArmState(arm)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    runners, socks = [], []
    try:
        r = web.AppRunner(make_control_app(), access_log=None, shutdown_timeout=SHUTDOWN_S)
        await r.setup()
        await web.TCPSite(r, "127.0.0.1", CFG.gateway_port).start()
        runners.append(r)
        for st in ARMS.values():
            sock = st.arm.gateway_sock
            os.makedirs(os.path.dirname(sock), exist_ok=True)
            os.chmod(os.path.dirname(sock), 0o755)
            if os.path.exists(sock):
                os.unlink(sock)
            r = web.AppRunner(make_sandbox_app(st), access_log=None, shutdown_timeout=SHUTDOWN_S)
            await r.setup()
            await web.UnixSite(r, sock).start()
            os.chmod(sock, 0o666)
            runners.append(r)
            socks.append(sock)
        log.info("listening on 127.0.0.1:%d (trusted) and %s (sandbox)", CFG.gateway_port,
                 ", ".join(f"{st.arm.name}={st.arm.gateway_sock}->{st.arm.served_model}" for st in ARMS.values()))
        keep = asyncio.create_task(keeper())
        await stop.wait()
        log.info("shutting down")
        keep.cancel()
    finally:
        # in parallel: each runner gives its in-flight requests SHUTDOWN_S before cancelling them
        await asyncio.gather(*(r.cleanup() for r in runners), return_exceptions=True)
        for sock in socks:
            try:
                os.unlink(sock)
            except OSError:
                pass
        await SESSION.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(serve())


if __name__ == "__main__":
    main()
