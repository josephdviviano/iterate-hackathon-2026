#!/usr/bin/env python3
"""Logging reverse proxy for an OpenAI-compatible (vLLM) server.

Sits between pi and vLLM and records, per /v1/chat/completions call, the exact
request body pi sent plus what vLLM streamed back (text, reasoning, tool-call
deltas, usage, timings). With --token-ids it also injects
`return_token_ids: true` into the request so vLLM returns `prompt_token_ids`
(first chunk) and per-chunk `choices[0].token_ids`; these are recorded so that
prefix-extension between consecutive calls can be checked in token space
(see check_prefix.py).

The response bytes are forwarded unchanged (pi ignores the extra vLLM fields).

Usage:
  python capture_proxy.py --listen 127.0.0.1:8199 --upstream http://127.0.0.1:8000 \
      --out calls.jsonl --token-ids
"""
import argparse
import asyncio
import json
import os
import time

import aiohttp
from aiohttp import web

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "content-length", "host", "content-encoding"}


class CallRecorder:
    """Accumulates one streamed chat completion from its SSE `data:` payloads."""

    def __init__(self, idx, request_body):
        self.rec = {
            "idx": idx, "t_start": time.time(), "t_first_chunk": None, "t_first_token": None, "t_end": None,
            "request": request_body, "status": None, "prompt_token_ids": None, "completion_token_ids": [],
            "reasoning": "", "content": "", "tool_calls": {}, "finish_reason": None, "usage": None,
            "n_chunks": 0, "error_body": None,
        }

    def feed(self, payload: dict):
        r = self.rec
        now = time.time()
        r["n_chunks"] += 1
        if r["t_first_chunk"] is None:
            r["t_first_chunk"] = now
        if payload.get("prompt_token_ids") is not None:
            r["prompt_token_ids"] = payload["prompt_token_ids"]
        if payload.get("usage"):
            r["usage"] = payload["usage"]
        for ch in payload.get("choices") or []:
            ids = ch.get("token_ids")
            if ids:
                if r["t_first_token"] is None:
                    r["t_first_token"] = now
                r["completion_token_ids"].extend(ids)
            delta = ch.get("delta") or {}
            for key in ("reasoning", "reasoning_content"):
                if delta.get(key):
                    r["reasoning"] += delta[key]
            if delta.get("content"):
                r["content"] += delta["content"]
            for tc in delta.get("tool_calls") or []:
                slot = r["tool_calls"].setdefault(str(tc.get("index", 0)), {"id": None, "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                r["finish_reason"] = ch["finish_reason"]


class Proxy:
    def __init__(self, upstream, out_path, token_ids):
        self.upstream = upstream.rstrip("/")
        self.out_path = out_path
        self.token_ids = token_ids
        self.counter = 0
        self.lock = asyncio.Lock()
        self.session = None

    async def start(self, app):
        # No total/read timeout: a paused upstream or a long generation must not be cut by the proxy.
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=None))

    async def stop(self, app):
        await self.session.close()

    async def write(self, rec):
        async with self.lock:
            with open(self.out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    async def handle(self, request: web.Request):
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        url = self.upstream + request.path_qs
        is_chat = request.method == "POST" and request.path.endswith("/chat/completions")
        recorder = None
        if is_chat:
            payload = json.loads(body)
            self.counter += 1
            recorder = CallRecorder(self.counter, json.loads(body))
            if self.token_ids:
                payload["return_token_ids"] = True
            body = json.dumps(payload).encode()
        status = None
        raw_error = b""
        client_gone = False
        try:
            async with self.session.request(request.method, url, data=body, headers=headers) as up:
                status = up.status
                resp = web.StreamResponse(status=up.status, headers={k: v for k, v in up.headers.items()
                                                                      if k.lower() not in HOP_BY_HOP})
                await resp.prepare(request)
                buf = b""
                async for chunk in up.content.iter_any():
                    try:
                        await resp.write(chunk)
                    except (ConnectionResetError, aiohttp.ClientConnectionResetError):
                        # Client went away (aborted). Stop reading; closing upstream aborts the generation.
                        client_gone = True
                        break
                    if recorder is None:
                        continue
                    if up.status != 200:
                        raw_error += chunk
                        continue
                    buf += chunk
                    # SSE events are separated by blank lines; a data line may be split across TCP chunks.
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if line.startswith(b"data:"):
                            data = line[5:].strip()
                            if data and data != b"[DONE]":
                                try:
                                    recorder.feed(json.loads(data))
                                except json.JSONDecodeError:
                                    pass
                try:
                    await resp.write_eof()
                except (ConnectionResetError, aiohttp.ClientConnectionResetError):
                    # pi closes the socket as soon as it has the final SSE event, often before the
                    # chunked-encoding terminator is written. The response is complete; ignore.
                    pass
            return resp
        finally:
            if recorder is not None:
                recorder.rec["status"] = status
                recorder.rec["t_end"] = time.time()
                recorder.rec["client_disconnected_early"] = client_gone and recorder.rec["finish_reason"] is None
                if raw_error:
                    recorder.rec["error_body"] = raw_error.decode(errors="replace")[:4000]
                await self.write(recorder.rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:8199")
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True, help="JSONL file, one record per chat completion call")
    ap.add_argument("--token-ids", action="store_true", help="inject return_token_ids:true into requests")
    args = ap.parse_args()
    host, port = args.listen.rsplit(":", 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    proxy = Proxy(args.upstream, args.out, args.token_ids)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_route("*", "/{tail:.*}", proxy.handle)
    web.run_app(app, host=host, port=int(port), access_log=None, print=None)


if __name__ == "__main__":
    main()
