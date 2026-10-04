#!/usr/bin/env python3
"""Fakes for the h2h supervisor tests (serve env: aiohttp). No GPU, no vLLM.

  h2h_supervisor_fakes.py --llm-sock S --runner-sock S2 --runner-port P --log calls.jsonl [--control ctl.json]

* A scripted OpenAI-compatible chat server on the Unix socket `--llm-sock` (what the arm's gateway socket would
  be): `GET /v1/models`, `POST /v1/chat/completions` (SSE streaming like vLLM: reasoning deltas, text deltas, one
  tool call with chunked arguments, finish_reason, a final usage chunk, `[DONE]`). The reply is chosen from the
  conversation so that a pi session exercises the supervisor:
    - last message is a tool result                   -> short text, stop  (the agent run ends -> settled)
    - last user prompt asks to kick off / restart     -> one `bash` tool call running `./run.sh "<desc>"`
    - any other prompt (the "continue" nudge)         -> text only (no run: drives the nudge back-off), or one
                                                         `bash` call running `nudge_cmd` from the control file
  With `{"stall": true}` in the control file, chat requests hang (no bytes) until the client disconnects.
* A fake experiment runner: `POST /run` on `--runner-sock` (sleeps `run_s` from the control file, default 0, then
  returns an upstream-style summary) and `GET /control/state` on 127.0.0.1:`--runner-port`, shaped like
  h2h_runner's ({"arm", "busy", "job", "queued", "n_runs", "last"}); `{"state_down": true}` makes it answer 503
  (the supervisor then treats the runner as unreachable).
Every chat request is appended to `--log` (one JSON line: ts, n_messages, last_role, last_user, tools, model).
"""
import argparse
import asyncio
import json
import os
import time

from aiohttp import web

ARGS = None
JOB = {"job": None, "n_runs": 0, "last": None}


def control() -> dict:
    try:
        with open(ARGS.control) as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return {}


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def chunk(delta=None, finish=None, usage=None):
    c = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()), "model": "policy",
         "choices": [] if usage else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return b"data: " + json.dumps(c).encode() + b"\n\n"


def plan(messages):
    """-> ("text", str) or ("tool", bash command)."""
    last = messages[-1] if messages else {}
    if last.get("role") == "tool":
        return "text", "The step finished. Moving on."
    user = next((text_of(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
    if "kick off" in user:
        return "tool", './run.sh "baseline" > run.log 2>&1; grep "^val_bpb:" run.log'
    if "interrupted by a restart" in user:
        return "tool", 'cd . && ./run.sh "after restart" > run.log 2>&1'
    if control().get("nudge_cmd"):
        return "tool", control()["nudge_cmd"]
    return "text", "Understood, thinking about the next idea."


async def chat(request):
    body = await request.json()
    msgs = body.get("messages") or []
    user = next((text_of(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"), "")
    rec = {"ts": time.time(), "n_messages": len(msgs), "last_role": (msgs[-1] if msgs else {}).get("role"),
           "last_user": user[:300], "model": body.get("model"), "stream": body.get("stream"),
           "tools": [t.get("function", {}).get("name") for t in body.get("tools") or []],
           "first_user": next((text_of(m.get("content"))[:200] for m in msgs if m.get("role") == "user"), "")}
    with open(ARGS.log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    if control().get("stall"):
        await asyncio.sleep(3600)          # cancelled when the client goes away
        return resp
    kind, payload = plan(msgs)
    await resp.write(chunk({"role": "assistant", "content": ""}))
    for piece in ("Let me ", "think about this."):     # U+2028 inside JSON: must not split records
        await resp.write(chunk({"reasoning_content": piece}))
        await asyncio.sleep(0.01)
    n_out = 8
    if kind == "text":
        for piece in payload.split(" "):
            await resp.write(chunk({"content": piece + " "}))
        await resp.write(chunk({}, finish="stop"))
    else:
        args = json.dumps({"command": payload})
        await resp.write(chunk({"tool_calls": [{"index": 0, "id": f"call_{int(time.time() * 1000)}", "type": "function",
                                                "function": {"name": "bash", "arguments": ""}}]}))
        for i in range(0, len(args), 16):
            await resp.write(chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[i:i + 16]}}]}))
        await resp.write(chunk({}, finish="tool_calls"))
    n_in = sum(len(json.dumps(m)) for m in msgs) // 4
    await resp.write(chunk(usage={"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}))
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def models(_):
    return web.json_response({"object": "list", "data": [{"id": "policy", "object": "model"}]})


async def run(request):
    d = await request.json()
    JOB["job"] = {"desc": d.get("desc"), "t0": time.time()}
    try:
        await asyncio.sleep(float(control().get("run_s", 0)))
    finally:
        JOB["job"] = None
        JOB["n_runs"] += 1
        JOB["last"] = {"desc": d.get("desc"), "t0": time.time(), "t1": time.time(), "status": "ok"}
    with open(ARGS.log + ".runs", "a") as f:
        f.write(json.dumps({"ts": time.time(), "desc": d.get("desc"), "n_bytes": len(d.get("train_py") or "")}) + "\n")
    out = ("---\nval_bpb:          0.990000\ntraining_seconds: 300.1\ntotal_seconds:    325.9\n"
           "peak_vram_mb:     45060.2")
    return web.json_response({"rc": 0, "output": out})


async def state(_):
    if control().get("state_down"):
        return web.json_response({"error": "down"}, status=503)
    return web.json_response({"arm": "base", "busy": JOB["job"] is not None, "job": JOB["job"], "queued": 0,
                              "n_runs": JOB["n_runs"], "last": JOB["last"]})


async def main():
    llm = web.Application()
    llm.add_routes([web.post("/v1/chat/completions", chat), web.get("/v1/models", models)])
    rsock = web.Application()
    rsock.add_routes([web.post("/run", run)])
    rctl = web.Application()
    rctl.add_routes([web.get("/control/state", state)])
    runners = []
    for app, site in ((llm, lambda r: web.UnixSite(r, ARGS.llm_sock)),
                      (rsock, lambda r: web.UnixSite(r, ARGS.runner_sock)),
                      (rctl, lambda r: web.TCPSite(r, "127.0.0.1", ARGS.runner_port))):
        r = web.AppRunner(app, access_log=None)
        await r.setup()
        await site(r).start()
        runners.append(r)
    for s in (ARGS.llm_sock, ARGS.runner_sock):
        os.chmod(s, 0o666)
    print("ready", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-sock", required=True)
    ap.add_argument("--runner-sock", required=True)
    ap.add_argument("--runner-port", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--control", default="")
    ARGS = ap.parse_args()
    asyncio.run(main())
