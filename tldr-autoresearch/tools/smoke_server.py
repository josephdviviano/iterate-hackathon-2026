"""Smoke test of the live vLLM server (127.0.0.1:8000): tool calling, reasoning, token ids and logprobs, streaming
and non-streaming, as the gateway and trainer use them.   python tools/smoke_server.py"""
import json, time, requests, sys
B = "http://127.0.0.1:8000"
tools = [{"type": "function", "function": {"name": "bash", "description": "Run a bash command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
msgs = [{"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "List the files in the current directory using the bash tool."}]
def chat(model, stream=False, **kw):
    body = dict(model=model, messages=msgs, tools=tools, max_tokens=2048, temperature=1.0, top_p=1.0, top_k=-1,
                return_token_ids=True, logprobs=True, top_logprobs=0, reasoning_effort="low", stream=stream, **kw)
    if stream: body["stream_options"] = {"include_usage": True}
    t = time.time(); r = requests.post(f"{B}/v1/chat/completions", json=body, stream=stream, timeout=600); r.raise_for_status()
    if not stream:
        d = r.json(); c = d["choices"][0]
        return d, time.time() - t
    pid, ids, lps, reasoning, content, tcs, fin, usage = None, [], [], "", "", [], None, None
    for line in r.iter_lines():
        if not line.startswith(b"data: ") or line[6:] == b"[DONE]": continue
        d = json.loads(line[6:]); pid = pid or d.get("prompt_token_ids"); usage = d.get("usage") or usage
        for c in d.get("choices", []):
            ids += c.get("token_ids") or []; lps += [x["logprob"] for x in ((c.get("logprobs") or {}).get("content") or [])]
            dl = c.get("delta", {}); reasoning += dl.get("reasoning") or dl.get("reasoning_content") or ""; content += dl.get("content") or ""
            tcs += dl.get("tool_calls") or []; fin = c.get("finish_reason") or fin
    return dict(pid=pid, ids=ids, lps=lps, reasoning=reasoning, content=content, tcs=tcs, fin=fin, usage=usage), time.time() - t

d, dt = chat("qwen3.8-27b-fp8")
c = d["choices"][0]; m = c["message"]
print("NONSTREAM finish:", c["finish_reason"], "| tool_calls:", json.dumps(m.get("tool_calls"))[:200])
print("  reasoning:", (m.get("reasoning") or m.get("reasoning_content") or "")[:150].replace("\n", " "))
print("  content:", (m.get("content") or "")[:100], "| prompt_ids:", len(d.get("prompt_token_ids") or []), "| comp ids:", len(c.get("token_ids") or []),
      "| logprobs:", len((c.get("logprobs") or {}).get("content") or []), "| usage:", d["usage"], f"| {dt:.1f}s")
s, dt = chat("qwen3.8-27b-fp8", stream=True)
print("STREAM finish:", s["fin"], "| tool_calls deltas:", json.dumps(s["tcs"])[:200])
print("  prompt_ids:", len(s["pid"] or []), "| comp ids:", len(s["ids"]), "| logprobs:", len(s["lps"]), "| usage:", s["usage"], f"| {dt:.1f}s",
      f"| {s['usage']['completion_tokens']/dt:.1f} tok/s")
print("  ids==usage:", len(s["ids"]) == s["usage"]["completion_tokens"], "| lps==ids:", len(s["lps"]) == len(s["ids"]))
