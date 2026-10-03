"""Chat backends that return the text and, where the server exposes them, token logprobs.

`openai_compatible` covers the vLLM servers on Modal. `claude_cli` runs the
Claude Code CLI with no tools, in a Modal container by default so a sweep
never loads the local machine; it exposes no logprobs.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Reply:
    text: str
    mean_logprob: float | None = None
    n_tokens: int | None = None
    first_token_top: dict[str, float] | None = None  # token -> probability, first output token
    meta: dict = field(default_factory=dict)


Backend = Callable[..., Reply]  # backend(messages, temperature=..., max_tokens=..., seed=..., logprobs=...)


def openai_compatible(base_url: str, api_key: str, model: str, extra_body: dict | None = None,
                      timeout: float = 600) -> Backend:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=4)

    def call(messages: list[dict], *, temperature: float, max_tokens: int, seed: int | None = None,
             logprobs: bool = True, top_logprobs: int = 0) -> Reply:
        t0 = time.time()
        kw: dict = {"logprobs": True, "top_logprobs": top_logprobs} if logprobs else {}
        if seed is not None:
            kw["seed"] = seed
        if extra_body:
            kw["extra_body"] = extra_body
        r = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                           max_tokens=max_tokens, **kw)
        ch = r.choices[0]
        text = ch.message.content or ""
        mean_lp = n = None
        top = None
        content = getattr(ch.logprobs, "content", None) if ch.logprobs else None
        if content:
            lps = [t.logprob for t in content if t.logprob is not None]
            n = len(lps)
            mean_lp = sum(lps) / n if n else None
            if content[0].top_logprobs:
                top = {t.token: math.exp(t.logprob) for t in content[0].top_logprobs}
        return Reply(text, mean_lp, n, top, {"finish_reason": ch.finish_reason, "wall_s": round(time.time() - t0, 2),
                                             "completion_tokens": getattr(r.usage, "completion_tokens", None),
                                             "reasoning": (getattr(ch.message, "reasoning_content", None) or "")[:2000]})
    return call


CLAUDE_FLAGS = ["--output-format", "json", "--max-turns", "1", "--no-session-persistence", "--strict-mcp-config",
                "--tools", ""]


def run_claude_cli(messages: list[dict], model: str, timeout_s: float = 300) -> dict:
    """One tool-free completion through `claude -p`. The system message goes in
    as the system prompt; the single user message is the prompt."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    cmd = ["claude", "-p", user, "--model", model, "--system-prompt", system] + CLAUDE_FLAGS
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    out: dict = {"wall_s": round(time.time() - t0, 2), "returncode": proc.returncode}
    try:
        j = json.loads(proc.stdout)
        if isinstance(j, list):
            j = next((m for m in j if m.get("type") == "result"), {})
        out.update({k: j.get(k) for k in ("total_cost_usd", "is_error", "subtype", "duration_ms")})
        out["text"] = j.get("result") or ""
    except json.JSONDecodeError:
        out["text"] = ""
        out["stdout_tail"] = proc.stdout[-500:]
        out["stderr_tail"] = proc.stderr[-500:]
    return out


def claude_cli(model: str, where: str = "modal") -> Backend:
    if where == "modal":
        import modal

        fn = modal.Function.from_name("bioprot-claude", "complete")

        def call(messages, *, temperature: float, max_tokens: int, seed=None, logprobs=True, top_logprobs=0) -> Reply:
            out = fn.remote({"messages": messages, "model": model})
            return Reply(out.pop("text", ""), None, None, None, out)
    else:
        def call(messages, *, temperature: float, max_tokens: int, seed=None, logprobs=True, top_logprobs=0) -> Reply:
            out = run_claude_cli(messages, model)
            return Reply(out.pop("text", ""), None, None, None, out)
    return call


def load_env() -> None:
    from committee.synth_api import load_env_file

    load_env_file()


MODELS = {
    "qwen": ("Qwen3-Coder-30B-A3B-Instruct-FP8", "vLLM on Modal"),
    "gptoss": ("gpt-oss-120b", "vLLM on Modal, reasoning effort low"),
    "mistral": ("Mistral-Small-24B-Instruct-2501", "vLLM on Modal"),
    "haiku": ("claude-haiku-4-5", "Claude Code CLI on Modal, no tools"),
    "sonnet": ("claude-sonnet-4-5", "Claude Code CLI on Modal, no tools"),
}


def make_backend(name: str, where: str = "modal") -> Backend:
    load_env()
    if name == "qwen":  # the dedicated server when one is deployed, else the shared committee server
        return openai_compatible(os.environ.get("BIOPROT_QWEN_URL") or os.environ["OPENAI_BASE_URL"], os.environ["VLLM_API_KEY"], "llm")
    if name == "gptoss":
        return openai_compatible(os.environ["BIOPROT_GPTOSS_URL"], os.environ["VLLM_API_KEY"], "llm",
                                 extra_body={"reasoning_effort": "low"})
    if name == "mistral":
        return openai_compatible(os.environ["BIOPROT_MISTRAL_URL"], os.environ["VLLM_API_KEY"], "llm")
    if name in ("haiku", "sonnet", "opus"):
        return claude_cli(name, where)
    raise KeyError(name)
