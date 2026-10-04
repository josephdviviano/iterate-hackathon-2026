"""TL;DR insight generation (paper App. A.1 / B), adapted to autoresearch.

After a failed attempt, the *current policy* is called in a fresh conversation (no contextual drag) with
the judge prompt below. It sees the task as the agent saw it (rules + current best + experiment history),
the agent's rollout (actions + observations, think traces dropped), the evaluated diff and the harness
("verifier") verdict, which the agent never saw in this form. It thinks, then emits JSON with
summary/feedback/wrong_step_id/corrected_step/hint; only the one-sentence `hint` is used.
"""
import json
import re
import unicodedata

import requests

JUDGE_TEMPLATE = """You are a verifier that is given a step-by-step solution of an agent that has to perform the following research task:
============================== TASK ==============================
{task}
============================== END OF TASK ==============================

Here is the step-by-step solution that the agent proposed. At each step, it called a tool (read a file, edit a file, run a bash command, ...) and then got an observation from the environment that executed that tool call.
============================== AGENT ROLLOUT ==============================
{steps}
============================== END OF AGENT ROLLOUT ==============================
The final change to train.py that was evaluated (unified diff against the current best version):
```diff
{diff}
```
The experiment harness (verifier) ran and returned this output:
{verifier}
As a verifier, you now have to produce a json with FIVE parts: 1) Summarize the agent attempt, 2) Give feedback, 3) Give the step at which it went wrong, 4) Provide the corrected code for that step, 5) Give a one-sentence TL;DR hint.
1) Summarize the attempt
- Give a summary of roughly one paragraph that describes what the agent did.
- You can skip very generic steps like listing or reading files.
- Focus on which idea the agent tried and what exactly it changed in train.py.
- Also mention the final outcome of the experiment.
- Do NOT critique yet, just summarize.
2) Give feedback
- For successful attempts, just return an empty string.
- For failed attempts, look at the harness output.
- Write a paragraph on what went wrong in the attempt.
- Try to point out where exactly in the code or in the reasoning the error is. Stop at the FIRST thing that went wrong.
- Make it independent of the agent rollout, do NOT assume that the rollout will be shown along with your feedback.
3) Give the step at which it went wrong
- Return the ID of the step where the agent made the decisive mistake.
- If you cannot identify a specific step, output -1.
4) Provide the corrected code for that step
- Return ONLY the code that the agent should have written for the wrong step in place of what it actually wrote. Not the prior steps, not the following steps.
- Do NOT include explanatory prose inside the code; use code-comments only if essential.
- If you cannot identify a specific wrong step, return an empty string.
5) Give a one-sentence TL;DR hint
- After creating all the previous four parts, output a single-sentence hint on what went wrong.
- Keep it to ONE sentence, plain language, no code.
- This should be a concise pointer the agent can act on (e.g. "You changed the setting in one place but the code reads it from another.").
- For successful attempts, just return an empty string.
Output your answer in a JSON. Do NOT output any text except the JSON. The format should be:
{{
"summary": "...",
"feedback": "...",
"wrong_step_id": "integer",
"corrected_step": "...",
"hint": "..."
}}"""


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))


def _trunc(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    h = n // 2
    return s[:h] + f"\n... [{len(s) - n} characters omitted] ...\n" + s[-h:]


def rollout_steps(calls: list, tool_chars: int) -> str:
    """Render the agent rollout from the last recorded agent call (its messages + final response)."""
    agent_calls = [c for c in calls if c.get("kind") == "agent" and c.get("messages")]
    if not agent_calls:
        return "(the agent made no LLM calls)"
    last = agent_calls[-1]
    msgs = list(last["messages"])
    if last.get("response"):
        msgs.append(last["response"])
    # skip system, the task message and injected insight messages: start after the leading non-agent turns
    i = 0
    while i < len(msgs) and msgs[i].get("role") in ("system", "developer", "user"):
        i += 1
    out, step = [], 0
    for m in msgs[i:]:
        role = m.get("role")
        if role == "assistant":
            step += 1
            parts = [f"Step {step}:"]
            txt = _text(m.get("content")).strip()
            if txt:
                parts.append(_trunc(txt, 3000))
            for tc in m.get("tool_calls") or []:
                f = tc.get("function", {})
                parts.append(f"[tool call] {f.get('name')}({_trunc(f.get('arguments') or '', 6000)})")
            out.append("\n".join(parts))
        elif role == "tool":
            out.append(f"Observation {step}:\n{_trunc(_text(m.get('content')), tool_chars)}")
        elif role == "user":
            out.append(f"[user message]\n{_trunc(_text(m.get('content')), 1500)}")
    return "\n".join(out) if out else "(the agent took no actions)"


def _stats(e: dict) -> str:
    m = (e or {}).get("metrics") or {}
    return (f"num_steps={m.get('num_steps')}, total_tokens_M={m.get('total_tokens_M')}, "
            f"peak_vram_mb={m.get('peak_vram_mb')}, num_params_M={m.get('num_params_M')}, depth={m.get('depth')}, "
            f"time_to_eval_s={m.get('t_to_eval')}")


def verifier_text(verdict: list, outcome: dict, parent_entry: dict) -> str:
    """verdict: plain-language lines from the driver, the first one starting with 'Rollout FAILED'."""
    lines = list(verdict)
    if outcome and outcome.get("status") == "ok":
        lines.append(f"Statistics of the evaluated run: {_stats(outcome)}.")
        if parent_entry:
            lines.append(f"Statistics of a run of the current best version, for comparison: {_stats(parent_entry)}.")
    elif outcome and outcome.get("error_tail"):
        lines.append("Last lines of the training log:\n" + _trunc(outcome["error_tail"], 3000))
    return "\n".join(lines)


def parse_insight(text: str):
    """Extract the hint from the judge output; tolerant to code fences / leading prose."""
    if not text:
        return None, None
    for c in reversed(re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            d = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("hint"):
            return d, str(d["hint"]).strip()
    m = re.search(r'"hint"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.S)
    if m:
        try:
            return None, json.loads(f'"{m.group(1)}"').strip()
        except json.JSONDecodeError:
            return None, m.group(1).strip()
    return None, None


def clean_hint(h: str, max_chars: int) -> str:
    # NFC + no control characters: the hint is injected into prompts and located again in token space
    h = unicodedata.normalize("NFC", h)
    h = "".join(ch for ch in h if ch == " " or not unicodedata.category(ch).startswith("C"))
    h = " ".join(h.split())
    return h if len(h) <= max_chars else h[: max_chars - 3].rstrip() + "..."


def generate_insight(cfg, attempt_id: str, task: str, calls: list, diff: str, verdict: list,
                     outcome: dict, parent_entry: dict, retries: int = 2):
    """Call the current policy through the gateway (kind=insight; recorded, never trained on)."""
    prompt = JUDGE_TEMPLATE.format(
        task=task,
        steps=rollout_steps(calls, cfg.insight_tool_output_chars),
        diff=_trunc(diff or "(no change was evaluated)", 12000),
        verifier=verifier_text(verdict, outcome, parent_entry),
    )
    body = {"model": "policy", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.insight_max_tokens, "thinking_token_budget": cfg.insight_thinking_budget,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20}
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(f"{cfg.gateway_url}/v1/chat/completions", json=body, timeout=3600,
                              headers={"X-RL-Kind": "insight", "X-RL-Attempt": attempt_id})
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            full, hint = parse_insight(msg.get("content") or "")
            if hint:
                hint = clean_hint(hint, cfg.insight_max_chars)
                if hint:
                    return {"hint": hint, "judge": full, "raw": msg.get("content"), "prompt_chars": len(prompt)}
            last_err = f"unparseable judge output: {(msg.get('content') or '')[:300]!r}"
        except Exception as e:  # network / server errors: retry
            last_err = repr(e)
    return {"hint": None, "error": last_err, "prompt_chars": len(prompt)}
