#!/usr/bin/env python3
"""Analyse calls captured by capture_proxy.py (run with --token-ids) for one sequential agent session.

Per call: prompt/completion token counts, token-id completeness (sum of streamed token_ids vs
usage.completion_tokens), time to first token, decode speed, finish reason, and whether the
request replays reasoning for every earlier assistant message.

Per consecutive pair (k, k+1): does prompt_{k+1} start with prompt_k + completion_k in TOKEN space?
If not, the first divergence is decoded with the model tokenizer so the cause is visible.

Usage: python check_prefix.py calls.jsonl [--tokenizer ~/models/Qwen3.8-27B-FP8] [--json out.json]
"""
import argparse
import json
import os


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calls")
    ap.add_argument("--tokenizer", default="~/models/Qwen3.8-27B-FP8", help="model dir containing tokenizer.json")
    ap.add_argument("--json", help="write a machine-readable summary here")
    args = ap.parse_args()
    calls = [c for c in load(args.calls) if c.get("status") == 200]
    tok = None
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(os.path.expanduser(args.tokenizer), "tokenizer.json"))
    except Exception as e:  # analysis still works without decoding
        print(f"(tokenizer unavailable: {e})")
    dec = (lambda ids: tok.decode(ids, skip_special_tokens=False)) if tok else (lambda ids: str(ids))

    summary = {"calls": [], "pairs": []}
    print(f"{'k':>2} {'prompt':>7} {'compl':>6} {'ids==usage':>10} {'ttft_s':>7} {'dur_s':>6} {'dec_tok/s':>9} {'finish':>10} {'reasoning_replayed':>18}")
    for k, c in enumerate(calls):
        p, o = c.get("prompt_token_ids") or [], c.get("completion_token_ids") or []
        usage = c.get("usage") or {}
        ttft = (c["t_first_token"] - c["t_start"]) if c.get("t_first_token") else None
        dur = c["t_end"] - c["t_start"]
        decode_s = (c["t_end"] - c["t_first_token"]) if c.get("t_first_token") else None
        speed = (len(o) - 1) / decode_s if decode_s and len(o) > 1 else None
        msgs = c["request"]["messages"]
        assistants = [m for m in msgs if m.get("role") == "assistant"]
        with_reasoning = sum(1 for m in assistants if m.get("reasoning") or m.get("reasoning_content"))
        row = {"k": k, "prompt_tokens": len(p), "completion_tokens": len(o), "usage": usage,
               "ids_match_usage": len(o) == usage.get("completion_tokens"), "ttft_s": ttft, "dur_s": dur,
               "decode_tok_s": speed, "finish_reason": c.get("finish_reason"),
               "assistant_msgs": len(assistants), "assistant_msgs_with_reasoning": with_reasoning,
               "reasoning_chars": len(c.get("reasoning") or ""), "tool_calls": list((c.get("tool_calls") or {}).values())}
        summary["calls"].append(row)
        print(f"{k:>2} {len(p):>7} {len(o):>6} {str(row['ids_match_usage']):>10} {ttft or float('nan'):>7.2f} {dur:>6.2f} "
              f"{speed or float('nan'):>9.1f} {str(c.get('finish_reason')):>10} {f'{with_reasoning}/{len(assistants)}':>18}")

    print("\nPrefix extension in token space (prompt_{k+1} vs prompt_k + completion_k):")
    for k in range(len(calls) - 1):
        a = (calls[k].get("prompt_token_ids") or []) + (calls[k].get("completion_token_ids") or [])
        b = calls[k + 1].get("prompt_token_ids") or []
        lcp = common_prefix(a, b)
        ok = lcp == len(a)
        pair = {"k": k, "prev_len": len(a), "next_prompt_len": len(b), "lcp": lcp, "exact_extension": ok}
        if ok:
            print(f"  {k}->{k+1}: EXACT extension ({len(a)} tokens reused, {len(b) - len(a)} new)")
        else:
            plen = len(calls[k].get("prompt_token_ids") or [])
            where = "in prompt_k" if lcp < plen else f"at completion token {lcp - plen}/{len(a) - plen}"
            pair.update({"diverge_where": where, "prev_ctx": dec(a[max(0, lcp - 12):lcp]),
                         "prev_next": dec(a[lcp:lcp + 12]), "new_next": dec(b[lcp:lcp + 12])})
            print(f"  {k}->{k+1}: DIVERGES at token {lcp} ({where}); "
                  f"context={pair['prev_ctx']!r}\n        generated: {pair['prev_next']!r}\n        re-rendered: {pair['new_next']!r}")
        summary["pairs"].append(pair)

    total_out = sum(r["completion_tokens"] for r in summary["calls"])
    total_dur = sum(r["dur_s"] for r in summary["calls"])
    exact = sum(1 for p in summary["pairs"] if p["exact_extension"])
    print(f"\ncalls={len(calls)} completion_tokens={total_out} llm_time_s={total_dur:.1f} "
          f"exact_extensions={exact}/{len(summary['pairs'])} ids_complete={sum(r['ids_match_usage'] for r in summary['calls'])}/{len(calls)}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
