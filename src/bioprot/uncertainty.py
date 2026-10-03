"""Four uncertainty signals per sample, stored side by side.

    uv run python -m bioprot.uncertainty --model qwen

1. Self-consistency: mean symmetric Levenshtein distance between a sample's
   function sequence and the other k-1 samples of the same protocol (label-free).
2. Verbalised confidence with an explicit abstain channel, in a second call
   against the stored plan.
3. Sequence logprob: minus the mean token logprob of the generation.
4. Self-critique: a PASS or FAIL execution judgement in a second call; where
   the server exposes first-token probabilities the score is 1 - P(PASS).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .backends import Reply, make_backend
from .data import load_protocols, strip_definitions
from .generate import ART, add_condition_args, condition_name, read_rows, task_inputs
from .prompts import confidence_messages, critique_messages, parse_confidence, parse_verdict
from .score import symmetric_distance


# A reasoning model spends tokens on its analysis channel before the one-line answer.
ELICIT_TOKENS = {"gptoss": 3000}


def self_consistency(rows: list[dict]) -> dict[tuple[str, int], dict]:
    by_p: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_p[r["protocol_id"]].append(r)
    out = {}
    for pid, group in by_p.items():
        seqs = {r["sample_idx"]: r["parsed_function_sequence"] for r in group}
        pairs = {(i, j): symmetric_distance(seqs[i], seqs[j]) for i in seqs for j in seqs if i < j}
        protocol_mean = sum(pairs.values()) / len(pairs) if pairs else None
        for i in seqs:
            partners = [d for (a, b), d in pairs.items() if i in (a, b)]
            out[(pid, i)] = {"self_consistency": sum(partners) / len(partners) if partners else None,
                             "self_consistency_protocol": protocol_mean, "k": len(seqs)}
    return out


def critique_score(reply: Reply) -> tuple[str | None, float | None]:
    """The verdict and P(PASS). From the first token's alternatives when the
    server gives them and they carry the verdict tokens, else 1 or 0."""
    verdict = parse_verdict(reply.text)
    top = reply.first_token_top or {}
    mass = {"PASS": 0.0, "FAIL": 0.0}
    for tok, p in top.items():
        t = tok.strip().upper()
        for v in mass:
            if t and v.startswith(t) and len(t) >= 1 and (t == v or len(t) <= 2):
                mass[v] += p
    if mass["PASS"] + mass["FAIL"] > 0.5:
        return verdict, mass["PASS"] / (mass["PASS"] + mass["FAIL"])
    return verdict, (1.0 if verdict == "PASS" else 0.0 if verdict == "FAIL" else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_condition_args(ap)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--where", default="modal", choices=["modal", "local"])
    a = ap.parse_args()
    shuffled = not a.unshuffled
    cond = condition_name(a.model, shuffled, a.description, a.temperature)
    gens = read_rows(ART / cond / "generations.jsonl")
    out = ART / cond / "uncertainty.jsonl"
    done = {(r["protocol_id"], r["sample_idx"]) for r in read_rows(out)}
    sc = self_consistency(gens)
    by_id = {p.id: p for p in load_protocols()}
    backend = make_backend(a.model, a.where)
    jobs = [r for r in gens if (r["protocol_id"], r["sample_idx"]) not in done]
    print(f"{cond}: {len(done)} stored, {len(jobs)} to elicit", flush=True)

    def run(r: dict) -> dict:
        p = by_id[r["protocol_id"]]
        defs, desc, _ = task_inputs(p, a.seed, shuffled, a.description)
        plan = strip_definitions(r["raw_output"]) or r["raw_output"]
        budget = ELICIT_TOKENS.get(a.model, 80)
        conf = backend(confidence_messages(p, defs, desc, plan), temperature=0.0, max_tokens=budget, seed=0, logprobs=False)
        crit = backend(critique_messages(p, defs, desc, plan), temperature=0.0, max_tokens=budget, seed=0,
                       logprobs=True, top_logprobs=5)
        verdict, p_pass = critique_score(crit)
        key = (r["protocol_id"], r["sample_idx"])
        return {"protocol_id": r["protocol_id"], "model": r["model"], "condition": cond, "sample_idx": r["sample_idx"],
                **sc[key],
                **parse_confidence(conf.text), "confidence_reply": conf.text[:400],
                "logprob_uncertainty": (-r["mean_logprob"]) if r.get("mean_logprob") is not None else None,
                "critique_verdict": verdict, "critique_p_pass": p_pass, "critique_reply": crit.text[:400],
                "critique_top": crit.first_token_top}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        futures = [ex.submit(run, r) for r in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as exc:
                print("error:", type(exc).__name__, str(exc)[:200], flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
