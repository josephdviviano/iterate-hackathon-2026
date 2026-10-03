"""Full protocol generation, k samples per protocol, every raw reply stored.

    uv run python -m bioprot.generate --model qwen --k 5
    uv run python -m bioprot.generate --model qwen --k 5 --unshuffled        # order-leak ablation
    uv run python -m bioprot.generate --model qwen --k 5 --description ai   # GPT-4 descriptions

One row per sample in artifacts/bioprot/<condition>/generations.jsonl, keyed
by (protocol_id, sample_idx); a re-run fills in only the missing keys. The
feedback loop is never on: the artifact judged is the first reply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .backends import MODELS, make_backend
from .data import Protocol, function_sequence, load_protocols, shuffled_definitions, strip_definitions
from .prompts import generation_messages

ART = Path(__file__).resolve().parents[2] / "artifacts" / "bioprot"
MAX_TOKENS = {"gptoss": 6000}


def condition_name(model: str, shuffled: bool, description: str, temperature: float = 0.7) -> str:
    name = f"{model}_{'shuf' if shuffled else 'ord'}_{description}"
    return name if temperature == 0.7 else f"{name}_t{temperature:g}"


def task_inputs(p: Protocol, seed: int, shuffled: bool, description: str) -> tuple[list[str], str, str]:
    """The definitions in the order shown, the description text, and which source it came from."""
    defs = shuffled_definitions(p, seed) if shuffled else list(p.definitions)
    if description == "ai" and p.ai_description:
        return defs, p.ai_description, "ai"
    return defs, p.description, "human"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def add_condition_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--unshuffled", action="store_true", help="definitions in call order (leaks the answer)")
    ap.add_argument("--description", default="human", choices=["human", "ai"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_condition_args(ap)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="first N protocols only (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--where", default="modal", choices=["modal", "local"], help="where the claude CLI runs")
    a = ap.parse_args()
    shuffled = not a.unshuffled
    cond = condition_name(a.model, shuffled, a.description, a.temperature)
    out = ART / cond / "generations.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["protocol_id"], r["sample_idx"]) for r in read_rows(out)}
    protocols = load_protocols()[: a.limit or None]
    backend = make_backend(a.model, a.where)
    max_tokens = MAX_TOKENS.get(a.model, 2000)

    jobs = []
    for p in protocols:
        defs, desc, used = task_inputs(p, a.seed, shuffled, a.description)
        messages = generation_messages(p, defs, desc)
        prompt_hash = hashlib.sha1(json.dumps(messages).encode()).hexdigest()[:12]
        for i in range(a.k):
            if (p.id, i) in done:
                continue
            jobs.append((p, i, messages, prompt_hash, used))
    print(f"{cond}: {len(done)} stored, {len(jobs)} to generate", flush=True)

    def run(job):
        p, i, messages, prompt_hash, used = job
        seed = a.seed * 1000 + i
        reply = backend(messages, temperature=a.temperature, max_tokens=max_tokens, seed=seed, logprobs=True)
        seq, parser = function_sequence(strip_definitions(reply.text))
        return {"protocol_id": p.id, "stratum": "bioprot", "model": a.model, "condition": cond, "seed": seed,
                "sample_idx": i, "prompt_hash": prompt_hash, "shuffled": shuffled, "description_source": a.description,
                "description_used": used, "feedback": False, "temperature": a.temperature,
                "raw_output": reply.text, "parsed_function_sequence": seq, "parser": parser,
                "mean_logprob": reply.mean_logprob, "n_tokens": reply.n_tokens, "meta": reply.meta}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        futures = [ex.submit(run, j) for j in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as exc:  # one failed sample is re-tried on the next run, the sweep continues
                print("error:", type(exc).__name__, str(exc)[:200], flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
