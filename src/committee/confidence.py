"""Verbalized-confidence baseline: ask a language model how sure it is of one prediction.

For each held-out transition the judge sees the committee's plurality
program, the before state, the action and the program's predicted after
state, and answers with a number from 0 to 100. The complement of that
number is the uncertainty score compared against the committee's own
disagreement on AUROC. Kadavath et al. 2022 (P(True)) and Tian et al. 2023
(verbalized confidence) are the references; Spiess et al. 2024 report such
scores are overconfident for code.
"""

from __future__ import annotations

import json
import os
import re
import time

from .committee import Committee, Member, auroc, description_length
from .evaluate import load_runs
from .experiment import condition_dir
from .loader import build_buffer, temporal_split
from .synth_api import load_env_file
from .verify import canonical

PROMPT = """You are judging a program that models an unknown grid game. The program must map
(state, action) to the exact next state, every field of every object.

Program:
```python
{program}
```

Before state:
{before}

Action: {action}

The program predicts this after state:
{after}

How confident are you that this predicted after state is exactly right?
Reply with a single integer from 0 to 100 and nothing else."""


def judge(game: str, level: int, train_frac: float, condition: str, model: str = "llm",
          base_url: str | None = None, test_level: int | None = None, max_calls: int | None = None) -> dict:
    from openai import OpenAI

    load_env_file()
    client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL_GPTOSS") or os.environ.get("OPENAI_BASE_URL"),
                    api_key=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY"), timeout=600)
    train, test = temporal_split(build_buffer(game), level, train_frac, test_level)
    cond = condition_dir(game, level, train_frac, condition, test_level)
    runs = load_runs(cond, train, test)
    members = [Member(n, s, p, description_length(s)) for n, m, s, p in runs if m["consistent"] and p]
    com = Committee(members)
    top = members[com.weights.index(max(com.weights))]
    truth = [json.dumps(canonical(t.after_objs)) for t in test]
    rows = []
    t0 = time.time()
    for attempt in range(30):  # a scaled-to-zero server answers 503 until its container is up
        try:
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Reply with 1."}],
                                           max_tokens=4)
            break
        except Exception:
            time.sleep(30)
    for i, t in enumerate(test[: max_calls or len(test)]):
        pred = top.test_preds[i]
        text = PROMPT.format(program=top.source[:12000], before=json.dumps(t.before_objs)[:6000], action=json.dumps(t.action),
                             after=json.dumps(pred)[:6000] if pred is not None else "no prediction (error)")
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}],
                                               temperature=0.0, max_tokens=12)
            m = re.search(r"\d+", r.choices[0].message.content or "")
            conf = min(100, max(0, int(m.group()))) if m else None
        except Exception:
            conf = None
        correct = pred is not None and json.dumps(canonical(pred)) == truth[i]
        rows.append({"step": t.step, "confidence": conf, "correct": correct,
                     "disagreement": com.uniform_disagreement(i)})
    scored = [r for r in rows if r["confidence"] is not None]
    errors = [not r["correct"] for r in scored]
    out = {
        "game": game, "level": level, "condition": condition, "judge_model": model, "n": len(scored),
        "program_accuracy": round(sum(r["correct"] for r in scored) / max(1, len(scored)), 4),
        "auroc_verbalized": auroc([100 - r["confidence"] for r in scored], errors),
        "auroc_disagreement_same_rows": auroc([r["disagreement"] for r in scored], errors),
        "confidence_values": sorted({r["confidence"] for r in scored}),
        "mean_confidence_when_right": round(sum(r["confidence"] for r in scored if r["correct"]) / max(1, sum(r["correct"] for r in scored)), 1),
        "mean_confidence_when_wrong": round(sum(r["confidence"] for r in scored if not r["correct"]) / max(1, sum(not r["correct"] for r in scored)), 1),
        "wall_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    (cond / "verbalized_confidence.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verbalized-confidence baseline over the plurality program.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--test-level", type=int, default=None)
    parser.add_argument("--condition", default="committee_devin")
    parser.add_argument("--model", default="llm")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    args = parser.parse_args(argv)
    out = judge(args.game, args.level, args.train_frac, args.condition, args.model, args.base_url, args.test_level,
                args.max_calls)
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
