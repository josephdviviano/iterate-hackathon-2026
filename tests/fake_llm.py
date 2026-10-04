#!/usr/bin/env python3
"""Stand-in for `claude -p ... --json-schema ...` in tests: canned structured output per ROLE.

Logs each call (role, tools flag, cwd, files in cwd) to $FAKE_LLM_LOG as JSON lines.
$FAKE_LLM_REFUTE=H1 makes revise refute that hypothesis when the tree is not empty.
$FAKE_COMMITTEE: committee members forecast wide ranges (default), "far" ranges no result can hit,
or "harm": ranges entirely worse than the tip.
"""
import json
import os
import re
import sys

argv = sys.argv[1:]
prompt = sys.stdin.read()  # research.py sends the prompt on stdin
tools = argv[argv.index("--tools") + 1]
role = re.search(r"^ROLE: ([\w-]+)", prompt, re.MULTILINE).group(1)
with open(os.environ["FAKE_LLM_LOG"], "a") as f:
    f.write(json.dumps({"role": role, "tools": tools, "cwd_files": os.listdir("."),
                        "model": argv[argv.index("--model") + 1], "has_results": "=== RESULTS" in prompt, "prompt_bytes": len(prompt.encode()), "has_literature": "=== LITERATURE" in prompt,
                        "has_committee": "=== UNCERTAINTY COMMITTEE" in prompt}) + "\n")
tree = re.findall(r"^(H[\d.]+)\t\d\t[^\t]*\topen\t", prompt, re.MULTILINE)
if role == "committee-member":
    mode = os.environ.get("FAKE_COMMITTEE", "wide")
    metrics = re.search(r"^The forecast metrics: (.*?)\. ", prompt, re.MULTILINE).group(1).split(", ")
    tip = json.loads(re.search(r"The current best \(the branch tip\): (\{.*?\})", prompt).group(1) or "{}")
    ideas = re.findall(r"^(I\d+)\tH", prompt, re.MULTILINE)

    def rng(m):
        if mode == "far":
            return 1e6, 1e6 + 1, 1e6 + 2
        if mode == "harm":  # worse than the tip on every forecast metric (all minimized in the toy task)
            b = float(tip.get(m, 1.0))
            return 5 * b + 10, 6 * b + 11, 7 * b + 12
        return -1e9, 0.0, 1e9

    out = {"world_model": "world model of " + re.search(r"You are member (W\d+)", prompt).group(1),
           "forecasts": [{"idea": i, "metrics": [dict(zip(("metric", "lo", "point", "hi"), (m, *rng(m)))) for m in metrics],
                          "verdict": "supports", "reason": "r"} for i in ideas],
           "hypotheses": [{"id": h, "outcome": "unknown"} for h in tree], "cruxes": ["is the step size the bottleneck?"]}
elif role == "lit-plan":
    out = {"question": "Does a lower learning rate help short schedules?", "queries": ["learning rate short schedule"],
           "rubric": ["reports learning rate ablations"], "exclude": ["pretrained"]}
elif role == "lit-triage":
    idx = [int(i) for i in re.findall(r"^\[(\d+)\]", prompt, re.MULTILINE)]
    out = {"scores": [{"index": i, "score": 3 if i == 0 else (2 if i == 1 else 0), "reason": "r"} for i in idx]}
elif role == "lit-extract":
    out = {"cards": [
        {"claim": "lower lr converges faster", "conditions": "small nets", "effect": "-10% steps", "evidence": "ablation",
         "quote": "Lower learning rates converge faster in short schedules on small networks.", "hypotheses": ["H1.1"]},
        {"claim": "invented", "conditions": "-", "effect": "-", "evidence": "-",
         "quote": "This sentence does not appear anywhere in the paper text at all.", "hypotheses": []}]}
elif role == "lit-synthesize":
    cards = re.findall(r"^(C\d{4}) ", prompt, re.MULTILINE)
    out = {"digest": "The literature says lower learning rates help in short schedules.",
           "claims": [{"card": c, "kind": "conditional", "hypotheses": ["H1.1"], "use": "try it"} for c in cards],
           "summary": "one claim"}
elif role == "search":
    out = {"findings": [{"source": "arXiv:0000.00000", "claim": "lower lr helps", "relevance": "H1",
                         "hypotheses": ["H1"]}], "summary": "one finding"}
elif role == "review":
    out = {"critiques": [{"target": "H1", "issue": "single run", "suggestion": "replicate"}], "summary": "ok"}
elif role == "revise":
    if not tree:
        out = {"updates": [], "new": [
            {"key": "a", "parent": "", "statement": "the learning rate is too high", "confidence": 0.6, "source": "seed"},
            {"key": "b", "parent": "a", "statement": "lower lr reduces loss", "confidence": 0.5, "source": "seed"},
            {"key": "c", "parent": "", "statement": "width matters", "confidence": 0.4, "source": "seed"}],
            "summary": "seeded"}
    else:
        target = os.environ.get("FAKE_LLM_REFUTE", "")
        out = {"updates": [{"id": target, "status": "refuted", "confidence": 0.1, "reason": "results"}] if target else [],
               "new": [{"key": "x", "parent": "", "statement": "epochs matter", "confidence": 0.5, "source": "review"}],
               "summary": "revised"}
else:  # ideate
    leaf = sorted(tree, key=lambda h: -h.count("."))[0] if tree else "H1"
    out = {"ideas": [{"hypothesis": leaf, "description": f"lr {v}", "expected": "loss down", "priority": 3}
                     for v in ("0.05", "0.03", "0.02", "0.01", "0.008")] + [
               {"hypothesis": "H99", "description": "bogus", "expected": "-", "priority": 9}],
           "drop": [], "summary": "five ideas"}
print(json.dumps({"type": "result", "is_error": False, "total_cost_usd": 0.01, "structured_output": out,
                  "result": json.dumps(out)}))
