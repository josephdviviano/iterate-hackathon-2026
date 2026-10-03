#!/usr/bin/env python3
"""Stand-in for `claude -p ... --json-schema ...` in tests: canned structured output per ROLE.

Logs each call (role, tools flag, cwd, files in cwd) to $FAKE_LLM_LOG as JSON lines.
$FAKE_LLM_REFUTE=H1 makes revise refute that hypothesis when the tree is not empty.
"""
import json
import os
import re
import sys

argv = sys.argv[1:]
prompt = argv[argv.index("-p") + 1]
tools = argv[argv.index("--tools") + 1]
role = re.search(r"^ROLE: (\w+)", prompt, re.MULTILINE).group(1)
with open(os.environ["FAKE_LLM_LOG"], "a") as f:
    f.write(json.dumps({"role": role, "tools": tools, "cwd_files": os.listdir("."),
                        "model": argv[argv.index("--model") + 1], "has_results": "=== RESULTS" in prompt}) + "\n")
tree = re.findall(r"^(H[\d.]+)\t\d\t[^\t]*\topen\t", prompt, re.MULTILINE)
if role == "search":
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
