"""Correctness of each generation against the ground-truth call sequence.

    uv run python -m bioprot.score --model qwen

The headline risk is the Levenshtein distance between the predicted and the
ground-truth function sequences, one symbol per call, divided by the number
of ground-truth calls (BioPlanner's metric). A protocol is acceptable at or
below THRESHOLD, set before any generation was scored.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from typing import Sequence

from .data import load_protocols, split_pseudocode, strip_definitions
from .generate import ART, add_condition_args, condition_name, read_rows

THRESHOLD = 0.4


def levenshtein(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalized_levenshtein(pred: Sequence, gt: Sequence) -> float:
    return levenshtein(pred, gt) / len(gt)


def symmetric_distance(a: Sequence, b: Sequence) -> float:
    """Levenshtein over the longer length, in [0, 1]; for comparing two samples."""
    return levenshtein(a, b) / max(len(a), len(b), 1)


def precision_recall(gt: Sequence, pred: Sequence) -> tuple[float | None, float]:
    """Repeated calls count as many times as they occur, as in the reference harness."""
    tp = sum((Counter(gt) & Counter(pred)).values())
    return (tp / len(pred) if pred else None), tp / len(gt)


def verbatim_ratio(pred_text: str, gt_body: str) -> float:
    """Character similarity of the call blocks after whitespace is collapsed. A
    value near 1 marks a protocol reproduced from memory, not planned."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    return difflib.SequenceMatcher(None, norm(pred_text), norm(gt_body), autojunk=False).ratio()


def score_row(row: dict, protocol) -> dict:
    pred = row["parsed_function_sequence"]
    gt = list(protocol.calls)
    p, r = precision_recall(gt, pred)
    names = set(protocol.function_names)
    risk = normalized_levenshtein(pred, gt)
    body = strip_definitions(row["raw_output"])
    return {k: row[k] for k in ("protocol_id", "model", "condition", "sample_idx", "seed")} | {
        "risk": risk, "acceptable": risk <= THRESHOLD, "threshold": THRESHOLD,
        "levenshtein": levenshtein(pred, gt), "n_pred": len(pred), "n_gt": len(gt),
        "function_precision": p, "function_recall": r,
        "admissible_fraction": (sum(c in names for c in pred) / len(pred)) if pred else None,
        "verbatim_ratio": verbatim_ratio(body, split_pseudocode(protocol.pseudocode)[1]),
        "empty": not pred,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    add_condition_args(ap)
    a = ap.parse_args()
    cond = condition_name(a.model, not a.unshuffled, a.description, a.temperature)
    by_id = {p.id: p for p in load_protocols()}
    rows = read_rows(ART / cond / "generations.jsonl")
    scores = [score_row(r, by_id[r["protocol_id"]]) for r in rows]
    out = ART / cond / "scores.jsonl"
    out.write_text("".join(json.dumps(s) + "\n" for s in scores))
    n = len(scores)
    mean = lambda k: sum(s[k] for s in scores if s[k] is not None) / max(1, sum(s[k] is not None for s in scores))
    print(f"{cond}: {n} samples, risk {mean('risk'):.3f}, acceptable {mean('acceptable'):.2f}, "
          f"function precision {mean('function_precision'):.3f}, recall {mean('function_recall'):.3f}, "
          f"empty {sum(s['empty'] for s in scores)}, verbatim>0.9: {sum(s['verbatim_ratio'] > 0.9 for s in scores)}")


if __name__ == "__main__":
    main()
