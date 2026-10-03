"""Ledger of every idea the loop has seen: proposed, scored, decided, measured.

One row per idea. `decision` is run or rejected (the committee's veto) or audit
(a rejected idea run anyway to measure the veto). `outcome` is pending,
success (measured positive at matched cost), failed (run, no improvement),
or vetoed (rejected and not run). Rendered to Markdown for the record.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "artifacts" / "ideafilter" / "ledger.jsonl"
SUCCESS_PP, SUCCESS_TIME = 0.10, -0.02


def load() -> list[dict]:
    return [json.loads(l) for l in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


def save(rows: list[dict]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text("".join(json.dumps(r) + "\n" for r in rows))


def outcome_of(dpp: float | None, dtime: float | None, se: float | None, paired: bool = True) -> str:
    """Screening runs (unpaired time, one container per config) decide on accuracy only; the time rule needs paired timing."""
    if dpp is None:
        return "pending"
    if not paired:
        return "success" if dpp >= max(SUCCESS_PP, 2 * (se or 0)) else "failed"
    acc_win = dpp >= max(SUCCESS_PP, 2 * (se or 0)) and (dtime is None or dtime <= 0.01)
    time_win = dtime is not None and dtime <= SUCCESS_TIME and dpp >= -0.05
    return "success" if (acc_win or time_win) else "failed"


def add(rows: list[dict], round_no: int, idea: dict, decision: str, rule: str) -> dict:
    row = {"round": round_no, "id": f"r{round_no}-{idea['id']}", "levers": idea["levers"], "proposition": idea.get("proposition", ""),
           "p_mean": idea.get("p_mean"), "pred_dpp": idea.get("dpp_mean"), "pred_dtime": idea.get("dtime_mean"), "spread": idea.get("spread"),
           "decision": decision, "rule": rule, "dpp": None, "dtime": None, "se": None, "n": None,
           "outcome": "vetoed" if decision == "rejected" else "pending"}
    rows.append(row)
    return row


def record_measurement(row: dict, dpp: float, dtime: float | None, se: float | None, n: int, paired: bool = True) -> None:
    row.update({"dpp": round(dpp, 3), "dtime": None if dtime is None else round(dtime, 4), "se": se, "n": n, "paired_time": paired,
                "outcome": outcome_of(dpp, dtime, se, paired)})


def render(rows: list[dict]) -> str:
    counts = {k: sum(r["outcome"] == k for r in rows) for k in ("success", "failed", "vetoed", "pending", "invalid")}
    confirmed = sum(r["outcome"] == "success" and r["decision"] == "confirm" for r in rows)
    audited = [r for r in rows if r["decision"] == "audit" and r["dpp"] is not None]
    false_rejects = sum(r["outcome"] == "success" for r in audited)
    out = ["# Idea ledger", "",
           f"Success {counts['success']} (confirmed on fresh seeds with paired timing: {confirmed}), failed {counts['failed']}, "
           f"vetoed {counts['vetoed']}, pending {counts['pending']}, invalid or duplicate {counts['invalid']}. Audited rejections {len(audited)}, of which false rejects {false_rejects}. "
           "Screening rows decide on accuracy only (unpaired time); a success counts for adoption once confirmed.", "",
           "| Round | Id | Levers | Committee p / pred dpp / pred dtime | Decision | Measured dpp / dtime (n) | Outcome |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        meas = "" if r["dpp"] is None else f"{r['dpp']:+.2f} / {('%+.3f' % r['dtime']) if r['dtime'] is not None else 'n/a'} ({r['n']})"
        pred = f"{r['p_mean']:.2f} / {r['pred_dpp']:+.2f} / {r['pred_dtime']:+.3f}" if r["p_mean"] is not None else ""
        out.append(f"| {r['round']} | {r['id']} | `{json.dumps(r['levers'])}` | {pred} | {r['decision']} | {meas} | {r['outcome']} |")
    return "\n".join(out) + "\n"


def write_md(rows: list[dict]) -> None:
    (LEDGER.parent / "ledger.md").write_text(render(rows))
