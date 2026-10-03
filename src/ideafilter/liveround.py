"""Live round: propose new arms over the lab recipe's lever space, score them with the committee,
select a committee set and a random set, and write the sweep that the Modal runner evaluates.

The proposer is one model call in the X-005 format (proposition, mechanism, prediction) restricted
to levers the lab recipe already implements, excluding settings the programme has tested. The
committee's predictions are stored before any sweep runs, so the comparison is preregistered.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .backtest import LOOP, S, findings
from .committee import _ask, aggregate, predict

LAB = LOOP / "ctx/lab_config.py"


def lever_space() -> dict:
    """Field name -> default literal, parsed from the lab RecipeConfig dataclass."""
    out = {}
    for m in re.finditer(r"^    (\w+): [^=\n]+ = (.+?)(?:\s+#.*)?$", LAB.read_text(), re.M):
        out[m.group(1)] = m.group(2).strip()
    return out


def tested_settings(arms: list[dict]) -> list[str]:
    return sorted({json.dumps(a["levers"], sort_keys=True) for a in arms})


PROPOSER = """\
You propose changes to a CIFAR-100 speedrun recipe for a 10-seed A100 sweep. Score: mean
prepare+train time at a 40-seed mean top-1 of at least 75.2 percent. Only levers that the lab
recipe already implements may be used (the dataclass below); no new code. Propose {n} distinct arms,
each one or two lever settings, that the programme has NOT tested (the tested settings are listed).
Prefer arms whose mechanism is specific to this short, underfit regime. Answer with a JSON list only:
[{{"id": "L1", "levers": {{"<field>": <value>, ...}}, "proposition": "...", "mechanism": "...",
  "prediction": "<expected pp and time change>"}}, ...]
"""


def propose(n: int, model: str, ledger_rows: list[dict] | None = None, base: dict | None = None) -> list[dict]:
    arms = json.load(open(S / "ideas_dataset.json"))
    fnd = findings()
    lines = [f"- {f['id']} ({f['kind']}): {f['observation'][:600]}" for f in fnd]
    recent = ""
    if base:
        recent += ("\n\n# The current base (the control every arm is compared to). Do not propose a lever at the value it already has here; "
                   "a change must differ from this.\n\n" + json.dumps(base) +
                   "\n\nScreening measures accuracy only (compilation off, unpaired hosts). Levers whose only effect is time "
                   "(compile_loss, coordinate_descent, cudnn_benchmark_limit, whiten_grad_off) cannot be scored here; do not propose them.")
    if ledger_rows:
        recent = "\n\n# This loop's own rounds so far (measured against the control; do not repeat these settings or near-variants of the clear losers)\n\n" + "\n".join(
            f"- round {r['round']} {r['id']}: {json.dumps(r['levers'])} -> " + (f"measured {r['dpp']:+.2f} pp, time {r['dtime']:+.3f}" if r.get("dpp") is not None else r["outcome"])
            for r in ledger_rows)
    prompt = ("# Lab recipe levers\n\n```python\n" + LAB.read_text() + "\n```\n\n# Already tested settings (do not repeat)\n\n" +
              "\n".join(tested_settings(arms)) + "\n\n# Findings so far\n\n" + "\n".join(lines) + recent + "\n\nReply with the JSON list only.")
    text = _ask(prompt, model, timeout_s=300, system=PROPOSER.format(n=n) + "You have no tools.")
    start, end = text.find("["), text.rfind("]")
    ideas = json.loads(text[start:end + 1])
    space = lever_space()
    ok = [i for i in ideas if i.get("levers") and all(k in space for k in i["levers"])]
    print(f"proposed {len(ideas)}, {len(ok)} use only implemented levers")
    return ok


def score(ideas: list[dict], model: str) -> list[dict]:
    fnd = findings()
    from .backtest import context_for
    ctx = context_for("none", fnd)  # every finding: these arms are new
    out = []
    for i in ideas:
        text = (f"Proposed arm {i['id']}: {json.dumps(i['levers'])}\nProposition: {i.get('proposition')}\nMechanism: {i.get('mechanism')}\n"
                f"Author's prediction: {i.get('prediction')}\nSweep: 10 seeds against the submitted defaults.")
        agg = aggregate(predict(text, ctx, model=model, parallel=4))
        out.append({**i, **agg})
        print(f"{i['id']:4s} p_mean {agg['p_mean']:.2f} spread {agg['spread']:.2f} dpp {agg['dpp_mean']:+.2f} dtime {agg['dtime_mean']:+.3f} | {json.dumps(i['levers'])}", flush=True)
    return out


def select(scored: list[dict], k: int, seed: int) -> dict:
    by_committee = sorted(scored, key=lambda d: -d["p_mean"])[:k]
    rest = [d for d in scored if d not in by_committee]
    rng = random.Random(seed)
    by_random = rng.sample(scored, k)  # random draws from the whole pool, may overlap the committee set
    return {"committee": [d["id"] for d in by_committee], "random": [d["id"] for d in by_random]}


def toml_value(v):
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, str): return json.dumps(v)
    if isinstance(v, (list, tuple)): return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return repr(v)


def write_sweep(scored: list[dict], chosen: dict, path: Path, seeds: list[int]) -> None:
    ids = sorted(set(chosen["committee"]) | set(chosen["random"]))
    lines = ["# Live round: arms proposed by the generator, chosen by the committee or at random; preregistered in liveround.json.",
             'name = "l1-live"', 'submission = "research/lab_recipe"', 'python = ".venv-blackwell/bin/python"',
             f"seeds = {seeds}", "devices = [1]", "slots_per_device = 1", "", "[base]",
             "widths = [128, 384, 640]", "block_depth = 3", "translate = 2", "res_schedule = [[0.0, 20], [0.5, 32]]",
             "scaling_factor = 0.16666666666666666", 'global_pool = "flatmax"', "compile = true", "fused_sgd = true",
             "epochs = 8.25", "whiten_grad_off = true", "", "[[configs]]  # control"]
    for i in scored:
        if i["id"] in ids:
            lines.append(f"[[configs]]  # {i['id']}: {i.get('proposition', '')[:80]}")
            for k, v in i["levers"].items():
                lines.append(f"{k} = {toml_value(v)}")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default="sonnet"); ap.add_argument("--proposer-model", default="opus")
    ap.add_argument("--sweep-out", default=str(S / "l1-live.toml"))
    a = ap.parse_args()
    ideas = propose(a.n, a.proposer_model)
    scored = score(ideas, a.model)
    chosen = select(scored, a.k, seed=1)
    json.dump({"ideas": scored, "chosen": chosen}, open(S / "liveround.json", "w"), indent=1)
    write_sweep(scored, chosen, Path(a.sweep_out), seeds=list(range(4000, 4010)))
    print("chosen", chosen); print("sweep written to", a.sweep_out)


if __name__ == "__main__":
    main()
