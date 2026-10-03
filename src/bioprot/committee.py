"""The program-committee method on BioProt: admission, the plurality plan, disagreement, and the agreed-prefix gate.

    uv run python -m bioprot.committee            # committees of 5 per model and the cross-family committee of 15

The members of a protocol are its stored samples. A member is admitted when
its plan is non-empty and calls only the given functions (the function set is
the verifier here; there is no replay). Admitted members vote with equal
weight over identical call sequences: the plurality sequence is the
committee's plan and the normalised entropy of the vote is its disagreement,
as in `committee.committee`. The step gate executes the longest prefix that
every admitted member shares and stops at the first disagreement, which is
the step to put to a human.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from .conformal import aci
from .data import Protocol, load_protocols
from .generate import ART, read_rows
from .metrics import aurc, bootstrap, oracle_aurc, random_aurc, selective_risk_at
from .score import THRESHOLD, normalized_levenshtein, symmetric_distance

MODELS = ("qwen", "gptoss", "mistral")


@dataclass
class ProtocolCommittee:
    protocol_id: str
    n_total: int
    members: list[list[str]]
    plan: list[str]
    share: float
    disagreement: float
    spread: float
    n_distinct: int
    agreed_prefix: int


def admitted(seq: list[str], names: set[str]) -> bool:
    return bool(seq) and all(c in names for c in seq)


def common_prefix(seqs: list[list[str]]) -> int:
    n = 0
    for calls in zip(*seqs):
        if len(set(calls)) > 1:
            break
        n += 1
    return n


def build(rows: list[dict], protocol: Protocol) -> ProtocolCommittee | None:
    names = set(protocol.function_names)
    seqs = [list(r["parsed_function_sequence"]) for r in sorted(rows, key=lambda r: (r["model"], r["sample_idx"]))]
    members = [s for s in seqs if admitted(s, names)]
    if not members:
        return None
    counts = Counter(tuple(s) for s in members)
    plan = list(max(counts, key=lambda k: (counts[k], -[tuple(s) for s in members].index(k))))
    n = len(members)
    h = -sum(c / n * math.log(c / n) for c in counts.values()) / math.log(n) if len(counts) > 1 and n > 1 else 0.0
    pairs = [symmetric_distance(a, b) for i, a in enumerate(members) for b in members[i + 1:]]
    return ProtocolCommittee(protocol.id, len(seqs), members, plan, counts[tuple(plan)] / n, h,
                             float(np.mean(pairs)) if pairs else 0.0, len(counts), common_prefix(members))


def auroc(scores, labels) -> float | None:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or not y.any():
        return None
    pos, neg = s[y], s[~y]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def step_gate(c: ProtocolCommittee, gt: list[str]) -> dict:
    """Execute the agreed prefix of the plurality plan. e is the first step where the
    plan departs from the expert sequence; d is where the members first disagree."""
    e = common_prefix([c.plan, gt])
    d = c.agreed_prefix if c.n_distinct > 1 else len(c.plan)
    return {"n_steps": len(c.plan), "first_error": e, "first_disagreement": d, "executed": d,
            "wrong_executed": max(0, d - e), "wrong_if_all": len(c.plan) - e,
            "disagreement_not_after_error": (d <= e) if c.n_distinct > 1 else None}


def evaluate(committees: list[ProtocolCommittee], by_id: dict[str, Protocol], single: dict[str, dict],
             label: str, n_boot: int = 1000) -> dict:
    ids = [c.protocol_id for c in committees]
    gt = {pid: list(by_id[pid].calls) for pid in ids}
    risk = np.array([normalized_levenshtein(c.plan, gt[c.protocol_id]) for c in committees])
    err = risk > THRESHOLD
    member_risk = np.array([np.mean([normalized_levenshtein(m, gt[c.protocol_id]) for m in c.members]) for c in committees])
    best_risk = np.array([min(normalized_levenshtein(m, gt[c.protocol_id]) for m in c.members) for c in committees])
    single_risk = np.array([single[pid]["risk"] for pid in ids])
    dis = np.array([c.disagreement for c in committees])
    spread = np.array([c.spread for c in committees])
    tb = np.array(ids)
    unanimous = dis == 0

    def curve(r, u):
        return {"aurc": aurc(r, u, tb), "aurc_ci": list(bootstrap(tb, lambda i: aurc(r[i], u[i], tb[i]), n_boot)),
                "risk_at_0.5": selective_risk_at(r, u, tb, 0.5), "random": random_aurc(r), "oracle": oracle_aurc(r, tb)}

    rb = err.astype(float)
    sb = (single_risk > THRESHOLD).astype(float)
    curves = {"committee: vote entropy": curve(rb, dis), "committee: mean pairwise distance": curve(rb, spread)}
    for key, name in (("verbal", "single sample: verbalised confidence"), ("logprob", "single sample: logprob"),
                      ("critique", "single sample: self-critique")):
        u = np.array([single[pid].get(key) if single[pid].get(key) is not None else np.nan for pid in ids])
        keep = ~np.isnan(u)
        if keep.sum() >= 20:
            curves[name] = curve(sb[keep], u[keep]) | {"n": int(keep.sum())}
            curves[name]["aurc_ci"] = list(bootstrap(tb[keep], lambda i: aurc(sb[keep][i], u[keep][i], tb[keep][i]), n_boot))
    gates = [step_gate(c, gt[c.protocol_id]) for c in committees]
    executed = sum(g["executed"] for g in gates)
    steps = sum(g["n_steps"] for g in gates)
    cov = executed / steps
    matched = sum(max(0, int(round(cov * g["n_steps"])) - g["first_error"]) for g in gates) / max(1, sum(int(round(cov * g["n_steps"])) for g in gates))
    split_gates = [g for g in gates if g["disagreement_not_after_error"] is not None]
    return {
        "committee": label, "n_protocols": len(committees),
        "members_per_protocol": float(np.mean([len(c.members) for c in committees])),
        "admission_rate": float(sum(len(c.members) for c in committees) / sum(c.n_total for c in committees)),
        "n_distinct_mean": float(np.mean([c.n_distinct for c in committees])),
        "acceptable": {"committee_plan": float(np.mean(~err)), "single_sample": float(np.mean(single_risk <= THRESHOLD)),
                       "member_mean": float(np.mean(member_risk <= THRESHOLD)), "any_member": float(np.mean(best_risk <= THRESHOLD))},
        "mean_risk": {"committee_plan": float(risk.mean()), "single_sample": float(single_risk.mean()),
                      "member_mean": float(member_risk.mean()), "best_member": float(best_risk.mean())},
        "unanimous": {"n": int(unanimous.sum()), "error": float(err[unanimous].mean()) if unanimous.any() else None},
        "split": {"n": int((~unanimous).sum()), "error": float(err[~unanimous].mean()) if (~unanimous).any() else None},
        "auroc_entropy_vs_error": auroc(dis, err), "auroc_spread_vs_error": auroc(spread, err),
        "curves": curves,
        "conformal_entropy": aci(dis, ~err, alpha=0.1),
        "step_gate": {"coverage": cov, "wrong_executed_share": sum(g["wrong_executed"] for g in gates) / executed,
                      "wrong_share_if_all": sum(g["wrong_if_all"] for g in gates) / steps,
                      "wrong_share_matched_prefix": matched,
                      "split_protocols": len(split_gates),
                      "disagreement_not_after_error": float(np.mean([g["disagreement_not_after_error"] for g in split_gates])) if split_gates else None},
    }


def single_signals(cond: str) -> dict[str, dict]:
    """Sample 0 of a condition: its risk and the three single-sample uncertainties."""
    sc = {r["protocol_id"]: r for r in read_rows(ART / cond / "scores.jsonl") if r["sample_idx"] == 0}
    un = {r["protocol_id"]: r for r in read_rows(ART / cond / "uncertainty.jsonl") if r["sample_idx"] == 0}
    out = {}
    for pid, s in sc.items():
        u = un.get(pid, {})
        conf = u.get("confidence")
        out[pid] = {"risk": s["risk"],
                    "verbal": 1.0 if u.get("abstained") else (None if conf is None else 1 - conf),
                    "logprob": u.get("logprob_uncertainty"),
                    "critique": None if u.get("critique_p_pass") is None else 1 - u["critique_p_pass"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="shuf_human")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    by_id = {p.id: p for p in load_protocols()}
    rows = {m: read_rows(ART / f"{m}_{a.suffix}" / "generations.jsonl") for m in MODELS}
    results = []
    for label, pool, single_cond in [(m, rows[m], f"{m}_{a.suffix}") for m in MODELS] + \
                                    [("cross-family (15)", sum(rows.values(), []), f"qwen_{a.suffix}")]:
        by_p = defaultdict(list)
        for r in pool:
            by_p[r["protocol_id"]].append(r)
        committees = [c for pid, rs in sorted(by_p.items()) if (c := build(rs, by_id[pid]))]
        results.append(evaluate(committees, by_id, single_signals(single_cond), label, a.n_boot))
    (ART / "committee.json").write_text(json.dumps(results, indent=1, default=float))
    print("| Committee | Admitted | Distinct plans | Acceptable: plan / single / member mean / any member | Unanimous n, error | Split n, error | AUROC entropy | AUROC spread |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        acc, u, s = r["acceptable"], r["unanimous"], r["split"]
        print(f"| {r['committee']} | {r['admission_rate']:.2f} | {r['n_distinct_mean']:.1f} | {acc['committee_plan']:.2f} / {acc['single_sample']:.2f} / "
              f"{acc['member_mean']:.2f} / {acc['any_member']:.2f} | {u['n']}, {u['error']:.2f} | {s['n']}, {s['error']:.2f} | "
              f"{r['auroc_entropy_vs_error']:.2f} | {r['auroc_spread_vs_error']:.2f} |")
    print("\n| Committee | Uncertainty | AURC [95% CI] | Random | Oracle | Risk at 0.5 coverage |")
    print("|---|---|---|---|---|---|")
    for r in results:
        for name, c in r["curves"].items():
            print(f"| {r['committee']} | {name} | {c['aurc']:.3f} [{c['aurc_ci'][0]:.3f}, {c['aurc_ci'][1]:.3f}] | {c['random']:.3f} | {c['oracle']:.3f} | {c['risk_at_0.5']:.2f} |")
    print("\n| Committee | Conformal on entropy: coverage, committed, accuracy, abstain | Step gate: coverage | wrong among executed | wrong if all executed | wrong at matched prefix | split protocols where disagreement is not after the first error |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        c, g = r["conformal_entropy"], r["step_gate"]
        print(f"| {r['committee']} | {c['coverage']:.3f}, {c['committed']:.2f}, {c['accuracy_when_committed']:.2f}, {c['abstain']:.2f} | "
              f"{g['coverage']:.2f} | {g['wrong_executed_share']:.3f} | {g['wrong_share_if_all']:.3f} | {g['wrong_share_matched_prefix']:.3f} | "
              f"{g['disagreement_not_after_error']:.2f} ({g['split_protocols']}) |")


if __name__ == "__main__":
    main()
