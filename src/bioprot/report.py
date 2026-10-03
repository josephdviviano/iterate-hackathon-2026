"""Risk-coverage curves, AURC with intervals, both baselines, calibration, and the severity sample.

    uv run python -m bioprot.report                 # every condition under artifacts/bioprot

Writes artifacts/bioprot/summary.json, risk_coverage.png and
severity_template.csv, and prints the table rows for RESULTS.md. Risk is
binary (not acceptable) for the headline and continuous (normalised
Levenshtein) as the check that the conclusion does not hinge on the cut.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import load_protocols
from .generate import ART, read_rows
from .metrics import (aurc, bootstrap, brier, coverage_at_risk, ece, oracle_aurc, random_aurc, risk_coverage_curve,
                      selective_risk_at, tie_fraction)
from .uncertainty import self_consistency

SIGNALS = {"self_consistency": "Self-consistency", "verbal": "Verbalised confidence",
           "logprob": "Sequence logprob", "critique": "Self-critique"}
COVERAGES = (0.9, 0.75, 0.5)
RISK_TARGET = 0.1
VERBATIM = 0.9
# Reference categorical palette, fixed slot order; baselines in ink tones.
COLORS = {"self_consistency": "#2a78d6", "verbal": "#eb6834", "logprob": "#1baf7a", "critique": "#eda100"}


def load_condition(d: Path) -> list[dict]:
    unc = {(r["protocol_id"], r["sample_idx"]): r for r in read_rows(d / "uncertainty.jsonl")}
    items = []
    for s in read_rows(d / "scores.jsonl"):
        u = unc.get((s["protocol_id"], s["sample_idx"]))
        if u:
            items.append({**u, **s, "tiebreak": f"{s['protocol_id']}:{s['sample_idx']:02d}"})
    return items


def uncertainty_of(item: dict, signal: str) -> float | None:
    if signal == "self_consistency":
        return item["self_consistency"]
    if signal == "verbal":
        if item["declined"]:
            return None
        return 1.0 if item["abstained"] else (None if item["confidence"] is None else 1 - item["confidence"])
    if signal == "logprob":
        return item["logprob_uncertainty"]
    if signal == "critique":
        return None if item["critique_p_pass"] is None else 1 - item["critique_p_pass"]
    raise KeyError(signal)


def evaluate(items: list[dict], signal: str, risk_kind: str, n_boot: int = 1000) -> dict | None:
    keep = [(it, uncertainty_of(it, signal)) for it in items]
    keep = [(it, u) for it, u in keep if u is not None]
    if len(keep) < 10:
        return None
    risk = np.array([(0.0 if it["acceptable"] else 1.0) if risk_kind == "binary" else it["risk"] for it, _ in keep])
    unc = np.array([u for _, u in keep])
    tb = np.array([it["tiebreak"] for it, _ in keep])
    groups = np.array([it["protocol_id"] for it, _ in keep])

    def stats(idx):
        r, u, t = risk[idx], unc[idx], tb[idx]
        return [aurc(r, u, t), oracle_aurc(r, t)] + [selective_risk_at(r, u, t, c) for c in COVERAGES]

    point = stats(np.arange(len(risk)))
    lo, hi = bootstrap(groups, stats, n_boot=n_boot)
    full = random_aurc(risk)
    out = {"signal": signal, "risk": risk_kind, "n": len(risk), "n_protocols": len(set(groups)),
           "aurc": point[0], "aurc_ci": [lo[0], hi[0]], "random_aurc": full,
           "oracle_aurc": point[1], "oracle_aurc_ci": [lo[1], hi[1]],
           "coverage_at_risk_0.1": coverage_at_risk(risk, unc, tb, RISK_TARGET), "tie_fraction": tie_fraction(unc)}
    for j, c in enumerate(COVERAGES):
        out[f"risk_at_{c}"] = point[2 + j]
        out[f"risk_at_{c}_ci"] = [lo[2 + j], hi[2 + j]]
    out["criterion_aurc_below_random"] = hi[0] < full
    out["criterion_half_coverage_third_lower"] = point[4] <= (2 / 3) * full
    return out


def calibration(items: list[dict]) -> dict:
    n = len(items)
    conf = [(it["confidence"], 1.0 if it["acceptable"] else 0.0) for it in items if it["confidence"] is not None]
    attempted = [it for it in items if not it["abstained"] and not it["declined"]]
    out = {"n": n, "abstain_rate": sum(it["abstained"] for it in items) / n,
           "decline_rate": sum(it["declined"] for it in items) / n,
           "coverage_attempted": len(attempted) / n,
           "precision_attempted": (sum(it["acceptable"] for it in attempted) / len(attempted)) if attempted else None,
           "critique_fail_rate": sum(it["critique_verdict"] == "FAIL" for it in items) / n}
    if conf:
        c, y = zip(*conf)
        out |= {"ece": ece(c, y), "brier": brier(c, y), "mean_confidence": float(np.mean(c)), "accuracy": float(np.mean(y))}
    return out


def memorisation(items: list[dict]) -> dict:
    flagged = [it for it in items if it["verbatim_ratio"] > VERBATIM]
    rest = [it for it in items if it["verbatim_ratio"] <= VERBATIM]
    e = evaluate(rest, "self_consistency", "binary", n_boot=200) if len(rest) >= 10 else None
    return {"n_flagged": len(flagged), "protocols_flagged": sorted({it["protocol_id"] for it in flagged}),
            "aurc_self_consistency_without": e["aurc"] if e else None, "random_without": e["random_aurc"] if e else None}


def highest_risk(items: list[dict], n: int = 20) -> dict:
    """What the top of the risk scale is made of: unrolled repeats and extra
    steps (length ratio above 1 with high recall) or wrong function choices."""
    top = sorted(items, key=lambda it: -it["risk"])[:n]
    return {"length_ratio": float(np.mean([it["n_pred"] / it["n_gt"] for it in top])),
            "function_recall": float(np.mean([it["function_recall"] for it in top])),
            "admissible_fraction": float(np.mean([it["admissible_fraction"] or 0.0 for it in top])),
            "n_empty": sum(it["empty"] for it in top)}


def k_ablation(d: Path, items: list[dict], ks=(3, 5)) -> dict:
    """Self-consistency AURC when only the first k samples of each protocol exist."""
    gens = read_rows(d / "generations.jsonl")
    out = {}
    for k in ks:
        sub = [g for g in gens if g["sample_idx"] < k]
        sc = self_consistency(sub)
        its = [{**it, "self_consistency": sc[(it["protocol_id"], it["sample_idx"])]["self_consistency"]}
               for it in items if (it["protocol_id"], it["sample_idx"]) in sc]
        e = evaluate(its, "self_consistency", "binary", n_boot=200)
        out[str(k)] = {"aurc": e["aurc"], "aurc_ci": e["aurc_ci"], "random": e["random_aurc"], "n": e["n"]} if e else None
    return out


def figure(conditions: dict[str, list[dict]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c for c in conditions if c.endswith("_shuf_human")] or list(conditions)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        items = conditions[name]
        risk_all = np.array([0.0 if it["acceptable"] else 1.0 for it in items])
        tb_all = np.array([it["tiebreak"] for it in items])
        for sig, label in SIGNALS.items():
            keep = [(it, uncertainty_of(it, sig)) for it in items]
            keep = [(it, u) for it, u in keep if u is not None]
            if len(keep) < 10:
                continue
            r = np.array([0.0 if it["acceptable"] else 1.0 for it, _ in keep])
            u = np.array([u for _, u in keep])
            t = np.array([it["tiebreak"] for it, _ in keep])
            cov, sel = risk_coverage_curve(r, u, t)
            ax.plot(cov, sel, color=COLORS[sig], lw=2, label=label)
        cov, sel = risk_coverage_curve(risk_all, risk_all, tb_all)
        ax.plot(cov, sel, color="#0b0b0b", lw=1.5, ls=":", label="Oracle (true risk)")
        ax.axhline(risk_all.mean(), color="#8a8984", lw=1.5, ls="--", label="Random order")
        ax.set_title(name.split("_")[0], fontsize=11)
        ax.set_xlabel("Coverage (fraction of plans retained)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("Selective risk (share not acceptable)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("BioProt protocol generation: does the agent know which plans are bad?", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(out, dpi=160, bbox_inches="tight")


def severity_template(items: list[dict], gens: list[dict], out: Path, n: int = 20) -> None:
    """One sample per protocol (sample 0), 20 protocols spread evenly over the risk
    ranks, for a wet-lab reader to mark: correct, cosmetic, or wasteful/dangerous."""
    by_id = {p.id: p for p in load_protocols()}
    raw = {(g["protocol_id"], g["sample_idx"]): g["raw_output"] for g in gens}
    first = sorted((it for it in items if it["sample_idx"] == 0), key=lambda it: (it["risk"], it["protocol_id"]))
    picks = [first[int(round(i))] for i in np.linspace(0, len(first) - 1, n)] if first else []
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol_id", "title", "risk", "acceptable", "ground_truth_calls", "generated_plan",
                    "label(correct|cosmetic|wasteful_or_dangerous)", "notes"])
        for it in picks:
            p = by_id[it["protocol_id"]]
            w.writerow([p.id, p.title, f"{it['risk']:.3f}", it["acceptable"], "\n".join(p.calls),
                        raw.get((p.id, 0), ""), "", ""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--headline", default="qwen_shuf_human", help="condition for the severity template")
    a = ap.parse_args()
    conditions = {d.name: load_condition(d) for d in sorted(ART.iterdir())
                  if (d / "scores.jsonl").exists() and (d / "uncertainty.jsonl").exists()}
    conditions = {k: v for k, v in conditions.items() if v}
    summary = {"threshold": 0.4, "coverages": COVERAGES, "verbatim_flag": VERBATIM, "conditions": {}}
    print("| Condition | Signal | Risk | n | AURC [95% CI] | Random | Oracle | Risk at 0.9 / 0.75 / 0.5 coverage | Cov. at risk 0.1 | Ties |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, items in conditions.items():
        entry = {"n": len(items), "n_protocols": len({it["protocol_id"] for it in items}),
                 "mean_risk": float(np.mean([it["risk"] for it in items])),
                 "acceptable_rate": float(np.mean([it["acceptable"] for it in items])),
                 "function_precision": float(np.mean([it["function_precision"] for it in items if it["function_precision"] is not None])),
                 "function_recall": float(np.mean([it["function_recall"] for it in items])),
                 "n_empty": sum(it["empty"] for it in items), "highest_risk_20": highest_risk(items),
                 "signals": {}, "calibration": calibration(items), "memorisation": memorisation(items),
                 "k_ablation": k_ablation(ART / name, items)}
        for sig in SIGNALS:
            for kind in ("binary", "continuous"):
                e = evaluate(items, sig, kind, a.n_boot)
                if e is None:
                    continue
                entry["signals"][f"{sig}/{kind}"] = e
                ci = e["aurc_ci"]
                pts = " / ".join(f"{e[f'risk_at_{c}']:.2f}" for c in COVERAGES)
                print(f"| {name} | {SIGNALS[sig]} | {kind} | {e['n']} | {e['aurc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                      f"{e['random_aurc']:.3f} | {e['oracle_aurc']:.3f} | {pts} | {e['coverage_at_risk_0.1']:.2f} | {e['tie_fraction']:.2f} |")
        summary["conditions"][name] = entry
        c = entry["calibration"]
        print(f"  {name}: risk {entry['mean_risk']:.3f}, acceptable {entry['acceptable_rate']:.2f}, "
              f"precision {entry['function_precision']:.3f}, recall {entry['function_recall']:.3f}, "
              f"abstain {c['abstain_rate']:.2f}, declined {c['decline_rate']:.2f}, "
              f"ECE {c.get('ece', float('nan')):.3f}, Brier {c.get('brier', float('nan')):.3f}, "
              f"critique FAIL {c['critique_fail_rate']:.2f}, verbatim flagged {entry['memorisation']['n_flagged']}, "
              f"empty plans {entry['n_empty']}, top-20 risk: length x{entry['highest_risk_20']['length_ratio']:.1f}, "
              f"recall {entry['highest_risk_20']['function_recall']:.2f}")
    (ART / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    figure(conditions, ART / "risk_coverage.png")
    if a.headline in conditions:
        severity_template(conditions[a.headline], read_rows(ART / a.headline / "generations.jsonl"),
                          ART / "severity_template.csv")


if __name__ == "__main__":
    main()
