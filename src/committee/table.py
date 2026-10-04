"""One table of every committee result, the same columns for every benchmark.

    uv run python -m committee.table                      # prints the table, writes artifacts/uncertainty_table.{md,csv}

Each row is one committee on one task. The score columns use the benchmark's own
metric (named in the row). The uncertainty columns are the point: does the
committee's disagreement predict its own errors? "Wrong" is the benchmark's
per-item failure: a wrong held-out transition (ARC), a world that misses full
credit or claims on a null (ONC), an unacceptable plan (BioProt), one minus the
reaction-set F1 (SciGym). Agreed means zero disagreement; the halves split the
items at the median disagreement where per-item records exist.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from committee.committee import auroc

ART = Path("artifacts")
COLUMNS = [
    "Benchmark", "Task", "Committee", "n", "Metric", "Single", "Committee score", "Best member", "Gain",
    "AUROC disagreement vs wrong", "Wrong when agreed (n)", "Wrong when split (n)",
    "Wrong, most-agreed half", "Wrong, least-agreed half", "ECE", "Conformal coverage / commit", "Entry",
]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def wilson(p: float | None, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion (Wilson, 1927)."""
    if p is None or n <= 0:
        return None
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def auroc_ci(auc: float | None, n_pos: int, n_neg: int, z: float = 1.96) -> tuple[float, float] | None:
    """Normal interval from the Hanley and McNeil (1982) standard error of an AUROC."""
    if auc is None or n_pos <= 0 or n_neg <= 0:
        return None
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg))
    return (max(0.0, auc - z * se), min(1.0, auc + z * se))


def bootstrap_auroc(dis: list[float], wrong: list[int], draws: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    rng = np.random.default_rng(seed)
    d, w = np.asarray(dis, float), np.asarray(wrong, int)
    vals = []
    for _ in range(draws):
        idx = rng.integers(0, len(d), len(d))
        if 0 < w[idx].sum() < len(idx):
            a = auroc([float(x) for x in d[idx]], [bool(x) for x in w[idx]])
            if a is not None:
                vals.append(a)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 20 else None


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _halves(dis: list[float], wrong: list[int]) -> tuple[str, str, str, str, str]:
    """AUROC, wrong when agreed (n), wrong when split (n), wrong on the most- and least-agreed halves."""
    d, w = np.asarray(dis, float), np.asarray(wrong, int)
    agreed = d == 0
    au = auroc([float(x) for x in d], [bool(x) for x in w]) if 0 < w.sum() < len(w) else None
    order = np.argsort(d, kind="stable")
    half = order[: len(d) // 2], order[len(d) // 2:]
    return (
        _f(au),
        f"{_f(w[agreed].mean())} ({int(agreed.sum())})" if agreed.any() else "0 agreed",
        f"{_f(w[~agreed].mean())} ({int((~agreed).sum())})" if (~agreed).any() else "0 split",
        _f(w[half[0]].mean()),
        _f(w[half[1]].mean()),
        {
            "_auroc": au, "_auroc_ci": bootstrap_auroc(list(d), list(w)) if au is not None else None,
            "_agreed": (float(w[agreed].mean()), int(agreed.sum())) if agreed.any() else None,
            "_split": (float(w[~agreed].mean()), int((~agreed).sum())) if (~agreed).any() else None,
        },
    )


# ---------------------------------------------------------------------- ARC, OPINE-World's program contract


def arc_rows() -> list[dict]:
    summary, cal = _load(ART / "summary_opine.json"), _load(ART / "calibration_opine.json")
    if not summary:
        return []
    rows = []
    for r in summary["rows"]:
        per = r.get("per_transition") or []
        au, agreed, split, top, bottom, num = _halves([p["disagreement"] for p in per], [int(p["error"]) for p in per]) if per else ("n/a",) * 5 + ({},)
        lvl = (cal or {}).get("per_level", {}).get(r["level"], {})
        aci = lvl.get("aci", {})
        rows.append({
            **num, "_n": r["n_test"], "_single": r["single_mean"], "_single_ci": wilson(r["single_mean"], r["n_test"]), "_single_ci_kind": "95% CI",
            "_committee": r["vote"], "_committee_ci": wilson(r["vote"], r["n_test"]),
            "Benchmark": "ARC (OPINE-World)", "Task": r["level"], "Committee": f"{r['n_members']} Devin programs", "n": r["n_test"],
            "Metric": "held-out transition accuracy", "Single": _f(r["single_mean"], 3), "Committee score": _f(r["vote"], 3),
            "Best member": _f(r["member_max"], 3), "Gain": f"{r['vote'] - r['single_mean']:+.3f}",
            "AUROC disagreement vs wrong": au, "Wrong when agreed (n)": agreed, "Wrong when split (n)": split,
            "Wrong, most-agreed half": top, "Wrong, least-agreed half": bottom,
            "ECE": _f(lvl.get("ece_vote_share")), "Conformal coverage / commit": f"{_f(aci.get('coverage'))} / {_f(1 - aci['abstain_rate']) if aci else 'n/a'}",
            "Entry": "R34, R35",
        })
    pooled = summary["pooled_auroc"]; pu, ps = summary["pooled_unanimous"], summary["pooled_split"]
    cp = (cal or {}).get("pooled", {})
    rows.append({
        "_n": pooled["n"], "_single": summary["mean_single"], "_single_ci": wilson(summary["mean_single"], pooled["n"]), "_single_ci_kind": "95% CI",
        "_committee": summary["mean_vote"], "_committee_ci": wilson(summary["mean_vote"], pooled["n"]),
        "_auroc": pooled["auroc"], "_auroc_ci": tuple(pooled["ci95"]), "_agreed": (pu["error"], pu["n"]), "_split": (ps["error"], ps["n"]),
        "Benchmark": "ARC (OPINE-World)", "Task": "7 levels pooled", "Committee": "8 Devin programs", "n": pooled["n"],
        "Metric": "held-out transition accuracy", "Single": _f(summary["mean_single"], 3), "Committee score": _f(summary["mean_vote"], 3),
        "Best member": _f(summary["mean_best_member"], 3), "Gain": f"{summary['mean_vote'] - summary['mean_single']:+.3f}",
        "AUROC disagreement vs wrong": f"{_f(pooled['auroc'])} [{_f(pooled['ci95'][0])}, {_f(pooled['ci95'][1])}]",
        "Wrong when agreed (n)": f"{_f(pu['error'])} ({pu['n']})", "Wrong when split (n)": f"{_f(ps['error'])} ({ps['n']})",
        "Wrong, most-agreed half": _f(1 - cp.get("selective_by_disagreement", {}).get("selective_acc_50", np.nan)),
        "Wrong, least-agreed half": "n/a",
        "ECE": _f(cp.get("ece_vote_share")), "Conformal coverage / commit": f"{_f((cal or {}).get('aci_pooled', {}).get('coverage'))} / n/a",
        "Entry": "R34, R35",
    })
    return rows


# ---------------------------------------------------------------------- ONC-AGI


def _onc_conditions(path: Path) -> dict[str, dict]:
    d = _load(path)
    if not d:
        return {}
    return {("templates" if r["acquisition"] == "disagreement" else r["acquisition"]): r for r in d["conditions"]}


def _onc_wrong(w: dict) -> int:
    return int((not w["is_null"] and w["find"] < 0.999) or (w["is_null"] and not w["restrained"]))


def onc_rows() -> list[dict]:
    rows = []
    sets = [
        ("first 120 full-access worlds", "first120", ["eval_bench_first120_full-access.json", "eval_bench_synth_first120_full-access.json"], "eval_bench_singles_first120_full-access.json", "O9, O10"),
        ("first 30 full-access worlds", "first30", ["eval_bench_synth_first30_full-access.json"], "eval_bench_singles_first30_full-access.json", "O10"),
        ("all 995 full-access worlds", "all", ["eval_bench_all_full-access.json"], None, "O9"),
    ]
    labels = {"templates": "8 templates", "synth:8:qwen": "8 Qwen programs", "synth:4:qwen": "4 Qwen programs", "synth:1:qwen": "1 Qwen program",
              "synth:4:devin": "4 Devin programs", "synth:1:devin": "1 Devin program", "both:8:qwen": "8 templates + 8 Qwen programs",
              "both:4:devin": "8 templates + 4 Devin programs"}
    for task, _, files, singles_file, entry in sets:
        conds: dict[str, dict] = {}
        for f in files:
            conds.update(_onc_conditions(ART / "onc" / f))
        singles = _onc_conditions(ART / "onc" / singles_file) if singles_file else {}
        for key, r in conds.items():
            if key.startswith("single:"):
                continue
            w = r["worlds"]
            au, agreed, split, top, bottom, num = _halves([x["disagreement"][0] if x["disagreement"] else 0.0 for x in w], [_onc_wrong(x) for x in w])
            model = key.split(":")[2] if ":" in key else "templates"
            fams = [model] + (["templates"] if key.startswith("both:") else [])
            s_ds = [s["discovery_score"] for k, s in singles.items() if any(k.startswith(f"single:{m}:") for m in fams) and s["discovery_score"] is not None]
            rows.append({
                **num, "_n": len(w), "_single": float(np.mean(s_ds)) if s_ds else None,
                "_single_ci": (min(s_ds), max(s_ds)) if s_ds else None, "_single_ci_kind": "range over single programs",
                "_committee": r["discovery_score"], "_committee_ci": tuple(r["interval"]),
                "Benchmark": "ONC-AGI v1.0.0rc3", "Task": task, "Committee": labels.get(key, key), "n": len(w),
                "Metric": "Discovery Score", "Single": f"{_f(np.mean(s_ds), 3)} ({min(s_ds):.2f} to {max(s_ds):.2f})" if s_ds else "n/a",
                "Committee score": f"{_f(r['discovery_score'], 3)} [{_f(r['interval'][0])}, {_f(r['interval'][1])}]",
                "Best member": _f(max(s_ds), 3) if s_ds else "n/a", "Gain": f"{r['discovery_score'] - np.mean(s_ds):+.3f}" if s_ds else "n/a",
                "AUROC disagreement vs wrong": au, "Wrong when agreed (n)": agreed, "Wrong when split (n)": split,
                "Wrong, most-agreed half": top, "Wrong, least-agreed half": bottom,
                "ECE": _f(r["p_signal"].get("ece")), "Conformal coverage / commit": "n/a (full access)", "Entry": entry,
            })
    return rows


# ---------------------------------------------------------------------- BioProt


def bioprot_rows() -> list[dict]:
    recs = _load(ART / "bioprot" / "committee.json")
    if not recs:
        return []
    rows = []
    for r in recs:
        acc, conf = r["acceptable"], r.get("conformal_entropy", {})
        n = r["n_protocols"]
        n_wrong = int(round(n * (1 - acc["committee_plan"])))
        rows.append({
            "_n": n, "_single": acc["single_sample"], "_single_ci": wilson(acc["single_sample"], n), "_single_ci_kind": "95% CI",
            "_committee": acc["committee_plan"], "_committee_ci": wilson(acc["committee_plan"], n),
            "_auroc": r.get("auroc_entropy_vs_error"), "_auroc_ci": auroc_ci(r.get("auroc_entropy_vs_error"), n_wrong, n - n_wrong),
            "_agreed": (r["unanimous"]["error"], r["unanimous"]["n"]), "_split": (r["split"]["error"], r["split"]["n"]),
            "Benchmark": "BioProt", "Task": r["committee"], "Committee": f"{r['members_per_protocol']:.1f} sampled plans per protocol", "n": r["n_protocols"],
            "Metric": "acceptable plan rate", "Single": _f(acc["single_sample"], 3), "Committee score": _f(acc["committee_plan"], 3),
            "Best member": f"{_f(acc['any_member'], 3)} (any member)", "Gain": f"{acc['committee_plan'] - acc['single_sample']:+.3f}",
            "AUROC disagreement vs wrong": _f(r.get("auroc_entropy_vs_error")),
            "Wrong when agreed (n)": f"{_f(r['unanimous']['error'])} ({r['unanimous']['n']})", "Wrong when split (n)": f"{_f(r['split']['error'])} ({r['split']['n']})",
            "Wrong, most-agreed half": "n/a", "Wrong, least-agreed half": "n/a", "ECE": "n/a (entropy is not a probability)",
            "Conformal coverage / commit": f"{_f(conf.get('coverage'))} / {_f(conf.get('committed'))}", "Entry": "B4",
        })
    return rows


# ---------------------------------------------------------------------- SciGym


def scigym_rows() -> list[dict]:
    rows = []
    for model in ("qwen", "gptoss"):
        for tag, tol in (("", "tolerance 0.15"), ("_eps50", "tolerance 0.5")):
            s, cal = _load(ART / "scigym" / f"summary_{model}{tag}.json"), _load(ART / "scigym" / f"calibration_{model}{tag}.json")
            if not s or "single_fixed" not in s:
                continue
            single = s["single_fixed"]["rms_f1"]
            for arm in ("committee_probe", "committee_fixed"):
                a = s.get(arm)
                if not a:
                    continue
                c = ((cal or {}).get(arm) or {}).get("system", {})
                aci = c.get("aci", {})
                una, spl = a["unanimous"], a["split"]
                n_wrong = int(round(a["n"] * (1 - c["share_correct"]))) if c.get("share_correct") is not None else 0
                rows.append({
                    "_n": a["n"], "_single": single, "_single_ci": tuple(s["single_fixed"]["rms_f1_ci"]), "_single_ci_kind": "95% CI",
                    "_committee": a["rms_f1"], "_committee_ci": tuple(a["rms_f1_ci"]),
                    "_auroc": a.get("auroc_spread_vs_wrong"), "_auroc_ci": auroc_ci(a.get("auroc_spread_vs_wrong"), n_wrong, a["n"] - n_wrong),
                    "_agreed": (1 - una["f1"], una["n"]) if una.get("f1") is not None else None,
                    "_split": (1 - spl["f1"], spl["n"]) if spl.get("f1") is not None else None,
                    "Benchmark": "SciGym", "Task": f"{model}, {tol}, {a['n']} systems", "Committee": f"4 programs, {arm.split('_')[1]} design", "n": a["n"],
                    "Metric": "reaction-set F1", "Single": _f(single, 3), "Committee score": f"{_f(a['rms_f1'], 3)} [{_f(a['rms_f1_ci'][0])}, {_f(a['rms_f1_ci'][1])}]",
                    "Best member": _f(a.get("rms_f1_best_member"), 3), "Gain": f"{a['rms_f1'] - single:+.3f}",
                    "AUROC disagreement vs wrong": _f(a.get("auroc_spread_vs_wrong")),
                    "Wrong when agreed (n)": f"{_f(1 - una['f1']) if una.get('f1') is not None else 'n/a'} ({una['n']})",
                    "Wrong when split (n)": f"{_f(1 - spl['f1']) if spl.get('f1') is not None else 'n/a'} ({spl['n']})",
                    "Wrong, most-agreed half": "n/a", "Wrong, least-agreed half": "n/a", "ECE": _f(c.get("ece")),
                    "Conformal coverage / commit": f"{_f(aci.get('coverage'))} / {_f(aci.get('committed'))}", "Entry": "B5",
                })
    return rows


def all_rows() -> list[dict]:
    return arc_rows() + onc_rows() + bioprot_rows() + scigym_rows()


def main() -> int:
    rows = all_rows()
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in COLUMNS) + " |" for r in rows]
    text = "\n".join(lines) + "\n"
    (ART / "uncertainty_table.md").write_text(text)
    with (ART / "uncertainty_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(text)
    print(f"{len(rows)} rows; wrote artifacts/uncertainty_table.md and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
