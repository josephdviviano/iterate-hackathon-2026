"""Figures for SciGym: the arms' reaction F1 at each admission tolerance, the committee's
agreement against being right at the system level, and the per-reaction share against truth.

    uv run python -m scigym.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .loop import ART  # noqa: E402

BLUE, LIGHT, GREY, GREEN, ORANGE = "#1E6FD9", "#7FB2F0", "#8A8A8A", "#2E9E5B", "#E8871E"
ARM_STYLE = {"committee_probe": ("committee, probe by disagreement", BLUE), "committee_fixed": ("committee, fixed order", LIGHT),
             "single_fixed": ("single member", GREY), "committee_probe_nocx": ("committee, probe, no counterexample", GREEN)}


def _style(ax, title, ylabel=""):
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def _load(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    out = ART.parent / "figures"
    out.mkdir(exist_ok=True)
    tags = [("", "tolerance 0.15"), ("_eps50", "tolerance 0.5")]
    summaries = {(m, t): _load(f"summary_{m}{t}.json") for m in ("qwen", "gptoss") for t, _ in tags}
    calib = {m: _load(f"calibration_{m}.json") for m in ("qwen", "gptoss")}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 2, 2]})
    # (a) F1 per arm, model and tolerance
    ax = axes[0]
    groups = [(m, t, lab) for m in ("qwen", "gptoss") for t, lab in tags if summaries[(m, t)]]
    x = np.arange(len(groups))
    arms = [a for a in ARM_STYLE if any(a in summaries[(m, t)] for m, t, _ in groups)]
    w = 0.8 / max(1, len(arms))
    for j, arm in enumerate(arms):
        xs, ys, lo, hi = [], [], [], []
        for i, (m, t, _) in enumerate(groups):
            e = summaries[(m, t)].get(arm)
            if e:
                xs.append(x[i] + (j - (len(arms) - 1) / 2) * w); ys.append(e["rms_f1"]); lo.append(e["rms_f1"] - e["rms_f1_ci"][0]); hi.append(e["rms_f1_ci"][1] - e["rms_f1"])
        ax.bar(xs, ys, w, color=ARM_STYLE[arm][1], label=ARM_STYLE[arm][0], yerr=[lo, hi], capsize=2, error_kw={"linewidth": 0.8})
    ax.set_xticks(x, [f"{'Qwen3-Coder-30B' if m == 'qwen' else 'gpt-oss-120b'}\n{lab}" for m, t, lab in groups], fontsize=8.5)
    ax.axhline(0.18, color=GREY, linestyle=":", linewidth=1)
    ax.text(-0.4, 0.47, "dotted line: paper's frontier models (137 systems, 20 iterations): 0.17 to 0.18", ha="left", va="bottom", fontsize=7.5, color=GREY)
    ax.set_ylim(0, 0.5)
    _style(ax, "Reaction F1 per arm (B5, B7), 95% bootstrap intervals", "RMS F1 of the committee's answer")
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.93))
    # (b) system level: share right by agreement quartile
    ax = axes[1]
    for m, mk in (("qwen", "o"), ("gptoss", "s")):
        c = calib[m]
        if not c:
            continue
        for arm in ("committee_probe", "committee_fixed"):
            rel = c.get(arm, {}).get("system", {}).get("reliability")
            if rel:
                ax.plot([b["mean_conf"] for b in rel], [b["share_correct"] for b in rel], marker=mk, color=ARM_STYLE[arm][1],
                        label=f"{'Qwen' if m == 'qwen' else 'gpt-oss'}, {ARM_STYLE[arm][0].split(', ')[1]}", linewidth=1.3)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("committee agreement, 1 - spread (quartile mean)")
    _style(ax, "Systems: agreement vs right (B6)", "share of answers with F1 >= 0.5")
    ax.legend(frameon=False, fontsize=7.5)
    # (c) reaction level: precision by share quartile
    ax = axes[2]
    for m, mk in (("qwen", "o"), ("gptoss", "s")):
        c = calib[m]
        if not c:
            continue
        for arm in ("committee_probe", "committee_fixed"):
            rel = c.get(arm, {}).get("reaction", {}).get("reliability")
            if rel:
                ax.plot([b["mean_share"] for b in rel], [b["precision"] for b in rel], marker=mk, color=ARM_STYLE[arm][1], linewidth=1.3)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("share of members proposing it (quartile mean)")
    _style(ax, "Reactions: share vs truth (B6)", "precision of the proposed reactions")
    fig.tight_layout()
    fig.savefig(out / "scigym.png", dpi=160)
    print(out / "scigym.png")


if __name__ == "__main__":
    main()
