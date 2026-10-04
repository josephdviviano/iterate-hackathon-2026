"""The headline figure: one committee per environment, score against a single program,
whether disagreement predicts error, and the error rate when the committee agrees.

    uv run python -m committee.headline        # writes artifacts/figures/headline.{png,svg}

Rows come from ``committee.table``, so the figure follows the table as results land.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from committee.table import all_rows, wilson  # noqa: E402

BLUE, ORANGE, GREY, INK, MUTED = "#1E6FD9", "#E8871E", "#8A8A8A", "#222222", "#6B6B6B"
OUT = Path("artifacts/figures/headline")

# (benchmark prefix, task prefix, committee label) -> row label
PICKS = [
    (("ARC", "7 levels pooled", "8 Devin programs"), "ARC, OPINE-World\n8 Devin programs, 376 transitions"),
    (("ONC", "first 120", "8 templates"), "ONC-AGI\n8 templates, 120 worlds"),
    (("ONC", "first 30", "4 Devin programs"), "ONC-AGI\n4 Devin programs, 30 worlds"),
    (("BioProt", "cross-family", ""), "BioProt\n15 sampled plans, 100 protocols"),
    (("SciGym", "qwen, tolerance 0.15", "probe"), "SciGym\n4 Qwen programs, 27 systems"),
    (("SciGym", "gptoss, tolerance 0.15", "probe"), "SciGym\n4 gpt-oss programs, 27 systems"),
]


def pick(rows: list[dict]) -> list[tuple[str, dict]]:
    out = []
    for (bench, task, committee), label in PICKS:
        for r in rows:
            if r["Benchmark"].startswith(bench) and r["Task"].startswith(task) and committee in r["Committee"]:
                out.append((label, r))
                break
    return out


def _bar(ax, y, value, ci, color, filled: bool, label: str | None = None, text: str | None = None, dy: float = 0.0):
    if value is None:
        return
    if ci:
        ax.plot([ci[0], ci[1]], [y + dy, y + dy], color=color, linewidth=1.6, solid_capstyle="butt", zorder=2)
    ax.plot(value, y + dy, marker="o", markersize=8, color=color, markerfacecolor=color if filled else "white",
            markeredgewidth=1.8, linestyle="none", label=label, zorder=3)
    if text:
        ax.annotate(text, (value, y + dy), xytext=(0, 7 if dy >= 0 else -12), textcoords="offset points", ha="center", fontsize=7.5, color=INK)


def _panel(ax, title: str, xlabel: str):
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_xlim(-0.02, 1.1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)


def main() -> int:
    picked = pick(all_rows())
    n = len(picked)
    ys = np.arange(n)[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.95 * n + 1.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax in axes:
        ax.set_yticks(ys)
    axes[0].set_yticklabels([label for label, _ in picked], fontsize=8.5, color=INK)

    ax = axes[0]
    _panel(ax, "Score: single program against committee", "the benchmark's metric (0 to 1)")
    for y, (_, r) in zip(ys, picked):
        _bar(ax, y, r.get("_single"), r.get("_single_ci"), ORANGE, False, text=f"{r['_single']:.2f}" if r.get("_single") is not None else None, dy=0.16)
        _bar(ax, y, r.get("_committee"), r.get("_committee_ci"), BLUE, True, text=f"{r['_committee']:.2f}", dy=-0.16)
    ax.plot([], [], marker="o", color=ORANGE, markerfacecolor="white", markeredgewidth=1.8, linestyle="none", label="single program (mean)")
    ax.plot([], [], marker="o", color=BLUE, linestyle="none", label="committee")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = axes[1]
    _panel(ax, "Does disagreement predict error?", "AUROC of disagreement against a wrong item")
    ax.axvline(0.5, color=GREY, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.5, ys[0] + 0.62, "chance", ha="center", fontsize=8, color=MUTED)
    for y, (_, r) in zip(ys, picked):
        _bar(ax, y, r.get("_auroc"), r.get("_auroc_ci"), BLUE, True, text=f"{r['_auroc']:.2f}" if r.get("_auroc") is not None else None)

    ax = axes[2]
    _panel(ax, "Error when the committee agrees or splits", "share of items wrong")
    for y, (_, r) in zip(ys, picked):
        ag, sp = r.get("_agreed"), r.get("_split")
        if ag:
            _bar(ax, y, ag[0], wilson(ag[0], ag[1]), BLUE, True, text=f"{ag[0]:.2f} (n {ag[1]})", dy=0.16)
        if sp:
            _bar(ax, y, sp[0], wilson(sp[0], sp[1]), ORANGE, False, text=f"{sp[0]:.2f} (n {sp[1]})", dy=-0.16)
    ax.plot([], [], marker="o", color=BLUE, linestyle="none", label="committee agrees")
    ax.plot([], [], marker="o", color=ORANGE, markerfacecolor="white", markeredgewidth=1.8, linestyle="none", label="committee splits")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    for ax in axes:
        ax.set_ylim(-0.7, n - 0.3)
    fig.suptitle("A committee of programs: a small gain in score, and a signal of when it is wrong, across four environments",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.995)
    caption = (
        "Bars are 95% intervals: Wilson on items (ARC, BioProt), the benchmark's bootstrap (ONC, SciGym); the single program on ONC shows the range over programs. "
        "AUROC: bootstrap over items (ARC, ONC), Hanley and McNeil (BioProt, SciGym). Error rates: Wilson. Agreement is zero disagreement among members. "
        "Metric per row: held-out transition accuracy (ARC), Discovery Score (ONC), acceptable-plan rate (BioProt), reaction-set F1 (SciGym). "
        "Wrong: a wrong transition, a world short of full credit or claimed on a null, an unacceptable plan, one minus F1. Every row is in artifacts/uncertainty_table.md."
    )
    fig.text(0.01, 0.01, textwrap.fill(caption, 215), fontsize=7.2, color=MUTED, va="bottom")
    fig.subplots_adjust(left=0.165, right=0.99, top=0.87, bottom=0.2, wspace=0.12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    fig.savefig(OUT.with_suffix(".svg"))
    print(f"wrote {OUT}.png and .svg with {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
