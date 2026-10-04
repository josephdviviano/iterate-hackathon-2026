"""Figures for the report: the five questions, the environment and committee-size ablations,
and what agreement, disagreement and a shared blind spot look like on the frames."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .cegis import report  # noqa: E402
from .evaluate import members_from  # noqa: E402
from .experiment import ARTIFACTS, condition_dir  # noqa: E402
from .loader import build_buffer, temporal_split  # noqa: E402

LEVELS = [("ar25", 3), ("m0r0", 3), ("sk48", 2), ("ar25", 7), ("ls20", 3), ("ka59", 2), ("g50t", 1)]
# OPINE-World's frame palette, so the frames look as they do in its viewer.
PALETTE = ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000", "#E53AA3", "#FF7BCC",
           "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00", "#FF851B", "#921231", "#4FCC30", "#A356D6"]
CMAP = matplotlib.colors.ListedColormap(PALETTE)
NORM = matplotlib.colors.BoundaryNorm(np.arange(-0.5, 16.5, 1), 16)
BLUE, ORANGE, GREY, GREEN, RED = "#1E6FD9", "#E8871E", "#8A8A8A", "#2E9E5B", "#D23B3B"


def _label(g: str, l: int) -> str:
    return f"{g} L{l}"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _style(ax, title: str, ylabel: str = ""):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_accuracy(summary: dict, out: Path) -> Path:
    """Question 1 and the environment ablation: single programs, vote and best member per level, with
    the same committees' vote in the object contract."""
    rows = summary["rows"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, r in enumerate(rows):
        ax.scatter([i] * len(r["singles"]), r["singles"], color=GREY, s=28, zorder=3, label="single program (3 runs)" if i == 0 else None)
        g, l = r["level"].split(" L")
        e = _load(condition_dir(g, int(l), 0.4, "committee_devin") / "evaluation.json")
        if e:
            ax.scatter(i, e["committee"]["vote_accuracy"], marker="x", color=RED, s=60, zorder=3,
                       label="committee vote, object contract (R4, R27)" if i == 0 else None)
    ax.plot(x, [r["member_max"] for r in rows], "o", mfc="white", mec=BLUE, mew=1.8, ms=9, zorder=4, label="best member")
    ax.plot(x, [r["vote"] for r in rows], "o", color=BLUE, ms=8, zorder=5, label="committee vote (8 members)")
    ax.set_xticks(x, [r["level"] for r in rows])
    ax.set_ylim(0.3, 1.03)
    _style(ax, "Held-out accuracy per level, OPINE-World's environment (R34)", "accuracy")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "q1_accuracy.png", dpi=160)
    plt.close(fig)
    return out / "q1_accuracy.png"


def fig_flag(summary: dict, out: Path) -> Path:
    """Question 3 and the ablations: unanimous against split error per level, AUROC against K, and the
    object-contract baselines."""
    rows = summary["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 2, 1.6]})
    ax = axes[0]
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w / 2, [r["unanimous_err"] for r in rows], w, color=BLUE, label="unanimous transitions")
    ax.bar(x + w / 2, [r["split_err"] or 0 for r in rows], w, color=ORANGE, label="split transitions")
    for i, r in enumerate(rows):
        ax.text(i - w / 2, (r["unanimous_err"] or 0) + 0.015, f"n={r['unanimous_n']}", ha="center", fontsize=7.5, color="#333")
        ax.text(i + w / 2, (r["split_err"] or 0) + 0.015, f"n={r['split_n']}", ha="center", fontsize=7.5, color="#333")
    ax.set_xticks(x, [r["level"] for r in rows], rotation=20)
    ax.set_ylim(0, 1.0)
    _style(ax, "Error rate when the committee agrees against when it splits", "error rate")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    p = summary["pooled_unanimous"], summary["pooled_split"]
    ax.text(0.99, 0.97, f"pooled: unanimous {p[0]['error']:.3f} (n {p[0]['n']}), split {p[1]['error']:.3f} (n {p[1]['n']})\n"
            f"AUROC {summary['pooled_auroc']['auroc']} [{summary['pooled_auroc']['ci95'][0]}, {summary['pooled_auroc']['ci95'][1]}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    ax = axes[1]
    for r in rows:
        g, l = r["level"].split(" L")
        ks = _load(condition_dir(g, int(l), 0.4, "committee_opine_devin") / "k_sweep.json")
        if not ks:
            continue
        pts = [(b["k"], b["auroc"][0], b["auroc"][1] or 0) for b in ks["by_k"] if b["auroc"][0] is not None]
        if len(pts) < 2:
            continue
        ax.errorbar([k for k, _, _ in pts], [a for _, a, _ in pts], yerr=[s for _, _, s in pts], marker="o", ms=4,
                    capsize=2, label=r["level"], linewidth=1.3)
    ax.axhline(0.5, color=GREY, linestyle=":", linewidth=1)
    ax.set_xticks([2, 4, 8])
    ax.set_ylim(0.2, 1.05)
    _style(ax, "AUROC against committee size K", "AUROC, disagreement vs error")
    ax.set_xlabel("K (mean and sd over member subsets)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax = axes[2]
    base = [("committee\n(K=8)", 0.75, BLUE), ("judge model\nconfidence", 0.71, GREY), ("bagged\ntrees", 0.47, GREY), ("MLP\nensemble", 0.48, GREY)]
    ax.bar(range(4), [v for _, v, _ in base], color=[c for _, _, c in base])
    ax.set_xticks(range(4), [n for n, _, _ in base], fontsize=8)
    ax.axhline(0.5, color=GREY, linestyle=":", linewidth=1)
    ax.set_ylim(0.2, 1.05)
    _style(ax, "Baselines, object contract\n(R9 to R11, 2 levels)", "AUROC")
    fig.tight_layout()
    fig.savefig(out / "q3_flag.png", dpi=160)
    plt.close(fig)
    return out / "q3_flag.png"


def fig_conformal(calibration: dict, out: Path) -> Path:
    """Question 2: coverage against the target, abstain rate and singleton accuracy per level."""
    per = calibration["per_level"]
    names = list(per)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    w = 0.38
    ax.bar(x - w / 2, [per[n]["aci"]["coverage"] for n in names], w, color=BLUE, label="coverage")
    ax.bar(x + w / 2, [per[n]["aci"]["abstain_rate"] for n in names], w, color=ORANGE, label="abstain rate")
    ax.axhline(0.9, color=BLUE, linestyle="--", linewidth=1)
    ax.text(-0.45, 0.905, "target 0.90", ha="left", va="bottom", fontsize=8, color=BLUE)
    for i, n in enumerate(names):
        ax.text(i - w / 2, per[n]["aci"]["coverage"] + 0.01, f"{per[n]['aci']['coverage']:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylim(0, 1.1)
    _style(ax, f"Adaptive conformal sets, target 0.90, pooled coverage {calibration['aci_pooled']['coverage']} (R35)", "rate")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    ax = axes[1]
    ax.bar(x, [per[n]["aci"]["mean_set_size"] for n in names], color=GREY, width=0.5)
    for i, n in enumerate(names):
        acc = per[n]["aci"]["singleton_accuracy"]
        ax.text(i, per[n]["aci"]["mean_set_size"] + 0.04, f"singleton\nacc {acc:.2f}" if acc is not None else "", ha="center", fontsize=7)
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylim(0, 3)
    _style(ax, "Mean set size (abstention counts one extra)", "next states in the set")
    fig.tight_layout()
    fig.savefig(out / "q2_conformal.png", dpi=160)
    plt.close(fig)
    return out / "q2_conformal.png"


def fig_rounds(out: Path) -> Path:
    """Questions 4 and 5: round 1, the mechanism and object-diff rounds and the passive control on the
    held-out rows no arm trained on, and the live round."""
    arms_by_level = {}
    for g, l in [("m0r0", 3), ("ka59", 2), ("sk48", 2)]:
        r = report(g, l, 0.4, "committee_opine_devin", "cegis_opine_devin", "passive_opine_devin",
                   ["cegis_opine_devin", "cegisobj_opine_devin"])
        arms = {}
        for name, a in r["arms"].items():
            key = ("round 1" if name == "round1" else "passive, no counterexample" if name.startswith("passive")
                   else "object-diff counterexample" if "cegisobj" in name else "mechanism counterexample")
            arms[key] = a["vote"]
        arms_by_level[f"{_label(g, l)} ({r['n_common']} rows)"] = arms
    live1 = _load(ARTIFACTS / "ar25" / "live" / "ar25_L3_f40_committee_opine_devin_seed0.json")
    live2 = _load(ARTIFACTS / "ar25" / "live" / "ar25_L3_f40_probe0_live70_live_opine_devin_seed0.json")
    if live1 and live2:
        arms_by_level["ar25 L3 live (75 moves)"] = {"round 1": live1["vote_accuracy"], "mechanism counterexample": live2["vote_accuracy"]}
    order = ["round 1", "mechanism counterexample", "object-diff counterexample", "passive, no counterexample"]
    colors = [GREY, BLUE, "#7FB2F0", GREEN]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    levels = list(arms_by_level)
    x = np.arange(len(levels))
    w = 0.2
    for j, (arm, c) in enumerate(zip(order, colors)):
        vals = [arms_by_level[lv].get(arm) for lv in levels]
        xs = [x[i] + (j - 1.5) * w for i in range(len(levels)) if vals[i] is not None]
        ys = [v for v in vals if v is not None]
        ax.bar(xs, ys, w, color=c, label=arm)
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, levels)
    ax.set_ylim(0, 1.12)
    _style(ax, "Resynthesis after the probes: vote accuracy on the rows no arm trained on (R36)", "accuracy")
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "q4q5_rounds.png", dpi=160)
    plt.close(fig)
    return out / "q4q5_rounds.png"


def _draw(ax, frame, title, diff_from=None):
    ax.imshow(np.array(frame), cmap=CMAP, norm=NORM, interpolation="nearest")
    if diff_from is not None:
        d = np.argwhere(np.array(frame) != np.array(diff_from))
        for y, x in d:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=RED, linewidth=1.2))
        title += f"\n{len(d)} cells differ" if len(d) else "\nexact"
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_examples(game: str, level: int, picks: list[tuple[int, str]], out: Path, name: str) -> Path:
    """Rows of held-out transitions: before, observed after, and each distinct predicted after frame
    with the share of members behind it; differing cells outlined."""
    train, test = temporal_split(build_buffer(game), level, 0.4)
    members = members_from(condition_dir(game, level, 0.4, "committee_opine_devin"), train, test, game)
    k = len(members)
    ncol = 2 + max(len(Counter(json.dumps(m.test_preds[i]) for m in members)) for i, _ in picks)
    ncol = min(ncol, 5)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(2.3 * ncol, 2.5 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, (i, caption) in enumerate(picks):
        t = test[i]
        groups = Counter(json.dumps(m.test_preds[i]) for m in members)
        _draw(axes[row][0], t.before_grid, f"{caption}\nstep {t.step}, before, action {json.dumps(t.action)}")
        _draw(axes[row][1], t.after_grid, "observed after")
        for col, (key, n) in zip(range(2, ncol), groups.most_common()):
            pred = json.loads(key)
            right = pred == t.after_grid
            _draw(axes[row][col], pred, f"predicted by {n} of {k} ({'right' if right else 'wrong'})", diff_from=t.after_grid)
        for col in range(2 + len(groups), ncol):
            axes[row][col].axis("off")
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    return out / name


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Figures for the report.")
    parser.add_argument("--out", default="artifacts/figures")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path("artifacts/summary_opine.json").read_text())
    calibration = json.loads(Path("artifacts/calibration_opine.json").read_text())
    made = [fig_accuracy(summary, out), fig_flag(summary, out), fig_conformal(calibration, out), fig_rounds(out)]
    e = json.loads((condition_dir("sk48", 2, 0.4, "committee_opine_devin") / "evaluation.json").read_text())["committee"]["per_transition"]
    split_right = max((i for i, r in enumerate(e) if r["n_distinct"] > 1 and r["correct"]), key=lambda i: e[i]["uniform_disagreement"])
    una_right = next(i for i, r in enumerate(e) if r["n_distinct"] == 1 and r["correct"])
    una_wrong = next(i for i, r in enumerate(e) if r["n_distinct"] == 1 and not r["correct"])
    made.append(fig_examples("sk48", 2, [(split_right, "disagreement, vote right"), (una_right, "agreement, right"),
                                         (una_wrong, "agreement, wrong: shared blind spot")], out, "examples_sk48.png"))
    e = json.loads((condition_dir("ka59", 2, 0.4, "committee_opine_devin") / "evaluation.json").read_text())["committee"]["per_transition"]
    split_wrong = max((i for i, r in enumerate(e) if r["n_distinct"] > 1 and not r["correct"]), key=lambda i: e[i]["uniform_disagreement"], default=None)
    split_right = max((i for i, r in enumerate(e) if r["n_distinct"] > 1 and r["correct"]), key=lambda i: e[i]["uniform_disagreement"])
    picks = [(split_right, "disagreement, vote right")] + ([(split_wrong, "disagreement, vote wrong")] if split_wrong is not None else [])
    made.append(fig_examples("ka59", 2, picks, out, "examples_ka59.png"))
    for p in made:
        print(p)


if __name__ == "__main__":
    main()
